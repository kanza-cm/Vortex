# REPORT — lavoro di supporto "perf 2 PC" (mpi_fissa)

Aggiornato: fine lavoro. Tutti i file sono in
`C:\Users\zorka\AppData\Local\Temp\mpi_work\perf\`.

Vincoli rispettati: nessuna scrittura nel workspace WSL, nessuna compilazione,
nessun `mpirun`/`ssh`. Tutto viene da lettura di file (`read`/`glob`/`grep`) o da
calcolo chiuso sul sorgente. Nota tecnica: `pwsh` non riesce ad aprire il path
UNC del workspace (`GetNamedSecurityInfoW failed (Win32 1)`), quindi **non e'
stato eseguito nulla** — nemmeno `git`, `wc`, `bash -n`.

## File prodotti

| file | cosa | stato |
|---|---|---|
| `OTIMIZZAZIONI.md` | analisi scambi ghost / V-cycle / `correct_faces` / allreduce; conteggio chiuso per passo **validato** su 3 `timing.dat` reali; 10 ottimizzazioni quantificate (O1-O10) + cosa non fare; stima scaling L8 | completo |
| `run_2pc.sh` | sostituto di `mpi_fissa/scripts/run_2pc.sh`: pre-flight (ssh, `orted`, versione OpenMPI, `$HOME`, md5 del binario), test `mpirun -np 2 hostname`, log su file, riepilogo da `timing.dat` | completo |
| `hostfile_2pc.txt` | `localhost slots=1` + `slave1 slots=1`, con il perche' di ogni riga | completo |
| `README_2PC.md` | testo sostitutivo delle sezioni 4.4 e 5 del README di `mpi_fissa` (percorsi `~/progetto/mpi_fissa`, mirrored networking, ssh, firewall porte MPI, lettura di `timing.dat`) | completo |
| `CHECKLIST_2PC.md` | checklist passo-passo in 7 fasi + rubrica errori (sintomo → causa → fix) | completo |
| `benchmark.sh` | campagna np1 vs np2: `-poisson_nit 1/2/3` × 3 ripetizioni, `timing.dat`, `nvidia-smi` dei due PC, hostfile configurabile, `out/perf_2pc/tabella.txt` + `dati.csv` + `note.txt` | completo |
| `mpi_pingpong.c` | (extra) latenza/banda reali fra i due PC per le taglie del solver: e' il test che separa "rete lenta" da "pack lento" | completo |

Nessuno degli script e' stato eseguito (non c'e' shell utilizzabile). Sono
scritti in modo difensivo (pre-flight, `timeout`, fallback senza `run_2pc.sh`),
ma la prima esecuzione va fatta come dice `CHECKLIST_2PC.md`, non alla cieca.

---

## 1. I cinque risultati che cambiano le decisioni

### 1.1 A L8 si scambiano 25.7 MB e 388 messaggi per passo, non 13-20 MB

Conteggio chiuso dal sorgente, per `POISSON_NIT` e per passo:

| voce | NIT=1 | NIT=2 (default) | NIT=3 |
|---|---|---|---|
| halo `u` (4 chiamate, 5 piani) | 7.86 MB / 4 scambi | 7.86 / 4 | 7.86 / 4 |
| pressione (V-cycle) | 8.39 / 94 | 16.77 / 188 | 25.16 / 282 |
| `bc_pressure` post-proiezione | 1.05 / 2 | 1.05 / 2 | 1.05 / 2 |
| **totale** | **17.3 MB / 100 scambi** | **25.7 MB / 194 scambi** | **34.1 MB / 288** |
| messaggi point-to-point | 200 | **388** | 576 |

Il modello riproduce **esattamente** il numero di scambi (e i byte entro il 5%,
sempre per difetto) in tre `timing.dat` reali del workspace:

| run | modello | misurato |
|---|---|---|
| `out/L8_np2` (NX=256, 1 passo) | 196 scambi / 27.8 MB | **196 / 28.3 MB** |
| `out/m2_128` (NX=128, 1 passo) | 172 / 6.94 | **172 / 7.1** |
| `out/g2` (NX=64, 1 passo) | 148 / 1.70 | **148 / 1.8** |

La struttura che genera i numeri: `smooth_lvl` chiama `bc_pressure` **dentro**
il ciclo sulle parita' (2 scambi per sweep) e il V-cycle ha 8 livelli
distribuiti (47 scambi per V-cycle, 4 V-cycle per passo, 2 proiezioni). I
livelli grossolani costano 2 messaggi ciascuno e muovono 4 KB in tutto: **il 36%
degli scambi (17 su 47) trasporta 4 KB e potrebbe essere replicato** invece che
distribuito (O4).

### 1.2 I "2-8 MB, 151-175 scambi, comm 0.01-0.09 s" sono un run a **L6**

`out/m2_128` (8.4M celle) misura 7.1 MB / 172 scambi, e 7.1 MB a ~100 MB/s fanno
0.07 s. A L8 lo stesso tempo implicherebbe 300-2800 MB/s. Verifica in 5 secondi:

```sh
head -3 out/<run>/diagnostics.dat    # colonna cells: 67108864 = L8, 8388608 = L6
```

### 1.3 I due numeri del Lead a L8 implicano `comm ≈ 0.52 s` (42% del passo)

```
T(np=1) = 1.407 s                            [dato del Lead]
T(np=2) = 1.407/2 + comm = 0.704 + 0.522 = 1.226 s   [combacia col dato]
speedup = 1.15x        comm = 42% del passo a 2 PC
```
E' coerente con 25.7 MB a ~49 MB/s "tutto compreso" e con il `comm_s = 0.538 s`
di `out/L8_np2`; **non** e' coerente con 0.01-0.09 s.

### 1.4 `comm_s` **non e' (quasi tutta) rete**: e' in buona parte attesa della GPU

`out/L8_np2` e' un run a 2 rank sulla **stessa** GPU (loopback): `comm_s` =
0.538 s su 0.5595 s di wall. Ma il costo *marginale* del secondo rank e' solo
`0.5595 − 0.3466 = 0.213 s` (≈1.1 ms/scambio, con rete locale ~0). Il resto e'
tempo di GPU che il cronometro si attribuisce perche' **ogni `pack_plane` fa una
`cudaMemcpy2D` sincrona** (256 DMA da 1 KB a L8). Prova indipendente: la build
**CPU** fa 172 scambi in **4.8 ms** (28 µs/scambio), la build GPU a NX=64 ne fa
148 in **82.5 ms** (557 µs/scambio).

Conseguenza: prima di ottimizzare la rete va strumentato `comm_s`
(`OTIMIZZAZIONI.md` O1: separare `T_PACK`, `T_MPI`, e i delta per passo di
`comm_MB`/`n_exch` che oggi sono **cumulativi** in `timing.dat`).

### 1.5 Il tetto realistico a L8 e' ~1.5x, e conviene di piu' sul solver vero

Modello a due termini tarato sul dato del Lead (`t_l` = costo fisso per scambio,
`BW` = banda):

```
comm = n_scambi · t_l + MB / BW          (t_l = 0.9 ms, BW ≈ 75 MB/s)
```

| configurazione | scambi | MB/passo | comm | `T(np=2)` | speedup |
|---|---|---|---|---|---|
| oggi (NIT=2) | 194 | 25.7 | 0.50 s | 1.20 s | **1.17x** (= dato Lead 1.15x) |
| `-poisson_nit 1` | 100 | 17.3 | 0.32 s | 1.02 s | 1.37x |
| + un solo `Sendrecv` per R=2 | 100 | 17.3 | 0.29 s | 0.99 s | 1.42x |
| + replica del coarse da `l=5` + niente scambio di `GRD` | 63 | 16.2 | 0.25 s | 0.96 s | 1.47x |
| + pack contiguo | 63 | 16.2 | 0.24 s | 0.94 s | 1.50x |

Dopo le prime due ottimizzazioni **dominano i byte** (16.2 MB / 75 MB/s = 0.22 s
di pavimento): il batch di sweeps (O8, il piu' rischioso) vale ~0.01 s e
**non conviene**. Il valore sicuro della decomposizione resta la **memoria**
(67M → 134M celle); per la velocita' il numero da guardare e' `saetta3d`
(2.8 s/passo): con lo stesso `comm` lo speedup e' **1.47x oggi e 1.70x dopo
O2-O4**, molto meglio del prototipo perche' il prototipo e' troppo veloce e ha
un rapporto comm/calcolo sfavorevole.

---

## 2. Cosa ho corretto nella parte 2-PC (e perche')

| punto | prima | adesso | perche' |
|---|---|---|---|
| percorsi | `~/mpi_fissa` | `~/progetto/mpi_fissa`, identico sui due PC | e' il percorso che usa gia' `cluster/push_project.sh`; il workspace locale contiene uno spazio e non va usato |
| pre-flight | nessuno | ssh batch, `orted`, versione OpenMPI **uguale**, `$HOME` uguale, **md5 del binario** uguale | 5 secondi invece di scoprire a meta' run che la NX e' diversa o che `orted` manca |
| test MPI | manuale nel README | `mpirun -np 2 hostname` **dentro** lo script | e' il test che distingue "ssh ok" da "MPI ok" |
| firewall | solo porta 22 | **porte dati MPI 50000-51023** documentate + MCA per fissarle | con la sola 22 aperta `hostname` funziona e il **run resta appeso**: e' il guasto che fa perdere piu' tempo |
| `hostfile` | 2 righe | 2 righe + commenti | `slots=1` non e' opzionale (senza, 2 rank finiscono sul master) |
| `mpirun` | `exec mpirun` | `mpirun` + log su file + riepilogo | con `exec` non si vede nulla se il run muore |
| `plm_rsh_args` | assente | `BatchMode=yes -o ConnectTimeout=10` | niente prompt appesi |
| GPU diverse | non menzionato | nota su `-gencode`/PTX in checklist | un binario `sm_86` non parte su una GPU diversa |
| `timing.dat` | letto come per-passo | documentato che `comm_MB`/`n_exch` sono **cumulativi** e che l'ultimo passo contiene la diagnostica | sono due trappole che falsano ogni confronto |

## 3. Come si usa (ordine consigliato)

1. copiare i 4 file nel progetto sui due PC (tabella in testa a
   `CHECKLIST_2PC.md`), `chmod +x scripts/*.sh`;
2. seguire `CHECKLIST_2PC.md` fasi 1-5 (una volta, ~1 h): rete, ssh, progetto,
   firewall porte MPI, hostfile, `mpi_pingpong`;
3. fase 6: un run a 1 GPU e uno a 2 PC, confronto campi a 1e-5;
4. fase 7: `NX=256 ./scripts/benchmark.sh` → `out/perf_2pc/tabella.txt`;
5. solo dopo: `OTIMIZZAZIONI.md` §7 (O1 strumentazione, poi O2-O5).

Nota sul "1 passo, 3 ripetizioni": il default di `benchmark.sh` e' `TMAX=0.1`,
cioe' 5 passi a L8 con l'ultimo scartato (contiene la diagnostica di fine run) —
4 passi utili per ripetizione, mediana. E' piu' robusto di un singolo passo, che
include l'avvio a freddo e la diagnostica finale. Per il letterale "1 passo":
`FAST=1` (o `TMAX=0.02`), con l'avvertenza che quel valore e' gonfiato.

## 4. Cosa non e' stato verificato (dichiarazione esplicita)

- **Nessuno script e' stato eseguito**: niente shell sul workspace. Errori di
  sintassi bash restano possibili; la prima esecuzione va fatta a mano come in
  checklist, non in un loop di produzione.
- **Tutte le stime di tempo sono modello**, non misure. Ogni numero in
  `OTIMIZZAZIONI.md` e' etichettato [DATO] / [MODELLO] / [STIMA].
- Le proposte di modifica al sorgente (O1-O11) descrivono *dove* e *quanto*, non
  sono patch: nessuna riga di `mpi_fissa.cu` e' stata toccata.
- La stima "`comm = 0.52 s`" e' **dedotta** dalla coppia di tempi del Lead, non
  misurata: se la media di 1.407 s/passo include la diagnostica, il conto cambia
  (§5 punto 1).

## 5. Domande aperte per il Lead

1. **`out/L8_np1` = 0.347 s/passo contro 1.407 s/passo del Lead** (stesso L8,
   4x). Ipotesi: la media include la diagnostica (`-diag 1` di default → 50
   passi per output; `download_u` = 3×137 MB per rank non pinnati + loop host
   single-thread su 34.4M celle), oppure macchina/`-tmax` diversi. Da chiarire:
   tutte le stime di speedup poggiano su 1.407 s.
2. **I "2-8 MB / 151-175 scambi"** vengono da un run a L6: confermare con la
   colonna `cells` di `diagnostics.dat`.
3. **La GPU del secondo PC e' lo stesso modello della prima?** Se no serve il
   PTX (`-gencode`) o un `ARCH` per macchina, altrimenti il rank sul secondo PC
   muore con "no kernel image is available".
4. **Il dump finale (`omega3d_matrix.bin`, 268 MB/run a L8) serve in campagna?**
   `benchmark.sh` lo manda su `/dev/null` per default (`KEEP_DUMP=1` per
   tenerlo): il `MPI_Gather` costa 1-3 s per run e 4.8 GB su disco in 18 run.

---

## Appendice — numeri di riferimento usati

| grandezza | valore | fonte |
|---|---|---|
| griglia L8 | 256×256×1024 = 67 108 864 celle | `out/L8_np*/diagnostics.dat` (colonna `cells`) |
| celle per piano di interfaccia (L8) | 65 536 (256 KB) | `mpi_fissa.cu` |
| livelli MG distribuiti (L8, 2 rank) | 8 (`LD=7`) + 1 replicato `1×1×4` | `mg_init`, righe 784-839 |
| scambi per V-cycle | 47 | §1.2 di `OTIMIZZAZIONI.md` |
| np1 / np2 a L8, 1 passo | 0.3466 s / 0.5595 s | `out/L8_np1`, `out/L8_np2` `timing.dat` |
| comm np2 locale (2 rank, 1 GPU) | 0.5384 s / 196 scambi / 28.3 MB | `out/L8_np2/timing.dat` |
| `mgp_resa` (NIT=2, L8) | 78.2 (identico a np1 e np2) | `out/L8_np*/diagnostics.dat` |
| np1 / np2 a L8 (2 PC) | 1.407 / 1.226 s/passo | misura del Lead |
| `saetta3d` a L8, 1 GPU | 2.8 s/passo | `DESIGN.md` §3 (da `run7/log.txt`) |
| banda Wi-Fi attesa | 40-150 MB/s | `cluster/README.md` §5 |
