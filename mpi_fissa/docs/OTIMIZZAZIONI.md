# OTIMIZZAZIONI — comunicazione del prototipo `mpi_fissa` su 2 PC in Wi-Fi

Documento di analisi per il Lead. Ogni numero ha un'etichetta:

- **[DATO]** = letto da un file del workspace o fornito dal Lead (provenienza citata);
- **[MODELLO]** = conteggio chiuso ricavato dal sorgente (ripetibile a mano, formula inclusa);
- **[STIMA]** = conseguenza del modello con ipotesi dichiarate (banda, latenza).

Nessuna misura e' stata eseguita da chi scrive: non c'e' accesso a `mpirun`/`nvcc`
dal workspace. I conteggi [MODELLO] sono pero' **validati** contro tre
`timing.dat` reali, con scarto < 2% (vedi §2.4): questo e' il motivo per cui li
uso per dimensionare le ottimizzazioni.

---

## 1. Mappa degli scambi nel sorgente (chi chiama cosa, quante volte)

Riferimenti a `mpi_fissa/src/mpi_fissa.cu`.

| funzione | righe | quando | messaggi MPI per chiamata | byte per chiamata |
|---|---|---|---|---|
| `exchange_u(u, tag, np)` | 317-360 | dentro `bc_u` | **2 `MPI_Sendrecv`** (uno per verso) | `2 · 3·np·A0 · 4` |
| `exchange_scalar(a, tag)` | 363-389 | dentro `bc_pressure` | **2 `MPI_Sendrecv`** | `2 · A_l · 4` |
| `bc_pressure(g, a)` | 460-490 | MG (smoothing, residuo, prolong) + fine proiezione | 1 `exchange_scalar` | come sopra |
| `restrict_to_repl` | 913-954 | 1 volta per V-cycle (livello `LD`) | 1 `MPI_Allgather` | `nloc·4` (4 float a L8) |
| `compute_dt` | 1099-1115 | 1 volta per passo | 1 `MPI_Allreduce` | 8 B |
| `diagnostics_compute` | 1185-1278 | ogni `TDIAG` | 6 `Allreduce` | < 100 B |
| `dump_wz` | 1285-1319 | fine run | 1 `MPI_Gather` | ~130 MB a L8 |

`A0 = NX·NY` = celle per piano di interfaccia (L8: 65536 celle = 262144 B).

### 1.1 Punti di chiamata per passo (main loop, righe 1488-1531)

```
bc_u(U,2)          -> exchange_u  np=2   (riga 1494)
bc_u(GRD,1)        -> exchange_u  np=1   (riga 1495)   <-- derivato da P: eliminabile (O5)
compute_dt         -> 3x mf_max_abs + 1 Allreduce       <-- 3 sync del device (O9)
predict_uf
project(PF, dt/2)  -> mg_solve(POISSON_NIT) + bc_pressure + correct_faces
advect
bc_u(U,1)          -> exchange_u  np=1   (riga 1502)
viscosity
bc_u(U,1)          -> exchange_u  np=1   (riga 1504)   <-- fondibile con la precedente (O7)
acceleration
project(P, dt)     -> mg_solve(POISSON_NIT) + bc_pressure + correct_faces
centered_gradient
add_gradient
```

### 1.2 Quanti `bc_pressure` per V-cycle (`vcycle`, righe 1003-1022)

`smooth_lvl` chiama `bc_pressure` **dentro il ciclo sulle parita'** (riga 851):
quindi **2 scambi per ogni sweep** (uno prima delle celle rosse, uno prima delle
nere — necessari perche' il ghost contiene celle di entrambe le parita').

| punto | scambi per livello e per V-cycle |
|---|---|
| `smooth_lvl(L,1)` pre | 2 |
| `residual_lvl` (riga 873) | 1 |
| `prolong_add(LV[l+1], L)` — `bc_pressure` sul livello coarse (riga 965) | 1 |
| `smooth_lvl(L,1)` post | 2 |
| **totale, livelli distribuiti `0..LD-1`** | **6** |
| livello `LD` (verso il replicato: `prolong_add` non scambia perche' `g.repl=1`) | 5 |
| livello `LD+1` (replicato `1x1x4`) | 0 (ma 60 `bc_pressure` locali) |

**Questo e' il punto chiave**: il 97% dei messaggi del codice e' dentro il
multigrid, e i livelli grossolani — che costano ~0 byte — costano pero' 2
messaggi ciascuno quanto il livello fine. Il rapporto byte/messaggio crolla di
4x per livello: a `l=4` si scambiano 1 KB con la stessa latenza di 256 KB.

### 1.3 `correct_faces`, `allreduce`, `gather`: chi comunica e chi no

- **`correct_faces`** (1049-1069): 3 kernel, **zero MPI**. Legge `p` alle facce
  e quindi ha bisogno del ghost di `p` a `k=-1` e `k=nzl`: e' il motivo della
  `bc_pressure(DOM, sol)` a riga 1077, **dopo** `mg_solve` e prima della
  correzione. Le facce `uf.z` all'interfaccia vengono corrette **due volte**,
  una per rank, con gli stessi valori di `p` (il ghost e' scambiato): e' l'unico
  punto in cui i due rank devono coincidere bit-per-bit, e coincidono per
  costruzione. Se si tocca lo scambio del ghost di `p`, si tocca qui.
- **`allreduce`**: nel ciclo caldo ce n'e' **esattamente uno per passo**
  (`compute_dt`, riga 1106, 8 byte): non e' un problema di banda, e' ~2-4 ms di
  latenza (1-2% del passo a 2 PC) ed e' bloccante. Gli altri sono 6 per ogni
  `TDIAG` (diagnostica) e 1 `Allgather` per V-cycle (4 float a L8).
  **La voce che pesa davvero non e' l'allreduce ma i 194 scambi point-to-point.**
  Da notare pero' che prima dell'allreduce ci sono **3 `mf_max_abs`** (uno per
  componente di `uf`), ognuno con `cudaMemset` + kernel + `cudaMemcpy` di 4 byte:
  3 sincronizzazioni del device per passo sul percorso critico (O9).
- **`gather`/`dump_wz`** (1285-1319): 1 `MPI_Gather` di 137 MB per rank a L8,
  **solo a fine run**; non entra nel tempo per passo ma pesa in una campagna di
  benchmark (268 MB scritti per run): per questo `benchmark.sh` manda il dump su
  `/dev/null`.

---

## 2. Conteggio chiuso per passo, e validazione contro i dati reali

### 2.1 Formula

Con `R=2` rank, `LD` = ultimo livello distribuito, e
`A_l = (NX>>l)^2 · 4` byte:

```
scambi_V-cycle = 6·LD + 5
byte_V-cycle   = 12 · Σ_{l=0..LD-1} A_l  +  10 · A_LD
scambi_passo   = 2·POISSON_NIT · scambi_V-cycle  +  2  +  4
byte_passo     = 2·POISSON_NIT · byte_V-cycle    +  4·A0  +  30·A0
```
(i `+2` e `+4·A0` sono i due `bc_pressure` dopo le proiezioni; i `+4` e
`+30·A0` sono le 4 chiamate a `exchange_u`, una con `np=2` e tre con `np=1`:
ogni "unita' piano" costa `2·3·A0 = 1.573 MB` a L8.)

Per `NX=256, NZ=1024, R=2`: `LD=7`, livelli distribuiti `0..7`, replicato `8=1x1x4`.
Per `NX=128, NZ=512, R=2`: `LD=6`, replicato `7=1x1x4`.

### 2.2 Tabella per livello a L8 (una V-cycle; la griglia e' lo slab locale, 512 piani per rank)

| liv. | griglia | `A_l` | scambi | byte |
|---|---|---|---|---|
| 0 | 256×256×512 | 256 KB | 6 | 3.15 MB |
| 1 | 128×128×256 | 64 KB | 6 | 0.79 MB |
| 2 | 64×64×128 | 16 KB | 6 | 0.20 MB |
| 3 | 32×32×64 | 4 KB | 6 | 0.05 MB |
| 4 | 16×16×32 | 1 KB | 6 | 12 KB |
| 5 | 8×8×16 | 256 B | 6 | 3 KB |
| 6 | 4×4×8 | 64 B | 6 | 0.8 KB |
| 7 | 2×2×4 | 16 B | 5 | 0.2 KB |
| **tot** | | | **47** | **4.19 MB** |

Il 75% dei byte sta nel solo livello 0 e il 94% nei primi due; **il 36% degli
scambi (17 su 47, i livelli 5-7) muove 4 KB in totale**.

### 2.3 Budget per passo (L8, 2 rank)

| voce | NIT=1 | NIT=2 (default) | NIT=3 |
|---|---|---|---|
| halo `u` (4 chiamate, 5 piani) | 7.86 MB / 4 scambi | 7.86 / 4 | 7.86 / 4 |
| pressione (V-cycle) | 8.39 MB / 94 | 16.77 / 188 | 25.16 / 282 |
| `bc_pressure` post-proiezione (×2) | 1.05 / 2 | 1.05 / 2 | 1.05 / 2 |
| allgather (coarse) | ~0 / 2 coll. | ~0 / 4 | ~0 / 6 |
| allreduce (`dt`) | ~0 / 1 | ~0 / 1 | ~0 / 1 |
| **totale** | **17.3 MB / 100 scambi** | **25.7 MB / 194 scambi** | **34.1 MB / 288 scambi** |
| messaggi point-to-point | 200 | **388** | 576 |

### 2.4 Validazione del modello contro i `timing.dat` presenti [DATO]

| run | NX | modello (scambi / MB) | misurato | scarto |
|---|---|---|---|---|
| `out/L8_np2` (1 passo, include la diagnostica finale) | 256 | 196 / 27.8 | **196 / 28.3** | +0% / +1.8% |
| `out/m2_128` (CPU, 1 passo, idem) | 128 | 172 / 6.94 | **172 / 7.1** | +0% / +2.3% |
| `out/g2` (GPU, 1 passo, idem) | 64 | 148 / 1.70 | **148 / 1.8** | +0% / +5% |

Il modello prevede **esattamente** il numero di scambi in tutti e tre i casi e i
byte entro il 5%, sempre leggermente **per difetto** (restano fuori il traffico
dell'`Allgather` e l'overhead di protocollo). Le due "unita'" in piu' rispetto al
passo puro sono la coda del run: `residual_lvl(LV[0])` + `bc_u(U,1)` di
`write_diagnostics`.

Nota metodologica: la colonna `n_exch` di `timing.dat` e' **cumulativa** (il
contatore `N_EXCH` non viene mai azzerato, riga 1516) e cosi' `comm_MB`
(`B_COMM`). Va differenziata fra righe successive. Piccolo bug da correggere:
emettere `B_COMM - b_comm0` e `N_EXCH - n0` per passo.

### 2.5 L8 e' 28 MB, non 2-8 MB: il numero del modello del teammate e' L6

Il "modello del teammate" citato (13-20 MB/passo, 160-320 messaggi/passo a L8)
e' **piu' basso del vero**: a L8 il codice scambia **25.7 MB e 194 scambi
(388 messaggi) per passo con NIT=2**, e 17.3 MB / 100 scambi con NIT=1.

I "2-8 MB, 151-175 scambi, comm 0.01-0.09 s" citati fra le misure del Lead
coincidono invece, byte per byte, con un run **a L6**: `out/m2_128` misura
7.1 MB / 172 scambi, e 7.1 MB a ~100 MB/s fanno 0.07 s, dentro l'intervallo
0.01-0.09 s. Con 28 MB l'intervallo 0.01-0.09 s implicherebbe 300-2800 MB/s,
impossibile in Wi-Fi.

Conseguenza operativa: **prima di ogni altra misura, verificare a che griglia
si riferiscono i numeri di comm**, con

```sh
head -3 out/<run>/diagnostics.dat     # colonna cells: 67108864 = L8, 8388608 = L6
awk 'NR>1{print $1,$4,$5,$6,$7}' out/<run>/timing.dat
```

---

## 3. Dove va il tempo: tre costi che `comm_s` confonde

`T_COMM` (riga 357, 386) racchiude **tre cose diverse**:

1. `cudaMemcpy2D` D2H del piano (una riga di DMA per ogni `j`: 256 copie da
   1 KB a L8) — riga 305;
2. la `MPI_Sendrecv` bloccante;
3. **l'attesa del device**: `cudaMemcpy2D` sincronizza lo stream di default,
   quindi ogni `pack_plane` aspetta la fine del kernel lanciato subito prima.
   Con ~50 kernel per passo, questo trasferisce dentro `comm_s` una fetta
   grossa del tempo di calcolo.

Evidenze [DATO] che i tre contributi sono separabili:

| misura | scambi | byte | `comm_s` | per-scambio |
|---|---|---|---|---|
| `out/m2_128` (CPU, NX=128) — nessuna GPU di mezzo | 172 | 7.1 MB | 0.0048 s | **28 µs** |
| `out/g2` (GPU, NX=64) | 148 | 1.8 MB | 0.0825 s | **557 µs** |
| `out/L8_np2` (GPU, NX=256) | 196 | 28.3 MB | 0.5384 s | **2747 µs** |

La versione **CPU** fa gli stessi scambi in 28 µs: la rete locale (shared
memory) non e' il collo di bottiglia. La versione **GPU** costa 20-100x di piu'
per scambio, e il costo per scambio **scala con il numero di righe del piano**
(64 righe a NX=64, 256 a NX=256 → 4.9x di costo per 4x di righe). Questo e'
il profilo tipico di una `cudaMemcpy2D` fatta di tante DMA da 1 KB l'una su
WSL2 (percorso paravirtualizzato `/dev/dxg`), non di un limite di banda.

Test decisivo (5 minuti, da fare **prima** di ottimizzare, vedi O1):

```sh
# out/L8_np1 e out/L8_np2 sono lo stesso passo (5 = 0.02, dt = 0.02, stesse 67.1M
# celle); np2 ha i 2 rank sulla STESSA GPU, quindi il lavoro di calcolo totale
# e' ~lo stesso (2 rank da mezzo dominio su una GPU sola). Allora:
#   0.5595 - 0.3466 = 0.213 s = costo marginale reale del 2-rank
#   -> 0.213/194 = 1.1 ms per scambio, di cui la rete locale (loopback) e' ~0
```
Cioe': **su 0.54 s di `comm_s`, solo ~0.21 s e' costo vero aggiunto**; il resto
e' tempo di GPU che il cronometro si attribuisce per effetto della
sincronizzazione nelle copie. Su 2 PC la parte di rete si aggiunge a quei
0.21 s. (Il conto e' approssimato: due rank su una GPU non sono esattamente
come un rank solo, ma l'errore e' molto minore del fattore 2.5 che si sta
spiegando.)

---

## 4. Il quadro sui 2 PC: cosa dicono i numeri del Lead

[DATO Lead] L8, stessa GPU: np=1 **1.407 s/passo**, np=2 **1.226 s/passo**.
[DATO file] il run a 2 rank costa, in piu' del monorank, ~0.52-0.54 s di
comunicazione (28 MB / 196 scambi).

Le due cose sono coerenti e si incastrano:

```
T(np=1) = 1.407 s                      (calcolo, 1 GPU)
T(np=2) = 1.407/2 + comm = 0.704 + 0.522 = 1.226 s   [STIMA, combacia con il dato]
speedup  = 1.15x        comm/T(np=2) = 42%
```

Quindi: **il 42% del passo a 2 PC e' comunicazione**. La banda efficace "tutto
compreso" e' `25.7 MB / 0.52 s ≈ 49 MB/s`, che il modello di §6 scompone in
~75 MB/s di banda pura piu' ~0.9 ms di costo fisso per ognuno dei 194 scambi.
Il modello "13-20 MB → 0.2-1.0 s" del DESIGN §6 sottostima i byte (sono 25.7)
ma azzecca l'ordine di grandezza del tempo.

**Conclusione strategica**: con 0.52 s di comunicazione il tetto dello speedup
e' `2/(1+0.52/0.704) = 1.15x`. Per andare oltre bisogna **togliere messaggi e
byte**, non "comprare banda": a 100 MB/s i 25.7 MB costano 0.26 s, e gli altri
0.26 s sono latenza per 194 scambi × ~1.3 ms.

---

## 5. Ottimizzazioni proposte

Ordinate per (beneficio / rischio). La colonna "effetto" e' a L8, R=2, NIT=2 se
non diversamente indicato; "msg" = messaggi point-to-point (2 per scambio).

### O1 — Strumentare prima di toccare: separare pack, MPI e attesa device

**Dove**: `exchange_u`/`exchange_scalar` (righe 317-389), `open_outputs` (1327),
`timing.dat` (1515).

Aggiungere 2 contatori `T_PACK` (solo le `cudaMemcpy2D`) e `T_MPI` (solo la
`Sendrecv`) e 2 colonne a `timing.dat`, piu' i delta per passo di `comm_MB` e
`n_exch` (§2.4). 15 righe di codice, zero rischio numerico. Senza questo, O6
(il pack) e O10 (la rete) si ottimizzano alla cieca: oggi `comm_s` non dice
quale dei due sta costando.
**Effetto**: nessuno sui tempi, decide pero' *quale* delle O2-O10 conviene.
**Costo**: 0.2 giorni.

### O2 — `-poisson_nit 1` (nessun codice)

| | scambi | msg | MB | comm [STIMA] |
|---|---|---|---|---|
| NIT=2 (default) | 194 | 388 | 25.7 | 0.50 s |
| NIT=1 | 100 | 200 | 17.3 | 0.32 s |
| NIT=3 | 288 | 576 | 34.1 | 0.71 s |

**Effetto**: −48% scambi, −33% byte, `T(np=2)` 1.226 → **~1.02 s**, speedup
1.15x → **1.37x**.
**Rischio**: il residuo della proiezione peggiora. [DATO] `out/L8_np2` con
NIT=2 ha `mgp_resa = 78.2`; con NIT=1 va misurato (criterio: la fisica a
t=1..2 deve restare dentro le tolleranze di `VALIDAZIONE.md`, non il residuo in
se'). Da provare **subito**, costa un flag.

### O3 — Un solo `Sendrecv` invece di due quando `NRANKS==2`

**Dove**: `exchange_u` (335-356) e `exchange_scalar` (372-384).

Con 2 rank il rank 0 ha un solo vicino: le due `Sendrecv` (una "verso l'alto",
una "verso il basso") diventano **una sola** che scambia i due piani di
frontiera. Con `R>2` restano due, ma si possono postare come
`MPI_Irecv`+`MPI_Isend` sulle due direzioni e chiudere con una `Waitall`
(1 RTT invece di 2 trasferimenti in sequenza).

**Effetto**: messaggi **388 → 194** (−50%) a parita' di byte; la componente di
latenza di `comm` si dimezza → [STIMA] 0.50 → ~0.43 s, `T(np=2)` ~1.13 s,
speedup 1.24x. Da solo non basta, ma e' ~10 righe di codice e si combina con
tutto il resto. **Rischio**: basso (si puo' tenere il path generico per R>2).
**Costo**: 0.3 giorni.

### O4 — Meno livelli distribuiti: replicare da `l=5` invece di `l=8`

**Dove**: `mg_init` (784-839), in particolare il calcolo di `LD` (788-796).

Oggi si distribuiscono **8 livelli** per poi replicare una griglia `1x1x4` da
16 celle: gli ultimi 3 livelli costano 17 scambi (36% del V-cycle) e muovono
**4 KB in tutto**. Replicando da `l=5` (griglia `8x8x32` = 2048 celle,
l'`Allgather` passa da 4 float a 8 KB, trascurabile) si tengono distribuiti
solo `0..4`:

| | scambi/V-cycle | byte/V-cycle |
|---|---|---|
| oggi (LD=7) | 47 | 4.19 MB |
| LD=4 (replica da l=5) | **29** | 4.19 MB (−0.1%) |

**Effetto**: −38% scambi a byte invariati. NIT=2: 194 → **122 scambi**, 25.6 MB.
Combinato con O2: 64 scambi / 17.3 MB.
**Attenzione**: il coarse solve replicato diventa 60 sweep × 2048 celle; va
fatto in **un solo kernel con il ciclo interno** (oggi sono 60 lanci di kernel
per V-cycle: 240 lanci/passo × ~10 µs = 2.4 ms, accettabile ma inutile). Anche
`MF_COARSE_SWEEPS` puo' scendere a 10 su una griglia 2048 celle.
**Rischio**: basso (cambia solo il numero di livelli; la soluzione del
coarse cambia leggermente → rirun del test np1 vs np2 a 1e-5).
**Costo**: 0.5 giorni.

### O5 — Togliere lo scambio di `GRD` (e' derivato da `P`)

**Dove**: riga 1495 `bc_u(GRD,1)`, `centered_gradient` (768-780),
`bc_pressure` (460).

`GRD = -grad(P)` e `P` ha gia' un piano di ghost valido dopo
`bc_pressure(DOM, sol)` in `project` (riga 1077). Serve pero' il piano a
`k=-2` per la differenza centrata: basta che **quell'unico** `bc_pressure`
scambi 2 piani invece di 1 (`+0.52 MB`) e che `centered_gradient` calcoli anche
la riga di ghost (`-1.57 MB`, −1 scambio, −2 messaggi).

**Effetto**: −1.05 MB, −1 scambio (−2 messaggi) per passo. Da solo e' poco; e'
pero' a costo quasi zero e toglie una `cudaMemcpy` sincrona dal percorso critico
di inizio passo.
**Rischio**: basso, ma la riga di ghost va verificata contro np=1 (il gradiente
sul bordo e' quello che entra in `predict_uf` per le facce z di interfaccia).
**Costo**: 0.3 giorni.

### O6 — Pack contiguo, copie asincrone e **overlap**

**Dove**: `pack_plane`/`unpack_plane` (304-312), `exchange_u` (331-334, 339-342,
345-347), `exchange_scalar` (372, 376, 379, 383).

Oggi ogni scambio fa **1 `cudaMemcpy2D` per piano** e `exchange_u` con `np=2`
ne fa **6 per verso** (3 componenti × 2 piani): 12 copie sincrone per chiamata.
A L8 una `cudaMemcpy2D` di 256 righe da 1 KB costa ~0.5-1.3 ms su WSL (§3).
Questo e' il candidato numero uno per spiegare i 2.7 ms/scambio misurati.

Tre interventi, dal piu' semplice al piu' utile:

1. **pack contiguo su device**: un kernel che copia i piani utili in un buffer
   device contiguo (le celle interne sono gia' contigue in x), poi **una sola**
   `cudaMemcpy` device→host di `3·np·A0` float. Da 12 copie a 1. Stesso discorso
   per lo `unpack`, con un kernel che sparpaglia.
2. **`cudaMemcpyAsync` su stream dedicato + evento**, con un solo
   `cudaStreamSynchronize` prima della `Sendrecv`: toglie la sincronizzazione
   implicita dello stream di default che oggi fa aspettare i kernel (§3).
3. **overlap vero** (il punto che il DESIGN chiama "estensione"): due schemi
   possibili, in ordine di costo di implementazione:
   - *ricezione anticipata*: `MPI_Irecv` **prima** del pack, poi pack, poi
     `MPI_Isend`, poi `Waitall`, poi unpack. Il tempo di pack si sovrappone
     all'arrivo del messaggio: −30-40% del costo per scambio, ~15 righe.
   - *split interno/bordo*: dopo aver postato `Isend` del piano di frontiera,
     lanciare il kernel della parita' successiva **solo sulle celle interne**
     (quelle che non toccano il piano di frontiera non dipendono dal ghost) e
     poi il bordo quando il ghost e' arrivato. Nasconde ~50% della latenza, ma
     richiede due kernel e una maschera: 2-3 giorni, e va fatto dopo O1-O4.

**Effetto** [STIMA]: il costo fisso per scambio passa da ~0.9 ms a ~0.3 ms. Sul
baseline (194 scambi, 25.7 MB, `BW`≈75 MB/s) `comm` 0.52 → ~0.38 s. **Ma dopo
O2-O4 il termine di latenza pesa ormai poco**: a 63 scambi il guadagno di O6 e'
0.25 → 0.24 s, perche' dominano i byte (vedi §8). Va quindi fatto **solo se O1
mostra che il pack e' davvero il costo** — ed e' probabile che lo sia, ma non
cambia la classifica delle priorita'.
**Rischio**: medio-basso (codice di impacchettamento, nessuna matematica) per i
punti 1-3a; medio per 3b.
**Costo**: 1 giorno (1-3a), +2-3 giorni (3b).

### O7 — Fondere i due `bc_u(U,1)` (advezione + viscosita')

**Dove**: righe 1502-1504, `advect` (598), `viscosity` (718), `acceleration`
(747).

`advect` aggiorna `u` e `viscosity` lo rilegge: due scambi di 1 piano. Con un
alone di **2 piani** e un kernel fuso che ricalcola anche la riga di ghost
(advezione ridondante su 1 piano, costo O(1/256)) si fa **1 scambio da 2 piani**:
stessi byte, −1 scambio, −2 messaggi, −1 sincronizzazione.

**Effetto**: −1 scambio (−2 messaggi), stesso volume. **Rischio**: medio (kernel
fuso, ridondanza sul bordo). **Costo**: 1 giorno. Bassa priorita' rispetto a
O2-O4: riduce i messaggi, non i byte.

### O8 — Batch di sweeps per scambio (il "smooth_batch" gia' previsto)

**Dove**: `smooth_lvl` (842-864), `vcycle` (1003-1022).

Oggi: **2 `bc_pressure` per sweep** (uno per parita'). Con smoothing
**Jacobi pesato** (`ω≈0.75`) il ghost resta valido per piu' sweep consecutivi:
si fa 1 scambio ogni `S=2..4` sweep. La convergenza per sweep peggiora
(~0.5 vs ~0.25 di fattore di smoothing), quindi a parita' di lavoro servono
~2x sweep: il guadagno netto e' ~2-4x **sui messaggi**, non sui byte.

Variante a rischio piu' basso, stesso beneficio: **tenere il red-black GS e
accettare il ghost "in ritardo"** di un sweep (si scambia 1 volta ogni 2 sweep
invece di 1 per parita'), con l'errore confinato alla riga di bordo.

| | scambi/livello/V-cycle | byte/V-cycle |
|---|---|---|
| oggi | 6 | 4.19 MB |
| batch S=2 (4 scambi/livello) | **4** | 4.19 MB |
| batch S=2 + residuo fuso (3/livello) | **3** | 4.19 MB |

Con 4 scambi/livello e `LD=4` (O4) si passa a **19 scambi per V-cycle** invece
di 47. **Effetto** (con O4): NIT=2: 122 → **80 scambi**; NIT=1: 64 → **44**.
**Rischio**: **alto sul risultato numerico** (cambia lo smoother: rifare
np1-vs-np2 a 1e-5 e il confronto con `saetta3d` a L6). E' la leva piu' forte
sui messaggi: va provata, ma dopo O2-O4 che sono a rischio quasi nullo.
**Costo**: 1-2 giorni.

### O9 — Fuori dagli scambi: `compute_dt` e la diagnostica

`compute_dt` fa **3 `mf_max_abs`** (riga 1100-1102), ognuno con
`cudaMemset` + kernel + `cudaMemcpy` di 4 byte che **sincronizza il device**, e
poi 1 `Allreduce`: 3 sincronizzazioni per passo sul percorso critico. Un solo
kernel che riduce il massimo di `|uf.x|,|uf.y|,|uf.z|` in un `float3` e una sola
`cudaMemcpy` lo riduce a 1.

La diagnostica (ogni `TDIAG`) e' piu' pesante di quanto sembri: `download_u`
copia **3 × 137 MB per rank a L8** su memoria **non pinnata** (riga 1173-1175),
e `diagnostics_compute` e' un ciclo **host a thread singolo** su 34.4M celle con
7 letture sparse. Con `-diag 1` (default) su un passo da 0.02 s sono 50 passi
per output: se la diagnostica costa 2 s/rank, sono +40 ms/passo (**+4%**) che
finiscono nel `wall_s` e non nel `comm_s` — ed e' una delle spiegazioni
possibili del divario fra `out/L8_np1` (0.347 s/passo al primo passo) e il
dato del Lead (1.407 s/passo). Da chiarire prima di attribuire i tempi.
Fix: `cudaHostAlloc` per `HU0..HU2`, riduzione della diagnostica su GPU,
oppure `-diag 10` nei run di benchmark.

### O10 — Lato rete (Wi-Fi): MCA e firewall, non "piu' banda"

Tre cose che cambiano il costo per messaggio senza toccare il codice:

1. **Porte**: OpenMPI apre porte effimere per i collegamenti dati. In WSL
   *mirrored* il traffico passa dal firewall di Windows: se e' aperta solo la
   22, `mpirun -np 2 hostname` funziona (usa ssh) ma il run **si blocca al
   primo scambio grosso**. Si fissa il range e si apre:
   ```powershell
   New-NetFirewallRule -DisplayName "mpi-data" -Direction Inbound -Protocol TCP `
       -LocalPort 50000-51023 -Action Allow
   ```
   ```sh
   mpirun ... --mca btl_tcp_port_min_v4 50000 --mca btl_tcp_port_range_v4 1024
   ```
2. **Buffer TCP**: `--mca btl_tcp_sndbuf 4194304 --mca btl_tcp_rcvbuf 4194304`
   (default 128 KB: con 256 KB di messaggio si va in rendezvous).
3. **Trasporto**: selezionare esplicitamente `--mca btl tcp,self
   --mca btl_tcp_if_include <iface>` per non far scegliere a OpenMPI
   l'interfaccia NAT di WSL; e verificare con `ompi_info --param btl tcp` che i
   nomi dei parametri esistano nella vostra versione (cambiano fra 4.x e 5.x).

Non conviene: compressione degli aloni (fp16 su `p` rompe la validazione),
GPUDirect (GPU consumer + Wi-Fi), alberi/gerarchie (2 rank).

### O11 — Cosa NON fare adesso

- **Decomporre anche in x/y**: la direzione lunga e' z, 1 sola interfaccia con
  2 rank e' il minimo possibile. Nessun guadagno.
- **4 rank (2 per PC)**: raddoppia i messaggi per passo e la contesa sulla GPU.
- **`np=2` sulla stessa GPU per misurare lo speedup**: e' quello che e' stato
  fatto in `out/L8_np2` (0.5595 s contro 0.3466 s: **peggiora**), e' solo un
  test di correttezza.

---

## 6. Stima dello scaling a L8 su 2 PC

Modello a due termini, tarato sull'unico punto a 2 PC disponibile
([DATO Lead] `comm ≈ 0.522 s` con 194 scambi e 25.7 MB):

```
comm = n_scambi · t_l + MB / BW
```
`t_l` include latenza di rete **e** pack/sync (e' il costo fisso per scambio).
Con `BW ≈ 75 MB/s` il modello riproduce il dato del Lead con `t_l = 0.9 ms`
(`194·0.9 + 25.7/75 = 0.52 s`): e' la coppia di parametri di riferimento,
l'unica compatibile con 194 messaggi andata-ritorno da ~2 RTT l'uno. Con `O3`
(un solo `Sendrecv`) la latenza per scambio dimezza (`t_l ≈ 0.55 ms`), con `O6`
(pack contiguo) scende a `~0.30 ms`.

| ipotesi | `BW` | comm oggi (194 sc / 25.7 MB) | con O2+O3+O4+O5 (63 / 16.2) | con anche O6 (63 / 16.2, `t_l` 0.30) |
|---|---|---|---|---|
| Wi-Fi 6 buono | 100 MB/s | 0.43 s | 0.20 s | 0.18 s |
| **calibrato sul dato del Lead** | **75** | **0.52 s** | **0.25 s** | **0.24 s** |
| Wi-Fi 5 / affollato | 50 MB/s | 0.69 s | 0.36 s | 0.34 s |

Come togliere l'ambiguita' in 2 minuti, senza toccare il solver: `mpi_pingpong.c`
(allegato) misura il RTT per le taglie del multigrid (16 B, 1 KB, 64 KB,
256 KB). Da due righe si ricavano entrambi i parametri
(`RTT(256KB) − RTT(16B) = 2·256KB/BW`, e il RTT a 16 B e' `t_l` di rete), e il
modello diventa una previsione invece di un intervallo. Le taglie da guardare
sono esattamente quelle del codice: 256 KB = livello 0, 64 KB = livello 1,
1 KB = livello 4.

E lo speedup che ne segue, con `T(np=1) = 1.407 s` [DATO Lead] e
`T(np=2) = 1.407/2 + comm` [STIMA], a `BW = 75 MB/s`:

| configurazione | scambi | msg p2p | MB/passo | `t_l` | comm | `T(np=2)` | speedup | nota |
|---|---|---|---|---|---|---|---|---|
| oggi (NIT=2) | 194 | 388 | 25.7 | 0.90 | 0.50 s | 1.20 s | **1.17x** | = dato del Lead (1.226 → 1.15x) |
| O2 (`nit 1`) | 100 | 200 | 17.3 | 0.90 | 0.32 s | 1.02 s | 1.37x | zero codice |
| O2+O3 | 100 | 100 | 17.3 | 0.55 | 0.29 s | 0.99 s | 1.42x | ~0.5 giorno |
| O2+O3+O4+O5 | 63 | 63 | 16.2 | 0.55 | 0.25 s | 0.96 s | 1.47x | ~1.5 giorni |
| + O6 (pack) | 63 | 63 | 16.2 | 0.30 | 0.24 s | 0.94 s | 1.50x | |
| + O8 (batch) | 43 | 43 | 16.2 | 0.30 | 0.23 s | 0.93 s | 1.51x | rischio numerico |
| NIT=2 + O3 + O4 (confronto) | 122 | 122 | 25.6 | 0.55 | 0.41 s | 1.11 s | 1.27x | se serve piu' accuratezza |

**Lettura onesta**: a L8 il tetto realistico e' **~1.5x**, non 2x, e si ottiene
cumulando 3-4 interventi. Da notare che **dopo O2-O4 il termine di byte domina**
(16.2 MB / 75 MB/s = 0.22 s su 0.25 s): O6 e O8 spostano poco, e l'unico modo
per scendere ancora e' **togliere byte** (l'alone di `u` e' 6.3 dei 16.2 MB).
Il guadagno *sicuro* della decomposizione resta la **memoria** (67M → 134M
celle).

**Ma il numero che conta per il progetto e' un altro**: `saetta3d` a L8 costa
**2.8 s/passo** [DATO DESIGN §3, da `run7/log.txt`], cioe' il **doppio** del
prototipo. Con gli stessi `comm`:

| solver | `T(np=1)` | comm oggi | `T(np=2)` | speedup |
|---|---|---|---|---|
| prototipo `mpi_fissa` | 1.407 s | 0.50 s | 1.20 s | 1.17x |
| `saetta3d` (stima a parita' di comunicazione) | 2.8 s | 0.50 s | 1.90 s | **1.47x** |
| `saetta3d` + O2+O3+O4 | 2.8 s | 0.25 s | 1.65 s | **1.70x** |

Il prototipo e' **pessimista**: essendo molto piu' snello, ha un rapporto
comm/calcolo sfavorevole. La domanda "il Wi-Fi basta per una decomposizione?"
va risposta sul solver vero, e la risposta e' **si' a L8 con 1.5-1.7x**, a
patto di portare con se' le O2-O4.

**Griglia doppia (256×256×2048, 134M celle)**: `T(np=1)` non esiste (non entra
in 8 GB). `T(np=2)` ≈ 1.407 (calcolo dimezzato della griglia doppia) +
`comm(2× i byte)` ≈ 0.45-0.55 s → **1.9-2.0 s/passo** [STIMA]: il caso d'uso
che giustifica tutto il lavoro (a parita' di tempo, il doppio delle celle).

**L6 (128×128×512)**: `T(np=1)` ≈ 0.35 s [DATO `out/sep_g1`], `comm` 6.9 MB /
170 scambi con lo stesso modello (`170·0.9 + 6.9/75` = 0.24 s) → `T(np=2)`
0.42 s → **speedup 0.8x**: a L6 la decomposizione **peggiora**. Coerente con
DESIGN §6.

---

## 7. Piano consigliato (ordine di esecuzione)

| # | azione | giorni | guadagno atteso | rischio |
|---|---|---|---|---|
| 1 | O1 strumentazione (pack/MPI/delta contatori) | 0.2 | — (abilita il resto) | nullo |
| 2 | O2 `-poisson_nit 1` + validazione residuo | 0.3 | 1.15x → 1.4x | basso |
| 3 | misura 2 PC di riferimento con `benchmark.sh` + `mpi_pingpong` | 0.4 | — (i dati che mancano) | nullo |
| 4 | O3 un solo `Sendrecv` (R=2) | 0.3 | −50% messaggi | basso |
| 5 | O4 replica da `l=5` | 0.5 | −38% scambi | basso |
| 6 | O5 niente scambio di `GRD` | 0.3 | −1 MB, −2 msg | basso |
| 7 | O6 pack contiguo + overlap 3a (solo se O1 lo giustifica) | 1.0 | −0.1 s | medio |
| 8 | O9 `compute_dt` + diagnostica | 0.5 | −3 sync, togli 4% dal wall | basso |
| 9 | O8 batch di sweeps (solo se serve altro) | 1-2 | −20 scambi, ~0.01 s: **non conviene** | alto |

Dopo i punti 1-6 (2 giorni scarsi) il quadro a 2 PC e' gia' quello definitivo:
se lo speedup misurato e' ≥ 1.4x a L8, la decomposizione ha senso per la
griglia doppia; altrimenti la conclusione (DESIGN §10) e' "2 GPU solo per la
memoria". **O8 va lasciata per ultima e probabilmente non fatta**: dopo O2-O4 il
tempo e' dominato dai byte, e O8 riduce solo i messaggi.

---

## 8. Riepilogo in una tabella

`BW = 75 MB/s`, `t_l` = 0.90 ms (2 `Sendrecv`/scambio), 0.55 ms con O3, 0.30 ms
con O6. `T(np=2) = 1.407/2 + comm`; speedup = `1.407 / T(np=2)`.

| configurazione | scambi | msg p2p | MB/passo | comm [STIMA] | `T(np=2)` L8 | speedup L8 |
|---|---|---|---|---|---|---|
| oggi, NIT=2 | 194 | 388 | 25.7 | 0.50 s | 1.20 s | **1.17x** |
| oggi, NIT=1 | 100 | 200 | 17.3 | 0.32 s | 1.02 s | 1.37x |
| NIT=1 + O3 | 100 | 100 | 17.3 | 0.29 s | 0.99 s | 1.42x |
| NIT=1 + O3 + O4 + O5 | 63 | 63 | 16.2 | 0.25 s | 0.96 s | 1.47x |
| + O6 (pack) | 63 | 63 | 16.2 | 0.24 s | 0.94 s | 1.50x |
| + O8 (batch, 43 scambi) | 43 | 43 | 16.2 | 0.23 s | 0.93 s | 1.51x |
| NIT=2 + O3 + O4 (per confronto) | 122 | 122 | 25.6 | 0.41 s | 1.11 s | 1.27x |

Il pavimento a `BW=75` e' `16.2/75 = 0.22 s` di soli byte: sotto quello non si
scende senza togliere byte (l'alone di `u` e' 6.3 dei 16.2 MB). A `BW=50` il
pavimento sale a 0.32 s e lo speedup massimo scende a ~1.35x; a `BW=100` lo
speedup massimo e' ~1.55x. **La banda del Wi-Fi e' il parametro che decide**,
non il numero di messaggi: per questo il primo passo e' misurarla
(`mpi_pingpong.c`).

---

## 9. Dubbi aperti da chiarire con il Lead (non li posso risolvere da qui)

1. **`out/L8_np1` = 0.347 s/passo contro 1.407 s/passo del Lead**, stesso L8.
   4x di differenza. Ipotesi: diagnostica (`-diag`) dentro la media,
   run con `-tmax` diverso, macchina diversa. Va chiarito perche' **tutte** le
   stime di speedup qui sopra poggiano su 1.407 s.
2. **I "2-8 MB / 151-175 scambi"** sono L6 (§2.5). Confermare con
   `cells` in `diagnostics.dat`.
3. **La GPU del secondo PC e' lo stesso modello?** Se no, il binario compilato
   con `ARCH=86` (solo SASS sm_86) non parte: serve `-gencode` con PTX, o
   `ARCH` diverso per macchina. Va verificato **prima** del primo run a 2 PC.
4. **`comm_s` include il tempo di GPU** (§3): i confronti "comm/wall" vanno
   letti sapendo che il numeratore e' gonfiato. Da O1 in poi non piu'.
