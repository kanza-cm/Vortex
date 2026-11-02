# PROTOCOLLO — Validazione temporale (CFL) di `mpi_fissa` contro `saetta3d`

**Stato:** piano di validazione + script `validate_tempo.sh` (v1).
**Ambito:** griglia fissa `64 x 64 x 256`, `TMAX = 1.0`, `nu = 1e-5`,
`poisson_nit = 2`, `np = 1`; matrice CFL `0.3 / 0.15 / 0.1`.
**Non e':** uno studio di convergenza spaziale, una validazione a L8, un test
MPI np1-vs-np2, una misura di speedup.

Questo documento distingue esplicitamente:

- **misurato** = numero letto da un file del workspace o fornito dal Lead;
- **dedotto** = conseguenza del codice/delle formule, da confermare con il run;
- **da misurare** = risultato che produrra' `validate_tempo.sh`.

---

## 0. Sintesi per il Lead

Lo script:

1. compila il prototipo `NX=64` in versione GPU (fallback CPU se la build o lo
   smoke test GPU falliscono), senza sovrascrivere i binari in `mpi_fissa/bin`;
2. per ogni CFL `0.3`, `0.15`, `0.1` compila `saetta3d.c` in
   `/tmp/saetta3d_cfl<tag>` con i flag del riferimento
   (`-DCFL_USER=<cfl> -DNU=1e-5 -DMOVIE=0 -DVTK=0 -DMATRIX=1 -DNDIV0=1
   -DMAXLEVEL=6 -DTMAX=1.0`) e lo esegue;
3. esegue il prototipo con `-cfl <cfl> -tmax 1.0 -nu 1e-5 -poisson_nit 2`;
4. confronta i campi finali `omega3d_matrix.bin` con `confronta_campi.py` e le
   diagnostiche con `confronta_diagnostics.py` (file completo, solo `t=0`,
   solo `t=1`);
5. scrive `out/validazione_tempo/.../tabella.txt` con le colonne richieste e
   `ordine.txt` con la stima dell'ordine osservato.

La tabella **non decide** da sola se la fisica e' corretta: i criteri sono in
§4. Il verdetto va letto insieme ai rischi di §6, in particolare il diverso
scheduling di `dt` tra i due codici.

Comando previsto (dettagli in §9):

```sh
cd "/home/zorka/simulazioni/Test 1"
bash /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/validazione/validate_tempo.sh
```

Output: `mpi_fissa/out/validazione_tempo/` (prima esecuzione: direttamente in
questa cartella se e' vuota; esecuzioni successive: sottocartella con
timestamp, per non sovrascrivere).

Durata attesa (stima, non misurata): dell'ordine di 15-45 minuti, dominata
dalle tre build qcc e dai tre run saetta; i timeout per comando sono in §9.1.

---

## 1. Dati di partenza (forniti dal Lead, non rimisurati qui)

| dato | valore | provenienza |
|---|---|---|
| IC a `t=0`, griglia `64x64x256` | accordo campi ~`1e-7` | misura del Lead (contesto) |
| np1 vs np2, `NX=64` | `rms_rel 2.2e-7` | misura del Lead |
| np1 vs np2, `NX=128` | `rms_rel 3.6e-7` | misura del Lead |
| np1 vs np2 GPU | `rms_rel 2.1e-7` | misura del Lead |
| np1 vs np2 L8 | `rms_rel 2.8e-5` | misura del Lead |
| prototipo vs saetta3d a `t=1`, stessa griglia | `rms_rel 1.4e-2` | misura del Lead |
| passi a `t=1` | prototipo 14, saetta3d 24 | misura del Lead |

Dati letti dai file del workspace, usati solo come controllo/contesto:

| dato | valore | file |
|---|---|---|
| `t=0` prototipo `NX=64` | `Z=1.24323318e-1`, `E=1.00171256e-3`, `umax=5.74290698e-2`, `Wmax=1.94093272`, `G=5.14048109e-2`, `sep=0` | `mpi_fissa/out/g1_t1/diagnostics.dat` |
| `t=1` prototipo `NX=64` (GPU, 14 passi) | `Z=1.23387947e-1`, `E=9.47984709e-4`, `umax=6.28683516e-2`, `Wmax=1.90167767`, `G=5.13476644e-2`, `sep=2.390165e-1`, `mgp_resa=2.767e-2` | `mpi_fissa/out/g1_t1/diagnostics.dat` |
| run6 storico (L6 dichiarato) | `cells=8388608` -> griglia `128x128x512`, `dt0=9.907987e-3` | `gpu_nvidia_3d/risultati/error_study/error_study_fissa_3d_gpu/run6/diagnostics.dat` |
| L8 produzione | `cells=67108864` -> `256x256x1024`, `dt0=4.947802e-3` | `gpu_nvidia_3d/risultati/diagnostica/diagnostics.dat` |

**Attenzione alla griglia.** Il run6 storico ha `cells=8388608`, quindi
`128x128x512`: non e' la griglia di questo test. Con i flag richiesti
(`NDIV0=1`, `MAXLEVEL=6`) `saetta3d` gira su `64x64x256` (`cells=1048576`),
la stessa del prototipo `NX=64`. Le tabelle "L6 = 128x128x512" di
`VALIDAZIONE.md`/`riferimento/` non vanno usate come riferimento a questa
griglia: a `t=0` il valore atteso di `Z` a `64x64x256` e' circa
`1.243e-1` (dal prototipo), non `1.2563e-1`. La verifica di `cells` e'
obbligatoria nello script.

---

## 2. Setup congelato

| parametro | valore | note |
|---|---|---|
| griglia | `64 x 64 x 256` (`Delta=1/64`) | `NX=64` prototipo; `NDIV0=1, MAXLEVEL=6` saetta |
| tempo finale | `TMAX=1.0` | entrambi |
| viscosita' | `nu=1e-5` | entrambi; esplicita nel prototipo, implicita in saetta |
| proiezione | `poisson_nit=2` | solo prototipo; saetta usa multigrid adattivo |
| ranghi | `np=1` | nessun effetto di decomposizione |
| backend prototipo | GPU, fallback CPU | il fallback deve essere registrato nella tabella |
| CFL | `0.3`, `0.15`, `0.1` | stesso valore passato ai due codici |
| output campi | `omega3d_matrix.bin` a `t=1` | stesso formato `int32 nx,ny,nz + float32` |

Lo script registra in `ambiente.txt` data, host, percorsi, versioni di
`mpirun`/`mpicxx`/`python3`/`qcc`/`nvcc`, e lo `sha256` di
`mpi_fissa/src/mpi_fissa.cu` e `gpu_nvidia_3d/saetta3d.c`. Se i sorgenti
cambiano, i risultati non sono confrontabili con misure precedenti.

---

## 3. Cosa viene confrontato e come

### 3.1 Campi finali (`t=1`)

`confronta_campi.py` sui due `omega3d_matrix.bin`:

- `rms_rel = rms(omega_proto - omega_saetta) / rms(omega_saetta)`;
- `linf = max |omega_proto - omega_saetta|`.

Le griglie devono essere **identiche** (`64x64x256`): se non lo sono, il
confronto e' invalido (lo script controlla `cells` e le dimensioni del dump).
`rms_rel` e' la metrica primaria; `linf` e' sensibile ai massimi locali e va
letto solo come supporto.

### 3.2 Diagnostiche

`confronta_diagnostics.py` confronta, con accoppiamento per tempo:
`enstrophy`, `energy`, `umax`, `wmax`, `circulation_slice`, `separation`.
Vengono prodotti tre confronti:

1. **file completo** (`t=0` e `t=1`): serve da record, ma contiene l'artefatto
   di `separation` a `t=0` (vedi §6.6), quindi il suo "FUORI TOLLERANZA" non
   e' un fallimento;
2. **solo `t=0`**: verifica dell'IC e delle formule diagnostiche; `separation`
   va ignorata;
3. **solo `t=1`**: confronto che conta per la fisica a fine run.

Le tolleranze di default di `confronta_diagnostics.py` (`Z,E 5%`, `umax,wmax
10%`, `G 3%`, `sep 5%`) sono usate solo come riferimento: i criteri proposti
sono in §4.

### 3.3 `dt` e numero di passi

- `dt_proto_medio`: media della colonna `dt` di
  `out/.../proto/timing.dat` (tutti i passi);
- `passi_proto`: colonna `it` dell'ultima riga di `diagnostics.dat`;
- `passi_saetta`: colonna `it` dell'ultima riga di `diagnostics.dat` di
  saetta;
- `dt_saetta_medio` (in nota, non in colonna): stima grossolana
  `TMAX / passi_saetta`, perche' saetta non scrive un `timing.dat`.

Il rapporto `dt_proto_medio / dt_saetta_medio` e' informativo: i due codici
**non** hanno lo stesso scheduling di `dt` (§6.1).

### 3.4 Cosa NON si confronta

- `dt` puntuale passo-passo: i due codici non hanno la stessa sequenza;
- `mgp_resa` numerico tra i due codici: definizioni/scaling diversi; nel
  prototipo e' un residuo assoluto interno, in saetta `mgp.resa` e' un residuo
  del multigrid adattivo. Utile solo come indicatore interno del prototipo;
- valori L6/L8 storici a `128`/`256`: griglia diversa;
- `separation` a `t=0` (artefatto, §6.6);
- serie temporali a tempi intermedi: entrambi scrivono solo a `t=0` e `t=1`
  (`-diag 1`).

---

## 4. Criteri di accettazione (proposti, da confermare sul primo run)

I criteri sono ordinati: A e' prerequisito tecnico; B/C sono il cuore dello
studio temporale; D/E sono diagnostici.

### A. Validita' tecnica (obbligatorio)

- tutte e 3 le righe CFL presenti, senza timeout;
- entrambi i codici a `t=1` entro `1e-6`;
- entrambi `cells=1048576` e dump `64x64x256`;
- nessuna sovrascrittura di risultati precedenti.

Se A fallisce, la tabella va considerata incompleta e non si trae alcuna
conclusione sull'ordine.

### B. Monotonia (obbligatorio)

Con `rms_rel(CFL)`:

```
rms_rel(0.3) > rms_rel(0.15) > rms_rel(0.1)
```

La diminuzione deve essere **stretta**. Una riduzione inferiore al 10% tra due
CFL consecutivi viene trattata come *plateau* (sospetto errore indipendente dal
passo), non come conferma. La soglia del 10% e' prudenziale rispetto al rumore
np1-vs-np2 misurato (`2e-7`), che e' trascurabile qui.

### C. Ordine osservato (obbligatorio, stima)

Con `p_ij = ln(rms_i/rms_j) / ln(CFL_i/CFL_j)` (i piu' grande di j):

| `p_13` (0.3 -> 0.1) | interpretazione proposta |
|---|---|
| `>= 1.5` | errore temporale coerente con schema del 2o ordine; accettabile |
| `0.8 .. 1.5` | monotono ma ordine ~1: errore temporale non dominante o splitting; indagare §8.1 |
| `< 0.8` | plateau / errore indipendente dal CFL; indagare §8.2 |

Si riportano anche `p_12`, `p_23`, la pendenza log-log ai minimi quadrati e,
se i `dt_proto_medio` sono disponibili, la pendenza rispetto a `dt` invece che
a CFL. L'incertezza con 3 punti e' almeno `+-0.3`: non si pretende un ordine
"esatto".

### D. Plateau / residuo (diagnostico)

Se `p_13 < 0.8`, stimare se `rms` tende a un valore positivo per `CFL -> 0`.
Un plateau indica un errore **non temporale** (spaziale, BC, proiezione,
viscosita', IC/diagnostica). In quel caso l'ordine temporale non e'
dimostrabile con questa matrice: non e' un fallimento del prototipo, e' un
limite dell'esperimento.

### E. Diagnostiche a `t=1` (secondario)

- `Z`, `E`, `G`: entro 2% relativo;
- `separation`: entro 3%;
- `umax`, `wmax`: entro 5%.

Sono le tolleranze T4 di `VALIDAZIONE.md` per `t<=2`. Se i campi convergono ma
una diagnostica locale no, la causa probabile e' la selezione dei massimi
(`separation`, `wmax`), non il campo nel suo complesso.

### F. Tolleranza assoluta (guida, non criterio formale)

`rms_rel(0.1) <= 5e-3` e' un obiettivo ragionevole: con `p=2` e
`rms(0.3)=1.4e-2` si attenderebbe `~1.6e-3`; la soglia `5e-3` lascia margine
per il termine spaziale. **Non e' una misura**: va verificata.

---

## 5. Stima dell'ordine: formule e incertezza

Per ogni coppia di CFL:

```
p_ij = ln(rms(CFL_i) / rms(CFL_j)) / ln(CFL_i / CFL_j)
```

La stima complessiva e' la pendenza della retta ai minimi quadrati di
`ln(rms)` vs `ln(CFL)` sui tre punti. Si riporta anche la pendenza vs
`ln(dt_proto_medio)`: se il prototipo scala `dt ~ CFL`, le due stime devono
coincidere; se differiscono, il troncamento a `t=1` o il clamp dell'ultimo
passo stanno pesando.

Con tre punti:

- `p_12` e `p_23` sono stime indipendenti; la loro dispersione e' un indicatore
  di quanto il comportamento e' "pulito";
- `p_13` e' la stima piu' robusta perche' usa il rapporto di CFL piu' grande;
- un `p > 2` puo' indicare cancellazione/errore di compensazione o rumore;
  non va "festeggiato" come ordine superiore senza controllare i residui;
- un `p` negativo o non monotono invalida l'ipotesi di convergenza temporale.

La stima riguarda l'**errore di troncamento temporale** del prototipo rispetto
a saetta, non l'ordine formale assoluto di uno dei due schemi.

---

## 6. Rischi e mitigazioni

### 6.1 `dt` diverso a parita' di CFL (rischio alto)

**Dedotto dal codice, non misurato in questo run.** Il prototipo usa
`dt = CFL*DH/max|u|` con crescita massima del 10%/passo e clamp a `tnext`
(`compute_dt`, righe 1099-1115 di `mpi_fissa.cu`): al primo passo parte gia'
al limite CFL. Saetta usa `timestep()` di Basilisk
(`basilisk/src/timestep.h`):

```c
dtmax /= CFL;
... dtmax = CFL * min(Delta/|u_f|);
if (dtmax > previous)
  dtmax = (previous + 0.1*dtmax)/1.1;
```

cioe' parte da una frazione del limite e si avvicina al limite con crescita
<10%/passo; inoltre `timestep()` viene invocata piu' volte durante
l'inizializzazione (`init` + `stability` e il primo ciclo di eventi). Il `dt`
iniziale di saetta e' infatti ~0.24 del limite CFL nei run storici, compatibile
con 2-3 applicazioni della formula di smoothing:

- L8: `dt0=4.947802e-3`, limite `0.3*Delta/umax = 2.04e-2`, rapporto `0.2425`;
- run6 (128): `dt0=9.907987e-3`, limite `4.08e-2`, rapporto `0.2427`.

Per questo il prototipo fa 14 passi e saetta 24 a `t=1` (misura del Lead),
pur con lo stesso `-cfl`. **Conseguenza:** la matrice CFL confronta due
successioni di `dt` diverse; l'errore a `t=1` include la fase di ramp-up di
saetta. La scala con CFL resta confrontabile, ma non si puo' pretendere che
`dt_proto = dt_saetta` ne' confrontare passo-passo.
**Mitigazione:** la tabella riporta `dt_proto_medio`, `passi_proto`,
`passi_saetta` e il rapporto; l'ordine viene stimato anche vs `dt_proto`.
Se serve un test "a parita' di dt", va fatto a parte (§8.6).

### 6.2 Proiezione a iterazioni fisse (rischio alto)

Il prototipo usa `POISSON_NIT=2` V-cycle fissi; saetta usa un multigrid
adattivo con tolleranza. A `NX=64` due V-cycle possono lasciare un residuo
non trascurabile: nel run esistente `out/g1_t1/diagnostics.dat` il
`mgp_resa` a `t=1` e' `2.767e-2` (indicatore interno, non confrontabile con
saetta). Un errore di proiezione **non scala necessariamente con CFL** e puo'
produrre un plateau in `rms`: e' il primo sospetto se `p_13 < 0.8`.
**Mitigazione:** test ausiliario con `-poisson_nit 4` a CFL fissato (§9.4);
se `rms` scende molto, la proiezione e' la causa, non il tempo.

### 6.3 Viscosita' esplicita vs implicita (rischio basso a questi parametri)

Numero di diffusione `nu*dt/Delta^2`:

- `NX=64`, `dt~7.1e-2`: `1e-5*7.1e-2/(1.5625e-2)^2 ~= 2.9e-3`;
- `NX=128`, `dt~3.6e-2`: `~5.9e-3`.

Sono molto sotto `0.1`: l'esplicitazione non minaccia la stabilita'. La
differenza tra i due schemi e' un termine di ordine superiore nel numero di
diffusione: per le scale risolte (gradiente ~`2*pi`) il numero di diffusione
efficace e' ~`3e-5`, per le mode di griglia arriva a `3e-3`. Non puo' spiegare
da solo un `rms` di `1.4e-2` su un campo liscio. Attenzione a non confondere
questo con la **decadenza viscosa fisica**, che a `t=1` vale circa
`2*nu*Z/E ~ 2.5e-3` relativo sull'energia: quella c'e' in entrambi i codici e
non e' una differenza. **Non e' la prima leva**; va rivalutata solo se il
plateau resta anche con `nu -> 0` o a `dt` molto piu' grandi. La soluzione
implicita "decoupled" di Basilisk e' un cambio piu' invasivo di quanto questi
numeri giustifichino.

### 6.4 fp32 e sensibilita' caotica (rischio medio a tempi lunghi)

Entrambi i codici sono fp32 (Basilisk usa `float` per i campi; il prototipo
usa fp32 nei kernel). A `t=1` il flusso e' ancora liscio, ma i massimi locali
(`wmax`, `separation`) sono sensibili a dettagli di arrotondamento. `rms_rel`
e le diagnostiche globali (`Z`, `E`) sono piu' robuste di `linf`/`wmax`.
**Mitigazione:** non inseguire `linf`; usare `rms_rel` come metrica primaria.

### 6.5 Etichette L6/L8 e griglia (rischio medio)

`VALIDAZIONE.md` chiama "L6" il run6 storico a `128x128x512`
(`cells=8388608`). Questo test usa `MAXLEVEL=6, NDIV0=1` ->
`64x64x256` (`cells=1048576`). Confrontare i valori diagnostici del test con
la tabella "L6" sarebbe un errore di griglia. **Mitigazione:** lo script
controlla `cells` e le dimensioni del dump; la tabella riporta la griglia.

### 6.6 `separation = 0` a `t=0` nel prototipo (rischio medio per le diagnostiche)

Nel prototipo la fetta nodo ha due massimi simmetrici di `WZ`; la procedura
"media di tutte le celle con `WZ == m1`" porta `px=py=0`, e il test
`x*px + y*py < 0` non seleziona nessuna cella del secondo nucleo ->
`w2=0` -> `sep=0` (`diagnostics_compute`, righe 1231-1275). Nei run esistenti
`out/g1_t1/diagnostics.dat` ha `sep=0` a `t=0`. Saetta, con lo stesso algoritmo
ma valori fp32 leggermente asimmetrici, nei run storici a `128x128x512`
riporta `sep=0.2578` a `t=0` (`riferimento/valori_riferimento.md`); il valore
esatto a `64x64x256` **va misurato**, ma non sara' zero.
**Conseguenza:** il confronto diagnostiche completo a `t=0` segnalera'
`separation` fuori tolleranza; non e' un errore di fisica. A `t=1` il
prototipo riporta `sep=2.390e-1`; il valore di saetta a `t=1` e' **da
misurare** in questo run (non e' nella tabella di riferimento a 64).
**Mitigazione:** il confronto che conta e' quello a `t=1`; il `t=0` serve solo
per IC e per `Z,E,Umax,Wmax,G`.

### 6.7 Ambiente qcc/CUDA (rischio medio)

`saetta3d.c` richiede:

- `BASILISK` (default `$HOME/basilisk/src`) e `qcc` nel `PATH`;
- header e librerie CUDA (`cuda_runtime.h`, `libnvrtc`, `libcudart`) in
  `$CUDA/targets/x86_64-linux` o `$HOME/cuda-13.1`;
- `liberrors.a` e `libbuda.a` gia' compilate in `$BASILISK/grid/...`;
- percorsi **senza spazi** per qcc: lo script compila in
  `/tmp/saetta3d_cfl<tag>`, non nel workspace.

Se manca qualcosa, lo script si ferma con un messaggio e il comando `make`
suggerito (o lo esegue, se la libreria manca). La build del prototipo usa
`nvcc`; se `nvcc` manca, il fallback CPU e' automatico.

### 6.8 Fallback CPU (rischio basso)

Se la build/smoke GPU fallisce, il prototipo gira su CPU. La differenza
CPU-vs-GPU **non e' stata rimisurata in questo studio**: il test T3 di
`VALIDAZIONE.md` propone `<1e-5` sul campo; qui il fallback va registrato e,
se possibile, confrontato con una riga GPU dello stesso CFL. Il confronto con
saetta resta valido in linea di principio, ma i tempi cambiano. La tabella
registra `GPU` o `CPU` nella nota. Con `NX=64` e 14-20 passi il run CPU e'
comunque breve.

### 6.9 Timeout, sovrascritture, run interrotti (rischio basso)

Ogni comando esterno e' avvolto in `timeout` (build, run, post-processing).
Le directory di output sono uniche: la prima esecuzione usa
`out/validazione_tempo/` se e' vuota, altrimenti una sottocartella
`YYYYmmdd_HHMMSS`; le build temporanee usano `/tmp/saetta3d_cfl<tag>` e, se
esistono, un suffisso. Nessun `rm -rf` sul workspace. Una riga incompleta
riporta `NA` e il motivo nella nota; lo script termina con exit 1 se una riga
e' tecnicamente incompleta (l'esito scientifico resta nella tabella).

---

## 7. Interpretazione dei possibili risultati

| osservazione | interpretazione | azione |
|---|---|---|
| `rms` decresce, `p_13 ~ 2` | errore temporale del 2o ordine; il residuo e' probabilmente spaziale/BC | accettare il test temporale; per ridurre `rms` serve raffinare la griglia, non il CFL |
| `rms` decresce, `p_13 ~ 1` | split/proiezione o BC dominano, oppure schema effettivamente 1o ordine | §8.1: `poisson_nit 4`, controllo BC, confronto a `dt` fissato |
| `rms` quasi costante (`p_13 < 0.8`) | plateau: errore indipendente da CFL | §8.2: proiezione, BC, viscosita', IC/diagnostica; non e' un ordine temporale |
| `rms` aumenta riducendo CFL | confronto invalido (tempi/griglie/dump diversi) o instabilita' | controllare `cells`, `t_finale`, header; rifare |
| campi convergono ma `sep`/`wmax` no | selezione dei massimi sensibile | usare `rms_rel` e `Z/E/G`; non inseguire i massimi |
| `t=0` diagnostiche fuori solo per `sep` | artefatto noto §6.6 | ignorare; guardare `t=1` |

---

## 8. Se dai risultati emerge un errore di ordine: correzioni suggerite

### 8.1 Ordine ~1 (monotono, `0.8 <= p_13 < 1.5`)

1. **Proiezione.** Rilanciare a CFL fissato (es. `0.1`) con
   `-poisson_nit 4` e confrontare `rms` e `mgp_resa`. Se `rms` scende
   sensibilmente, l'errore dominante e' la proiezione approssimata, non
   l'avvezione. Costo: ~2x per passo nella proiezione.
2. **BC.** Verificare le facce normali alle pareti nel prediction
   (`bc_u(U,2)` e `uf` forzata a 0) e il segno delle BC free-slip. Un segno
   sbagliato cambia `Z` in modo sistematico, non solo l'ordine.
3. **Splitting temporale.** Il prototipo replica lo schema di `centered.h`
   (prediction a `t+dt/2`, project, advect, viscosity, project); un errore
   nell'ordine delle proiezioni o nell'uso di `dt/2` puo' degradare a 1o
   ordine. Confrontare con la sequenza di `centered.h` (gia' documentata in
   `DESIGN.md`) e con un caso a `dt` fissato.
4. **Viscosita' esplicita.** Solo se il numero di diffusione e' grande
   (non e' questo caso): implementare il solve implicito decoupled come
   Basilisk (`viscosity.h`) o almeno verificare con `-nu 0` che il termine
   viscoso non sia la causa.

### 8.2 Plateau (`p_13 < 0.8`)

Ordine di indagine consigliato, dal meno invasivo:

1. **`poisson_nit`**: 2 -> 4 (o 3) a CFL fissato; guardare `mgp_resa` e `rms`.
   Se il plateau sparisce, e' la proiezione.
2. **BC**: caso a `nu=0`, `t` breve, confronto con saetta; se `Z` differisce
   gia' a `t=0.1`, il problema e' nel prediction/BC, non nel tempo.
3. **Viscosita'**: `-nu 0` e `-nu 1e-5` a confronto; se `rms` non cambia, la
   viscosita' non c'entra.
4. **IC/diagnostica**: confronto campi a `t=0` (deve essere ~`1e-7`) e
   `Z/E` a `t=0`; se l'IC non coincide, il plateau e' falso.
5. **Griglia**: se il plateau resta a `NX=64`, un test a `NX=128` (se la
   memoria lo consente) dice se e' un errore spaziale. Fuori ambito di questo
   script.

### 8.3 Non monotono

- Controllare che entrambi i run siano a `t=1` e sulla stessa griglia
  (`cells=1048576`, dump `64x64x256`).
- Controllare che il dump non sia di un run precedente (timestamp/hash).
- Controllare timeout e `done_0.txt`; un run troncato produce un dump a un
  tempo diverso.
- Se tutto e' coerente, un `rms` che cresce riducendo il CFL e' un segnale
  forte di problema nel confronto o di instabilita': non accettare.

### 8.4 Viscosita' implicita vs esplicita

La differenza tra esplicito e implicito non e' "la viscosita' mancante": la
decadenza viscosa fisica (`2*nu*Z/E` per unita' di tempo, ~`2.5e-3` a `t=1`) e'
presente in entrambi. La differenza e' il diverso errore di discretizzazione
del termine viscoso, di ordine superiore nel numero di diffusione (piccolo
sulle scale risolte a `nu=1e-5`). **Non e' la spiegazione piu' probabile** di
un ordine ~1. Va considerata solo se:

- si aumenta `nu` o `dt` di ordini di grandezza;
- il test a `nu=0` mostra che il termine viscoso cambia l'ordine;
- si passa a griglie molto piu' fine con `dt` grande (diffusione numero
  vicino a 0.1).

### 8.5 Iterazioni di proiezione

`POISSON_NIT=2` e' il default del prototipo. A `NX=64` il costo di 4 V-cycle
e' modesto; se il plateau e' dovuto alla proiezione, 4 e' il primo test. Se
4 risolve, per la produzione si puo' valutare:

- criterio di convergenza (residuo) invece di iterazioni fisse;
- inizializzazione con `p` del passo precedente (gia' presente);
- multigrid piu' profondo o smoother diverso.

Non cambiare `poisson_nit` tra i CFL della matrice: la matrice deve isolare il
CFL.

### 8.6 Allineamento del `dt`

Per separare l'effetto del tempo da quello dello scheduling:

- **Opzione A (minima):** far partire anche il prototipo con una rampa di
  `dt` simile a `timestep()` di Basilisk (stessa formula), senza cambiare lo
  schema. Utile solo come test diagnostico, non per la produzione.
- **Opzione B:** imporre una sequenza di `dt` fissata (stesso array di passi)
  a entrambi i codici; richiede una modifica al prototipo e non e' banale in
  saetta.
- **Opzione C:** confrontare a `dt` medio simile scegliendo il CFL del
  prototipo in modo che `passi_proto ~ passi_saetta`; e' una forzatura, ma
  isola l'errore spaziale a parita' di costo computazionale.

Nessuna di queste e' necessaria per l'accettazione di §4; servono se il
risultato e' ambiguo.

---

## 9. Elenco comandi

### 9.1 Lancio dello script

```sh
cd "/home/zorka/simulazioni/Test 1"
bash /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/validazione/validate_tempo.sh
```

Variabili utili:

```sh
# percorso del progetto (default: cerca ./mpi_fissa o usa il path noto)
PROJECT="/home/zorka/simulazioni/Test 1/mpi_fissa" \
OUTROOT="/home/zorka/simulazioni/Test 1/mpi_fissa/out/validazione_tempo" \
bash /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/validazione/validate_tempo.sh

# forzare il prototipo su CPU (se la GPU da' problemi)
FORCE_CPU=1 bash .../validate_tempo.sh

# timeout personalizzati (secondi)
T_BUILD_PROTO=1800 T_BUILD_SAETTA=1800 T_RUN_PROTO=1800 T_RUN_SAETTA=3600 \
  bash .../validate_tempo.sh
```

### 9.2 Comandi manuali equivalenti (per debug)

```sh
PROTO="/home/zorka/simulazioni/Test 1/mpi_fissa"
SAETTA="/home/zorka/simulazioni/Test 1/gpu_nvidia_3d/saetta3d.c"
OUT="/home/zorka/simulazioni/Test 1/mpi_fissa/out/validazione_tempo/manuale"
mkdir -p "$OUT"

# prototipo GPU (binari in /tmp, non in mpi_fissa/bin)
NX=64 OUT=/tmp/mf_bin bash "$PROTO/scripts/build.sh" gpu
BIN=/tmp/mf_bin/mpi_fissa_gpu bash "$PROTO/scripts/run_gpu.sh" 1 \
  -tmax 1.0 -cfl 0.3 -nu 1e-5 -poisson_nit 2 -out "$OUT/cfl030/proto"

# saetta3d (in /tmp, path senza spazi)
mkdir -p /tmp/saetta3d_cfl030 && cp "$SAETTA" /tmp/saetta3d_cfl030/
cd /tmp/saetta3d_cfl030
export BASILISK="$HOME/basilisk/src"
export PATH="$PATH:$BASILISK"
CUDA=/usr/local/cuda-13.1/targets/x86_64-linux
export LD_LIBRARY_PATH="$CUDA/lib:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
qcc -autolink -disable-dimensions -O3 -grid=cuda/multigrid3D \
  -I"$CUDA/include" -L"$CUDA/lib" -L"$CUDA/lib/stubs" \
  -DCFL_USER=0.3 -DNU=1e-5 -DMOVIE=0 -DVTK=0 -DMATRIX=1 \
  -DNDIV0=1 -DMAXLEVEL=6 -DTMAX=1.0 \
  saetta3d.c -o saetta3d_gpu -lm -lz
./saetta3d_gpu
```

### 9.3 Confronti

```sh
python3 "$PROTO/scripts/confronta_campi.py" \
  "$OUT/cfl030/proto/omega3d_matrix.bin" \
  /tmp/saetta3d_cfl030/omega3d_matrix.bin \
  --tol-rms 1e-2 --label "proto cfl=0.3" --label "saetta cfl=0.3"

python3 "$PROTO/scripts/confronta_diagnostics.py" \
  "$OUT/cfl030/proto/diagnostics.dat" \
  /tmp/saetta3d_cfl030/diagnostics.dat \
  --label "proto cfl=0.3" --label "saetta cfl=0.3"
```

### 9.4 Test ausiliario: proiezione piu' convergente

Solo se il plateau lo suggerisce (non fa parte della matrice):

```sh
BIN=/tmp/mf_bin/mpi_fissa_gpu bash "$PROTO/scripts/run_gpu.sh" 1 \
  -tmax 1.0 -cfl 0.1 -nu 1e-5 -poisson_nit 4 -out "$OUT/aux_nit4_cfl010"
python3 "$PROTO/scripts/confronta_campi.py" \
  "$OUT/aux_nit4_cfl010/omega3d_matrix.bin" \
  /tmp/saetta3d_cfl010/omega3d_matrix.bin --tol-rms 1e-2
```

### 9.5 Ispezione

```sh
cat "$OUT/tabella.txt"
cat "$OUT/ordine.txt"
head -5 "$OUT"/cfl*/proto/diagnostics.dat
head -5 "$OUT"/cfl*/saetta/diagnostics.dat
awk 'NR>1 {s+=$3; n++} END{printf "dt medio %.6e su %d passi\n", s/n, n}' \
  "$OUT/cfl030/proto/timing.dat"
```

---

## 10. Dedotto vs da misurare

| affermazione | stato | fonte / come verificare |
|---|---|---|
| IC a `t=0` coincide ~`1e-7` | **misurato** | misura del Lead (contesto) |
| `rms_rel` a `t=1` e' `1.4e-2` a CFL 0.3 | **misurato** | misura del Lead |
| prototipo 14 passi, saetta 24 a `t=1` | **misurato** | misura del Lead |
| np1/np2 `~2e-7` (`NX=64` e GPU) | **misurato** | misura del Lead |
| saetta parte da ~0.24 del limite CFL e rampa | **dedotto** | `timestep.h`, `dt0` dei run storici; da confermare con `dt0` dei tre run |
| prototipo parte dal limite CFL | **dedotto** | `compute_dt`; confermato da `out/g1_t1` (`dt0=8.186e-2`) |
| numero di diffusione `~3e-3` | **dedotto/calcolato** | formula `nu*dt/Delta^2`; non misurato |
| `mgp_resa=2.767e-2` a `t=1` | **misurato** | `mpi_fissa/out/g1_t1/diagnostics.dat` (run esistente) |
| griglia del test = `64x64x256` | **dedotto dai flag** | da confermare con `cells=1048576` nel run |
| run6 storico = `128x128x512` | **misurato/letto** | `cells=8388608` in `run6/diagnostics.dat` |
| `sep=0` a `t=0` e' un artefatto | **dedotto** | codice `diagnostics_compute` + `g1_t1`; da confermare confrontando `t=1` |
| ordine atteso ~2 | **atteso, non misurato** | schema BCG/projection; il run lo stima |
| `rms(0.1) ~1.6e-3` se `p=2` | **estrapolazione, non misurata** | da confrontare con la tabella |
| il fallback CPU non cambia la fisica | **atteso, non misurato qui** | T3 propone `<1e-5`; da riconfermare nel log/confronto |
| `-DCFL_USER` e' efficace | **dedotto** | i tre `dt0` di saetta devono scalare con CFL; se non scalano, fermarsi |

---

## 11. Limiti

- Una sola griglia (`64x64x256`) e un solo tempo (`t=1`): non si separa
  l'errore spaziale da quello temporale oltre a quanto fa la matrice CFL.
- Tre soli CFL: la stima dell'ordine ha incertezza `+-0.3` e non distingue
  bene `p=1.8` da `p=2.2`.
- I due codici hanno scheduling di `dt` diversi: la matrice non e' un
  raffrescamento a `dt` identico.
- `poisson_nit=2` e' un parametro fissato; se la proiezione domina, il test
  misura la proiezione, non l'ordine dell'avvezione.
- `np=1`: nessuna informazione sulla decomposizione MPI (gia' validata a
  parte dal Lead).
- La "verita'" e' saetta3d alla stessa griglia, non una soluzione analitica;
  un accordo a `rms` piccolo non dimostra che entrambi siano giusti, solo che
  sono consistenti.

---

## 12. Stato dell'implementazione

- `validate_tempo.sh` (v1) e' in
  `C:\Users\zorka\AppData\Local\Temp\mpi_work\validazione\validate_tempo.sh`.
- **Non e' stato eseguito** in questa sessione: il workspace WSL e' in sola
  lettura per l'agente e non c'e' accesso a shell/compilatori. La sintassi
  bash e' stata rivista a mano; il primo run del Lead e' anche il primo test
  reale dello script.
- Lo script non scrive nel workspace tranne che in
  `mpi_fissa/out/validazione_tempo/` (risultati) e non sovrascrive: binari in
  `/tmp/mpi_fissa_bin_<timestamp>_<pid>`, build saetta in
  `/tmp/saetta3d_cfl<tag>`.
- La tabella viene prodotta anche se una riga fallisce (con `NA` e nota);
  l'exit code e' 1 se una riga e' tecnicamente incompleta, 0 se i dati sono
  completi (l'esito scientifico e' in `tabella.txt`/`ordine.txt`).
- Ordine di esecuzione dello script:
  1. controlla `mpirun`, `mpicxx`, `python3`, `qcc`, CUDA e le librerie
     `liberrors.a`/`libbuda.a` (se mancano, prova a costruirle una volta);
  2. compila il prototipo in `/tmp` e fa lo smoke test `-tmax 0`;
     se la GPU fallisce, passa alla CPU;
  3. per ogni CFL compila saetta in `/tmp/saetta3d_cfl<tag>`, esegue entrambi,
     confronta campi e diagnostiche;
  4. scrive `ambiente.txt`, `tabella.txt`, `ordine.txt`, `dati_ordine.txt`,
     i log per run e i tre confronti diagnostici (`completo`, `t0`, `t1`).
- La nota di ogni riga contiene: `dt_saetta~...`, `dt_ratio~...`,
  `saetta_dt0=...`, `diag_t1_max=...(Z=...,sep=...)`, `backend=gpu|cpu`.
  `saetta_dt0` serve a verificare che `-DCFL_USER` sia davvero efficace: se i
  tre `dt0` sono identici, il flag non e' stato applicato e la matrice CFL di
  saetta e' costante (va segnalato al Lead).
- **Verifiche consigliate prima del run** (lo script non le puo' fare da
  solo):
  ```sh
  bash -n /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/validazione/validate_tempo.sh
  command -v qcc mpirun mpicxx python3 nvcc
  ls "$HOME/basilisk/src/grid/gpu/liberrors.a" \
     "$HOME/basilisk/src/grid/cuda/libbuda.a"
  df -h /tmp "/home/zorka/simulazioni/Test 1"
  nvidia-smi -L
  ```
  Se `nvcc` manca, lo script usa la CPU per il prototipo ma **non** puo'
  compilare saetta: in quel caso serve qcc/CUDA comunque.
- **Cosa non fa lo script:** non riprova un run fallito, non cambia
  `poisson_nit`, non fa la matrice a `NX=128`, non pulisce `/tmp`. I build
  temporanei restano per debug; si possono rimuovere a mano dopo aver letto i
  log.
- Se `out/validazione_tempo/` e' gia' piena, la prima esecuzione crea una
  sottocartella con timestamp: il percorso esatto e' stampato all'inizio e
  alla fine ed e' in `ambiente.txt`.
