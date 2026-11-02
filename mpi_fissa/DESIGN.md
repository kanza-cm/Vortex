# DESIGN — MPI (domain decomposition) per il solver 3D a griglia fissa su GPU

**Stato documento**: 2026-11-02. Scritto in una sessione di progetto che **non**
ha potuto compilare/eseguire nel workspace (vedi §9 "cosa non e' verificato").
Il codice del prototipo e' in `src/mpi_fissa.cu`; le istruzioni in `README.md`;
il piano di lavoro in `PIANO_MILESTONE.md`; la validazione in `VALIDAZIONE.md`.

---

## 1. Obiettivo

Far girare **una singola simulazione** del caso "vortex coalescing" 3D a
griglia fissa (dominio 1x1x4, celle cubiche, `nx=ny=NDIV0*2^MAXLEVEL`,
`nz=4*nx`) su **2 PC con WSL, una GPU da 8 GB ciascuno**, collegati in Wi-Fi
locale a 1 m. Riferimento fisico/numerico:
`gpu_nvidia_3d/saetta3d.c` (Basilisk `navier-stokes/centered.h`, backend
`grid=cuda/multigrid3D`, singola precisione), che e' **single-GPU**.

Caso di produzione attuale: `MAXLEVEL=8`, `NDIV0=1` → 256x256x1024 =
67.1M celle, ~7.2-7.5 GB di VRAM (dato misurato, riportato nella testata di
`saetta3d.c`) su una RTX 3070 da 8 GB. E' quindi **al limite della memoria**:
oggi non si puo' ne' aumentare la griglia ne' tenere piu' campi.

## 2. Perche' non basta "usare 2 GPU in task parallel"

Il `cluster/README.md` documenta gia' il task parallelism (2 run indipendenti,
uno per GPU): raddoppia il *throughput* di una campagna di run, ma **non fa
girare una singola simulazione su 2 GPU** e non aumenta la memoria per run.
Per una singola simulazione servono, in un modo o nell'altro:
(i) decomposizione di dominio con scambio di aloni, e
(ii) una **proiezione (Poisson) distribuita**.

## 3. Cosa c'e' e cosa manca (verificato sui sorgenti in questo workspace)

| elemento | stato |
|---|---|
| `gpu_nvidia_3d/saetta3d.c` (fisica, IC, diagnostica, output) | presente, stabile |
| backend CUDA di Basilisk `~/basilisk/src/grid/cuda/` | solo `cartesian.h`, `multigrid.h`, `multigrid3D.h`, `cuda.c`: **nessun supporto MPI, nessuna decomposizione** |
| `centered.h` | proiezione **globale**: `project()` = multigrid su tutto il dominio, ad ogni passo (2 proiezioni/passo: mezzio passo per l'avvezione + passo intero) |
| `gpu_amr_3d/` (incflo/AMReX, MPI+CUDA) | esiste ed e' gia' MPI, ma e' un **altro solver** (MOL + proiezione incflo, AMR): non e' la "fissa" e i risultati non coincidono con `saetta3d` senza una nuova validazione |
| `cluster/` (ssh+rsync, hostfile, dispatch) | presente: riusabile per il run a 2 PC |

Numeri utili (dai log gia' presenti, non misurati in questa sessione):

- L8 (`256x256x1024`): 67.1M celle, `2.4e7 points.step/s` misurati su RTX 3070
  → **~2.8 s per passo** (log `error_study.../run7/log.txt`, 21 passi in
  59 s reali); la produzione (`risultati/diagnostica/`) ha fatto ~38000 passi
  fino a t=1500, coerente con ~1 giorno di run.
- L6 (`128x128x512`): 8.4M celle, `5.26e6 points.step/s` → ~1.6 s/passo
  (run6, 13 passi in 20.7 s reali), con multigrid che converge in 1 iterazione
  (`mgp.i=1`) quasi sempre.
- Il costo per passo e' dominato dal **multigrid della proiezione** e
  dall'accesso memoria (24 campi fp32), non dal calcolo puro: c'e' quindi
  margine per un solver prototipo piu' snello, ma il punto e' che **la
  quantita' di comunicazione per passo e' piccola rispetto a 2.8 s**.

## 4. Opzioni considerate

| opzione | cosa | sforzo | rischio | verdetto |
|---|---|---|---|---|
| **A. prototipo standalone MPI+CUDA** (questo lavoro) | riscrivere il caso (avvezione BCG + proiezione MAC) in un solver autonomo con slab decomposition lungo z e multigrid distribuito; validarlo contro `saetta3d`; usarlo come banco di misura e, se la fisica torna, come nuovo solver | 2-3 settimane-uomo | medio (nuovo codice numerico, ma tutto sotto controllo) | **raccomandata adesso** |
| **B. patchare Basilisk** (`grid.h` + `cuda/multigrid3D.h` + `centered.h` + `qcc`) per l'MPI | mantenere esattamente la fisica/numerica attuale con 2 GPU | 6-12 settimane-uomo molto esperte | **alto**: il backend CUDA e il generatore di codice non hanno alcun gancio MPI; servirebbe un multigrid MPI *dentro* il backend, piu' boundary/riduzioni distribuite | non ora; valutare solo se A fallisce |
| **C. incflo/AMReX uniforme con MPI+CUDA** | usare il solver gia' MPI (`gpu_amr_3d`, `max_level=0`) | 3-5 giorni per configurare/validare | medio: schema diverso → serve validazione; overhead memoria AMR (~300-550 B/cella) limita la griglia | **fallback** se serve "subito" una singola simulazione multi-GPU |
| **D. task parallelism** | 2 run indipendenti | 0 | nessuno | gia' disponibile; non risolve l'obiettivo |

Il prototipo A e' anche il modo piu' economico per **misurare** la vera
domanda: *il Wi-Fi a 1 m basta per una decomposizione di dominio?* Se i numeri
di A dicono no, non ha senso spendere settimane su B.

## 5. Architettura del prototipo A (`src/mpi_fissa.cu`)

### 5.1 Griglia e decomposizione

- Griglia uniforme cell-centered per `u,p`, facce MAC per `uf` (come
  `centered.h`). Dominio `[-0.5,0.5]x[-0.5,0.5]x[0,4]`, `Delta=1/NX`,
  `nz=4*NX`.
- **Decomposizione a lastre (slab) lungo z**: ogni rank ha
  `[K0,K1)` con `K1-K0 = nz/nranks`. Scelta obbligata: la direzione lunga e'
  z (4x), quindi il numero di facce di interfaccia e' minimo (1 sola
  interfaccia con 2 rank) e il volume di alone per passo e'
  `2*nx*ny*4 B` per piano scambiato.
- Aloni: **2 piani in z** (servono al calcolo delle facce `uf.z`
  all'interfaccia con lo stencil upwind di 2° ordine), **1 in x/y** (pareti
  fisiche); pressione: **1 piano** (stencil a 7 punti).
- Pareti fisiche: x,y,z **free-slip** (come il default di Basilisk per `u`:
  componente normale specchiata con segno -, tangenziali con segno +;
  pressione Neumann omogeneo = ghost uguale all'interno).
- z **non periodico** (coerente con `saetta3d.c`: lo dice la testata del
  sorgente; il commento "periodic z" in `build.sh` e' un residuo).

### 5.2 Schema numerico (scelto per poter confrontare con `saetta3d`)

Replica di `centered.h`:
1. `dt` da CFL = 0.3 su `max|uf|` (crescita max 10%/passo, allineato ai tempi
   di diagnostica come fa `dtnext`);
2. `prediction()` di `uf` a `t+dt/2` (upwind con correzione di pendenza,
   termini tangenziali) — **fonte di §5.6**;
3. `project(uf, pf, dt/2)` (Poisson + correzione facce);
4. `advection(u, uf, dt, g)` — schema BCG "unsplit" conservativo, con termini
   tangenziali di velocita' di faccia;
5. viscosita': **esplicita** `u += nu*dt*lap(u)` (differenza voluta: a
   `nu=1e-5` il numero di diffusione e' ~1e-2, l'errore e' trascurabile; la
   versione implicita e' documentata come possibile estensione);
6. `uf = media facce di u` (event `acceleration`), `project(uf, p, dt)`;
7. `g = -grad(p)` centrato, `u += dt*g`.

La pressione `p` e `pf` sono **persistenti** tra i passi (stima iniziale del
multigrid), come in Basilisk.

### 5.3 Proiezione distribuita: multigrid geometrico V-cycle

- Livelli `l=0..LD` **distribuiti** (stessi dims dimezzati: `nx>>l`, `nz>>l`,
  fette z divisibili per `nranks`); `LD` e' l'ultimo livello in cui
  `nx>>l >= 2`, `nz>>l >= 2*nranks`.
- Livello `LD+1` **replicato**: e' minuscolo (per costruzione `1x1x4` celle),
  viene ricostruito su tutti i rank con un `MPI_Allgather` della fetta
  ristretta (poche centinaia di byte) e risolto **identicamente** su ogni rank
  con `MF_COARSE_SWEEPS=30` sweeps di red-black Gauss-Seidel. Questo evita
  qualunque caso speciale di "coarse solve distribuito" e mantiene i rank
  sincronizzati.
- Smoothing: red-black Gauss-Seidel (deterministico, adatto a GPU; come il
  backend CUDA di Basilisk), 1 sweep per ciclo V, 4 relaxation totali per
  V-cycle (2 prima, 2 dopo il coarse).
- **Iterazioni FISSE**: `POISSON_NIT` (default **2**) V-cycle per proiezione e
  **nessuna allreduce nel ciclo caldo** della proiezione: i rank restano
  sincronizzati per costruzione, non per riduzione di norme. Le norme/residui
  si calcolano solo ai tempi di diagnostica (1 allreduce/unità di tempo).
  L'unica riduzione globale per passo e' il **massimo di `|uf|` per il dt**
  (necessaria: il passo deve essere identico su tutti i rank).
  Con `-poisson_nit 1` si dimezza la comunicazione ma resta piu' divergenza
  residua (da misurare).
- Restrizione full-weighting 2x2x2, interpolazione trilineare cell-centered
  (pesi 0.75/0.25). Lo slab z di ogni rank e' allineato ai blocchi 2x2x2, per
  cui restrizione e prolungamento sono **locali**.

### 5.4 Comunicazione host-staging (CUDA <-> MPI)

- `MPI_Sendrecv` **solo host**; i campi restano su device. Si usano buffer
  host **pinned** (`cudaHostAlloc`): `cudaMemcpy2D` device→host del piano
  (pitch = `(nx+2)*4`, larghezza `nx*4`, altezza `ny`), `MPI_Sendrecv`, poi
  `cudaMemcpy2D` host→device nel ghost.
- Una sola `Sendrecv` per direzione e per piano (i 3 componenti di `u`
  impacchettati insieme): niente `MPI_Allreduce` nel ciclo caldo; una sola
  `MPI_Allgather` per V-cycle (livello grossolano).
- **Nessun GPUDirect RDMA** (GPU consumer + Wi-Fi): il passaggio da host e'
  inevitabile; il costo e' ~2 copie PCIe per piano (marginale rispetto alla
  rete).

### 5.5 Ottimizzazioni di comunicazione gia' previste

- `u` scambiata con 2 piani solo una volta per passo (prima del `prediction`);
  viscosita' e accelerazione usano **1 piano** (stencil a 7 punti) → alone
  `u` per passo: 2+1+1 piani/verso x 3 componenti.
- Facce `uf.z` all'interfaccia calcolate **su entrambi i lati con gli stessi
  dati** (i 2 aloni rendono l'espressione identica): niente scambio delle
  facce normali dopo il `prediction`.
- Il flusso alle facce z di interfaccia usa la velocita' **centrata** nella
  cella upwind invece della media delle due facce (le facce x/y del rank
  vicino non ci sono): differenza O(Delta^2) su **un solo piano**, i due rank
  calcolano comunque lo stesso numero (consistenza conservativa).
- Scambi solo in z (nessuna comunicazione x/y: la griglia e' intera in x/y).

### 5.6 Scelte di precisione

- Campi `float` (fp32) come `saetta3d`; IC calcolata in double e poi
  troncata a float; diagnostica in double a partire dai float, con le
  differenze finite calcolate **in float** come le macro `WX/WY/WZ` di
  `saetta3d.c` (per riprodurre i valori di riferimento).
- La proiezione distribuita non introduce riduzioni globali (l'unica
  allreduce per passo e' il massimo di `|uf|` per il dt); le differenze
  np=1 vs np=2 devono restare a livello di arrotondamento fp32.

## 6. Modello di costo della comunicazione

Sia `A = nx*ny` (facce di interfaccia), `np=2` rank.

| voce | byte per passo (POISSON_NIT=1) | L6 (nx=128) | L8 (nx=256) |
|---|---|---|---|
| aloni `u` (2+1+1 piani x 3 comp x 2 versi) | vedi nota | 1.5 MB | 6.3 MB |
| pressione: 5 scambi/livello x `2*A_l*4` x 8 livelli x 2 proiezioni | ~ | 1.8 MB | 7.0 MB |
| allgather coarse (per V-cycle) | trascurabile | <1 KB | <1 KB |
| **totale con `-poisson_nit 1`** | | **~3.3 MB/passo** | **~13.3 MB/passo** |
| **totale con default `-poisson_nit 2`** | x2 sulla pressione | **~5.1 MB/passo** | **~20 MB/passo** |

Nota: aloni `u` = (2 piani prima + 1 dopo advezione + 1 dopo viscosita') x
2 versi x 3 componenti x `A*4 B`; per L8: (2+1+1)*2*3*256*256*4 = 6.3 MB.

A 80 MB/s (Wi-Fi 5/6 misurata "40-150 MB/s" nel `cluster/README.md`) e
2-5 ms di latenza per messaggio:

| scenario | banda | latenza | tempo/passo solo comunicazione (L8, NIT=2) | note |
|---|---|---|---|---|
| Wi-Fi 5, 40-80 MB/s | 40-80 | 3-5 ms | **0.6-1.0 s** | troppi scambi piccoli: la latenza domina |
| Wi-Fi 6, 80-150 MB/s | 80-150 | 2-3 ms | **0.25-0.5 s** | plausibile |
| cavo 1 GbE | 110 | 0.3 ms | ~0.2 s | opzione se serve velocita' |
| cavo 2.5 GbE / USB4 | 250-1000 | 0.2 ms | 0.03-0.08 s | |

Con `-poisson_nit 1` i numeri di pressione dimezzano (L8: ~0.15-0.6 s di
solo comunicazione). Il numero di **messaggi** per passo e' il vero rischio:
con V-cycle a 8 livelli ci sono ~80 scambi (2 Sendrecv ciascuno) per
proiezione => ~320 messaggi per passo con NIT=2 (L8). Se la latenza e' 2 ms e
il send e il receive non si sovrappongono, sono ~0.5-0.6 s/passo. Per questo
il prototipo espone:
- `-poisson_nit 1`: dimezza la comunicazione della proiezione (default 2 per
  accuratezza: con 1+1 sweeps per livello un solo V-cycle lascia piu'
  divergenza residua);
- (estensione prevista, milestone M5) **smoothing con ghost "in ritardo"**
  (`smooth_batch`), cioe' piu' sweeps di Jacobi/GS per scambio, che riduce i
  messaggi di 2-4x barattando un po' di convergenza;
- (estensione) replica del livello grossolano da un livello piu' alto
  (es. 16x16x64 = solo 16k celle) per dimezzare i livelli scambiati.

**Conclusione del modello**: per il *solver Basilisk attuale* (2.8 s/passo a
L8) 13-20 MB/passo (~0.2-1.0 s di comunicazione) lascia comunque ~1.5-2x di
speedup: la decomposizione avrebbe senso. Per il **prototipo** (che e' molto
piu' snello, probabilmente 0.1-0.3 s/passo su una 3070) il Wi-Fi puo' invece
rendere il run a 2 GPU **non piu' veloce** di quello a 1 GPU: il guadagno
certo e' la **memoria** (2x 8 GB = griglie fino a ~130M celle invece di 67M),
non il tempo. Questa e' la conclusione onesta del design, da verificare con le
misure della milestone M6 (e, se serve, con `-poisson_nit 1`).

## 7. Memoria per rank e griglie possibili

Campi per cella (byte fp32): `u(3)+uf(3)+flux(3)+g(3)+p+pf+rhs+res` = 16
campi + gerarchia multigrid (~1.14x di 3 campi, ~3.4) ≈ **~19.4 float/cella =
78 B/cella** (piu' aloni ~2%). Contando il contesto CUDA (~0.4-0.8 GB) e i
buffer pinned (~10 MB):

| griglia | celle totali | per rank (2 rank) | 2 GPU 8 GB | 1 GPU 8 GB |
|---|---|---|---|---|
| L6 `128x128x512` | 8.4M | 0.34 GB | comodo | comodo |
| L7 `256x256x1024` = L8 | 67.1M | 2.7 GB | **si'** | al limite (reference usa 7.5 GB) |
| `256x256x2048` (LZ=8) | 134M | 5.3 GB | **si'** (con attenzione) | no |
| `320x320x1280` (NDIV0=5/L6) | 131M | 5.2 GB | si' | no |
| L9 `512x512x2048` | 537M | 21 GB | no | no |

Quindi il valore concreto del progetto: **griglia doppia** (134M celle) a
parita' di tempo, oppure L8 con piu' campi diagnostici.

## 8. Rischi e mitigazioni

| rischio | probabilita' | impatto | mitigazione |
|---|---|---|---|
| **proiezione distribuita**: divergenza/convergenza non basta con 1 V-cycle fisso | media | alto (fisica sbagliata) | validare prima in CPU np=1 vs np=2 (invarianza), poi contro saetta3d; `-poisson_nit 2..3`; confrontare residuo massimo |
| **latenza Wi-Fi** (troppi messaggi piccoli) | alta | alto (2 GPU non piu' veloci) | batch di sweeps per scambio; meno livelli scambiati; misurare `timing.dat` (colonna comm_s); in caso estremo: cavo |
| **banda Wi-Fi** (13-20 MB/passo a L8) | media | medio | `u` con 1 piano dove basta; scambi solo z; `-poisson_nit 1`; eventualmente compressione (non ora) |
| memoria per rank a griglie grandi | media | medio | 2 GB/rank a L8: ampio margine su 8 GB; guardia su `-NX`; controllo con `nvidia-smi` |
| bug di indici/aloni (silenziosi!) | alta | alto | confronto np=1 vs np=2 **bit-per-bit-like** (rms < 1e-5) obbligatorio in milestone M2; poi CPU vs GPU |
| differenze fisiche vs saetta3d (schema leggermente diverso: viscosita' esplicita, facce uf forzate a 0 alle pareti, flusso O(Delta^2) all'interfaccia) | alta | medio | VALIDAZIONE.md: t=0 deve coincidere a ~1e-6; evoluzione entro tolleranze dichiarate; in caso, si puo' tornare alla viscosita' implicita o allineare le scelte |
| percorso con spazi (`Test 1`) | certa | basso | copiare in `~/mpi_fissa` (nessuno spazio) sia su master sia su slave |
| WSL networking per MPI a 2 PC (mirrored vs NAT) | media | medio | `cluster/setup_slave.ps1` + `cluster/README.md`; hostfile con alias ssh; in alternativa port-proxy 2222 |
| driver/CUDA 13.1 e RTX 3070 in WSL (libnvrtc, libcuda) | bassa | basso | gia' risolto per `gpu_nvidia_3d`; riusare `~/cuda-env.sh` |

## 9. Cosa NON e' verificato in questa sessione (dichiarazione esplicita)

La sessione che ha scritto questi file **non ha potuto compilare ne' eseguire**
nulla nel workspace WSL: `pwsh` fallisce sul path UNC del workspace, `write`
funziona solo in `%TEMP%`, e non c'e' un altro modo di invocare `g++/nvcc/mpirun`
da qui. Quindi:

- `src/mpi_fissa.cu` **non e' stato compilato**: e' scritto per essere compilato
  con i comandi di `README.md`, ma possono esserci errori di compilazione da
  correggere (la sessione e' pronta a iterare sui log);
- **nessuna misura** di tempo, banda o scalabilita' e' stata fatta qui: i
  numeri di §6 sono un modello analitico dichiarato, non misure;
- i valori di riferimento della fisica (`VALIDAZIONE.md`,
  `riferimento/valori_riferimento.md`) sono invece **letti dai file gia'
  presenti** nel workspace (`gpu_nvidia_3d/risultati/...`), e la loro
  provenienza e' citata.

## 10. Decisioni da prendere dopo le misure (milestone M6)

1. Se `comm_s/wall_s < ~20%` a L8 su 2 PC: la decomposizione conviene →
   si puo' valutare di usare il prototipo come solver di produzione (dopo
   validazione fisica) oppure di portare la strategia in Basilisk (opzione B).
2. Se `comm_s/wall_s > ~50%`: 2 GPU solo per **memoria** (griglie > 67M);
   per la velocita' restare su 1 GPU; se serve velocita' a parita' di griglia,
   l'unica strada e' il cavo o l'opzione C (incflo, il cui multigrid AMReX e'
   molto piu' ottimizzato sulla comunicazione).
3. Se la fisica del prototipo non passa la validazione: tornare a `saetta3d`
   su 1 GPU e usare il prototipo solo come esperimento di scalabilita'.
