# PIANO DI IMPLEMENTAZIONE A MILESTONE — MPI fissa 3D su 2 GPU

Stime in **giorni-uomo** (mezza giornata = 0.5) per una persona che conosce
Basilisk/CUDA/WSL (il Lead). Ogni milestone ha: cosa fare, comandi, criterio di
accettazione, cosa riportare indietro (a me, per correggere il codice).
Le milestone sono in ordine: **non** saltare M2 (invarianza MPI) prima di
guardare la fisica.

Prerequisito comune: copiare questa cartella in `~/mpi_fissa` su **entrambi** i
PC (niente spazi nel percorso; `Test 1` non va bene).

---

## M0 — Ambiente e prerequisiti (0.5 g)

```sh
cd ~/mpi_fissa
chmod +x scripts/*.sh
./scripts/check_env.sh
nvidia-smi
mpirun --version
```

Criterio: `mpirun`, `mpicxx`, `nvcc` (CUDA 13.1), GPU visibile in WSL; il test
`mpirun -np 2 hostname` funziona.

Riportare: output di `check_env.sh`, versione CUDA e modello GPU.

---

## M1 — Build CPU + smoke test a 1 rank (0.5-1 g)

Serve a scoprire subito gli errori di compilazione del codice nuovo, senza
CUDA di mezzo.

```sh
NX=64 ./scripts/build.sh cpu
./scripts/run_cpu.sh 1 -tmax 0.05 -out out/m1_np1
cat out/m1_np1/diagnostics.dat
```

Criterio: compila senza errori; il run arriva a t=0.05 e scrive
`diagnostics.dat`, `omega3d_matrix.bin`, `timing.dat`, `done_0.txt`.
Il valore a t=0 delle colonne `enstrophy` e `energy` deve essere vicino a
quello del riferimento **alla stessa risoluzione** (per NX=128:
`Z=1.25630230e-01`, `E=1.00166653e-03`; per NX=64 il valore sara' diverso,
l'importante e' che sia finito e dell'ordine giusto).

Riportare: eventuali errori di compilazione (copiare **i primi 10 errori**,
non tutto), output del run, e i valori t=0.

---

## M2 — Invarianza MPI: np=1 vs np=2 in CPU (**la milestone chiave**) (1 g)

Se la decomposizione e' corretta, i due run devono coincidere a meno di
arrotondamento fp32 (le due partizioni fanno operazioni in ordine diverso):
**rms relativo < 1e-5**. Qualsiasi differenza piu' grande e' un bug di aloni,
proiezione o consistenza delle facce.

```sh
NX=64 ./scripts/build.sh cpu
./scripts/run_cpu.sh 1 -tmax 0.2 -out out/m2_np1
./scripts/run_cpu.sh 2 -tmax 0.2 -out out/m2_np2
python3 scripts/confronta_campi.py out/m2_np1/omega3d_matrix.bin out/m2_np2/omega3d_matrix.bin \
        --tol-rms 1e-5 --label np1 --label np2
python3 scripts/confronta_diagnostics.py out/m2_np1/diagnostics.dat out/m2_np2/diagnostics.dat --tiny
```

Criterio: entrambi i comandi danno "entro tolleranza".
Se fallisce, mandare: i due `diagnostics.dat` (o solo le righe t=0..0.2), il
valore di `rms_rel` e l'output di `timing.dat` (per capire se il problema e'
all'inizio o cresce nel tempo, e se e' solo all'interfaccia).

Nota: qui si puo' gia' misurare il costo della comunicazione in CPU (poco
significativo per la rete, ma utile per il conteggio dei byte in
`timing.dat`).

---

## M3 — Confronto fisico contro `saetta3d` a L6 (1-2 g)

Serve un riferimento alla **stessa risoluzione** del prototipo. Il solver
esistente gira a `MAXLEVEL=6` (`128x128x512`, ~1.4 GB VRAM) in pochi minuti
per t=1..10.

```sh
# 1) riferimento L6 con il solver esistente (in una cartella sua)
cd ~/mpi_fissa/ref_saetta3d/          # oppure la cartella gpu_nvidia_3d del progetto
EXTRA="-DMAXLEVEL=6 -DTMAX=2 -DMOVIE=0 -DVTK=0" ./build.sh
cp saetta3d_gpu run_ref_L6 && cd run_ref_L6 && ../saetta3d_gpu 2> log.txt
# accumula diagnostics.dat (t=0,1,2)

# 2) prototipo alla stessa risoluzione
cd ~/mpi_fissa
NX=128 ./scripts/build.sh cpu
./scripts/run_cpu.sh 1 -tmax 2 -out out/m3_np1_L6
python3 scripts/confronta_diagnostics.py out/m3_np1_L6/diagnostics.dat \
        <percorso>/run_ref_L6/diagnostics.dat --label prototipo --label saetta3d_L6
python3 scripts/confronta_campi.py out/m3_np1_L6/omega3d_matrix.bin \
        <percorso>/run_ref_L6/omega3d_matrix.bin --tol-rms 2e-2
```

Criterio (atteso, da confermare):
- t=0: `Z`, `E`, `Umax`, `Wmax`, `G`, `sep` devono coincidere a ~1e-6
  relativo (stessa IC, stessi stencil di derivata) — **questo e' il test
  forte** che l'IC e la diagnostica sono giuste;
- t=1,2: `Z` ed `E` entro pochi %; separazione entro pochi %;
  campo `omega_z` rms relativo <= 2e-2 (schemi leggermente diversi: viscosita'
  esplicita, pareti, ordine dei passi; la tolleranza 2e-2 e' una proposta,
  da tarare sui risultati).

Riportare: le due tabelle dei due script, piu' il log del prototipo
(redirigere lo stderr del rank 0 con `2> log.txt`: contiene `ld`, la memoria
per rank e il riepilogo). Se t=0 non coincide, e' un bug di IC o
di diagnostica: mandare i primi valori delle colonne e la griglia.

---

## M4 — Build CUDA + cross-check CPU/GPU (1-2 g)

```sh
cd ~/mpi_fissa
NX=64 ./scripts/build.sh gpu          # se -ccbin fallisce, lo script prova i 2 passi
./scripts/run_gpu.sh 1 -tmax 0.2 -out out/m4_gpu_np1
./scripts/run_cpu.sh 1 -tmax 0.2 -out out/m4_cpu_np1      # stessa NX=64!
python3 scripts/confronta_campi.py out/m4_cpu_np1/omega3d_matrix.bin \
        out/m4_gpu_np1/omega3d_matrix.bin --tol-rms 1e-5 --label cpu --label gpu
```

Criterio: compila per `sm_86`, gira, e CPU vs GPU coincide a < 1e-5 (stesso
algoritmo, differenze solo di arrotondamento; se le GPU usano FMA e la CPU no,
la differenza puo' essere ~1e-6 in rms: tolleranza 1e-5 va bene).
Controllare in `timing.dat` che non ci siano NaN/inf e che `wall_s` sia
plausibile (ms per passo, non secondi).

Riportare: eventuali errori nvcc (primi 10), `wall_s` medio per passo, e se
compare un fallback/errore CUDA.

---

## M5 — Test 2 rank sulla stessa GPU + prime ottimizzazioni (1-2 g)

```sh
NX=128 ./scripts/build.sh gpu
./scripts/run_gpu.sh 2 -tmax 1 -out out/m5_gpu_np2
./scripts/run_gpu.sh 1 -tmax 1 -out out/m5_gpu_np1
python3 scripts/confronta_campi.py out/m5_gpu_np1/omega3d_matrix.bin \
        out/m5_gpu_np2/omega3d_matrix.bin --tol-rms 1e-5 --label np1 --label np2
# tempi: confrontare le due timing.dat (wall_s medio, colonna comm_s)
```

Criterio: np1 vs np2 entro 1e-5; su una sola GPU il tempo non migliora (anzi
peggiora: c'e' la comunicazione in piu'), ma il **conteggio dei byte** in
`timing.dat` deve essere dell'ordine previsto (~5 MB/passo a NX=128 con
`-poisson_nit 2`, ~3 MB con `-poisson_nit 1`; ~20 MB/passo a NX=256 con
`-poisson_nit 2`, ~13 MB con 1).

Riportare: `wall_s` medio per passo per np=1 e np=2, `comm_s` medio, MB per
passo, e il numero di scambi. Questi dati servono a tarare il modello di §6
del DESIGN e a decidere le ottimizzazioni:
- se `comm_s/wall_s > 30%`: aggiungere il "batch di sweeps per scambio"
  (previsto come `MF_SMOOTH_BATCH`, da implementare a richiesta) e/o
  replicare il livello grossolano da un livello piu' alto;
- se la banda e' il limite: ridurre i piani di alone di `u`.

---

## M6 — Run su 2 PC + misure di scalabilita' (2-3 g)

Setup (una volta): ssh master→slave1; progetto in `~/mpi_fissa` **uguale** sui
due PC; `scripts/hostfile_2pc.txt` con gli alias giusti; WSL mirrored
networking (o port-proxy). Vedi `README.md` §5 e `cluster/README.md`.

```sh
# su entrambi i PC: binario compilato con la stessa NX
cd ~/mpi_fissa && NX=128 ./scripts/build.sh gpu

# test rete/MPI (pochi secondi)
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 hostname

# run di scalabilita': stesso TMAX, np=1 (1 GPU) e np=2 (2 PC)
./scripts/run_gpu.sh 1 -tmax 1 -out out/m6_1gpu_L6
./scripts/run_2pc.sh     -tmax 1 -out out/m6_2pc_L6
# ripetere a L8 (NX=256) se la memoria lo consente
```

Criterio: il run np=2 su 2 PC arriva a t=1 con log pulito; `timing.dat` di
np=2 contiene `comm_s` e MB; il confronto campi np=1 vs np=2 e' entro 1e-5.

**Misure da riportare** (sono il cuore del progetto):
1. s/passo a L6 (o L8) con 1 GPU e con 2 PC;
2. `comm_s/wall_s` (%) e MB/passo a 2 PC;
3. speedup = t(1 GPU)/t(2 PC) e la sua dipendenza dalla griglia;
4. se possibile, ripetere il test 2 PC con **cavo** (1 GbE o 2.5 GbE) per
   isolare il costo del Wi-Fi: e' la misura che decide le milestone successive.

Riportare i quattro numeri e i due `timing.dat` (anche solo le righe campione).

---

## M7 — Decisione e produzione (2 g, a valle delle misure)

| esito M6 | azione |
|---|---|
| speedup >= 1.6x e fisica valida | usare il prototipo come solver di produzione per le griglie grandi; migrare gli output (diagnostica gia' compatibile; filmati/VTK da aggiungere) |
| speedup 1.1-1.6x | usare 2 GPU solo quando serve memoria (griglie > 67M celle); altrimenti 1 GPU |
| speedup < 1.1x | tenere 1 GPU per la velocita'; il codice MPI resta per il caso "griglia doppia"; valutare cavo o opzione C (incflo) |
| fisica non valida | correggere il prototipo (viscosita' implicita, allineamento BCG, allineamento tempi dt) prima di ogni altra cosa |

Possibili estensioni (solo se servono): viscosita' implicita (Helmholtz
distribuito, stesso V-cycle); output VTK/movie distribuiti (raccolta con
`MPI_Gather` come per `omega3d_matrix.bin`); restart/checkpoint; 4 rank su 2
PC (2 GPU per PC) se un giorno ci sono 2 GPU per macchina; compressione degli
aloni (solo se la banda e' il collo di bottiglia).

---

## Riepilogo stime

| milestone | stima (giorni-uomo) | cumulato |
|---|---|---|
| M0 ambiente | 0.5 | 0.5 |
| M1 build CPU + smoke | 0.5-1 | 1-1.5 |
| M2 invarianza MPI | 1 | 2-2.5 |
| M3 confronto fisico L6 | 1-2 | 3-4.5 |
| M4 build CUDA + CPU/GPU | 1-2 | 4-6.5 |
| M5 2 rank 1 GPU + ottimizzazioni | 1-2 | 5-8.5 |
| M6 2 PC + misure | 2-3 | 7-11.5 |
| M7 decisione/produzione | 2 | 9-13.5 |

Senza contare l'eventuale **porting in Basilisk** (opzione B, 6-12 settimane se
si vuole la fisica identica al solver attuale) e senza l'eventuale
**viscosita' implicita** distribuita (+2-4 giorni).
