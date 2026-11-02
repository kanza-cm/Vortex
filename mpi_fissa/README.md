# mpi_fissa — prototipo MPI (slab lungo z) per il caso 3D a griglia fissa su GPU

Prototipo **standalone** (senza Basilisk) del caso "vortex coalescing" 3D di
`gpu_nvidia_3d/saetta3d.c`, con **decomposizione di dominio MPI** e backend
**CUDA** opzionale, pensato per far girare **una singola simulazione su 2 GPU
da 8 GB** (2 PC WSL in Wi-Fi locale). Non e' il solver di produzione: e' il
banco di prova per (a) validare la proiezione distribuita, (b) misurare se e
quanto conviene la decomposizione su questa rete, (c) decidere se portare la
strategia in Basilisk o usare questo codice.

- **Stato**: codice scritto e documentato, **non compilato** nella sessione che
  lo ha prodotto (nessun accesso a g++/nvcc/mpirun dal workspace WSL di
  quella sessione). La prima cosa da fare e' la milestone M1 di
  `PIANO_MILESTONE.md`.
- Documenti: `DESIGN.md` (architettura e modello di costo),
  `PIANO_MILESTONE.md` (passi e stime), `VALIDAZIONE.md` (come si valida),
  `riferimento/` (valori di riferimento letti dal workspace).

## 1. Contenuto

```
mpi_fissa/
├── README.md                  questo file
├── DESIGN.md                  fattibilita', architettura, modello di comunicazione
├── PIANO_MILESTONE.md         piano a milestone con stime e criteri
├── VALIDAZIONE.md             protocollo e tolleranze di validazione
├── src/mpi_fissa.cu           solver: un solo file, compila CPU e CUDA
├── scripts/
│   ├── build.sh               compila CPU e/o GPU (nvcc -ccbin mpicxx)
│   ├── run_cpu.sh             run CPU con 1 o 2 rank
│   ├── run_gpu.sh             run GPU con 1 o 2 rank (stessa macchina)
│   ├── run_2pc.sh             run su 2 PC (hostfile, ssh)
│   ├── hostfile_locale.txt    2 rank su un PC
│   ├── hostfile_2pc.txt       1 rank per PC
│   ├── check_env.sh           controllo prerequisiti
│   ├── confronta_diagnostics.py
│   └── confronta_campi.py     confronto omega3d_matrix.bin
└── riferimento/
    ├── valori_riferimento.md  tabelle t=0 L6/L8 e trend L8
    └── ref_L8_diagnostics_estratto.dat
```

## 2. Prerequisiti

Su **ogni** PC (master e slave), dentro WSL2 Ubuntu:

- NVIDIA driver su Windows, GPU visibile in WSL (`nvidia-smi`, `/dev/dxg`,
  `/usr/lib/wsl/lib/libcuda.so*`); va bene il setup gia' usato per
  `gpu_nvidia_3d` (`cluster/gpu_check.sh`).
- CUDA toolkit 13.1 in `~/cuda-13.1` (o `CUDA=/usr/local/cuda-...`):
  servono `nvcc` e `libcudart`; `libnvrtc` non serve a questo prototipo.
- **OpenMPI** (con `mpirun`, `mpicxx`): `sudo apt install openmpi-bin libopenmpi-dev`
  (o `mpich`; gli script usano `mpicxx`/`mpirun`).
- `python3` per gli script di confronto (numpy facoltativo, solo per
  `confronta_campi.py`; senza numpy c'e' un fallback lento).
- `ssh` master→slave senza password (solo per il run a 2 PC; vedi §5).
- **Percorso senza spazi**: copiare questa cartella in `~/mpi_fissa` su
  entrambi i PC. (Il workspace `Test 1` contiene uno spazio: `nvcc`, `qcc` e
  alcuni tool MPI non lo digeriscono.)

Controllo: `./scripts/check_env.sh`.

## 3. Compilazione

Il solver e' **un solo file** `src/mpi_fissa.cu`, compilabile in due modi:

- **CPU** (OpenMP): `mpicxx -x c++ -DMF_USE_CUDA=0` — serve a validare la
  decomposizione e a fare i confronti "np=1 vs np=2" senza GPU;
- **GPU**: `nvcc --extended-lambda -arch=sm_86 -ccbin mpicxx` — le stesse
  routine numeriche scritte una volta con `parallel_for` + lambda `__device__`.

```sh
cd ~/mpi_fissa
chmod +x scripts/*.sh

# default: NX=128 -> griglia 128x128x512 (come MAXLEVEL=6 di saetta3d)
./scripts/build.sh              # CPU + GPU

# griglie esplicite (il valore e' unico: nx=ny=NX, nz=4*NX, Delta=1/NX)
NX=64  ./scripts/build.sh cpu   # test veloce (1M celle)
NX=128 ./scripts/build.sh gpu   # L6-equivalente (8.4M celle)
NX=256 ./scripts/build.sh gpu   # L8-equivalente (67M celle) - serve per il run "vero"

# altre variabili
ARCH=86 CUDA=~/cuda-13.1 ./scripts/build.sh gpu
```

Note:
- se `-ccbin mpicxx` non funziona sul vostro nvcc, `build.sh` prova
  automaticamente la compilazione in due passi (`nvcc -c` + link con
  `mpicxx`);
- il valore di `NX` e' **compile-time**: i binari con NX diversi sono
  incompatibili; compilare con la stessa `NX` su tutti i PC del run;
- requisito sulla decomposizione: `nz = 4*NX` deve essere divisibile per il
  numero di rank (con 2 PC va bene; 1, 2 o 4 rank sono i casi previsti);
- ogni rank vede una sola GPU (`cudaSetDevice(0)`);
- le routine numeriche sono scritte una volta e girano sia su CPU sia su GPU
  tramite `parallel_for` + lambda `__device__` (richiede `--extended-lambda`,
  gia' in `build.sh`). Se il vostro nvcc rifiutasse le lambda `__device__`,
  la build **CPU** resta comunque valida (e' la via principale per M1-M3);
  in tal caso mandatemi i primi errori nvcc e converto le routine in kernel
  espliciti.

## 4. Esecuzione locale (1 PC)

### 4.1 CPU, 1 rank (smoke test)

```sh
./scripts/run_cpu.sh 1 -tmax 0.05 -out out/smoke_np1
cat out/smoke_np1/diagnostics.dat
```

### 4.2 CPU, 2 rank (test di invarianza MPI — **importante**)

```sh
./scripts/run_cpu.sh 1 -tmax 0.2 -out out/m2_np1
./scripts/run_cpu.sh 2 -tmax 0.2 -out out/m2_np2
python3 scripts/confronta_campi.py out/m2_np1/omega3d_matrix.bin \
        out/m2_np2/omega3d_matrix.bin --tol-rms 1e-5 --label np1 --label np2
python3 scripts/confronta_diagnostics.py out/m2_np1/diagnostics.dat \
        out/m2_np2/diagnostics.dat --tiny
```

Atteso: "entro tolleranza". Se no, **non** guardare la fisica: e' un bug di
aloni/proiezione (mandare i log e i valori di confronto).

### 4.3 GPU

```sh
./scripts/run_gpu.sh 1 -tmax 1   -out out/gpu_np1      # 1 GPU
./scripts/run_gpu.sh 2 -tmax 1   -out out/gpu_np2      # 2 rank sulla stessa GPU
```

Con 2 rank sulla stessa GPU la memoria per rank si dimezza (utile per testare
il caso L8 senza il PC slave), ma il tempo peggiora: non e' una misura di
speedup, solo di correttezza.

### 4.4 GPU, 2 PC

Prerequisiti: progetti in `~/mpi_fissa` su entrambi i PC, binario GPU
compilato con la stessa `NX`, ssh master→slave, `scripts/hostfile_2pc.txt`
con l'alias giusto (default `slave1`).

```sh
# test rete (2 secondi)
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 hostname

# run
./scripts/run_2pc.sh -tmax 1 -out out/2pc_L6
```

`run_2pc.sh` usa `mpirun --hostfile ... -np 2 --map-by ppr:1:node --bind-to none
--wdir $PWD -x LD_LIBRARY_PATH -x OMP_NUM_THREADS`. Se `hostname` o il run
falliscono, vedere §6.

## 5. Setup 2 PC (una volta)

1. **Rete**: stessa Wi-Fi (5/6 GHz); meglio il **mirrored networking** di WSL2
   (`%USERPROFILE%\.wslconfig` sul PC slave):
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   poi `wsl --shutdown` e riaprire. In alternativa usare `-PortProxy` di
   `cluster/setup_slave.ps1` (porta 2222).
2. **SSH**: su master
   ```sh
   cat >> ~/.ssh/config <<'CFG'
   Host slave1
       HostName 192.168.1.11      # IP del PC slave
       User zorka
       Port 22                   # 2222 se usate il port-proxy
   CFG
   ssh-copy-id slave1 && ssh slave1 hostname
   ```
3. **Progetto**: rsync/copia di questa cartella in `~/mpi_fissa` su entrambi i
   PC (stesso percorso), poi `./scripts/build.sh gpu` su entrambi (con la
   stessa `NX`).
4. **Hostfile**: `scripts/hostfile_2pc.txt` = `localhost slots=1` +
   `<alias> slots=1`.
5. **CUDA da ssh**: `run_2pc.sh` propaga `LD_LIBRARY_PATH` con `-x`; se il
   vostro setup usa `~/cuda-env.sh`, eseguite il run dentro `bash -lc` o
   adattate `LD_LIBRARY_PATH` (`/usr/lib/wsl/lib` deve esserci per `libcuda`).

## 6. Output e lettura dei risultati

Ogni run crea `out/<nome>/` con:

| file | contenuto |
|---|---|
| `diagnostics.dat` | serie temporale, **stesse colonne di saetta3d**: `t dt it enstrophy energy umax wmax circulation_slice separation xc yc cells mgp mgp_resa` |
| `omega3d_matrix.bin` | `omega_z` finale: `int32 nx,ny,nz` + `float32` (i piu' veloce), formato identico a `saetta3d` |
| `timing.dat` | per passo: `step t dt wall_s comm_s comm_MB n_exch` — **qui si legge il costo della comunicazione** |
| stderr del rank 0 | griglia, slab, livelli MG, memoria stimata, riepilogo finale (salvarlo con `2> log.txt`) |
| `done_0.txt` | marker di fine (per gli script) |

Esempio: `awk 'NR>1 {w+=$4; c+=$5; n++} END {printf "wall medio %.3f s, comm %.3f s (%.1f%%)\n", w/n, c/n, 100*c/w}' out/2pc_L6/timing.dat`.

## 7. Cosa NON fa (limiti noti)

- **viscosita' esplicita** invece che implicita: differenza controllata a
  `nu=1e-5` (vedi `VALIDAZIONE.md`);
- **solo** decomposizione lungo z (e' la scelta giusta per questo dominio, ma
  non generalizza a domini "cubi");
- niente restart/checkpoint, niente VTK/movie (per la validazione basta
  `omega3d_matrix.bin` + diagnostica);
- niente adattivita' (e' la "fissa" per definizione);
- le ottimizzazioni di comunicazione previste (batch di sweeps per scambio,
  replica di livelli piu' alti) **non sono implementate**: si aggiungono solo
  se le misure della milestone M5/M6 le rendono necessarie (vedi DESIGN §6/§10);
- se la fisica non passa T4, le leve sono: `-poisson_nit 2/3`, viscosita'
  implicita, allineamento della politica di `dt` (vedi `VALIDAZIONE.md` §4).

## 8. Primo giro da fare (riassunto)

```sh
cd ~/mpi_fissa
chmod +x scripts/*.sh
./scripts/check_env.sh
NX=64 ./scripts/build.sh cpu                   # M1: scopre gli errori di compilazione
./scripts/run_cpu.sh 1 -tmax 0.05 -out out/m1_np1
NX=64 ./scripts/build.sh cpu && ./scripts/run_cpu.sh 1 -tmax 0.2 -out out/m2_np1 \
  && ./scripts/run_cpu.sh 2 -tmax 0.2 -out out/m2_np2    # M2: invarianza MPI
python3 scripts/confronta_campi.py out/m2_np1/omega3d_matrix.bin out/m2_np2/omega3d_matrix.bin --tol-rms 1e-5
```

Poi si passa a M3 (`VALIDAZIONE.md` §5) e a M4/M5/M6 di
`PIANO_MILESTONE.md`. Il criterio per proseguire oltre M2 e' **solo** la
correttezza della decomposizione, non la velocita'.

## Stato validazione (misurato)

| test | risultato |
|---|---|
| IC vs `saetta3d` (64x64x256, t=0) | differenze ~1e-7 |
| MPI np=1 vs np=2 (CPU, t=0.2, CFL 0.1) | rms_rel 5.0e-7 |
| MPI np=1 vs np=2 (GPU, t=0.05) | rms_rel 2.1e-7 |
| L8 (256x256x1024) np=1 vs np=2 | rms_rel 2.8e-5 |
| CPU vs GPU (np=1, NX=64) | rms_rel 1.4e-6 |
| campo a t=1 vs `saetta3d` (CFL 0.1) | rms_rel 9.98e-3, linf 6.66e-2 |

Il residuo ~1% a t=1 e' concentrato vicino alle pareti x/z. Verificato: BC
free-slip identiche a `centered.h`, proiezione irrilevante (2-8 V-cycle),
operatore viscoso ora allineato (`div(2 mu D)`, vedi `docs/BC_VISCOSITA.md`).
Restano come cause note: flusso normale alle pareti forzato a zero nel
prototipo (O(Delta)), sequenze di dt diverse (Basilisk limita la crescita
del dt), dettagli di advezione. La diagnostica `separation` e' stata
corretta (tie-break dell'argmax, vedi `patch.py`).

Nota: su WSL il run GPU con 2 rank sulla STESSA scheda puo' bloccarsi in
`cudaMemcpy2D` per uno stato del driver dxg (succede anche col codice
pre-patch; i 2 PC con GPU separate non hanno questa contesa). Dopo un
`wsl --shutdown` i test shared-GPU tornano disponibili.
