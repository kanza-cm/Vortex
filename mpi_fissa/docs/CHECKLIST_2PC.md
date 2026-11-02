# CHECKLIST 2 PC — dal nulla al primo run L8 (per il Lead)

Ogni passo ha: comando, **cosa devi vedere**, e cosa fare se non lo vedi.
I passi 1-5 si fanno **una volta**; 6-7 a ogni run. Tempo: ~1 h la prima volta.

Legenda: `[M]` = sul master, `[S]` = sul secondo PC (slave1), `[M+S]` = su
entrambi. In PowerShell serve "Esegui come amministratore" dove indicato.

## File da copiare prima di iniziare

Dal la directory di lavoro (`%TEMP%\mpi_work\perf\`) al progetto
`~/progetto/mpi_fissa/` **su entrambi i PC**:

| file | dove va | cosa sostituisce |
|---|---|---|
| `run_2pc.sh` | `scripts/run_2pc.sh` | lo script attuale (pre-flight, log, riepilogo) |
| `hostfile_2pc.txt` | `scripts/hostfile_2pc.txt` | invariato nei contenuti, piu' commenti |
| `benchmark.sh` | `scripts/benchmark.sh` | nuovo |
| `mpi_pingpong.c` | `tools/mpi_pingpong.c` (o dove vuoi) | nuovo: misura la rete da sola |
| `README_2PC.md` | **non** va copiato: e' il testo da incollare in `README.md` §4.4/§5 | |

Poi: `chmod +x scripts/*.sh`.

---

## Fase 1 — Le due macchine (15 min)

### 1.1 `[M+S]` WSL2 e utente
```sh
wsl -l -v          # da PowerShell: VERSION deve essere 2
whoami; echo $HOME # dentro WSL: stesso utente e stesso $HOME sui due PC
```
**Devi vedere** lo stesso username (es. `zorka`) e lo stesso `$HOME`
(`/home/zorka`) sui due PC. `run_2pc.sh` lo controlla e si ferma se differisce.
Il progetto deve stare in un percorso senza spazi: `~/progetto/mpi_fissa`.

### 1.2 `[M+S]` GPU visibile in WSL
```sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
ls -l /dev/dxg /usr/lib/wsl/lib/libcuda.so.1
```
**Devi vedere** il nome della GPU (annotatene il modello: se i due PC hanno GPU
**diverse**, un binario `-arch=sm_86` non parte sul secondo) e `memory.total`
~8 GB. Se manca `/dev/dxg`, il problema e' il driver Windows, non MPI.

### 1.3 `[M+S]` OpenMPI identico
```sh
mpirun --version | head -1
command -v orted
dpkg -l openmpi-bin | tail -1
```
**Devi vedere** la **stessa identica** riga di versione sui due PC (es.
`mpirun (Open MPI) 5.0.3`) e un path per `orted`. Se differiscono:
```sh
sudo apt install --reinstall openmpi-bin libopenmpi-dev   # su entrambi
```

### 1.4 `[M+S]` CUDA e build (non serve per MPI, serve per il run)
```sh
ls ~/cuda-13.1/bin/nvcc && ~/cuda-13.1/bin/nvcc --version | tail -1
```
**Devi vedere** CUDA 13.x. Se il toolkit sta altrove, `CUDA=/usr/local/cuda-13.1`.

### 1.5 `[M+S]` Rete mirrored
Nel file `%USERPROFILE%\.wslconfig` (Blocco note, path `C:\Users\<utente>\.wslconfig`):
```ini
[wsl2]
networkingMode=mirrored
```
Poi, da PowerShell: `wsl --shutdown`, riaprire WSL. Verifica:
```sh
ip -4 addr show eth0 | grep inet     # deve dare l'IP della LAN (192.168.x o 10.x)
ping -c 3 <IP-dell-altro-PC>
```
**Devi vedere** un IP della stessa LAN (non `172.x`) e 3 ping con `time<5 ms`.
**Se vedi `172.x`**: sei in NAT. O passi a mirrored, o usi il port-proxy
(`cluster/setup_slave.ps1 -PortProxy 2222`) e la porta 2222 nell'hostfile.
In PowerShell, per sapere il link reale: `netsh wlan show interfaces`
(`Receive rate` deve essere >= 300 Mbps; se e' 54 Mbps il problema e' il Wi-Fi).

### 1.6 `[M+S]` Risparmio energetico Wi-Fi **spento**
Pannello di controllo → Gestione dispositivi → scheda Wi-Fi → Proprieta' →
Risparmio energia → **togliere** "Consenti al computer di spegnere il
dispositivo". Un adapter in power-save aggiunge decine di ms a singhiozzo: su
~200 scambi per passo si vede come run "lento e irregolare".

---

## Fase 2 — SSH dal master allo slave (10 min)

### 2.1 `[S]` sshd dentro WSL
```sh
sudo apt install -y openssh-server
sudo service ssh restart
ss -ltn | grep ':22'
```
**Devi vedere** `LISTEN 0 128 0.0.0.0:22`.

### 2.2 `[S]` In PowerShell (amministratore): firewall per la 22
```powershell
New-NetFirewallRule -DisplayName "ssh-wsl" -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow
```

### 2.3 `[S]` Niente banner (rompe `orted` in modo oscuro)
```sh
sudo sed -i 's/^Banner /#Banner /; s/^PrintMotd yes/PrintMotd no/' /etc/ssh/sshd_config
sudo service ssh restart
```

### 2.4 `[M]` Configurazione e chiave
```sh
cat >> ~/.ssh/config <<'CFG'
Host slave1
    HostName 192.168.1.11
    User zorka
    Port 22
    StrictHostKeyChecking accept-new
CFG
ssh-copy-id slave1
ssh -o BatchMode=yes slave1 hostname    # <-- deve stampare l'hostname dello slave
ssh -o BatchMode=yes slave1 'echo $HOME; command -v orted; mpirun --version | head -1'
```
**Devi vedere** l'hostname, `/home/zorka`, il path di `orted` e la stessa
versione OpenMPI del master. **Se `ssh` chiede la password**, `ssh-copy-id` non
ha funzionato o l'`AuthorizedKeysFile` e' diverso.

---

## Fase 3 — Progetto e binari identici (15 min + build)

### 3.1 `[M]` Copia nel percorso definitivo (niente spazi, identico)
```sh
mkdir -p ~/progetto
rsync -az --exclude out/ --exclude bin/ --exclude .git/ \
      "/percorso/del/workspace/mpi_fissa/" ~/progetto/mpi_fissa/
rsync -az --exclude out/ --exclude bin/ --exclude .git/ \
      "/percorso/del/workspace/mpi_fissa/" slave1:~/progetto/mpi_fissa/
```
**Devi vedere** nessun errore e, sul secondo PC, `ls ~/progetto/mpi_fissa/src`.
(`--exclude out/` non e' cosmetico: un `omega3d_matrix.bin` a L8 e' 268 MB.)

### 3.2 `[M+S]` Build con la **stessa** NX
```sh
cd ~/progetto/mpi_fissa && chmod +x scripts/*.sh
NX=256 ./scripts/build.sh gpu
md5sum bin/mpi_fissa_gpu
```
**Devi vedere** due md5 **uguali** sui due PC. Se differiscono, o la NX e'
diversa o uno dei due ha compilato un altro commit: rifare.
Se le GPU sono modelli diversi, aggiungere PTX:
`nvcc ... -gencode arch=compute_86,code=compute_86` (altrimenti sul secondo PC
il binario muore con "no kernel image is available for execution").

### 3.3 `[M+S]` Smoke test locale (1 GPU, 1 rank, griglia piccola)
```sh
./scripts/run_gpu.sh 1 -tmax 0.01 -out out/smoke_np1
tail -2 out/smoke_np1/timing.dat
```
**Devi vedere** una riga di timing e `out/smoke_np1/done_0.txt`.
Se questo non va, il problema non e' la rete: fermarsi qui.

---

## Fase 4 — Porte dati MPI nel firewall (10 min, **il passo che si dimentica**)

### 4.1 `[M+S]` In PowerShell (amministratore)
```powershell
New-NetFirewallRule -DisplayName "mpi-data" -Direction Inbound -Protocol TCP `
    -LocalPort 50000-51023 -Action Allow
Get-NetFirewallRule -DisplayName "mpi-*" | Select DisplayName,Enabled,Direction
```
**Devi vedere** la regola `mpi-data` abilitata in inbound.

### 4.2 `[M+S]` Coerenza del profilo di rete
```powershell
Get-NetConnectionProfile | Select Name,NetworkCategory
```
**Devi vedere** `Private` sulla Wi-Fi. Se e' `Public`, il firewall e' piu'
restrittivo: `Set-NetConnectionProfile -NetworkCategory Private`.

> Perche' serve: `mpirun` usa ssh solo per **lanciare** `orted`; i dati viaggiano
> su porte TCP effimere. Con la sola 22 aperta, `hostname` funziona e il run
> **resta appeso al primo scambio di aloni**.

---

## Fase 5 — Hostfile e primo MPI end-to-end (5 min)

### 5.1 `[M]` Hostfile
`~/progetto/mpi_fissa/scripts/hostfile_2pc.txt`:
```
localhost slots=1
slave1 slots=1
```
**`slots=1` non e' opzionale**: senza, OpenMPI puo' mettere 2 rank sul master.

### 5.2 `[M]` Test 1: i due host rispondono
```sh
cd ~/progetto/mpi_fissa
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 \
  --mca plm_rsh_agent ssh --mca plm_rsh_args "-o BatchMode=yes -o ConnectTimeout=10" \
  hostname
```
**Devi vedere due hostname diversi.** Se sono uguali → `slots` o `--map-by`.
Se va in timeout → ssh/firewall 22. Se `orted: not found` → openmpi-bin.

### 5.3 `[M]` Test 2: le due GPU sono raggiungibili dai rank
```sh
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 \
  --mca plm_rsh_agent ssh --mca plm_rsh_args "-o BatchMode=yes" nvidia-smi -L
```
**Devi vedere due righe `GPU 0: ...`** (modelli anche diversi, ma due).

### 5.4 `[M]` Test 3: primo scambio **grosso** (e' qui che si vede il firewall)
```sh
cd ~/progetto/mpi_fissa
./scripts/run_2pc.sh -tmax 0.05 -out out/2pc_smoke     # 3 passi con il binario a NX=256
```
**Devi vedere** il pre-flight, il comando `mpirun`, i due rank che partono e il
riepilogo (`wall_s`, `comm_s`, `MB/passo`). Sono ~10 s in tutto.
**Se resta appeso piu' di 60 s senza riepilogo → Fase 4** (porte dati).
Per vedere quale porta manca, sul secondo PC:
```sh
sudo ss -tnp | grep orted      # connessioni in SYN-SENT = firewall
```
Per un test che isola MPI dal resto, senza la GPU di mezzo:
`mpirun --hostfile scripts/hostfile_2pc.txt -np 2 ./mpi_pingpong 20` (5.5).

### 5.5 `[M]` Misura la rete **da sola** (2 min, serve per tarare il modello)
```sh
mpicxx -O3 -o mpi_pingpong mpi_pingpong.c
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 ./mpi_pingpong 200
```
**Devi vedere** una tabella `bytes RTT_us MB/s`. Le taglie che contano sono
quelle del multigrid: 256 KB (livello 0), 64 KB (livello 1), 1 KB (livello 4).
Quello che serve e' la **differenza** `RTT(256KB) - RTT(16B)`: se e' < 1 ms la
rete e' veloce e il costo per scambio del solver e' nel pack (`cudaMemcpy`), non
nel Wi-Fi. Riportare questa tabella: e' l'unico dato che separa latenza da banda
senza toccare il codice.

---

## Fase 6 — Primo run reale e confronto (20 min)

### 6.1 `[M]` 3 passi L8 a 1 GPU e a 2 PC, stesso `-poisson_nit`
```sh
cd ~/progetto/mpi_fissa
./scripts/run_gpu.sh 1 -tmax 0.06 -out out/ref_np1_L8                # 3 passi
./scripts/run_2pc.sh     -tmax 0.06 -out out/ref_np2_L8              # 3 passi
```
**Devi vedere** i due riepiloghi con `wall_s` e `comm_s` per passo.
Attesi (ordine di grandezza, [STIMA] da `OTIMIZZAZIONI.md`): `np1` ~0.35-1.4
s/passo secondo la macchina, `np2` con `comm` 0.3-0.6 s e **~28 MB, ~196
scambi** per passo. Se `comm_MB` e' 7 invece di 28, il binario e' a NX=128.

### 6.2 `[M+S]` Durante il run: le due GPU lavorano?
```sh
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```
**Devi vedere** ~90-100% di utilizzazione su **entrambi** i PC e ~2.7 GB
occupati per rank. Se su uno dei due la GPU e' al 0%, quel rank sta aspettando
(MPI) o non e' partito.

### 6.3 `[M]` Il risultato e' lo stesso a 1 e 2 rank?
```sh
python3 scripts/confronta_campi.py out/ref_np1_L8/omega3d_matrix.bin \
        out/ref_np2_L8/omega3d_matrix.bin --tol-rms 1e-5 --label np1 --label np2
python3 scripts/confronta_diagnostics.py out/ref_np1_L8/diagnostics.dat \
        out/ref_np2_L8/diagnostics.dat --tiny
```
**Devi vedere** "entro tolleranza". [DATO] a L8 i `diagnostics.dat` di
`mpi_fissa/out/L8_np1` e `L8_np2` coincidono a ~1e-7: la decomposizione e'
gia' corretta, questo test serve a confermarlo sulla nuova macchina.

---

## Fase 7 — La campagna di misure (30 min)

### 7.1 `[M]` Benchmark completo
```sh
cd ~/progetto/mpi_fissa
NX=256 ./scripts/benchmark.sh                 # ~10-20 min a L8
cat out/perf_2pc/tabella.txt
```
**Devi vedere** la tabella con `np=1` e `np=2` per `-poisson_nit 1,2,3`, 3
ripetizioni, mediana, MB/passo, `comm%` e il confronto np1/np2 (atteso ~1e-6).
Opzioni: `HOSTFILE=...`, `REPS=3`, `NIT_LIST="1 2 3"`, `TMAX=0.1` (5 passi a
L8), `BIN=bin/mpi_fissa_gpu`, `SKIP_2PC=1` (solo np1), `FAST=1` (1 passo,
1 reps), `KEEP_DUMP=1` (tiene `omega3d_matrix.bin`: 268 MB per run, di default
il dump va su `/dev/null` e non entra nel tempo).
Se il pre-flight del benchmark fallisce, lo script **prosegue con la sola
1 GPU** e lo scrive in tabella: in quel caso la diagnosi e' nella Fase 4.

### 7.2 `[M]` Cosa mandare indietro
1. `out/perf_2pc/tabella.txt`;
2. i `timing.dat` dei run `np2` con nit=1 e nit=2;
3. la tabella di `mpi_pingpong` (Fase 5.5);
4. una riga di `nvidia-smi --query-gpu=name` per PC;
5. il valore di `comm_MB` **per passo** (differenziato, non cumulativo).

---

## Rubrica degli errori (sintomo → causa → fix)

| sintomo | causa piu' probabile | fix |
|---|---|---|
| `ssh: connect to host ... port 22: Connection timed out` | firewall/porta 22, IP sbagliato, WSL in NAT | Fase 2.2, `ping`, mirrored o port-proxy |
| `ssh` chiede la password anche con `BatchMode=yes` | chiave non copiata, `AuthorizedKeysFile` | `ssh-copy-id`, `chmod 600 ~/.ssh/authorized_keys` |
| `mpirun -np 2 hostname` da' **due volte lo stesso host** | `slots` mancante / `--map-by` | hostfile con `slots=1` |
| `orted: command not found` | `openmpi-bin` assente/versione diversa | Fase 1.3 |
| `Open MPI detected an unmatching ...` / versioni diverse | pacchetti diversi | reinstallare uguale su entrambi |
| run parte, stampa l'header, poi **appeso** | **porte dati MPI nel firewall** | Fase 4 |
| `Unable to find a usable transport` | interfaccia sbagliata | `MPI_EXTRA="--mca btl tcp,self --mca btl_tcp_if_include eth0"` |
| `Permission denied` su `bin/mpi_fissa_gpu` | binario non eseguibile/copiato senza `-p` | `chmod +x`, ricompilare la' |
| `no kernel image is available for execution` | GPU diversa dall'`ARCH` di compilazione | Fase 3.2 (PTX) |
| il run a 2 PC e' **piu' lento** del previsto e `comm_s` ~0.5 s | e' il `pack` (copia sincrona), non la rete | `OTIMIZZAZIONI.md` O1/O6 |
| `wall_s` dell'ultima riga molto piu' alta | diagnostica di fine run | scartare l'ultima riga (Fase 7/`benchmark.sh`) |

---

## Quello che questa checklist **non** copre

- l'**accuratezza** della proiezione con `-poisson_nit 1` (va validata contro
  `saetta3d` a L6, `VALIDAZIONE.md` §4);
- le ottimizzazioni del codice (`OTIMIZZAZIONI.md`, O1-O11): qui si misura lo
  stato attuale, non lo si migliora;
- 3+ PC o 2 rank per PC: l'hostfile e `run_2pc.sh` reggono `NP=4` solo se
  `nz` e' divisibile e le porte sono aperte su tutti.
