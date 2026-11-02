# Parte 2-PC del README — testo sostitutivo

Questo file sostituisce, in `mpi_fissa/README.md`, le sezioni **4.4** e **5**.
Le modifiche rispetto alla versione attuale sono marcate con `[CAMBIATO]` o
`[NUOVO]`; sotto ogni punto c'e' il motivo, cosi' si sa cosa si sta perdendo a
toglierlo.

---

## 4.4 GPU su 2 PC `[CAMBIATO]`

Prerequisiti (li verifica `run_2pc.sh` in 5 secondi, ma vanno preparati una
volta — vedi `CHECKLIST_2PC.md`):

- progetto nello **stesso percorso** sui due PC, **senza spazi**:
  `~/progetto/mpi_fissa` (coerente con `cluster/push_project.sh`, che copia in
  `~/progetto`). `[CAMBIATO]`: la versione precedente diceva `~/mpi_fissa`, che
  non e' il percorso usato dal resto del progetto; e il workspace locale
  `Test 1` contiene uno spazio, quindi **non** va usato come sorgente diretto
  del run;
- binario GPU compilato **sulla stessa NX** sui due PC (NX e' compile-time: due
  binari con NX diversi fanno un run sbagliato, non un errore);
- **stessa versione di OpenMPI** sui due PC (`mpirun --version`), `orted` nel
  PATH di entrambi;
- `ssh` master→slave senza password e **senza banner/MOTD**;
- WSL **mirrored networking** su entrambi (consigliato) oppure port-proxy;
- **firewall di Windows aperto anche sulle porte dati di MPI** (`[NUOVO]`: con
  la sola 22 aperta `hostname` funziona e il run si blocca al primo scambio
  grosso — e' il guasto piu' difficile da diagnosticare di tutta la catena).

```sh
# test rete + MPI (2 s): devono comparire DUE hostname diversi
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 hostname

# test "le due GPU ci sono davvero"
mpirun --hostfile scripts/hostfile_2pc.txt -np 2 nvidia-smi -L

# run
./scripts/run_2pc.sh -tmax 1 -out out/2pc_L6
```

`run_2pc.sh` usa:

```
mpirun --hostfile scripts/hostfile_2pc.txt -np 2
       --map-by ppr:1:node --bind-to none --wdir $PWD
       --mca plm_rsh_agent ssh
       --mca plm_rsh_args "-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"
       -x LD_LIBRARY_PATH -x OMP_NUM_THREADS
       bin/mpi_fissa_gpu <opzioni>
```

e **prima** controlla: ssh batch, `orted`, versione OpenMPI uguale, `$HOME`
uguale, `md5` del binario uguale, binario presente nel percorso identico,
`mpirun -np 2 hostname`. Se qualcosa non torna si ferma con un messaggio che
dice cosa fare. `DRY=1` stampa solo il comando; `SKIP_CHECK=1` salta i
controlli (sconsigliato).

## 5. Setup 2 PC (una volta) `[CAMBIATO]`

### 5.0 Percorsi `[NUOVO]`

Il percorso del progetto deve essere **identico** sui due PC e **senza spazi**:
`--wdir "$HERE"` viene applicato a tutti i rank, quindi se sul secondo PC il
progetto sta altrove `mpirun` fallisce (o peggio, parte con la directory
sbagliata). Convenzione: `~/progetto/mpi_fissa` su entrambi.

```sh
# dal workspace (una volta, poi solo dopo modifiche al codice)
rsync -az --exclude out/ --exclude bin/ --exclude .git/ \
      "/percorso/Test 1/mpi_fissa/" slave1:~/progetto/mpi_fissa/
# e in locale:
rsync -az --exclude out/ --exclude bin/ "/percorso/Test 1/mpi_fissa/" ~/progetto/mpi_fissa/
cd ~/progetto/mpi_fissa && NX=256 ./scripts/build.sh gpu
ssh slave1 'cd ~/progetto/mpi_fissa && NX=256 ./scripts/build.sh gpu'
```

`--exclude out/` non e' cosmetico: `out/*/omega3d_matrix.bin` a L8 e' 268 MB
per run e `rsync` lo ripasseresti ogni volta.

### 5.1 Rete `[CAMBIATO]`

1. Stessa Wi-Fi 5/6 GHz, 1 m, **risparmio energetico della scheda Wi-Fi
   disattivato** su entrambi i PC (Gestione dispositivi → scheda Wi-Fi →
   Risparmio energia: togliere la spunta). Un adapter che va in power-save
   aggiunge 10-100 ms di latenza a singhiozzo: su 194 scambi per passo si vede.
2. **WSL mirrored networking su entrambi** (`%USERPROFILE%\.wslconfig`):
   ```ini
   [wsl2]
   networkingMode=mirrored
   ```
   poi `wsl --shutdown` e riaprire. Con mirrored WSL condivide l'indirizzo di
   Windows e `ssh utente@<ip-pc>` arriva direttamente a WSL (porta 22).
3. Verifica, **su entrambi**:
   ```sh
   ip -4 addr show eth0     # deve mostrare l'IP della LAN (192.168.x / 10.x)
   ip route | head -3
   ping -c 3 <ip-dell-altro-pc>
   ```
   Se l'IP di WSL e' `172.x` state in NAT, non in mirrored: o si passa a
   mirrored, o si usa il port-proxy di `cluster/setup_slave.ps1 -PortProxy 2222`.
4. Dal master: `netsh wlan show interfaces` (in PowerShell) per vedere rate e
   canale; se il rate e' < 100 Mbps, il collo di bottiglia e' il Wi-Fi e nessuna
   ottimizzazione del codice lo compensa.

### 5.2 SSH `[CAMBIATO]`

```sh
sudo apt install -y openssh-server
sudo service ssh restart          # oppure: sudo systemctl enable --now ssh
```

Su **entrambi** i PC, in PowerShell da amministratore (regola per la 22):

```powershell
New-NetFirewallRule -DisplayName "ssh-wsl" -Direction Inbound -Protocol TCP `
    -LocalPort 22 -Action Allow
```

Sul master:

```sh
cat >> ~/.ssh/config <<'CFG'
Host slave1
    HostName 192.168.1.11      # IP del PC slave (quello di `ip -4 addr`)
    User zorka                 # stesso utente del master: $HOME identico
    Port 22                    # 2222 se usate il port-proxy
    StrictHostKeyChecking accept-new
CFG
ssh-copy-id slave1
ssh -o BatchMode=yes slave1 hostname     # deve stampare l'hostname dello slave
```

**Niente banner**: se `/etc/ssh/sshd_config` ha `Banner /etc/issue.net` o un
MOTD chiassoso, OpenMPI puo' Interpretarlo come output dell'`orted` e fallire
in modo oscuro. Commentare `Banner`/`PrintMotd` sullo slave.

### 5.3 Progetto e binari `[CAMBIATO]`

```sh
# su entrambi i PC, stessa NX e stesso ARCH
cd ~/progetto/mpi_fissa
NX=256 ./scripts/build.sh gpu
md5sum bin/mpi_fissa_gpu          # i due md5 devono coincidere
```

Se le due GPU sono **modelli diversi** (es. RTX 3070 = sm_86 e RTX 4070 =
sm_89) il binario compilato con `ARCH=86` **non parte** sul secondo. In quel
caso compilare con PTX incluso:

```sh
# in build.sh, oppure a mano:
nvcc -O3 -std=c++17 --extended-lambda -arch=sm_86 -gencode arch=compute_86,code=compute_86 ...
```

`[NUOVO]` — verificare **prima** del primo run:
`mpirun --hostfile scripts/hostfile_2pc.txt -np 2 nvidia-smi -L`.

### 5.4 Hostfile `[CAMBIATO]`

`scripts/hostfile_2pc.txt`:

```
localhost slots=1
slave1 slots=1
```

`slots=1` **non e' opzionale**: senza, OpenMPI puo' mettere 2 rank su
`localhost` e il "run a 2 PC" gira su una macchina sola (e sembra funzionare).
La prima riga deve essere la macchina locale. Se `localhost` risolve a `::1` su
una rete solo IPv4, usare `127.0.0.1 slots=1`.

### 5.5 Firewall per le porte dati di MPI `[NUOVO — il punto che fa perdere piu' tempo]`

OpenMPI non usa solo la 22: dopo aver lanciato `orted` via ssh, apre **porte TCP
effimere** per i collegamenti dati, in entrambe le direzioni. In WSL *mirrored*
queste connessioni passano dal firewall di Windows. Sintomo:

- `mpirun -np 2 hostname` → **funziona** (usa solo ssh);
- il run parte, i due rank stampano l'intestazione, e poi **resta appeso per
  sempre** al primo scambio di aloni (o muore con `Unable to find a usable
  transport` / `timed out`).

Fix, su **entrambi** i PC, in PowerShell da amministratore:

```powershell
New-NetFirewallRule -DisplayName "mpi-data" -Direction Inbound -Protocol TCP `
    -LocalPort 50000-51023 -Action Allow
```

e fissare quel range lato OpenMPI (in `MPI_EXTRA` di `run_2pc.sh`):

```sh
export MPI_EXTRA="--mca btl_tcp_port_min_v4 50000 --mca btl_tcp_port_range_v4 1024"
```

In alternativa (piu' veloce ma meno selettivo): mettere il profilo di rete
Wi-Fi su **Privato** su entrambi i PC. Non usare "disattiva firewall": il
problema si nasconde e torna al primo cambio di rete.

### 5.6 CUDA via ssh `[INVARIATO]`

`run_2pc.sh` propaga `LD_LIBRARY_PATH` con `-x`; su WSL `/usr/lib/wsl/lib`
deve esserci per `libcuda`. Se il setup usa `~/cuda-env.sh`, o si esporta prima
del run, o si adatta `CUDA=`/`LD_LIBRARY_PATH` nello script.

---

## 6.1 Lettura di `timing.dat` — attenzione `[NUOVO]`

Colonne: `step t dt wall_s comm_s comm_MB n_exch`.

**`comm_MB` e `n_exch` sono cumulativi** (i contatori `B_COMM`/`N_EXCH` non
vengono azzerati a ogni passo): per il valore per-passo vanno **differenziati**
fra righe successive. `run_2pc.sh` lo fa gia' nel riepilogo finale.

Altra trappola: l'**ultimo passo** di ogni run contiene la diagnostica di fine
run (`residual_lvl` + `bc_u` + `download_u`), quindi `wall_s` e `comm_s`
dell'ultima riga sono gonfiati. Per una misura pulita: `-diag 1e9` (nessuna
diagnostica a meta' run) e scartare l'ultima riga.

```sh
# wall e comm medi per passo, escludendo l'ultima riga
awk 'NR>1{r++;W[r]=$4;C[r]=$5;M[r]=$6}
     END{ if (r<2) { print "pochi passi: rifai il run con piu passi"; exit }
          for(i=1;i<r;i++){w+=W[i];c+=C[i];n++}
          printf "passi %d  wall %.4f s  comm %.4f s (%.1f%%)  MB/passo %.2f\n",
                 n, w/n, c/n, 100*c/w, (M[r-1])/n }' out/2pc_L8/timing.dat
```
