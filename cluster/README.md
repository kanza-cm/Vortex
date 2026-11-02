# master + N slaves over Wi-Fi (Windows + WSL)

One master (this PC) drives several slave PCs (Windows + WSL) on the same
local network: commands go over SSH, files go over rsync. No cloud, no
internet: traffic stays on the Wi-Fi.

```
        master (WSL)                 slaves (Windows + WSL)
   ssh / rsync  ──────────►  slave1 WSL  (GPU, basilisk/incflo)
                            slave2 WSL
                            slave3 WSL
```

## 1) Network
Put every PC on the same Wi-Fi (5 GHz or 6 GHz channel). If there is no
router, use a Windows Mobile Hotspot on the master and connect the slaves.
Give every slave a fixed address (router DHCP reservation, or hotspot).
At 1 m the link is at full rate; aggregate throughput is shared, so an
N-slave fan-out is limited by the master's radio.

## 2) Slave setup (each PC, PowerShell as Administrator)
```powershell
.\setup_slave.ps1                 # mirrored networking (Windows 11 22H2+, recommended)
.\setup_slave.ps1 -PortProxy 2222 # fallback: forward Windows port 2222 -> WSL:22
```
The script installs and starts `sshd` inside WSL, opens the firewall and
prints the PC's IPv4 address.

For mirrored networking put this in `%USERPROFILE%\.wslconfig` on the slave:
```ini
[wsl2]
networkingMode=mirrored
```
then run `wsl --shutdown` and reopen WSL. WSL now shares the Windows
address, so `ssh <wsl-user>@<pc-ip>` reaches WSL directly (port 22).

Without mirrored networking WSL sits behind NAT: use `-PortProxy` and
connect to port 2222 (re-run the script after a reboot, because the WSL
address can change).

## 3) Master setup (once)
```bash
# aliases for every slave
cat >> ~/.ssh/config <<'CFG'
Host slave1
    HostName 192.168.1.11
    User zorka
    Port 22          # or 2222 with -PortProxy
Host slave2
    HostName 192.168.1.12
    User zorka
    Port 22
CFG
ssh-keygen -t ed25519 -N ""   # if you have no key
ssh-copy-id slave1 && ssh-copy-id slave2
ssh slave1 hostname           # check
```
List the same aliases, one per line, in `hosts.txt`.

## 4) Daily use
```bash
./cluster.sh status                        # hostname, GPU, load on every slave
./cluster.sh run 'cd ~/progetto && ./build.sh'
./cluster.sh push ./gpu_amr_3d ~/progetto/gpu_amr_3d/
./cluster.sh run 'cd ~/progetto/gpu_amr_3d && ./run.sh 4 6'
./cluster.sh pull '~/progetto/gpu_amr_3d/risultati' ./risultati_slaves
```
`push`/`pull`/`run` fan out in parallel (`PARALLEL=4` by default) and print
one line per host. Use `HOSTS_FILE=...` for a different host list.

## 5) Notes
- Each slave needs the same toolchain: CUDA (`~/cuda-env.sh`), Basilisk in
  `~/basilisk/src`, incflo in `~/src/incflo`; run `./build.sh` once there.
- Wi-Fi is half-duplex and shared: with N slaves the aggregate is roughly
  the master's link rate. A wired switch is better for large fan-out.
- Throughput at 1 m: ~30-80 MB/s (Wi-Fi 5), 80-150 MB/s (Wi-Fi 6/6E);
  5-6 GB of results move in 1-3 minutes.
- `rsync` is incremental, so re-pushing also works after small changes.

## GPU jobs on the slaves

The GPU solvers run on the slave's NVIDIA card: `gpu_nvidia_3d` (fixed
grid, Basilisk CUDA backend) and `gpu_amr_3d` (incflo/AMReX CUDA) are
compiled for CUDA, so a run uses the GPU once the machine is set up.

### Check a slave
```bash
./cluster.sh run 'cd ~/cluster && ./gpu_check.sh'
```
`gpu_check.sh` verifies the WSL GPU stack (`nvidia-smi`, `/dev/dxg`,
`libcuda`, CUDA toolkit, `qcc`, incflo) and prints the card name and VRAM.
It also prints a fast GPU self-test command.

Typical fix list when the GPU is not visible in WSL:
1. Windows NVIDIA driver installed and `nvidia-smi` works on Windows;
   WSL2 (not WSL1), Windows updated.
2. Inside WSL `nvidia-smi` must work, `/dev/dxg` must exist,
   `/usr/lib/wsl/lib/libcuda.so*` present. The CUDA stack comes from
   Windows, not from a package inside WSL.
3. CUDA toolkit in WSL (the project uses `~/cuda-env.sh` and
   `~/cuda-13.1`), `nvcc` working.
4. Basilisk in `~/basilisk/src` with the CUDA grid libs built
   (see `gpu_nvidia_3d/build.sh`).
5. incflo patched clone in `~/src/incflo`, `./build.sh` run once.

Confirm the GPU is really computing during a run (on the slave):
```bash
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1
```

### Dispatch jobs
Edit `cluster/jobs.txt` (one `host|command` per line), then:
```bash
./cluster/push_project.sh        # project -> ~/progetto on every slave
./cluster/dispatch.sh            # start all jobs in background
./cluster/dispatch.sh --wait     # start and follow them until the end
```
Logs: `~/job_logs/<host>-<n>.log` on each slave. Results stay under each
solver's `risultati/`; pull them with `./cluster/cluster.sh pull`.

### Grid sizes for a 16 GB card (fixed grid)
`gpu_nvidia_3d` grid is `NDIV0*2^MAXLEVEL` in x,y and 4x in z, cubic
cells, single precision, ~117 B/cell:
- MAXLEVEL=8, NDIV0=1 -> 256x256x1024 = 67M cells (~7.5 GB), safe
- MAXLEVEL=6, NDIV0=5 -> 320x320x1280 = 131M cells (~14.5 GB), close to full
- MAXLEVEL=9 would need 537M cells (~58 GB): not possible
