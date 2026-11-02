# VALIDAZIONE — prototipo MPI contro i risultati noti di `saetta3d`

Questo documento definisce **come** si stabilisce che il prototipo
`src/mpi_fissa.cu` e' corretto, con i valori di riferimento **letti dai file
gia' presenti nel workspace** (nessuna misura inventata).

---

## 1. Riferimenti disponibili nel workspace (provenienza dei dati)

| dato | percorso | cosa contiene |
|---|---|---|
| produzione L8 | `gpu_nvidia_3d/risultati/diagnostica/diagnostics.dat` | serie temporale t=0..1500, griglia 256x256x1024 (67.1M celle), `cells=67108864` su ogni riga; log in `.../diagnostica/log-saetta`, residuo max in `.../resa_max.dat` |
| L6 (stesso schema) | `gpu_nvidia_3d/risultati/error_study/error_study_fissa_3d_gpu/run6/diagnostics.dat` | solo riga t=0, griglia 128x128x512 (8.4M celle), `MAXLEVEL=6` |
| L7/L8 iniziale | `.../error_study.../run7/diagnostics.dat` | riga t=0, griglia 256x256x1024 (`MAXLEVEL=7, NDIV0=2`), identica all'L8 a t=0 |
| campi omega_z | `.../run6/omega3d_matrix.bin`, `.../run7/omega3d_matrix.bin`, `gpu_nvidia_3d/risultati/matrici/omega3d_matrix.bin` | formato `int32 nx,ny,nz + float32`, ma **il tempo esatto dei dump non e' documentato nei log** (run6/run7 hanno 13/21 passi, quindi t≈0.1-0.2; non usarli come riferimento a t=10 senza verificare) |

**Attenzione**: i valori "L7" di run7 coincidono a t=0 con L8 perche' hanno la
stessa `Delta` (NDIV0=2, MAXLEVEL=7 → 256x256x1024), non perche' siano lo
stesso run.

I riferimenti a t>0 **alla stessa risoluzione del prototipo (L6)** non ci sono
per tempi lunghi: vanno generati con il solver esistente (comando in §5). Gli
unici riferimenti "lunghi" sono a L8 (produzione).

## 2. Valori di riferimento a t=0 (forma forte del test)

A t=0 il campo e' **esattamente l'IC analitica**: se il prototipo implementa
la stessa IC e le stesse differenze finite della diagnostica, i valori devono
coincidere a livello di arrotondamento fp32 (**~1e-6 relativo**).

| quantita | L6 (128x128x512) | L8 (256x256x1024) |
|---|---|---|
| enstrophy `Z` | 1.25630230e-01 | 1.25961661e-01 |
| energy `E` | 1.00166653e-03 | 1.00165501e-03 |
| umax | 5.74348159e-02 | 5.74350879e-02 |
| wmax | 1.98497629e+00 | 1.99623787e+00 |
| circulation_slice `G` | 5.11479825e-02 | 5.10205626e-02 |
| separation | 2.578125e-01 | 2.539062e-01 |
| xc | 1.289062e-01 | 1.269531e-01 |
| yc | 0.000000e+00 | 0.000000e+00 |
| cells | 8388608 | 67108864 |
| dt iniziale (saetta3d) | 9.907987e-03 | 4.947802e-03 |

Il `dt` iniziale del prototipo **non** deve coincidere (saetta3d ha una
politica di primo passo diversa): si confronta solo a t=0, poi la fisica.
Il `wmax ≈ 2` a t=0 e' corretto: i due tubi si incrociano in z≈3 e li' le
vorticity si sommano (il massimo analitico di un tubo e' 1).

## 3. Serie temporale di riferimento a L8 (per il trend; produzione)

Da `gpu_nvidia_3d/risultati/diagnostica/diagnostics.dat` (estratto completo in
`riferimento/ref_L8_diagnostics_estratto.dat`):

| t | Z | E | umax | sep |
|---|---|---|---|---|
| 0 | 1.25961661e-01 | 1.00165501e-03 | 5.74350879e-02 | 2.539062e-01 |
| 1 | 1.25322402e-01 | 9.48302099e-04 | 6.41421825e-02 | 2.485922e-01 |
| 2 | 1.24729231e-01 | 9.47045861e-04 | 6.12396039e-02 | 2.511266e-01 |
| 5 | 1.23489544e-01 | 9.43308463e-04 | 5.43031245e-02 | 2.461248e-01 |
| 10 | 1.25003546e-01 | 9.37079545e-04 | 5.03172763e-02 | 2.400994e-01 |
| 20 | 1.32974207e-01 | 9.23971063e-04 | 5.40651157e-02 | 2.461248e-01 |
| 49 | 1.36307836e-01 | 8.84037931e-04 | 5.39599359e-02 | 3.070322e-01 |
| 99 | 1.36514828e-01 | 8.14752188e-04 | 4.09827083e-02 | 3.979010e-01 |
| 199 | 1.02464221e-01 | 6.94844697e-04 | 4.11416739e-02 | 6.236558e-01 |
| 499 | 9.24894884e-02 | 4.11153713e-04 | 4.05759774e-02 | 6.243771e-01 |
| 999 | 1.62650645e-02 | 1.61108910e-04 | 2.06097383e-02 | 2.807341e-01 |
| 1499 | 5.27574774e-03 | 1.20148361e-04 | 1.22950356e-02 | 8.304801e-02 |

Fisica attesa (per il controllo qualitativo): i tubi si fondono e la
separazione **cresce** fino a ~0.62 (t≈200-500), poi si ha decadimento
viscoso con enstrophy che crolla (0.125 → 0.005) e separazione che cala
(~0.08 a t=1500). Un prototipo che non riproduce questi trend ha un problema.

## 4. I quattro test di validazione (in ordine)

### T1 — IC e diagnostica (obbligatorio, veloce)

Prototipo `NX=128`, `-tmax 0` (o leggi la riga t=0 del primo run):
`Z,E,Umax,Wmax,G,sep,xc` devono coincidere con la colonna L6 di §2 entro
**1e-5 relativo** (il cast a float dell'IC puo' dare ~1e-7; margine per le
differenze di compilazione). Comando:

```sh
NX=128 ./scripts/build.sh cpu
./scripts/run_cpu.sh 1 -tmax 0.05 -out out/val_t1 && head -3 out/val_t1/diagnostics.dat
# confronto manuale con la tabella L6; oppure:
python3 - <<'EOF'
ref = dict(Z=1.25630230e-01, E=1.00166653e-03, u=5.74348159e-02, w=1.98497629e+00,
           G=5.11479825e-02, s=2.578125e-01)
line = [l for l in open('out/val_t1/diagnostics.dat') if l.strip() and not l.startswith('#')][0].split()
z,e,u,w,g,s = map(float, line[3:9])
for k, v, r in (('Z',z,ref['Z']),('E',e,ref['E']),('umax',u,ref['u']),
                ('wmax',w,ref['w']),('G',g,ref['G']),('sep',s,ref['s'])):
    print('%-5s %.8e  rif %.8e  err.rel %.2e' % (k, v, r, abs(v-r)/abs(r)))
EOF
```

Se T1 fallisce: controllare la formula dell'IC (tilt, PERT, segni di
`u.x=-ut*y/r`, `u.y=+ut*dx/r`), le coordinate dei centri di cella, la formula
di `WZ`/`WX`/`WY` e la fetta `z=2-Delta/2`.

### T2 — Invarianza MPI (obbligatorio)

np=1 vs np=2 (CPU o GPU) a `t=0.2-1`: campo `omega_z` con
`rms_rel < 1e-5` (idealmente ~1e-7); diagnostica idem (`--tiny`). Vedi M2 di
`PIANO_MILESTONE.md` per i comandi. Questo test isola i bug di
decomposizione/proiezione da quelli di fisica: **va fatto prima di T3**.

### T3 — CPU vs GPU (obbligatorio per usare la GPU)

Stessa `NX`, np=1: campo `omega_z` con `rms_rel < 1e-5`. Se fallisce: bug di
`parallel_for`/indici o di trasferimento host-device (tipicamente un alone non
copiato o un piano scambiato), non di fisica.

### T4 — Fisica contro `saetta3d` (accettazione)

Riferimento L6 generato con il solver esistente (vedi §5), t fino a 2 (poi 10).
Tolleranze proposte (da tarare sul primo risultato, non sono misure):

| quantita | tolleranza t<=2 | tolleranza t<=10 | note |
|---|---|---|---|
| Z (enstrophy) | <= 2% | <= 5% | integrale globale, robusto |
| E (energy) | <= 2% | <= 5% | |
| umax/wmax | <= 5% | <= 10% | valori locali, sensibili a dettagli |
| G (circolazione fetta) | <= 2% | <= 5% | |
| separation | <= 3% | <= 8% | posizione dei nuclei |
| campo omega_z (rms rel) | <= 1e-2 | <= 2e-2 | confronto L6 vs L6 |

Differenze attese e **accettate** (documentate nel DESIGN §5.2/§5.6):
- viscosita' **esplicita** invece che implicita (a `nu=1e-5` l'effetto e'
  ~1e-5 per unita' di tempo sull'energia: trascurabile sulle tolleranze);
- `uf` alle pareti normali forzata a 0 nel prediction (BC MAC corretta: la
  versione Basilisk puo' avere un valore leggermente diverso a meta' passo);
- flusso alle facce z di interfaccia con velocita' centrata (O(Delta^2), solo
  un piano);
- politica del passo (dt) diversa: nessuna pretesa di coincidenza punto-punto
  della serie temporale; si confrontano gli **integrali** e il campo a tempi
  tondi;
- il prototipo usa fp32 nei kernel, saetta3d fa alcune espressioni in double:
  differenze a livello di roundoff che crescono nel tempo caotico.

Se T4 fallisce in modo sistematico (es. Z troppo alta/bassa del 10% gia' a
t=1), le cause piu' probabili sono: (a) BC free-slip applicate male (segno
sbagliato: no-slip vs free-slip cambia l'enstrophy), (b) proiezione poco
convergente (`-poisson_nit 2` per verificare), (c) errore nel termine
tangenziale dell'avvezione (indice upwind vs faccia), (d) `dt` troppo grande.
In questi casi mandare i due `diagnostics.dat` e il log del prototipo
(stderr del rank 0, da salvare con `2> log.txt`).

## 5. Come generare il riferimento L6 con il solver esistente

In una cartella col progetto `gpu_nvidia_3d`:

```sh
cd gpu_nvidia_3d
EXTRA="-DMAXLEVEL=6 -DTMAX=10 -DMOVIE=0 -DVTK=0 -DMATRIX=1" ./build.sh
mkdir -p run_ref_L6 && cd run_ref_L6
../saetta3d_gpu 2> log.txt      # t=0..10, dt ~ 1e-2 => ~1000 passi
head -20 diagnostics.dat        # t=0,1,2,... 
```

Nota: con `MOVIE=0` non ci sono eventi a t+0.5 che cambiano `dtnext`; la
diagnostica resta a `t += 1`. Per un confronto a L8 servirebbe una GPU 8 GB
(il solver esistente usa ~7.5 GB): si puo' fare `-DMAXLEVEL=8 -DTMAX=1` per
un riferimento corto a L8.

## 6. Cosa fare se i riferimenti L6/L8 non fossero rigenerabili

In questo workspace `gpu_nvidia_3d/saetta3d_gpu` e' gia' compilato: si puo'
usare direttamente se la GPU e' quella giusta (stessa architettura), altrimenti
ricompilare con `build.sh` (richiede qcc e le librerie GPU di Basilisk gia'
presenti, vedi `gpu_nvidia_3d/build.sh`). Se non fosse possibile rigenerare il
riferimento L6, la validazione minima accettabile e':
- T1 con i valori L6/L8 di §2 (dai file esistenti);
- T2 e T3;
- confronto **qualitativo** col trend L8 di §3 (fusione, crescita della
  separazione fino a ~0.6, decadimento) e, a parita' di griglia, confronto
  con i campi `omega3d_matrix.bin` esistenti solo se se ne stabilisce il tempo.
