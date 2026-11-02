# REPORT — BC/viscosità di `mpi_fissa.cu` contro il riferimento `saetta3d`

Sessione 2026-11-03. Revisione **in sola lettura** del workspace WSL: nessuna
compilazione, nessuna scrittura in `/home/zorka/simulazioni/Test 1`.
Le misure citate come **misure** sono state fatte da me sui file già presenti
(dump `omega3d_matrix.bin` e `diagnostics.dat`); quelle marcate **previsione**
non sono state verificate eseguendo il prototipo patchato.

File di lavoro (in `%TEMP%\mpi_work\bc\`):

| file | contenuto |
|---|---|
| `REPORT.md` | questo documento |
| `patch.py` | patch con stringhe esatte, `--check`, backup, `--out` |
| `mpi_fissa_patched.cu` | **anteprima** del sorgente patchato (P1+P2), per il diff |
| `check_stencil.py`, `check_stencil2.py` | verifica numerica della discretizzazione di P1 |
| `probe_*.py` | script delle misure sotto (rieseguibili) |

---

## 0. Sintesi (leggere questa, poi i dettagli)

1. **Le BC free-slip del prototipo sono corrette e già allineate al
   riferimento.** Nessun errore di segno, nessun ghost mancante, nessuna
   differenza nell'ordine di aggiornamento dei ghost rispetto a
   prediction/proiezione/viscosità. L'ipotesi "no-slip vs free-slip" (che
   `VALIDAZIONE.md` §T4 e `PROTOCOLLO.md` §8.1 indicano come prima
   sospettata) **va scartata**: a `t=0` le diagnostiche dei due codici
   coincidono a 4·10⁻⁸–1.4·10⁻⁶ relativi, e `Z` contiene anche le celle di
   parete, che sono le uniche che "vedono" i ghost.

2. **La differenza residua (~1%) non è di tipo boundary-condition ma di
   operatore viscoso.** Il riferimento usa `viscosity.h` con l'operatore
   *accoppiato* `∇·(2μD) = μ∇²u + μ∇(∇·u)`; il prototipo usa il laplaciano
   semplice `ν∇²u`. Sul campo *centrato* `u` la divergenza discreta non è
   piccola (il campo solenoidale è `uf`, non `u`), e alle pareti il termine
   mancante **vale esattamente il doppio** della derivata normale, perché la
   componente normale è dispari rispetto alla parete. L'ordine di grandezza
   dell'effetto accumulato (10⁻³ in `ω_z` sulle celle di parete) coincide con
   quello misurato.

3. **La patch minima (P1)** sostituisce quella riga di operatore, senza
   toccare né lo schema né l'MPI: stesse ghost già esistenti, footprint ±1,
   stessi 6 punti di chiamata. **P2** (opzionale) allinea la politica del
   `dt` a `timestep.h`: senza di essa il confronto a "CFL uguale" non è a
   `dt` uguale (52 passi contro 41 a CFL=0.1).

4. **La proiezione è esclusa come causa** (misura: da 2 a 8 V-cycle il campo
   cambia di 5.6·10⁻⁴ relativi e la differenza dal riferimento resta
   1.006·10⁻² — identica).

5. **Previsione onesta**: P1 porta `rms_rel` da 1.01·10⁻² a ~6–9·10⁻³ e
   dimezza circa le medie ai bordi x/z; **non** a < 3·10⁻³. Il criterio
   < 3·10⁻³, se resta l'obiettivo, richiede anche P2 e — se non basta —
   l'allineamento del flusso normale alle pareti (§8), che però è una
   regressione fisica e va decisa dal Lead.

---

## 1. Metodo e limiti

Letto: `gpu_nvidia_3d/saetta3d.c` (riferimento), `mpi_fissa/src/mpi_fissa.cu`
(intero), `basilisk/src/navier-stokes/centered.h` (installato e
`documentazione/centered_original.h`, **identici**: 12031 byte, hash
`98F132F6…`), `basilisk/src/viscosity.h`, `basilisk/src/grid/cartesian*.h`,
`basilisk/src/timestep.h`, `basilisk/src/grid/events.h`,
`gpu_nvidia_3d/documentazione/HANDOFF_viscosita.md`, `mpi_fissa/DESIGN.md`,
`VALIDAZIONE.md`, `docs/PROTOCOLLO.md`, i backup `~/centered.h.*.bak`,
i run in `/tmp/saetta3d_cfl*` e `mpi_fissa/out/validazione_tempo/*`.

Non fatto (vincoli): compilare, eseguire, scrivere nel workspace.

Nota importante sul riferimento: il `centered.h` attivo è quello **stock**
(nessun `DECOUPLED_VISCOSITY`); l'header con la viscosità disaccoppiata
descritto in `HANDOFF_viscosita.md` è solo nei backup. Quindi il run di
riferimento a `t=1` usa `viscosity.h` stock, cioè l'operatore **accoppiato
implicito**. Questo è il punto chiave della diagnosi.

---

## 2. Confronto puntuale delle BC

Legenda: "rif" = `saetta3d` con `centered.h` stock; "proto" = `mpi_fissa.cu`.

### 2.1 Pareti free-slip per `u` (x = ±0.5, y = ±0.5, z = 0, 4)

Riferimento: BC di default del vettore `u` in Basilisk, che per la componente
normale è `antisymmetry` (ghost = −interno, quindi `u_n = 0` sulla faccia) e
per le tangenziali `symmetry` (ghost = +interno, quindi ∂u_t/∂n = 0). Il
commento in `centered.h` ("default symmetric boundary conditions") e la
verifica empirica in `HANDOFF_viscosita.md` (test `u.x[]=5` → ghost `+5` sulla
parete y sul CPU) confermano questa semantica.

Prototipo, `bc_u()` (righe 398–455):

| parete | componente | segno | righe |
|---|---|---|---|
| x = ±0.5 | `u.x` normale | − | 412–414 |
| x = ±0.5 | `u.y`, `u.z` tang. | + | 412–414 |
| y = ±0.5 | `u.y` normale | − | 425–427 |
| y = ±0.5 | `u.x`, `u.z` tang. | + | 425–427 |
| z = 0 | `u.z` normale | − (2 piani) | 438–441 |
| z = 4 | `u.z` normale | − (2 piani) | 449–452 |
| z = 0, 4 | `u.x`, `u.y` tang. | + | 438–452 |

**Verdetto: identiche.** Segni giusti, specchiatura giusta (ghost `k=−1` da
`k=0`, `k=−2` da `k=1`; `k=nz` da `nz−1`, `nz+1` da `nz−2`), e i due piani di
ghost in z sono quelli che servono a `predict_uf` (che a `fk=0` usa
`u.z[ju±1]` con `ju=−1`, quindi `u.z[−2]`).

### 2.2 Angoli e spigoli

Ordine in `bc_u`: prima lo scambio MPI, poi pareti x, poi y, poi z. Ogni
passata scrive **tutto** il piano di ghost, angoli compresi (`j ∈ [−1, ny]`,
`k ∈ [−np, nzl−1+np]`, ecc.), e le passate successive usano valori già
specchiati della passata precedente: gli spigoli risultano coerenti
(specchiatura sequenziale dei due lati). **Verdetto: identico** al
comportamento di `box_boundary_level` di Basilisk, che per `d > left` estende
il ciclo ai corner (`_start--; _end++`).

### 2.3 Pressione Neumann omogenea

Riferimento: `centered.h` righe 91–107, `p[boundary] = neumann(a.n*fm/alpha)`
= 0 (a = 0). Il ghost vale quindi `p_int`.

Prototipo: `bc_pressure()` (460–490) fa `ghost = interno` su tutte e sei le
facce, in **tutti** i punti in cui serve: `smooth_lvl` (851), `residual_lvl`
(873), `prolong_add` (965), `project` (1077), coarse replicato (472–477).
**Verdetto: identica.**

Unica differenza *non* di BC: il prototipo fa **iterazioni fisse** (2
V-cycle), il riferimento usa il multigrid adattivo con tolleranza. Misurato:
irrilevante (§4.1).

### 2.4 Facce MAC normali alle pareti

Riferimento: `prediction()` calcola `uf` su tutte le facce; alle pareti il
termine di pendenza si annulla perché `un = 0` (ghost normale dispari). Il
prototipo forza esplicitamente `uf.n = 0` (righe 510 x, 537 y, 588–589 z) e
`flux = 0` alle facce di parete in `advect` (614, 641; per z il flusso è nullo
perché `uf.z = 0`).

**Verdetto: differenza possibile ma di ordine O(Δ) e già documentata**
(`VALIDAZIONE.md` §T4: "la versione Basilisk può avere un valore leggermente
diverso a metà passo"). Non è un errore: il prototipo impone la BC MAC in
modo più pulito. Non la toccherei (§8).

### 2.5 Gradiente `g` (campo `GRD`)

Riferimento: `g` è un `vector`, quindi ha le stesse BC di default di `u`
(normale −, tangenziali +). Prototipo: `bc_u(GRD, 1)` (riga 1554), cioè
esattamente la stessa specchiatura, e con **1** piano di ghost, che è quanto
serve a `predict_uf` (`grd[*][fi]` e `grd[*][fi−1]`, `grd.z[fk−1]`).

Verifica di coerenza con `p` Neumann: `g_n = −∂p/∂n` è dispari rispetto alla
parete ⇒ ghost = −interno; `g_t` è pari ⇒ ghost = +interno. **Verdetto:
identica.**

### 2.6 Ordine di aggiornamento dei ghost rispetto alle operazioni

| passo | riferimento (`centered.h`) | prototipo (righe 1553–1567) |
|---|---|---|
| inizio passo | `boundary` implicito negli eventi | `bc_u(U,2)`, `bc_u(GRD,1)` |
| prediction | `prediction()` con ghost correnti | `predict_uf(...)` |
| proiezione a metà passo | `project(uf, pf, dt/2)` | `project(PF, 0.5*dt)` |
| avvezione | `advection(u, uf, dt, g)` | `advect(U, UF, GRD, FLUX, dt)` |
| viscosità | `correction(dt)`→`viscosity`→`correction(−dt)` | `bc_u(U,1)`; `viscosity(U,dt)`; `bc_u(U,1)` |
| facce | `uf = face_value(u) + dt*a` | `acceleration(U, UF)` |
| proiezione | `project(uf, p, dt)` | `project(P, dt)` |
| gradiente + correzione | `centered_gradient(p,g)`; `correction(dt)` | `centered_gradient(P,GRD)`; `add_gradient(U,GRD,dt)` |

**Verdetto: stesso ordine.** Le due `correction(±dt)` attorno alla viscosità
nel riferimento servono solo a far lavorare il solve su `u + dt·g`: la
differenza è O(ν·dt²·∇²g) ≈ 10⁻⁷ in `u` per passo, trascurabile.

### 2.7 **Differenza vera: l'operatore viscoso**

Riferimento (`basilisk/src/viscosity.h`, ramo non-TREE, quello usato dalla
griglia fissa):

```c
res.x[] = r.x[] - lambda*u.x[] + dt/rho[ ]*(
    2.*mu.x[1]*(u.x[1] - u.x[]) - 2.*mu.x[]*(u.x[] - u.x[-1])          // 2*d2 normale
  + mu.y[0,1]*(u.x[0,1] - u.x[] + (u.y[1,0] + u.y[1,1])/4.
                                 - (u.y[-1,0] + u.y[-1,1])/4.)          // d2 trasversale + trasposta
  - mu.y[]  *(u.x[] - u.x[0,-1] + (u.y[1,-1] + u.y[1,0])/4.
                                 - (u.y[-1,-1] + u.y[-1,0])/4.)
  + mu.z[0,0,1]*(...) - mu.z[]*(...) )/sq(Delta);
```

cioè `div(2μD)`, **non** `μ∇²u`. Prototipo (righe 728–739): laplaciano a 7
punti. La differenza è `μ∇(∇·u)`, e sul campo *centrato* `u` la divergenza
discreta non è piccola — è il punto già documentato in
`HANDOFF_viscosita.md` ("la divergenza CENTRATA del campo `u` è O(0.1)"),
dove infatti il termine `T = μ∇(div u)` è stato **aggiunto** alla versione
disaccoppiata perché "NON è trascurabile qui".

Perché conta soprattutto alle pareti: la componente normale è dispari
(ghost = −interno), quindi la seconda differenza normale sulla prima cella
vale `(u[1] − 3u[0])/Δ²`; l'operatore del riferimento la moltiplica per 2 e
il laplaciano no. Verificato numericamente (§4.4): a parete il fattore 2 è
esatto, e su un campo con `div u ≠ 0` il laplaciano semplice differisce
dall'operatore vero del **33%** del termine, mentre lo stencil proposto lo
riproduce a 1.2·10⁻³ (0.01%, ordine 2).

Viscosità esplicita vs implicita: a `ν = 1e-5` il numero di diffusione è
`ν·dt/Δ² = 4·10⁻⁴`, quindi il trattamento temporale (Eulero esplicito vs
backward) resta trascurabile **purché l'operatore spaziale sia lo stesso**.
Per questo la patch è esplicita e cambia solo lo stencil.

---

## 3. Cosa NON è la causa (misurato)

### 3.1 La proiezione
Run esistenti `pn2/pn4/pn6/pn8` (stesso caso, solo `POISSON_NIT`):

| confronto | rms_rel | linf |
|---|---|---|
| pn2 − pn4 | 5.1·10⁻⁴ | 4.37·10⁻³ |
| pn2 − pn8 | 5.6·10⁻⁴ | 4.84·10⁻³ |
| rif − pn2 | 1.0061·10⁻² | 7.189·10⁻² |
| rif − pn8 | 1.0060·10⁻² | 6.961·10⁻² |

La convergenza della proiezione sposta il campo di 5.6·10⁻⁴ e **non sposta**
la differenza dal riferimento. Esclusa.

### 3.2 Le BC: il controllo a t=0
Diagnostiche a `t=0`, NX=64 (dai `diagnostics.dat`):

| | rif | proto | rel |
|---|---|---|---|
| Z | 1.24323323e-01 | 1.24323318e-01 | 4·10⁻⁸ |
| E | 1.00171263e-03 | 1.00171256e-03 | 7·10⁻⁸ |
| Umax | 5.74290752e-02 | 5.74290698e-02 | 9·10⁻⁸ |
| Wmax | 1.94093537e+00 | 1.94093272e+00 | 1.4·10⁻⁶ |
| G | 5.14048114e-02 | 5.14048109e-02 | 1·10⁻⁸ |

`Z` integra `|ω|²` su **tutte** le celle, pareti comprese: un ghost di segno
sbagliato sulla componente tangenziale (che entra in `WZ` alle pareti z) o
sulla normale (che entrerebbe in `WZ` alle pareti x/y) cambierebbe `Z` di
percentuali, non di 10⁻⁸. **Nessun errore di segno.**

---

## 4. Misure nuove (fatte in questa sessione)

Dump a `t=1`, CFL=0.1, NX=64 (`/tmp/saetta3d_cfl010/omega3d_matrix.bin` vs
`out/validazione_tempo/20261103_000059/cfl010/proto/omega3d_matrix.bin`).

### 4.1 Dove sta la differenza
`rms(ω_rif) = 1.7494e-01`, `rms(d) = 1.760e-03` (rel 1.006·10⁻²),
`linf = 7.189·10⁻²` in `(i,j,k) = (0, 26, 0)`.

| zona | media \|d\| | note |
|---|---|---|
| i<3 (x = −0.5) | 1.104e-03 | |
| i>nx−4 (x = +0.5) | 1.039e-03 | |
| j<3 (y = −0.5) | 3.391e-04 | **sotto** l'interno |
| j>ny−4 (y = +0.5) | 3.077e-04 | **sotto** l'interno |
| k<3 (z = 0) | 1.085e-03 | |
| k>nz−4 (z = 4) | 4.835e-04 | |
| interno | 6.079e-04 | |

Per faccia intera: `x=0` 3.94·10⁻³, `x=1` 3.41·10⁻³, `z=0` 4.50·10⁻³,
`z=4` 9.34·10⁻⁴, `y=±0.5` ~4.0·10⁻⁴.
Distribuzione pesante: le prime 1000 celle (0.1%) valgono il 24% della
varianza, le prime 10 000 (1%) il 58%; il 91% delle 400 celle peggiori è entro
3 celle da una parete, con `i ∈ {0,1,60,62,63}`, `j ∈ [24,38]`, `k ∈ [0,11]`.

### 4.2 L'asimmetria z=0 / z=4 non è un bug: è la fisica dell'IC
Con `xc(s,z) = s·DD(1+PERT cos(2πz/LZ)) − s·slope·(z−LZ/2)`, `slope = 0.14`:

* `z = 0` → `xc = ±0.434`: i nuclei sono a **0.066 dalla parete x** (σ = 0.09),
  cioè la vasca di vorticità *sfiora* la parete;
* `z = 4` (e `z = 2`) → `xc = ∓0.126`: i nuclei sono a 0.374 dalla parete.

Quindi l'interazione con la parete x è massima **solo a z ≈ 0**: è lì che
qualunque differenza di trattamento di parete viene amplificata, ed è lì che
sta il massimo `(0,26,0)` (parete x ∩ parete z=0). Coerente anche il fatto che
le pareti y siano le migliori: a `y = ±0.5` i nuclei sono a 5.5σ, quindi la
velocità normale e i suoi gradienti sono trascurabili e non c'è nulla da
sbagliare.

### 4.3 La differenza somiglia all'errore temporale del riferimento
Confronto degli errori "contro se stessi" (stesso codice, CFL diverso):

| | rms_rel | i<3 | j<3 | k<3 | interno |
|---|---|---|---|---|---|
| rif CFL 0.3 − rif CFL 0.1 | 4.24e-03 | 4.53e-04 | 1.39e-04 | 4.93e-04 | 2.57e-04 |
| rif 0.15 − rif 0.1 | 1.52e-03 | 1.43e-04 | 4.80e-05 | 1.75e-04 | 9.21e-05 |
| proto 0.3 − proto 0.1 | 2.60e-03 | 4.83e-04 | 2.30e-04 | 3.39e-04 | 1.37e-04 |
| **rif 0.1 − proto 0.1** | **1.006e-02** | 1.10e-03 | 3.39e-04 | 1.09e-03 | 6.08e-04 |

La differenza proto-riferimento ha la **stessa firma spaziale** dell'errore
temporale del riferimento (pareti x/z amplificate ~1.8×, pareti y ~0.55×
dell'interno), e la correlazione puntuale fra i due campi è **0.929**. Non è
una prova che la causa sia il tempo (l'ampiezza è ~10× quella attesa dal solo
rapporto di `dt`, §4.5), ma è la ragione per cui il fit "plateau + O(dt)" del
Lead non separa bene i due contributi: **una parte della differenza è
temporale e si somma a quella spaziale**.

### 4.4 Validazione numerica dello stencil proposto (P1)
Campo analitico liscio, stencil implementato con gli stessi indici del codice:

| NX | err max vs `div(2D)` esatto | err/rms |
|---|---|---|
| 64 | 1.24e-03 | 1.9e-04 |
| 128 | 3.10e-04 | 4.7e-05 |
| 256 | 7.76e-05 | 1.2e-05 |

Ordine 2 confermato. Sullo stesso campo il **laplaciano semplice** sbaglia del
**32.7%** del termine. A parete, con `u_x` dispari: `(u[1]−3u[0])/Δ²` per il
laplaciano contro il doppio per l'operatore del riferimento.

### 4.5 Passi e dt: i due run non sono allo stesso CFL effettivo
| | passi | dt a t=0 | dt a t=1 |
|---|---|---|---|
| rif CFL 0.1 | **52** | 1.984570e-02 | 2.294280e-02 |
| proto CFL 0.1 | **41** | 2.728784e-02 | 1.664732e-02 |

Il `dt` del riferimento è più piccolo del 27% perché `timestep.h` limita la
crescita con `if (dt > previous) dt = (previous + 0.1*dt)/1.1` partendo da
~0.09·dt_CFL: servono ~30 passi per raggiungere il limite CFL. Il prototipo
usa `dt ≤ 1.1·dt_prev` e al primo passo è già a `dt_CFL`. Ne segue che
"CFL=0.1" nei due codici significa `dt` medi di 1.92·10⁻² e 2.44·10⁻².
Questo **non** spiega l'ampiezza della differenza (l'errore temporale del
riferimento fra CFL 0.3 e 0.1 è 4.2·10⁻³, quello atteso dal rapporto dei `dt`
è ~7·10⁻⁴), ma va allineato per rendere pulito il confronto.

Diagnostiche a `t=1`, CFL 0.1 (per contesto):

| | rif | proto | rel |
|---|---|---|---|
| Z | 1.23535857e-01 | 1.23534552e-01 | 1.1·10⁻⁵ |
| E | 9.48147033e-04 | 9.48236259e-04 | 9.4·10⁻⁵ |
| Umax | 6.31574243e-02 | 6.29239486e-02 | 3.7·10⁻³ |
| Wmax | 1.91248202e+00 | 1.90933932e+00 | 1.6·10⁻³ |
| G | 5.13694137e-02 | 5.13579991e-02 | 2.2·10⁻⁴ |

Gli integrali coincidono a 10⁻⁴–10⁻⁵ mentre il campo differisce dell'1%: la
differenza è **spazialmente localizzata**, non un errore globale di soluzione.

---

## 5. Diagnosi

La differenza è concentrata dove la vorticità **interagisce con le pareti**
(tubi che sfiorano x = ±0.5 a z ≈ 0), ha la firma di un effetto di strato
limite (prima cella 6.9% delle celle ma 15.6% della varianza; 91% delle celle
peggiori entro 3 celle da una parete) e non dipende dalla proiezione né dai
segni delle BC.

L'unica differenza *di operatore* fra i due codici che agisce esattamente
lì è la viscosità: `μ∇²u` (proto) contro `∇·(2μD) = μ∇²u + μ∇(∇·u)`
(riferimento). Alle pareti il termine aggiuntivo vale il doppio della derivata
normale, quindi la correzione non è un dettaglio formale: è un termine
O(1) dell'operatore viscoso *sulla prima cella*, dove `div u` discreta è
massima. Stima d'ordine: incremento per passo `ν·dt·2·(u[1]−3u[0])/Δ²` ~ 4·10⁻⁶
in `u`; su ~40 passi in modo coerente ~1.6·10⁻⁴, che in `ω_z` (divisione per
2Δ) vale ~5·10⁻³, cioè lo stesso ordine della differenza misurata sulle facce
x/z (3.4–4.5·10⁻³).

Cause concorrenti, in ordine di plausibilità decrescente:
1. operatore viscoso (§2.7) — **patch P1**;
2. politica del `dt` (§4.5) — **patch P2**, contributo atteso minore ma non nullo
   e responsabile della parte "temporale" della differenza (§4.3);
3. flusso normale alle pareti (proto lo forza a 0, rif no) — O(Δ), documentato,
   **non** lo toccherei: allinearlo significherebbe reintrodurre un flusso di
   massa attraverso le pareti;
4. trattamento dell'interfaccia z (O(Δ²), un piano) — trascurabile qui (np1 e
   np2 identici).

---

## 6. Patch proposta

Una sola modifica obbligatoria (**P1**) e una opzionale (**P2**), entrambe
sostituzioni di testo locali. **Nessuna modifica** a: decomposizione,
`exchange_u`/`exchange_scalar`, multigrid, ordine degli eventi, BC.
`patch.py --check` verifica che ogni stringa compaia **esattamente una volta**
(eseguito: OK su `mpi_fissa.cu`, 68911 byte).

### P1 — operatore viscoso `div(2μD)` (obbligatoria)

Sostituisce il blocco righe 728–739 (le tre `float lap•` + le tre
`u.•[p] += c*lap•;`) con:

```c
        /* div(2 D) e non lap(u): stessa discretizzazione di
         * basilisk/src/viscosity.h (ramo non-TREE) --
         *   x: 2*d2xx + d2yy + d2zz + d2(u.y)/dxdy + d2(u.z)/dxdz  ecc.
         * Il fattore 2 sulla direzione della componente e' il termine
         * grad(div u) che il laplaciano semplice non ha; alle pareti vale
         * esattamente il doppio della derivata normale (componente normale
         * dispari rispetto alla parete). Footprint +/-1: ghost gia' presenti. */
        float ux = u.x[p], uy = u.y[p], uz = u.z[p];
        float d2xx_x = u.x[IDX(g, k, j, i + 1)] - 2.f*ux + u.x[IDX(g, k, j, i - 1)];
        float d2yy_x = u.x[IDX(g, k, j + 1, i)] - 2.f*ux + u.x[IDX(g, k, j - 1, i)];
        float d2zz_x = u.x[IDX(g, k + 1, j, i)] - 2.f*ux + u.x[IDX(g, k - 1, j, i)];
        float d2xx_y = u.y[IDX(g, k, j, i + 1)] - 2.f*uy + u.y[IDX(g, k, j, i - 1)];
        float d2yy_y = u.y[IDX(g, k, j + 1, i)] - 2.f*uy + u.y[IDX(g, k, j - 1, i)];
        float d2zz_y = u.y[IDX(g, k + 1, j, i)] - 2.f*uy + u.y[IDX(g, k - 1, j, i)];
        float d2xx_z = u.z[IDX(g, k, j, i + 1)] - 2.f*uz + u.z[IDX(g, k, j, i - 1)];
        float d2yy_z = u.z[IDX(g, k, j + 1, i)] - 2.f*uz + u.z[IDX(g, k, j - 1, i)];
        float d2zz_z = u.z[IDX(g, k + 1, j, i)] - 2.f*uz + u.z[IDX(g, k - 1, j, i)];
        /* trasposte: 0.25*(f(+,+) - f(-,+) - f(+,-) + f(-,-)) sui 4 angoli */
        float cxy = 0.25f*(u.y[IDX(g, k, j + 1, i + 1)] - u.y[IDX(g, k, j + 1, i - 1)]
                         - u.y[IDX(g, k, j - 1, i + 1)] + u.y[IDX(g, k, j - 1, i - 1)]);
        float cxz = 0.25f*(u.z[IDX(g, k + 1, j, i + 1)] - u.z[IDX(g, k + 1, j, i - 1)]
                         - u.z[IDX(g, k - 1, j, i + 1)] + u.z[IDX(g, k - 1, j, i - 1)]);
        float cyx = 0.25f*(u.x[IDX(g, k, j + 1, i + 1)] - u.x[IDX(g, k, j - 1, i + 1)]
                         - u.x[IDX(g, k, j + 1, i - 1)] + u.x[IDX(g, k, j - 1, i - 1)]);
        float cyz = 0.25f*(u.z[IDX(g, k + 1, j + 1, i)] - u.z[IDX(g, k + 1, j - 1, i)]
                         - u.z[IDX(g, k - 1, j + 1, i)] + u.z[IDX(g, k - 1, j - 1, i)]);
        float czx = 0.25f*(u.x[IDX(g, k + 1, j, i + 1)] - u.x[IDX(g, k - 1, j, i + 1)]
                         - u.x[IDX(g, k + 1, j, i - 1)] + u.x[IDX(g, k - 1, j, i - 1)]);
        float czy = 0.25f*(u.y[IDX(g, k + 1, j + 1, i)] - u.y[IDX(g, k - 1, j + 1, i)]
                         - u.y[IDX(g, k + 1, j - 1, i)] + u.y[IDX(g, k - 1, j - 1, i)]);
        float lx = 2.f*d2xx_x + d2yy_x + d2zz_x + cxy + cxz;
        float ly = 2.f*d2yy_y + d2xx_y + d2zz_y + cyx + cyz;
        float lz = 2.f*d2zz_z + d2xx_z + d2yy_z + czx + czy;
        u.x[p] += c*lx/dh2;
        u.y[p] += c*ly/dh2;
        u.z[p] += c*lz/dh2;
```

Note:
* `c = NU*dt` resta invariato (la divisione per `dh2` è esplicita, come prima).
* servono solo ghost a ±1 in tutte le direzioni → **nessun cambio** a `bc_u`,
  che già riempie 1 piano in x/y e 2 in z.
* `dh2` resta usato; nessuna nuova allocazione; nessun nuovo scambio MPI.
* il numero di `IDX(` passa da 249 a 273 (atteso: 18 + 24).

### P2 — politica del `dt` come `basilisk/src/timestep.h` (opzionale)

Sostituisce la riga 1111

```c
    if (dt > 1.1*dt_prev) dt = 1.1*dt_prev;   /* crescita max 10%/passo */
```

con

```c
    /* politica di basilisk/src/timestep.h: la crescita e' limitata da
     * (previous + 0.1*dt)/1.1, quindi il primo passo vale ~0.09*dt_CFL e il
     * limite CFL si raggiunge dopo ~30 passi (come il riferimento). */
    {
        static double prev_ref = 0.;
        if (t == 0.) prev_ref = 0.;
        if (dt > prev_ref) dt = (prev_ref + 0.1*dt)/1.1;
        prev_ref = dt;
    }
```

Senza P2 il confronto a CFL nominale uguale avviene con `dt` diversi del 27%
(§4.5); con P2 i due codici fanno un numero di passi confrontabile (atteso
~50 contro 52). P2 **non** è una BC e non tocca lo schema: cambia solo quanti
passi si fanno.

### `patch.py`

```
python3 patch.py --check                       # verifica, non scrive
python3 patch.py --check --with-dt-policy --out /tmp/mpi_fissa_patched.cu
python3 patch.py --apply                       # P1 + backup .bak-<timestamp>
python3 patch.py --apply --with-dt-policy      # P1 + P2
```

Comportamento:
* legge il file (default: il path WSL del prototipo, fallback POSIX);
* verifica che ogni stringa `OLD` compaia **1** volta (e che la `NEW` non ci
  sia già → idempotenza: se già patchato lo dice e salta);
* controlli di sanità: bilanciamento parentesi/graffe invariato, `IDX(`/`MPI_`
  contati prima/dopo, nessuna scrittura se un controllo fallisce (exit 2);
* `--apply` fa il backup **prima** e scrive in modo atomico (`.tmp-patch` +
  `os.replace`);
* `--out` permette di generare una copia senza toccare il workspace (è quello
  che ho usato: `mpi_fissa_patched.cu` è già l'anteprima P1+P2).

`patch.py --check` eseguito sul file reale: **P1 OK (1 occorrenza)**, **P2 OK
(1 occorrenza)**, sbilanciamento parentesi/graffe invariato (-4, 0), scritta
l'anteprima. Non ho eseguito `--apply` (vincolo: non scrivo nel workspace).

---

## 7. Test di accettazione (le cifre sono PREVISIONI, non misure)

### T0 — invariante: `t=0` non cambia
La diagnostica e il dump a `t=0` sono scritti **prima del primo passo**
(`write_diagnostics(t, dt0, 0, 0, 0.0)`, riga 1544), quindi non passano né da
`viscosity()` né da P2, e `bc_u` non è toccato. Atteso: diagnostiche a `t=0`
identiche a prima (Z 1.24323318e-01 ecc.) e dump `t=0` identico bit-per-bit.
Se cambia qualcosa a `t=0`, la patch è stata applicata male.

### T1 — accettazione principale (NX=64, `-tmax 1 -cfl 0.1`, np=1)
```sh
# 0) prima la verifica a secco (non scrive), poi l'applicazione con backup
cd /mnt/c/Users/zorka/AppData/Local/Temp/mpi_work/bc     # cartella di patch.py
python3 patch.py --check
python3 patch.py --apply                                  # P1 (+ --with-dt-policy per P1+P2)
#    il backup finisce accanto al sorgente: mpi_fissa.cu.bak-<timestamp>

cd "/home/zorka/simulazioni/Test 1/mpi_fissa"
NX=64 ./scripts/build.sh cpu
./scripts/run_cpu.sh 1 -tmax 1 -cfl 0.1 -out out/acc_p1
python3 scripts/confronta_campi.py out/acc_p1/omega3d_matrix.bin \
        /tmp/saetta3d_cfl010/omega3d_matrix.bin --tol-rms 3e-3
```
(Se preferisci non toccare il workspace: `patch.py --check --out
/tmp/mpi_fissa_patched.cu` e poi compila la copia a mano; `--out` non scrive
nel workspace.)

| grandezza | prima (misura) | previsione P1 | previsione P1+P2 |
|---|---|---|---|
| `rms_rel` | 1.006e-02 | **6–9e-03** | **5–8e-03** |
| `linf` | 7.19e-02 | 4–6e-02 | 4–6e-02 |
| media \|d\| i<3 | 1.104e-03 | 0.5–0.75× (→ 5–8e-04) | 0.45–0.7× |
| media \|d\| k<3 | 1.085e-03 | 0.5–0.75× | 0.45–0.7× |
| media \|d\| j<3 | 3.39e-04 | invariata (~) | invariata (~) |
| media \|d\| interno | 6.08e-04 | 5–6e-04 | 4.5–5.5e-04 |

**Il criterio `rms_rel < 3e-3` NON è previsto essere raggiunto** da una patch
di BC/viscosità: la parte di differenza che resta è, per la mia analisi, la
somma di (a) la parte non catturata dallo stencil (trasposte alle pareti
trattate diversamente dal solve implicito), (b) il termine O(Δ) del flusso
normale alle pareti, (c) la parte temporale (§4.3). Chi ha scritto il criterio
lo consideri un obiettivo da verificare, non una previsione.

### T2 — MPI invariato
```sh
./scripts/run_cpu.sh 2 -tmax 1 -cfl 0.1 -out out/acc_p1_np2
python3 scripts/confronta_campi.py out/acc_p1/omega3d_matrix.bin \
        out/acc_p1_np2/omega3d_matrix.bin --tol-rms 1e-5
```
Atteso: `rms_rel` ≤ 1e-5 (idealmente ~1e-7), come prima della patch. La patch
non tocca indici, scambi né riduzioni; se questo test peggiora, il problema è
nel build/ambiente, non nella patch.

### T3 — coerenza fisica (non regressione)
`Z`, `E`, `G` a `t=1` devono restare a 10⁻⁵–10⁻⁴ relativi dai valori attuali
(la viscosità non cambia l'enstrofia in modo visibile a `ν=1e-5`). Se `Z`
cambia di >1%, la patch ha un errore di indice.

### Procedura di decisione
1. Applica **P1**, rebuild, T0/T1/T2.
2. Se `rms_rel` ≈ 6–9e-3 → diagnosi confermata, chiudi qui (o valuta P2).
3. Se `rms_rel` ≈ 1.0e-2 (nessun miglioramento) → l'operatore viscoso non è
   la causa dominante: passaggio successivo **P2** e, se ancora nulla,
   misurare il flusso normale alle pareti del riferimento (dump di `u` a
   `t=1` con saetta: non c'è, va prodotto) prima di toccare altro.
4. Se peggiora → ripristina dal `.bak` e considera che la differenza è
   dominata dalla parte temporale.

---

## 8. Rischi e cose da NON fare

* **Non** "allineare" il flusso normale alle pareti facendolo diventare
  `u_n[0] + O(dt)` come fa la formula di `prediction()` di Basilisk: dal punto
  di vista fisico è un flusso di massa attraverso una parete. È la differenza
  documentata in `VALIDAZIONE.md` §T4 e va lasciata com'è. (Non ho potuto
  verificare se nel run di riferimento quel flusso sia effettivamente nullo:
  servirebbe un dump di `u`.)
* **Non** passare alla viscosità implicita per "essere più simili": il
  termine temporale è già trascurabile (numero di diffusione 4·10⁻⁴) e
  l'implicito richiederebbe un solve in più per passo.
* **Non** toccare `POISSON_NIT`: misurato, non cambia nulla (§4.1).
* P2 cambia la traiettoria temporale: se il Lead ha un `dt`-target o fa
  confronti a passi fissi, P2 va tenuta spenta. È separabile per costruzione
  (`--with-dt-policy`).
* La patch P1 **non** è stata compilata né eseguita (vincolo esplicito). Il
  rischio di errore di sintassi è basso (sostituzione locale, nessuna nuova
  variabile fuori dallo scope della lambda) ma reale; il primo `build.sh`
  dirà subito. Il rischio di errore di *indice* è coperto dalla validazione
  numerica dello stencil (§4.4) e dal test T3.
* `viscosity()` aggiorna `u` in place leggendo i vicini: c'è una corsa
  preesistente (benigna: l'incremento è ~10⁻⁶ contro un campo ~10⁻²), non
  introdotta dalla patch e non corretta per non allargare il diff. Su CPU con
  `OMP_NUM_THREADS>1` può dare non determinismo a livello di 10⁻⁹; se serve
  riproducibilità bit-per-bit, va aggiunto un buffer temporaneo (patch
  separata).

---

## 9. Appendice — comandi usati per le misure

```sh
# differenza e distribuzione spaziale
python3 probe_wall.py          # profili per piano, medie per zona
python3 probe_variance.py      # percentili, facce, quota di varianza
python3 probe_structure.py     # fit d ~ grad(w) + a*w, residuo
python3 probe_temporal.py      # confronto con l'errore temporale del riferimento
python3 probe_poisson.py       # pn2/pn4/pn6/pn8 vs riferimento
python3 check_stencil.py       # stencil P1 vs derivata esatta
python3 check_stencil2.py      # ordine di convergenza dello stencil
```
