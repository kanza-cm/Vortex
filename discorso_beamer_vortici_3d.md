# Discorso per la presentazione Beamer — versione 10 slide

**Durata prevista:** circa 10-11 minuti (10 slide).
**Uso:** testo da dire; i numeri sono quelli delle slide. Le note tra parentesi quadre sono indicazioni per l'oratore.

---

## Slide 1 — Numerical Formulation & Setup (0:00-1:20)

Risolviamo Navier-Stokes incomprimibili in forma velocità-pressione. Lo schema è centered: proiezione della pressione con \(\nabla^2 p = \frac{1}{\Delta t}\nabla\cdot\mathbf{u}^*\), correzione delle velocità di faccia, avvezione BCG del secondo ordine. Lo stesso nucleo è implementato in due modi: **griglia fissa** con Basilisk su CUDA in singola precisione, e **griglia adattiva** con incflo/AMReX, sempre su CUDA.

I parametri sono gli stessi del caso 2D: \(\sigma=0.09\), separazione \(2d=0.28\) con \(D=0.14\), viscosità \(10^{-5}\), Reynolds di circolazione circa \(2.5\times10^3\). I livelli vanno da L4 a L8, cioè da \(64^3\) fino a \(256\times256\times1024\), con CFL tra 0.3 e 0.5. Per l'AMR: rapporto di raffinamento 4 e soglia VORTERR tra 0.3 e \(10^{-2}\).

## Slide 2 — Fixed-Grid GPU: Cost Model (1:20-2:10)

Il costo su griglia fissa è la tabella che avete davanti. A L6, un milione di celle, 0.15 GB e circa 0.2 secondi per passo; a L7, 8.4 milioni, 1.1 GB e 1.6 secondi; a L8, 67 milioni di celle, 7.5 GB e 2.8 secondi per passo. Il modello è semplice: circa 120 byte per cella, tra campi, ghost, gerarchia multigrid e contesto CUDA.

La singola precisione non è una scorciatoia: con 8 GB si arriva a 67 milioni di celle, con 16 GB a circa 140 milioni, ma il livello successivo, 537 milioni di celle, richiederebbe circa 58 GB. Il limite è di memoria prima che di calcolo.

Da dove viene 120 byte per cella? In singola precisione ogni valore costa 4 byte. Il solver alloca 26 variabili -- il log di avvio stampa '26 var': tre componenti di velocità, tre di velocità di faccia, tre di incremento viscoso Crank-Nicolson, pressione di cella e di faccia, accelerazione, mu/rho e i temporanei del multigrid. Sono 26 x 4 = 104 byte per cella; aggiungendo uno strato di ghost e l'overhead di allocatore e contesto CUDA si arriva a circa 120 byte per cella, cioè circa 30 valori fp32 per cella. Quindi 8 GB / 120 B ~ 67M celle (L8), 16 GB ~ 140M, e L9 (537M celle) richiederebbe ~58 GB.

## Slide 3 — Conservation Laws & Dynamics Diagnostics (2:10-3:50)

Ora le diagnostiche di conservazione a T = 1600. La circolazione non e' misurata su una singola fetta: e' la media di volume su tutto il dominio 3D, Gamma = (1/Lz) int_V omega_z dV, confrontata con il valore iniziale analitico Gamma_0 = 2 pi sigma^2 = 5.089e-2.

Cala del 42.9% negli inclinati e del 7.8% nei non inclinati. Non e' un errore del solver: un volume fisso non e' un volume materiale, e con le pareti int_V omega_z dV diventa un termine di bordo, quindi contribuiscono diffusione viscosa, tilting e shear alle pareti. A T = 50 la stessa media 3D era solo ~1% (inclinati) e 0.03% (non inclinati): la decadenza e' un effetto di tempi lunghi.

L'impulso angolare cresce (+3.8% inclinati, +134% non inclinati) perche' la vorticità si allarga radialmente, mentre la palinstrofia cala da 424 a 138 (inclinati) e da 399 a 363 (non inclinati), coerente con la diffusione viscosa.

---

## Slide 4 — Grid Convergence & Parameter Sensitivity (3:50-5:20)

Lo studio di convergenza. Metodo: stesso CFL, quindi Δt ∝ h; il campo ω_z di ogni livello viene confrontato con il riferimento L8 dopo restriction alla griglia Lf=5 e filtro gaussiano σ=3 celle, che elimina il contenuto di fase dei filamenti.

Ordini misurati: per gli inclinati p = 1.93 da L5 a L6 e 2.24 da L6 a L7 (fit 2.09 su L5-L7); per i non inclinati 2.22 e 2.50 (fit 2.36). Senza filtro l'inclinato scende a circa 1.4-2.0: lo schema è al secondo ordine e la perdita apparente è un errore di fase.

Messaggio operativo: l'accuratezza la guida MAXLEVEL; la tolleranza AMR comprime le celle, non sostituisce la risoluzione.

## Slide 5 — Poisson Solver: Elliptic Residuals (5:20-6:10)

La proiezione di pressione è la parte ellittica. La tabella mostra il residuo massimo a \(T=30\): cresce con il livello, perché ci sono più celle con la stessa tolleranza, ma resta sotto \(1.3\times10^{-2}\) a L8; nei non inclinati resta addirittura sotto \(2.3\times10^{-3}\) a tutti i livelli.

Il residuo va confrontato con la tolleranza del solver, \(10^{-3}/\Delta t^2\), non con 1. Quindi il solve è convergente e **non è il collo di bottiglia**: l'errore dominante è quello di discretizzazione spaziale. Ho anche verificato che con **5 V-cycle fissi** per la pressione gli errori dello studio di convergenza cambiano meno dello 0.4%: bastano, e il residuo resta piccolo.

## Slide 6 — Adaptive Grid: How It Is Built (6:10-7:00)

Passiamo all'AMR. Si parte da una base uniforme \(2^\text{MINLEVEL}\) e ogni livello raffinato moltiplica per 4. Il tagging è sulla vorticità: le celle con \(|\boldsymbol\omega|\) sopra la soglia VORTERR diventano nuove box, che vengono ricostruite periodicamente al regrid, con average down e up tra i livelli. C'è un budget massimo di celle: se la gerarchia esplode, le soglie si alzano automaticamente. Il disegno a destra mostra la gerarchia annidata: L0, L1, L2.

## Slide 7 — AMR on GPU: Data Structures & Regrid Cycle (7:00-7:50)

Su GPU ogni livello è una **BoxArray**: una lista di box non sovrapposti che coprono solo le regioni da raffinare. I dati sono MultiFab: BoxArray più distribution mapping più array sul device, in singola precisione. Il tagging è un kernel GPU che calcola \(|\omega|\) cella per cella; le celle flaggate vengono raggruppate in poche box grandi con un algoritmo tipo Berger-Oliger, per limitare il numero di kernel launch. Al regrid si costruisce la nuova gerarchia, si fa average down della soluzione fine e si riempiono i ghost coarse/fine.

## Slide 8 — AMR on GPU: Execution Model & Costs (7:50-8:40)

L'aggiornamento di un livello è una sequenza di kernel GPU: avvezione e diffusione sono una kernel per box, il solve di Poisson usa un multigrid tra i livelli. I ghost si riempiono copiando dai vicini dello stesso livello e interpolando dal coarse; dopo il regrid si fa average down e fill patches. Le riduzioni globali, come le norme e il conteggio delle celle taggate, devono completarsi prima di un nuovo regrid.

Il costo: circa 357 MB per milione di celle, tre volte la griglia fissa. Misurato, 11.7 milioni di celle occupano 6.4 GB; la campagna a \(T=30\) ha un tetto di 15 milioni di celle per stare in 8 GB.

Da dove viene il costo AMR? Ogni MultiFab conserva gli stessi campi con due o tre strati di ghost, più i metadati di BoxArray/distribution mapping e i buffer temporanei del multigrid tenuti nell'arena. La stima di progetto è circa 320 byte per cella, cioè circa 80 valori fp32 per cella (26 campi più ghost e temporanei), circa 2.7 volte la griglia fissa; il valore misurato in tabella è 357 MB per milione di celle. A 11.7M celle dà 6.4 GB, che corrispondono a ~550 byte per cella perché ai livelli fini si aggiungono i livelli grossolani duplicati e l'arena riserva buffer; per questo il tetto pratico è 15M celle in 8 GB (8 GB / 550 B ~ 15M), non i 22M che darebbe il solo 357 byte per cella.

## Slide 9 — Adaptive Grid: Error Map at T=30 (8:40-9:30)

Questa è la mappa d'errore a T = 30, per entrambe le configurazioni: MAXLEVEL 4-8 sull'asse verticale, la soglia su quello orizzontale.

A MAXLEVEL 4-5 l'errore è fissato dalla griglia, non dalla soglia: resta piatto, da +0.14 a -0.07 per gli inclinati e da -0.45 a -0.79 per i non inclinati.

Da L6 in poi conta la soglia, e i livelli più fini danno il guadagno maggiore; la gerarchia AMR arriva fino a 1.5×10⁷ celle a L8.

## Slide 10 — Methods & Software (9:30-10:30)

Prima di chiudere, due parole sul software. Per la griglia fissa uso Basilisk (Popinet, 2015) con lo schema centered e il backend CUDA; per l'AMR a blocchi uso incflo/AMReX (Zhang et al., 2019). Sopra questi ho implementato il termine viscoso al secondo ordine di Crank-Nicolson, le diagnostiche di palinstrofia e impulso angolare calcolate durante la run, e la pipeline delle mappe d'errore AMR e dei costi GPU. Tutte le run sono su una singola RTX 3070 da 8 GB, in singola precisione.


---

# Appendice — possibili domande

**Perché a T=1600 la circolazione sulla fetta cala del 43%?**
A T=200 il calo era sotto lo 0.5%: solo tilting e stretching. A T=1600 la vorticità si è diffusa e allargata, e il flusso attraverso la fetta cambia anche per decadimento viscoso e shear alle pareti: il calo misura la decadenza del momento, non un errore di schema.

**Perché a T=50 gli inclinati perdono l'ordine 2?**
A T=50 il flusso è dominato da filamenti: l'errore \(L_\infty\) è di fatto un errore di fase, non di dissipazione. \(L_2\) dà circa 1; servirebbe un riferimento più fine (L9 non entra in 8 GB) oppure norme filtrate: coarsenando entrambi i campi a L5/L6 l'ordine dovrebbe tornare vicino a 2.

**Quanto valgono i residui di Poisson a T=50?**
Massimi: 7.2e-2 per gli inclinati a L8, 2.2e-3 per i non inclinati, sempre sotto la tolleranza \(10^{-3}/\Delta t^2\). L'L8 inclinato è il caso più esigente, ma resta al 7% della tolleranza.

**5 iterazioni di pressione bastano?**
Sì: con 5 V-cycle fissi solo sulla pressione gli errori dello studio T=30 cambiano meno dello 0.4%, e il residuo finale a L8 è \(8\times10^{-5}\) (inclinati) e \(4\times10^{-5}\) (non inclinati).

**Perché niente MPI nella presentazione?**
Il prototipo MPI esiste ed è validato (decomposizione a lastre, V-cycle distribuito), ma il guadagno su due PC è memoria, non tempo (tetto ~1.5x). L'ho lasciato fuori per tenere il talk su una lunghezza gestibile; posso parlarne nelle domande.

**Perché la circolazione sulla fetta non è conservata?**
In 3D solo la circolazione lungo una curva materiale è conservata (Kelvin); la fetta fissa \(z=2\) non è materiale, e tilting/stretching ne cambiano il flusso. Per questo la uso come check, non come bilancio.

**Filmati e dati**
Animazione 3D e movie delle fette sono in `gpu_nvidia_3d/risultati/video/`; i dati delle diagnostiche e gli studi di errore sono nelle cartelle `risultati/error_study/` e `risultati/run_L7_T200_diag/`.
