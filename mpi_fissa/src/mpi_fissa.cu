/**
 * mpi_fissa.cu -- prototipo standalone: solver 3D incompressibile a GRIGLIA FISSA
 * con DECOMPOSIZIONE DI DOMINIO MPI (slab lungo z) e backend CUDA opzionale.
 *
 * Caso fisico: "vortex coalescing" 3D (due tubi di vorticita' co-rotanti con
 * perturbazione varicosa e assi inclinati a X), lo stesso caso di
 * gpu_nvidia_3d/saetta3d.c. Lo schema ricalca quello di Basilisk
 * navier-stokes/centered.h (BCG 2nd order unsplit + proiezione approssimata
 * su griglia MAC) in modo da poter confrontare i risultati con il riferimento.
 *
 * Differenze note e volute rispetto a saetta3d.c (documentate in VALIDAZIONE.md):
 *  - viscosita' ESPLICITA (nu*laplaciano(u)) invece che implicita decoupled:
 *    a nu=1e-5 il numero di diffusione e' ~1e-2, l'errore e' trascurabile;
 *  - calcolo in singola precisione nei kernel (i campi di Basilisk sono
 *    comunque float: la differenza e' a livello di arrotondamento fp32);
 *  - proiezione: multigrid geometrico distribuito a iterazioni FISSE (nessuna
 *    allreduce nel ciclo caldo) invece del multigrid adattivo di Basilisk;
 *  - face normali alle pareti forzate a zero nel prediction (BC MAC corretta).
 *
 * Decomposizione: SOLO lungo z (la direzione lunga, nz = 4*nx). Ogni rank
 * possiede un numero intero di piani z contigui, con 2 celle fantasma in z
 * (servono al prediction per il calcolo delle facce INTERNE all'interfaccia)
 * e 1 in x/y (pareti fisiche free-slip). La pressione usa 1 solo strato
 * fantasma (stencil a 7 punti).
 *
 * Uso:
 *   bin/mpi_fissa_cpu -tmax 10 -out out_cpu_np1
 *   mpirun -np 2 bin/mpi_fissa_gpu -tmax 10 -out out_gpu_np2
 *
 * La proiezione usa POISSON_NIT V-cycle FISSI (default 2; -poisson_nit 1
 * dimezza la comunicazione ma lascia piu' divergenza residua).
 *
 * Compilazione: vedere scripts/build.sh
 *   CPU: mpicxx -O3 -std=c++17 -fopenmp -x c++ -DMF_USE_CUDA=0 -DMF_NX=128 mpi_fissa.cu
 *   GPU: nvcc -O3 -std=c++17 --extended-lambda -arch=sm_86 -ccbin mpicxx \
 *             -DMF_USE_CUDA=1 -DMF_NX=128 mpi_fissa.cu -o mpi_fissa_gpu -lcudart
 *
 * Output (in -out DIR):
 *   diagnostics.dat      serie temporale, stesse colonne di saetta3d.c
 *   (stderr del rank 0)  log: griglia, decomposizione, livelli, tempi/comm, riepilogo
 *                        (per salvarlo: ... 2> log.txt)
 *   omega3d_matrix.bin   omega_z finale: int32 nx,ny,nz + float32 (i piu' veloce),
 *                        formato identico a quello di saetta3d.c
 *   timing.dat           tempo per passo, tempo di comunicazione, MB scambiati
 *   done_<rank>.txt      marker di fine (per gli script)
 */

#include <mpi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstddef>

#ifndef MF_NX
#  define MF_NX 64          /* nx = ny; nz = 4*nx (dominio 1x1x4, celle cubiche) */
#endif
#ifndef MF_USE_CUDA
#  define MF_USE_CUDA 0
#endif
#ifndef MF_COARSE_SWEEPS
#  define MF_COARSE_SWEEPS 30   /* sweeps di red-black GS sull'ultimo livello (4 celle) */
#endif

/* ============================ backend astratto ============================ */

#if MF_USE_CUDA
#  include <cuda_runtime.h>
#  define MF_HD __host__ __device__
#  define MF_DEV __device__
static void mf_cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
static float* mf_alloc(size_t n) {
    float* p = NULL;
    mf_cuda_ok(cudaMalloc((void**)&p, n*sizeof(float)), "cudaMalloc");
    return p;
}
static void mf_free(float* p) { if (p) mf_cuda_ok(cudaFree(p), "cudaFree"); }
static void mf_zero(float* p, size_t n) { mf_cuda_ok(cudaMemset(p, 0, n*sizeof(float)), "cudaMemset"); }
static void mf_h2d(float* d, const float* h, size_t n) {
    mf_cuda_ok(cudaMemcpy(d, h, n*sizeof(float), cudaMemcpyHostToDevice), "h2d");
}
static void mf_d2h(float* h, const float* d, size_t n) {
    mf_cuda_ok(cudaMemcpy(h, d, n*sizeof(float), cudaMemcpyDeviceToHost), "d2h");
}
static void mf_h2d_2d(float* d, size_t dpitch, const float* h, size_t hpitch,
                      size_t width, size_t height) {
    mf_cuda_ok(cudaMemcpy2D(d, dpitch, h, hpitch, width, height, cudaMemcpyHostToDevice), "h2d2d");
}
static void mf_d2h_2d(float* h, size_t hpitch, const float* d, size_t dpitch,
                      size_t width, size_t height) {
    mf_cuda_ok(cudaMemcpy2D(h, hpitch, d, dpitch, width, height, cudaMemcpyDeviceToHost), "d2h2d");
}
static void mf_sync(void) { mf_cuda_ok(cudaDeviceSynchronize(), "sync"); }
static float* mf_alloc_pinned(size_t n) {
    float* p = NULL;
    mf_cuda_ok(cudaHostAlloc((void**)&p, n*sizeof(float), cudaHostAllocDefault), "cudaHostAlloc");
    return p;
}
static void mf_free_pinned(float* p) { if (p) mf_cuda_ok(cudaFreeHost(p), "cudaFreeHost"); }

template <class F>
__global__ void mf_kernel(int n, F f) {
    int stride = gridDim.x*blockDim.x;
    for (int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) f(i);
}
template <class F>
static void parallel_for(int n, F f) {
    if (n <= 0) return;
    int threads = 256;
    int blocks = (n + threads - 1)/threads;
    if (blocks > 65535) blocks = 65535;
    mf_kernel<<<blocks, threads>>>(n, f);
}

__global__ void mf_max_abs_kernel(const float* a, size_t n, int* out) {
    size_t stride = (size_t)gridDim.x*blockDim.x;
    for (size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = fabsf(a[i]);
        atomicMax(out, __float_as_int(v));
    }
}
static float mf_max_abs(const float* a, size_t n) {
    static int* d = NULL;
    if (!d) mf_cuda_ok(cudaMalloc((void**)&d, sizeof(int)), "max:malloc");
    mf_cuda_ok(cudaMemset(d, 0, sizeof(int)), "max:memset");
    if (n > 0) {
        size_t threads = 256;
        size_t blocks = (n + threads - 1)/threads;
        if (blocks > 65535) blocks = 65535;
        mf_max_abs_kernel<<<(int)blocks, (int)threads>>>(a, n, d);
    }
    int h = 0;
    mf_cuda_ok(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost), "max:copy");
    float hf = 0.f;
    std::memcpy(&hf, &h, sizeof(float));
    return hf;
}

#else /* ---------------- CPU (OpenMP) ---------------- */

#  define MF_HD
#  define MF_DEV
static float* mf_alloc(size_t n) {
    float* p = (float*)std::malloc(n*sizeof(float));
    if (!p) { std::fprintf(stderr, "malloc %.1f MB fallita\n", n*sizeof(float)/1e6); std::exit(2); }
    return p;
}
static void mf_free(float* p) { std::free(p); }
static void mf_zero(float* p, size_t n) { std::memset(p, 0, n*sizeof(float)); }
static void mf_h2d(float* d, const float* h, size_t n) { std::memcpy(d, h, n*sizeof(float)); }
static void mf_d2h(float* h, const float* d, size_t n) { std::memcpy(h, d, n*sizeof(float)); }
static void mf_h2d_2d(float* d, size_t dpitch, const float* h, size_t hpitch,
                      size_t width, size_t height) {
    for (size_t j = 0; j < height; j++)
        std::memcpy((char*)d + j*dpitch, (const char*)h + j*hpitch, width);
}
static void mf_d2h_2d(float* h, size_t hpitch, const float* d, size_t dpitch,
                      size_t width, size_t height) {
    for (size_t j = 0; j < height; j++)
        std::memcpy((char*)h + j*hpitch, (const char*)d + j*dpitch, width);
}
static void mf_sync(void) {}
static float* mf_alloc_pinned(size_t n) {
    float* p = (float*)std::malloc(n*sizeof(float));
    if (!p) { std::fprintf(stderr, "malloc pinned fallita\n"); std::exit(2); }
    return p;
}
static void mf_free_pinned(float* p) { std::free(p); }

template <class F>
static void parallel_for(int n, F f) {
    if (n <= 0) return;
    long long nn = n;
#  pragma omp parallel for schedule(static)
    for (long long ii = 0; ii < nn; ii++) f((int)ii);
}

static float mf_max_abs(const float* a, size_t n) {
    float m = 0.f;
    long long nn = (long long)n;
#  pragma omp parallel for reduction(max:m) schedule(static)
    for (long long i = 0; i < nn; i++) {
        float v = std::fabs(a[i]);
        if (v > m) m = v;
    }
    return m;
}
#endif /* MF_USE_CUDA */

/* ============================ struttura griglia ============================ */

/* Celle fantasma: 2 strati in z (k = -2,-1 e nzloc,nzloc+1), 1 strato in x/y.
 * Layout flat: index(k,j,i) = (((k+2)*(ny+2) + (j+1))*(nx+2) + (i+1)).
 * Valori validi: k in [-2, nzloc+1], j in [-1, ny], i in [-1, nx]. */
struct Grid {
    int nx, ny, nzloc;   /* celle interne locali */
    int nxh, nyh;        /* nx+2, ny+2 */
    int nzg;             /* numero di celle z dell'intero dominio a questo livello */
    int k0;              /* indice z globale della cella locale k=0 */
    int repl;            /* 1 = griglia replicata in z (tutto il dominio su ogni rank) */
    size_t n;
};

/* array di 3 puntatori "catturabile per valore" nelle lambda device */
struct Arr3f { float* p[3]; };

struct Vec3f {
    float* x;
    float* y;
    float* z;
};

MF_HD inline size_t IDX(Grid g, int k, int j, int i) {
    return (((size_t)(k + 2) * (size_t)g.nyh) + (size_t)(j + 1)) * (size_t)g.nxh + (size_t)(i + 1);
}

static Grid mk_grid(int nx, int ny, int nzloc, int nzg, int k0, int repl) {
    Grid g;
    g.nx = nx; g.ny = ny; g.nzloc = nzloc;
    g.nxh = nx + 2; g.nyh = ny + 2;
    g.nzg = nzg; g.k0 = k0; g.repl = repl;
    g.n = (size_t)(nzloc + 4) * (size_t)g.nyh * (size_t)g.nxh;
    return g;
}

/* ============================ stato globale ============================ */

static int NX, NY, NZ;          /* griglia globale */
static float DH;                /* Delta = 1/NX */
static int RANK, NRANKS;
static int K0, K1, NZL;         /* slab locale [K0,K1), NZL = K1-K0 */
static Grid DOM;                /* griglia piu' fine (u, uf, p, ...) */

static double TMAX = 10.0;
static double CFL = 0.3;
static double NU = 1e-5;
static int POISSON_NIT = 2;     /* V-cycle per proiezione (1 = piu' veloce, 2 = piu' accurato) */
static int MG_VERBOSE = 0;
static int VERBOSE = 0;
static double TDIAG = 1.0;      /* passo della diagnostica */
static char OUTDIR[512] = "out";
static char DUMPFILE[1024] = ""; /* vuoto = OUTDIR/omega3d_matrix.bin */

static const double LZ = 4.0;
static const double ZNODE = 2.0;

/* parametri fisici (identici a saetta3d.c) */
static const double DD = 0.14, SIGMA = 0.09, PERT = 0.10, TILT = 2.0;
static const double PI_D = 3.14159265358979323846;

/* campi sulla griglia piu' fine */
static Vec3f U, UF, FLUX, GRD;
static float *P, *PF, *RHS, *RES;

/* buffer host per lo scambio di alone e per la diagnostica */
static float* EXSEND = NULL;
static float* EXRECV = NULL;
static size_t EXCAP = 0;

/* campi host per la diagnostica */
static float *HU0 = NULL, *HU1 = NULL, *HU2 = NULL;

/* contatori prestazioni */
static double T_COMM = 0.0;         /* secondi in MPI */
static double B_COMM = 0.0;         /* byte MPI */
static long   N_EXCH = 0;           /* scambi MPI */

/* ============================ multigrid (dichiarazioni) ============================ */

struct Lvl {
    Grid g;
    float dh, dh2;
    float *sol, *rhs, *res;
};

static Lvl LV[16];
static int NL = 0;      /* numero di livelli (0..NL-1) */
static int LD = 0;      /* ultimo livello distribuito (0..LD); LD+1 e' replicato */

static void vcycle(int l);
static void smooth_lvl(Lvl& L, int nsweeps);
static void residual_lvl(Lvl& L);

/* ============================ utility MPI ============================ */

static double now_s(void) { return MPI_Wtime(); }

static void ensure_exchange_capacity(size_t floats) {
    if (floats > EXCAP) {
        if (EXSEND) mf_free_pinned(EXSEND);
        if (EXRECV) mf_free_pinned(EXRECV);
        EXCAP = floats;
        EXSEND = mf_alloc_pinned(EXCAP);
        EXRECV = mf_alloc_pinned(EXCAP);
    }
}

/* copia un piano z (celle interne x/y) in un buffer contiguo [ny][nx] */
static void pack_plane(const Grid& g, const float* a, int k, float* dst) {
    mf_d2h_2d(dst, (size_t)g.nx*sizeof(float), a + IDX(g, k, 0, 0),
              (size_t)g.nxh*sizeof(float), (size_t)g.nx*sizeof(float), (size_t)g.ny);
}
static void unpack_plane(const Grid& g, float* a, int k, const float* src) {
    mf_h2d_2d(a + IDX(g, k, 0, 0), (size_t)g.nxh*sizeof(float),
              src, (size_t)g.nx*sizeof(float), (size_t)g.nx*sizeof(float), (size_t)g.ny);
}

/* Scambio di np piani in z per i 3 componenti di u (velocita' centrata).
 * np=2 serve al prediction delle facce z all'interfaccia tra rank;
 * np=1 basta a viscosita'/accelerazione (stencil a 7 punti). */
static void exchange_u(const Vec3f& u, int tag, int np) {
    if (NRANKS == 1) return;
    const Grid& g = DOM;
    size_t face = (size_t)g.nx*(size_t)g.ny;
    size_t tot = (size_t)3*np*face;
    ensure_exchange_capacity(2*tot);
    double t0 = now_s();

    int below = (RANK > 0) ? RANK - 1 : MPI_PROC_NULL;
    int above = (RANK < NRANKS - 1) ? RANK + 1 : MPI_PROC_NULL;
    MPI_Status st;
    Arr3f arr; arr.p[0] = u.x; arr.p[1] = u.y; arr.p[2] = u.z;

    /* invia i piani alti al rank superiore; ricevi i piani bassi dal rank inferiore */
    for (int p = 0; p < np; p++)
        for (int c = 0; c < 3; c++)
            pack_plane(g, arr.p[c], g.nzloc - np + p,
                       EXSEND + ((size_t)c*np + p)*face);
    MPI_Sendrecv(EXSEND, (int)tot, MPI_FLOAT, above, tag,
                 EXRECV, (int)tot, MPI_FLOAT, below, tag,
                 MPI_COMM_WORLD, &st);
    if (below != MPI_PROC_NULL)
        for (int p = 0; p < np; p++)
            for (int c = 0; c < 3; c++)
                unpack_plane(g, arr.p[c], -np + p,
                             EXRECV + ((size_t)c*np + p)*face);

    /* invia i piani bassi al rank inferiore; ricevi i piani alti dal rank superiore */
    for (int p = 0; p < np; p++)
        for (int c = 0; c < 3; c++)
            pack_plane(g, arr.p[c], p, EXSEND + ((size_t)c*np + p)*face);
    MPI_Sendrecv(EXSEND, (int)tot, MPI_FLOAT, below, tag + 1,
                 EXRECV, (int)tot, MPI_FLOAT, above, tag + 1,
                 MPI_COMM_WORLD, &st);
    if (above != MPI_PROC_NULL)
        for (int p = 0; p < np; p++)
            for (int c = 0; c < 3; c++)
                unpack_plane(g, arr.p[c], g.nzloc + p,
                             EXRECV + ((size_t)c*np + p)*face);

    T_COMM += now_s() - t0;
    B_COMM += 2.0*(double)(tot*sizeof(float));
    N_EXCH++;
}

/* Scambio di 1 piano in z per un campo scalare (pressione / correzione MG). */
static void exchange_scalar(const Grid& g, float* a, int tag) {
    if (NRANKS == 1 || g.repl) return;
    size_t face = (size_t)g.nx*(size_t)g.ny;
    ensure_exchange_capacity(2*face);
    double t0 = now_s();
    int below = (RANK > 0) ? RANK - 1 : MPI_PROC_NULL;
    int above = (RANK < NRANKS - 1) ? RANK + 1 : MPI_PROC_NULL;
    MPI_Status st;

    pack_plane(g, a, g.nzloc - 1, EXSEND);
    MPI_Sendrecv(EXSEND, (int)face, MPI_FLOAT, above, tag,
                 EXRECV, (int)face, MPI_FLOAT, below, tag,
                 MPI_COMM_WORLD, &st);
    if (below != MPI_PROC_NULL)
        unpack_plane(g, a, -1, EXRECV);

    pack_plane(g, a, 0, EXSEND);
    MPI_Sendrecv(EXSEND, (int)face, MPI_FLOAT, below, tag + 1,
                 EXRECV, (int)face, MPI_FLOAT, above, tag + 1,
                 MPI_COMM_WORLD, &st);
    if (above != MPI_PROC_NULL)
        unpack_plane(g, a, g.nzloc, EXRECV);

    T_COMM += now_s() - t0;
    B_COMM += 2.0*(double)(face*sizeof(float));
    N_EXCH++;
}

/* ============================ condizioni al contorno ============================ */

/* Free-slip su tutte le pareti fisiche per la velocita' centrata u:
 * componente normale alla parete: ghost = -interno (specchiato);
 * componenti tangenziali: ghost = +interno (specchiato).
 * Le pareti fisiche in z esistono solo per il primo e l'ultimo rank: gli
 * altri strati fantasma in z vengono dallo scambio MPI. */
static void bc_u(Vec3f u, int np) {
    const Grid g = DOM;
    const int NXl = g.nx, NYl = g.ny, NZLl = g.nzloc;
    if (NRANKS > 1) exchange_u(u, 100, np);
    const int klo = -np, khi = NZLl - 1 + np;   /* range z degli strati fantasma */

    /* pareti x: i = -1 e i = nx; u.x normale, u.y/u.z tangenziali */
    {
        int cnt = (khi - klo + 1)*(NYl + 2)*3;
        Arr3f arr; arr.p[0] = u.x; arr.p[1] = u.y; arr.p[2] = u.z;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int c = idx % 3; int t = idx / 3;
            int j = t % (NYl + 2) - 1; int k = t / (NYl + 2) + klo;
            float* f = arr.p[c];
            float sgn = (c == 0) ? -1.f : 1.f;
            f[IDX(g, k, j, -1)] = sgn * f[IDX(g, k, j, 0)];
            f[IDX(g, k, j, NXl)] = sgn * f[IDX(g, k, j, NXl - 1)];
        });
    }
    /* pareti y: j = -1 e j = ny; u.y normale */
    {
        int cnt = (khi - klo + 1)*(NXl + 2)*3;
        Arr3f arr; arr.p[0] = u.x; arr.p[1] = u.y; arr.p[2] = u.z;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int c = idx % 3; int t = idx / 3;
            int i = t % (NXl + 2) - 1; int k = t / (NXl + 2) + klo;
            float* f = arr.p[c];
            float sgn = (c == 1) ? -1.f : 1.f;
            f[IDX(g, k, -1, i)] = sgn * f[IDX(g, k, 0, i)];
            f[IDX(g, k, NYl, i)] = sgn * f[IDX(g, k, NYl - 1, i)];
        });
    }
    /* pareti fisiche z: solo primo/ultimo rank; u.z normale (np strati) */
    if (RANK == 0) {
        int cnt = (NYl + 2)*(NXl + 2)*3;
        Arr3f arr; arr.p[0] = u.x; arr.p[1] = u.y; arr.p[2] = u.z;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int c = idx % 3; int t = idx / 3;
            int i = t % (NXl + 2) - 1; int j = t / (NXl + 2) - 1;
            float* f = arr.p[c];
            float sgn = (c == 2) ? -1.f : 1.f;
            for (int p = 0; p < np; p++)
                f[IDX(g, -1 - p, j, i)] = sgn * f[IDX(g, p, j, i)];
        });
    }
    if (RANK == NRANKS - 1) {
        int cnt = (NYl + 2)*(NXl + 2)*3;
        Arr3f arr; arr.p[0] = u.x; arr.p[1] = u.y; arr.p[2] = u.z;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int c = idx % 3; int t = idx / 3;
            int i = t % (NXl + 2) - 1; int j = t / (NXl + 2) - 1;
            float* f = arr.p[c];
            float sgn = (c == 2) ? -1.f : 1.f;
            for (int p = 0; p < np; p++)
                f[IDX(g, NZLl + p, j, i)] = sgn * f[IDX(g, NZLl - 1 - p, j, i)];
        });
    }
}

/* Pressione: Neumann omogeneo su tutte le pareti fisiche (ghost = interno);
 * all'interfaccia tra rank il ghost e' il valore del rank vicino (scambio).
 * Usata sia per la pressione che per la correzione del multigrid. */
static void bc_pressure(Grid g, float* a) {
    exchange_scalar(g, a, 400 + (g.nx == NX ? 0 : 1));
    parallel_for((g.nzloc + 2)*(g.ny + 2), [=] MF_DEV (int idx) {
        int j = idx % (g.ny + 2) - 1; int k = idx / (g.ny + 2) - 1;
        a[IDX(g, k, j, -1)] = a[IDX(g, k, j, 0)];
        a[IDX(g, k, j, g.nx)] = a[IDX(g, k, j, g.nx - 1)];
    });
    parallel_for((g.nzloc + 2)*(g.nx + 2), [=] MF_DEV (int idx) {
        int i = idx % (g.nx + 2) - 1; int k = idx / (g.nx + 2) - 1;
        a[IDX(g, k, -1, i)] = a[IDX(g, k, 0, i)];
        a[IDX(g, k, g.ny, i)] = a[IDX(g, k, g.ny - 1, i)];
    });
    if (g.repl) {
        parallel_for((g.ny + 2)*(g.nx + 2), [=] MF_DEV (int idx) {
            int i = idx % (g.nx + 2) - 1; int j = idx / (g.nx + 2) - 1;
            a[IDX(g, -1, j, i)] = a[IDX(g, 0, j, i)];
            a[IDX(g, g.nzloc, j, i)] = a[IDX(g, g.nzloc - 1, j, i)];
        });
    } else {
        if (RANK == 0)
            parallel_for((g.ny + 2)*(g.nx + 2), [=] MF_DEV (int idx) {
                int i = idx % (g.nx + 2) - 1; int j = idx / (g.nx + 2) - 1;
                a[IDX(g, -1, j, i)] = a[IDX(g, 0, j, i)];
            });
        if (RANK == NRANKS - 1)
            parallel_for((g.ny + 2)*(g.nx + 2), [=] MF_DEV (int idx) {
                int i = idx % (g.nx + 2) - 1; int j = idx / (g.nx + 2) - 1;
                a[IDX(g, g.nzloc, j, i)] = a[IDX(g, g.nzloc - 1, j, i)];
            });
    }
}

/* ============================ prediction delle facce ============================ */
/* uf^{n+1/2} = P(u^n, g^n) secondo centered.h/prediction() (BCG):
 *  - normale: interpolazione upwind con correzione di pendenza (g = grad p + a);
 *  - tangenziali: correzione con le derivate trasversali di u (2a eq. della
 *    linearizzazione di u.grad u sul tempo mezzo).
 * Le facce normali alle pareti fisiche sono forzate a zero (BC MAC). */
static void predict_uf(Vec3f u, Vec3f uf, Vec3f grd, float dt) {
    const Grid g = DOM;
    const int NXl = NX, NYl = NY, NZLl = NZL;
    const float dh = DH;
    const int rk = RANK, nrk = NRANKS;

    /* facce x */
    {
        int cnt = (NXl + 1)*NYl*NZLl;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int fi = idx % (NXl + 1); int t = idx / (NXl + 1);
            int j = t % NYl; int k = t / NYl;
            if (fi == 0 || fi == NXl) { uf.x[IDX(g, k, j, fi)] = 0.f; return; }
            float un = dt*(u.x[IDX(g, k, j, fi - 1)] + u.x[IDX(g, k, j, fi)])/(2.f*dh);
            float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
            int iu = fi + ((un > 0.f) ? -1 : 0);
            float du = (u.x[IDX(g, k, j, iu + 1)] - u.x[IDX(g, k, j, iu - 1)])/(2.f*dh);
            float val = u.x[IDX(g, k, j, iu)]
                + (grd.x[IDX(g, k, j, fi)] + grd.x[IDX(g, k, j, fi - 1)])*(dt*0.25f)
                + s*(1.f - s*un)*du*(dh*0.5f);
            float uy = u.y[IDX(g, k, j, iu)];
            float fyy = (uy < 0.f)
                ? (u.x[IDX(g, k, j + 1, iu)] - u.x[IDX(g, k, j, iu)])
                : (u.x[IDX(g, k, j, iu)] - u.x[IDX(g, k, j - 1, iu)]);
            val -= dt*uy*fyy/(2.f*dh);
            float uz = u.z[IDX(g, k, j, iu)];
            float fzz = (uz < 0.f)
                ? (u.x[IDX(g, k + 1, j, iu)] - u.x[IDX(g, k, j, iu)])
                : (u.x[IDX(g, k, j, iu)] - u.x[IDX(g, k - 1, j, iu)]);
            val -= dt*uz*fzz/(2.f*dh);
            uf.x[IDX(g, k, j, fi)] = val;
        });
    }
    /* facce y */
    {
        int cnt = NXl*(NYl + 1)*NZLl;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int fj = idx % (NYl + 1); int t = idx / (NYl + 1);
            int i = t % NXl; int k = t / NXl;
            if (fj == 0 || fj == NYl) { uf.y[IDX(g, k, fj, i)] = 0.f; return; }
            float un = dt*(u.y[IDX(g, k, fj - 1, i)] + u.y[IDX(g, k, fj, i)])/(2.f*dh);
            float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
            int ju = fj + ((un > 0.f) ? -1 : 0);
            float du = (u.y[IDX(g, k, ju + 1, i)] - u.y[IDX(g, k, ju - 1, i)])/(2.f*dh);
            float val = u.y[IDX(g, k, ju, i)]
                + (grd.y[IDX(g, k, fj, i)] + grd.y[IDX(g, k, fj - 1, i)])*(dt*0.25f)
                + s*(1.f - s*un)*du*(dh*0.5f);
            /* tangenziale z */
            float uz = u.z[IDX(g, k, ju, i)];
            float fzz = (uz < 0.f)
                ? (u.y[IDX(g, k + 1, ju, i)] - u.y[IDX(g, k, ju, i)])
                : (u.y[IDX(g, k, ju, i)] - u.y[IDX(g, k - 1, ju, i)]);
            val -= dt*uz*fzz/(2.f*dh);
            /* tangenziale x */
            float ux = u.x[IDX(g, k, ju, i)];
            float fxx = (ux < 0.f)
                ? (u.y[IDX(g, k, ju, i + 1)] - u.y[IDX(g, k, ju, i)])
                : (u.y[IDX(g, k, ju, i)] - u.y[IDX(g, k, ju, i - 1)]);
            val -= dt*ux*fxx/(2.f*dh);
            uf.y[IDX(g, k, fj, i)] = val;
        });
    }
    /* facce z (interfacce tra rank calcolate su entrambi i lati con gli stessi
     * dati: gli strati fantasma in z rendono l'espressione identica) */
    {
        int cnt = NXl*NYl*(NZLl + 1);
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int rem = idx % (NXl*NYl);
            int fk = idx / (NXl*NYl);
            int j = rem / NXl; int i = rem - j*NXl;
            float un = dt*(u.z[IDX(g, fk - 1, j, i)] + u.z[IDX(g, fk, j, i)])/(2.f*dh);
            float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
            int ju = fk + ((un > 0.f) ? -1 : 0);
            float du = (u.z[IDX(g, ju + 1, j, i)] - u.z[IDX(g, ju - 1, j, i)])/(2.f*dh);
            float val = u.z[IDX(g, ju, j, i)]
                + (grd.z[IDX(g, fk, j, i)] + grd.z[IDX(g, fk - 1, j, i)])*(dt*0.25f)
                + s*(1.f - s*un)*du*(dh*0.5f);
            /* tangenziale x */
            float ux = u.x[IDX(g, ju, j, i)];
            float fxx = (ux < 0.f)
                ? (u.z[IDX(g, ju, j, i + 1)] - u.z[IDX(g, ju, j, i)])
                : (u.z[IDX(g, ju, j, i)] - u.z[IDX(g, ju, j, i - 1)]);
            val -= dt*ux*fxx/(2.f*dh);
            /* tangenziale y */
            float uy = u.y[IDX(g, ju, j, i)];
            float fyy = (uy < 0.f)
                ? (u.z[IDX(g, ju, j + 1, i)] - u.z[IDX(g, ju, j, i)])
                : (u.z[IDX(g, ju, j, i)] - u.z[IDX(g, ju, j - 1, i)]);
            val -= dt*uy*fyy/(2.f*dh);
            /* pareti fisiche z: solo primo/ultimo rank */
            if ((fk == 0 && rk == 0) || (fk == NZLl && rk == nrk - 1))
                val = 0.f;
            uf.z[IDX(g, fk, j, i)] = val;
        });
    }
}

/* ============================ advezione (BCG) ============================ */
/* flux_faccia = upwind(f) con pendenza limitata/di 2o ordine, trasporto della
 * quantita' f con la velocita' di faccia uf; poi u -= dt div(flux). */
static void advect(Vec3f u, Vec3f uf, Vec3f grd, Vec3f flux, float dt) {
    const Grid g = DOM;
    const int NXl = NX, NYl = NY, NZLl = NZL;
    const float dh = DH;
    const int interf_base = (RANK > 0) ? 1 : 0;   /* faccia z=0 e' interfaccia? */
    const int interf_top  = (RANK < NRANKS - 1) ? 1 : 0;

    for (int c = 0; c < 3; c++) {
        float* f = (c == 0) ? u.x : ((c == 1) ? u.y : u.z);
        float* src = (c == 0) ? grd.x : ((c == 1) ? grd.y : grd.z);
    /* --- flussi attraverso le facce x --- */
    {
        int cnt = (NXl + 1)*NYl*NZLl;
            parallel_for(cnt, [=] MF_DEV (int idx) {
                int fi = idx % (NXl + 1); int t = idx / (NXl + 1);
                int j = t % NYl; int k = t / NYl;
                if (fi == 0 || fi == NXl) { flux.x[IDX(g, k, j, fi)] = 0.f; return; }
                float un = dt*uf.x[IDX(g, k, j, fi)]/dh;
                float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
                int iu = fi + ((un > 0.f) ? -1 : 0);
                float du = (f[IDX(g, k, j, iu + 1)] - f[IDX(g, k, j, iu - 1)])/(2.f*dh);
                float f2 = f[IDX(g, k, j, iu)]
                    + (src[IDX(g, k, j, fi)] + src[IDX(g, k, j, fi - 1)])*(dt*0.25f)
                    + s*(1.f - s*un)*du*(dh*0.5f);
                float vn = 0.5f*(uf.y[IDX(g, k, j, iu)] + uf.y[IDX(g, k, j + 1, iu)]);
                float fyy = (vn < 0.f)
                    ? (f[IDX(g, k, j + 1, iu)] - f[IDX(g, k, j, iu)])
                    : (f[IDX(g, k, j, iu)] - f[IDX(g, k, j - 1, iu)]);
                f2 -= dt*vn*fyy/(2.f*dh);
                float wn = 0.5f*(uf.z[IDX(g, k, j, iu)] + uf.z[IDX(g, k + 1, j, iu)]);
                float fzz = (wn < 0.f)
                    ? (f[IDX(g, k + 1, j, iu)] - f[IDX(g, k, j, iu)])
                    : (f[IDX(g, k, j, iu)] - f[IDX(g, k - 1, j, iu)]);
                f2 -= dt*wn*fzz/(2.f*dh);
                flux.x[IDX(g, k, j, fi)] = f2*uf.x[IDX(g, k, j, fi)];
            });
    }
    /* --- flussi attraverso le facce y --- */
    {
        int cnt = NXl*(NYl + 1)*NZLl;
            parallel_for(cnt, [=] MF_DEV (int idx) {
                int fj = idx % (NYl + 1); int t = idx / (NYl + 1);
                int i = t % NXl; int k = t / NXl;
                if (fj == 0 || fj == NYl) { flux.y[IDX(g, k, fj, i)] = 0.f; return; }
                float un = dt*uf.y[IDX(g, k, fj, i)]/dh;
                float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
                int ju = fj + ((un > 0.f) ? -1 : 0);
                float du = (f[IDX(g, k, ju + 1, i)] - f[IDX(g, k, ju - 1, i)])/(2.f*dh);
                float f2 = f[IDX(g, k, ju, i)]
                    + (src[IDX(g, k, fj, i)] + src[IDX(g, k, fj - 1, i)])*(dt*0.25f)
                    + s*(1.f - s*un)*du*(dh*0.5f);
                float wn = 0.5f*(uf.z[IDX(g, k, ju, i)] + uf.z[IDX(g, k + 1, ju, i)]);
                float fzz = (wn < 0.f)
                    ? (f[IDX(g, k + 1, ju, i)] - f[IDX(g, k, ju, i)])
                    : (f[IDX(g, k, ju, i)] - f[IDX(g, k - 1, ju, i)]);
                f2 -= dt*wn*fzz/(2.f*dh);
                float unx = 0.5f*(uf.x[IDX(g, k, ju, i)] + uf.x[IDX(g, k, ju, i + 1)]);
                float fxx = (unx < 0.f)
                    ? (f[IDX(g, k, ju, i + 1)] - f[IDX(g, k, ju, i)])
                    : (f[IDX(g, k, ju, i)] - f[IDX(g, k, ju, i - 1)]);
                f2 -= dt*unx*fxx/(2.f*dh);
                flux.y[IDX(g, k, fj, i)] = f2*uf.y[IDX(g, k, fj, i)];
            });
    }
    /* --- flussi attraverso le facce z ---
     * Alle facce di interfaccia il termine tangenziale usa la velocita'
     * centrata nella cella upwind invece della media delle due facce:
     * le facce uf.x/uf.y del rank vicino non sono disponibili, mentre u e'
     * scambiata (e i due rank calcolano cosi' lo stesso flusso). La
     * differenza (media di facce vs valore di cella) e' O(Delta^2). */
    {
        int cnt = NXl*NYl*(NZLl + 1);
            parallel_for(cnt, [=] MF_DEV (int idx) {
                int rem = idx % (NXl*NYl);
                int fk = idx / (NXl*NYl);
                int j = rem / NXl; int i = rem - j*NXl;
                float un = dt*uf.z[IDX(g, fk, j, i)]/dh;
                float s = (un > 0.f) ? 1.f : ((un < 0.f) ? -1.f : 0.f);
                int ju = fk + ((un > 0.f) ? -1 : 0);
                float du = (f[IDX(g, ju + 1, j, i)] - f[IDX(g, ju - 1, j, i)])/(2.f*dh);
                float f2 = f[IDX(g, ju, j, i)]
                    + (src[IDX(g, fk, j, i)] + src[IDX(g, fk - 1, j, i)])*(dt*0.25f)
                    + s*(1.f - s*un)*du*(dh*0.5f);
                int at_interf = ((fk == 0 && interf_base) || (fk == NZLl && interf_top));
                float unx, uny;
                if (at_interf) {
                    unx = u.x[IDX(g, ju, j, i)];
                    uny = u.y[IDX(g, ju, j, i)];
                }
                else {
                    unx = 0.5f*(uf.x[IDX(g, fk, j, i)] + uf.x[IDX(g, fk, j, i + 1)]);
                    uny = 0.5f*(uf.y[IDX(g, fk, j, i)] + uf.y[IDX(g, fk, j + 1, i)]);
                }
                float fxx = (unx < 0.f)
                    ? (f[IDX(g, ju, j, i + 1)] - f[IDX(g, ju, j, i)])
                    : (f[IDX(g, ju, j, i)] - f[IDX(g, ju, j, i - 1)]);
                f2 -= dt*unx*fxx/(2.f*dh);
                float fyy = (uny < 0.f)
                    ? (f[IDX(g, ju, j + 1, i)] - f[IDX(g, ju, j, i)])
                    : (f[IDX(g, ju, j, i)] - f[IDX(g, ju, j - 1, i)]);
                f2 -= dt*uny*fyy/(2.f*dh);
                flux.z[IDX(g, fk, j, i)] = f2*uf.z[IDX(g, fk, j, i)];
            });
    }
    /* --- aggiornamento conservativo (componente c) --- */
    {
        int cnt = NXl*NYl*NZLl;
        parallel_for(cnt, [=] MF_DEV (int idx) {
            int i = idx % NXl; int t = idx / NXl;
            int j = t % NYl; int k = t / NYl;
            float d = (flux.x[IDX(g, k, j, i)] - flux.x[IDX(g, k, j, i + 1)])
                    + (flux.y[IDX(g, k, j, i)] - flux.y[IDX(g, k, j + 1, i)])
                    + (flux.z[IDX(g, k, j, i)] - flux.z[IDX(g, k + 1, j, i)]);
            f[IDX(g, k, j, i)] += dt*d/dh;
        });
    }
    } /* fine ciclo sulle componenti */
}

/* ============================ viscosita' esplicita ============================ */
static void viscosity(Vec3f u, float dt) {
    const Grid g = DOM;
    const int NXl = NX, NYl = NY, NZLl = NZL;
    const float dh2 = DH*DH;
    const float c = (float)(NU*dt);
    int cnt = NXl*NYl*NZLl;
    parallel_for(cnt, [=] MF_DEV (int idx) {
        int i = idx % NXl; int t = idx / NXl;
        int j = t % NYl; int k = t / NYl;
        size_t p = IDX(g, k, j, i);
        float lapx = (u.x[IDX(g, k, j, i + 1)] + u.x[IDX(g, k, j, i - 1)]
                    + u.x[IDX(g, k, j + 1, i)] + u.x[IDX(g, k, j - 1, i)]
                    + u.x[IDX(g, k + 1, j, i)] + u.x[IDX(g, k - 1, j, i)] - 6.f*u.x[p])/dh2;
        float lapy = (u.y[IDX(g, k, j, i + 1)] + u.y[IDX(g, k, j, i - 1)]
                    + u.y[IDX(g, k, j + 1, i)] + u.y[IDX(g, k, j - 1, i)]
                    + u.y[IDX(g, k + 1, j, i)] + u.y[IDX(g, k - 1, j, i)] - 6.f*u.y[p])/dh2;
        float lapz = (u.z[IDX(g, k, j, i + 1)] + u.z[IDX(g, k, j, i - 1)]
                    + u.z[IDX(g, k, j + 1, i)] + u.z[IDX(g, k, j - 1, i)]
                    + u.z[IDX(g, k + 1, j, i)] + u.z[IDX(g, k - 1, j, i)] - 6.f*u.z[p])/dh2;
        u.x[p] += c*lapx;
        u.y[p] += c*lapy;
        u.z[p] += c*lapz;
    });
}

/* ============================ accelerazione / gradiente ============================ */

/* uf = interpolazione alle facce di u (event acceleration di centered.h); le
 * facce normali alle pareti risultano 0 grazie ai ghost free-slip. */
static void acceleration(Vec3f u, Vec3f uf) {
    const Grid g = DOM;
    const int NXl = NX, NYl = NY, NZLl = NZL;
    parallel_for((NXl + 1)*NYl*NZLl, [=] MF_DEV (int idx) {
        int fi = idx % (NXl + 1); int t = idx / (NXl + 1);
        int j = t % NYl; int k = t / NYl;
        uf.x[IDX(g, k, j, fi)] = 0.5f*(u.x[IDX(g, k, j, fi - 1)] + u.x[IDX(g, k, j, fi)]);
    });
    parallel_for(NXl*(NYl + 1)*NZLl, [=] MF_DEV (int idx) {
        int fj = idx % (NYl + 1); int t = idx / (NYl + 1);
        int i = t % NXl; int k = t / NXl;
        uf.y[IDX(g, k, fj, i)] = 0.5f*(u.y[IDX(g, k, fj - 1, i)] + u.y[IDX(g, k, fj, i)]);
    });
    parallel_for(NXl*NYl*(NZLl + 1), [=] MF_DEV (int idx) {
        int rem = idx % (NXl*NYl); int fk = idx / (NXl*NYl);
        int j = rem / NXl; int i = rem - j*NXl;
        uf.z[IDX(g, fk, j, i)] = 0.5f*(u.z[IDX(g, fk - 1, j, i)] + u.z[IDX(g, fk, j, i)]);
    });
}

/* g = -(grad p) centrato (event centered_gradient di centered.h). */
static void centered_gradient(const float* p, Vec3f grd) {
    const Grid g = DOM;
    const int NXl = NX, NYl = NY, NZLl = NZL;
    const float dh = DH;
    parallel_for(NXl*NYl*NZLl, [=] MF_DEV (int idx) {
        int i = idx % NXl; int t = idx / NXl;
        int j = t % NYl; int k = t / NYl;
        size_t q = IDX(g, k, j, i);
        grd.x[q] = -(p[IDX(g, k, j, i + 1)] - p[IDX(g, k, j, i - 1)])/(2.f*dh);
        grd.y[q] = -(p[IDX(g, k, j + 1, i)] - p[IDX(g, k, j - 1, i)])/(2.f*dh);
        grd.z[q] = -(p[IDX(g, k + 1, j, i)] - p[IDX(g, k - 1, j, i)])/(2.f*dh);
    });
}

/* ============================ multigrid distribuito ============================ */

static void mg_init(void) {
    /* ultimo livello distribuito: finche' le dimensioni restano >=2 e lo slab
     * z resta divisibile per il numero di rank con almeno 2 piani locali */
    int ld = 0;
    for (;;) {
        int l = ld + 1;
        int nxl = NX >> l, nyl = NY >> l;
        int nzgl = NZ >> l;
        if (nxl < 2 || nyl < 2 || nzgl < 2) break;
        if (nzgl % NRANKS != 0) break;
        if (nzgl/NRANKS < 2) break;
        ld = l;
    }
    LD = ld;
    NL = LD + 2;
    if (NL > 16) { std::fprintf(stderr, "troppi livelli\n"); std::exit(2); }

    /* livello 0 = griglia piu' fine, puntatori assegnati in mg_solve */
    LV[0].g = DOM;
    LV[0].dh = DH;
    LV[0].dh2 = DH*DH;
    LV[0].sol = LV[0].rhs = LV[0].res = NULL;

    for (int l = 1; l <= LD; l++) {
        int nxl = NX >> l, nyl = NY >> l, nzgl = NZ >> l;
        int nzloc = nzgl/NRANKS;
        Lvl& L = LV[l];
        L.g = mk_grid(nxl, nyl, nzloc, nzgl, RANK*nzloc, 0);
        L.dh = DH*(float)(1 << l);
        L.dh2 = L.dh*L.dh;
        L.sol = mf_alloc(L.g.n); L.rhs = mf_alloc(L.g.n); L.res = mf_alloc(L.g.n);
        mf_zero(L.sol, L.g.n); mf_zero(L.rhs, L.g.n); mf_zero(L.res, L.g.n);
    }
    {
        int l = LD + 1;
        int nxl = NX >> l; if (nxl < 1) nxl = 1;
        int nyl = NY >> l; if (nyl < 1) nyl = 1;
        int nzgl = NZ >> l; if (nzgl < 1) nzgl = 1;
        if (nzgl % NRANKS != 0) {
            if (RANK == 0)
                std::fprintf(stderr, "ERRORE: livello grossolano nz=%d non divisibile per %d rank\n",
                             nzgl, NRANKS);
            MPI_Finalize(); std::exit(2);
        }
        Lvl& L = LV[l];
        L.g = mk_grid(nxl, nyl, nzgl, nzgl, 0, 1);
        L.dh = DH*(float)(1 << l);
        L.dh2 = L.dh*L.dh;
        L.sol = mf_alloc(L.g.n); L.rhs = mf_alloc(L.g.n); L.res = mf_alloc(L.g.n);
        mf_zero(L.sol, L.g.n); mf_zero(L.rhs, L.g.n); mf_zero(L.res, L.g.n);
    }
    if (RANK == 0 && VERBOSE) {
        std::fprintf(stderr, "# multigrid: %d livelli, distribuiti 0..%d, replicato %d (griglia %dx%dx%d)\n",
                     NL, LD, LD + 1, LV[LD + 1].g.nx, LV[LD + 1].g.ny, LV[LD + 1].g.nzg);
    }
}

/* Red-black Gauss-Seidel per l'operatore (sum_vicini - 6 a)/dh^2 = rhs. */
static void smooth_lvl(Lvl& L, int nsweeps) {
    const Grid g = L.g;
    float* a = L.sol;
    const float* b = L.rhs;
    const float dh2 = L.dh2;
    const int nx = g.nx, ny = g.ny, nzl = g.nzloc;
    int cnt = nx*ny*nzl;
    for (int sw = 0; sw < nsweeps; sw++) {
        for (int parity = 0; parity < 2; parity++) {
            bc_pressure(g, a);
            parallel_for(cnt, [=] MF_DEV (int idx) {
                int i = idx % nx; int t = idx / nx;
                int j = t % ny; int k = t / ny;
                if (((i + j + k) & 1) != parity) return;
                size_t q = IDX(g, k, j, i);
                float s = a[IDX(g, k, j, i + 1)] + a[IDX(g, k, j, i - 1)]
                        + a[IDX(g, k, j + 1, i)] + a[IDX(g, k, j - 1, i)]
                        + a[IDX(g, k + 1, j, i)] + a[IDX(g, k - 1, j, i)];
                a[q] = (s - dh2*b[q])*(1.f/6.f);
            });
        }
    }
}

static void residual_lvl(Lvl& L) {
    const Grid g = L.g;
    const float* a = L.sol;
    const float* b = L.rhs;
    float* r = L.res;
    const float dh2 = L.dh2;
    const int nx = g.nx, ny = g.ny, nzl = g.nzloc;
    bc_pressure(g, L.sol);
    parallel_for(nx*ny*nzl, [=] MF_DEV (int idx) {
        int i = idx % nx; int t = idx / nx;
        int j = t % ny; int k = t / ny;
        size_t q = IDX(g, k, j, i);
        float s = a[IDX(g, k, j, i + 1)] + a[IDX(g, k, j, i - 1)]
                + a[IDX(g, k, j + 1, i)] + a[IDX(g, k, j - 1, i)]
                + a[IDX(g, k + 1, j, i)] + a[IDX(g, k - 1, j, i)];
        r[q] = b[q] - (s - 6.f*a[q])/dh2;
    });
}

/* restrizione full-weighting (media 2x2x2) da livello l a l+1 (entrambi
 * distribuiti). Il blocco di celle fini e' tutto locale perche' lo slab z di
 * ogni rank e' allineato ai blocchi 2x2x2. */
static void restrict_dist(Lvl& F, Lvl& C) {
    const Grid gf = F.g, gc = C.g;
    const int nx = gc.nx, ny = gc.ny, nzl = gc.nzloc;
    const float* rf = F.res;
    float* bc_ = C.rhs;
    parallel_for(nx*ny*nzl, [=] MF_DEV (int idx) {
        int i = idx % nx; int t = idx / nx;
        int j = t % ny; int k = t / ny;
        float s = 0.f;
        for (int u = 0; u < 2; u++)
            for (int v = 0; v < 2; v++)
                for (int w = 0; w < 2; w++)
                    s += rf[IDX(gf, 2*k + w, 2*j + v, 2*i + u)];
        bc_[IDX(gc, k, j, i)] = s*0.125f;
    });
}

/* restrizione verso il livello grossolano REPLICATO: ogni rank calcola la
 * propria fetta, poi MPI_Allgather la concatena (ordine z = ordine dei rank). */
static float* ALLG_SEND = NULL;
static float* ALLG_RECV = NULL;
static float* ALLG_DEV = NULL;      /* buffer device per il calcolo della fetta */
static size_t ALLG_CAP = 0;
static size_t ALLG_DEVCAP = 0;

static void restrict_to_repl(Lvl& F, Lvl& C) {
    const Grid gf = F.g, gc = C.g;
    int nxl = gc.nx, nyl = gc.ny;
    int nzg = gc.nzg;                 /* griglia grossolana globale in z */
    int nzl_rank = nzg/NRANKS;        /* fetta locale (piani) */
    size_t nloc = (size_t)nxl*nyl*nzl_rank;
    size_t nglob = (size_t)nxl*nyl*nzg;
    if (nglob > ALLG_CAP) {
        if (ALLG_SEND) mf_free_pinned(ALLG_SEND);
        if (ALLG_RECV) mf_free_pinned(ALLG_RECV);
        ALLG_CAP = nglob;
        ALLG_SEND = mf_alloc_pinned(nloc);
        ALLG_RECV = mf_alloc_pinned(nglob);
    }
    if (nloc > ALLG_DEVCAP) {
        if (ALLG_DEV) mf_free(ALLG_DEV);
        ALLG_DEVCAP = nloc;
        ALLG_DEV = mf_alloc(nloc);
    }
    {
        const float* rf = F.res;
        float* out = ALLG_DEV;
        parallel_for((int)nloc, [=] MF_DEV (int idx) {
            int i = idx % nxl; int t = idx / nxl;
            int j = t % nyl; int k = t / nyl;
            float s = 0.f;
            for (int u = 0; u < 2; u++)
                for (int v = 0; v < 2; v++)
                    for (int w = 0; w < 2; w++)
                        s += rf[IDX(gf, 2*k + w, 2*j + v, 2*i + u)];
            out[idx] = s*0.125f;
        });
    }
    mf_d2h(ALLG_SEND, ALLG_DEV, nloc);
    MPI_Allgather(ALLG_SEND, (int)nloc, MPI_FLOAT,
                  ALLG_RECV, (int)nloc, MPI_FLOAT, MPI_COMM_WORLD);
    /* carica i valori nella griglia replicata (solo celle interne) */
    for (int k = 0; k < nzg; k++)
        mf_h2d_2d(C.rhs + IDX(gc, k, 0, 0), (size_t)gc.nxh*sizeof(float),
                  ALLG_RECV + (size_t)k*nxl*nyl, (size_t)nxl*sizeof(float),
                  (size_t)nxl*sizeof(float), (size_t)nyl);
}

/* prolongamento (interpolazione trilineare cell-centered 0.75/0.25) e somma
 * della correzione grossolana sulla soluzione fine. */
static void prolong_add(Lvl& C, Lvl& F) {
    const Grid gf = F.g, gc = C.g;
    const float* cs = C.sol;
    float* fs = F.sol;
    int nf_x = gf.nx, nf_y = gf.ny, nf_z = gf.nzloc;
    int rfx = gf.nx/gc.nx, rfy = gf.ny/gc.ny;
    int rfz = gf.nzg/gc.nzg;
    bc_pressure(gc, C.sol);
    /* i pesi dipendono solo dalla parita' della coordinata globale */
    parallel_for(nf_x*nf_y*nf_z, [=] MF_DEV (int idx) {
        int i = idx % nf_x; int t = idx / nf_x;
        int j = t % nf_y; int k = t / nf_y;
        float wgt[3][2]; int ind[3][2]; int n[3];
        /* x */
        if (rfx == 2) {
            int p = i >> 1;
            ind[0][0] = p; wgt[0][0] = 0.75f;
            ind[0][1] = p + ((i & 1) ? 1 : -1); wgt[0][1] = 0.25f; n[0] = 2;
        } else { ind[0][0] = i; wgt[0][0] = 1.f; n[0] = 1; }
        /* y */
        if (rfy == 2) {
            int p = j >> 1;
            ind[1][0] = p; wgt[1][0] = 0.75f;
            ind[1][1] = p + ((j & 1) ? 1 : -1); wgt[1][1] = 0.25f; n[1] = 2;
        } else { ind[1][0] = j; wgt[1][0] = 1.f; n[1] = 1; }
        /* z (coordinate globali: la fetta locale parte da gf.k0) */
        {
            int gz = gf.k0 + k;
            int pz = gz / rfz;
            int pzl = pz - gc.k0;
            if (rfz == 2) {
                ind[2][0] = pzl; wgt[2][0] = 0.75f;
                ind[2][1] = pzl + ((gz & 1) ? 1 : -1); wgt[2][1] = 0.25f; n[2] = 2;
            } else { ind[2][0] = pzl; wgt[2][0] = 1.f; n[2] = 1; }
        }
        float acc = 0.f;
        for (int a = 0; a < n[0]; a++)
            for (int b = 0; b < n[1]; b++)
                for (int c = 0; c < n[2]; c++)
                    acc += wgt[0][a]*wgt[1][b]*wgt[2][c]
                         * cs[IDX(gc, ind[2][c], ind[1][b], ind[0][a])];
        fs[IDX(gf, k, j, i)] += acc;
    });
}

static void vcycle(int l) {
    Lvl& L = LV[l];
    if (l == LD + 1) {
        /* ultimo livello: griglia minuscola replicata, risoluzione diretta
         * con molti sweeps di red-black GS (identica su tutti i rank) */
        smooth_lvl(L, MF_COARSE_SWEEPS);
        return;
    }
    smooth_lvl(L, 1);
    residual_lvl(L);
    if (l == LD) {
        restrict_to_repl(L, LV[LD + 1]);
    } else {
        restrict_dist(L, LV[l + 1]);
    }
    mf_zero(LV[l + 1].sol, LV[l + 1].g.n);
    vcycle(l + 1);
    prolong_add(LV[l + 1], L);
    smooth_lvl(L, 1);
}

/* Risolve (sum_vicini - 6 p)/dh^2 = b (Neumann omogeneo) con iterazioni FISSE
 * (nessuna allreduce): sol contiene la stima iniziale (p del passo precedente). */
static void mg_solve(float* sol, float* b, float* res, int nit) {
    LV[0].sol = sol; LV[0].rhs = b; LV[0].res = res;
    for (int it = 0; it < nit; it++) vcycle(0);
}

/* ============================ proiezione ============================ */

static void compute_divergence(float dt) {
    const Grid g = DOM;
    const int nx = NX, ny = NY, nzl = NZL;
    const float dh = DH;
    const Vec3f uf = UF;          /* copia locale: le lambda device non vedono i globals */
    float* b = RHS;
    parallel_for(nx*ny*nzl, [=] MF_DEV (int idx) {
        int i = idx % nx; int t = idx / nx;
        int j = t % ny; int k = t / ny;
        float d = (uf.x[IDX(g, k, j, i + 1)] - uf.x[IDX(g, k, j, i)])
                + (uf.y[IDX(g, k, j + 1, i)] - uf.y[IDX(g, k, j, i)])
                + (uf.z[IDX(g, k + 1, j, i)] - uf.z[IDX(g, k, j, i)]);
        b[IDX(g, k, j, i)] = d/(dt*dh);
    });
}

static void correct_faces(const float* p, float dt) {
    const Grid g = DOM;
    const int nx = NX, ny = NY, nzl = NZL;
    const float dh = DH;
    const Vec3f uf = UF;          /* copia locale per le lambda device */
    parallel_for((nx + 1)*ny*nzl, [=] MF_DEV (int idx) {
        int fi = idx % (nx + 1); int t = idx / (nx + 1);
        int j = t % ny; int k = t / ny;
        uf.x[IDX(g, k, j, fi)] -= dt*(p[IDX(g, k, j, fi)] - p[IDX(g, k, j, fi - 1)])/dh;
    });
    parallel_for(nx*(ny + 1)*nzl, [=] MF_DEV (int idx) {
        int fj = idx % (ny + 1); int t = idx / (ny + 1);
        int i = t % nx; int k = t / nx;
        uf.y[IDX(g, k, fj, i)] -= dt*(p[IDX(g, k, fj, i)] - p[IDX(g, k, fj - 1, i)])/dh;
    });
    parallel_for(nx*ny*(nzl + 1), [=] MF_DEV (int idx) {
        int rem = idx % (nx*ny); int fk = idx / (nx*ny);
        int j = rem / nx; int i = rem - j*nx;
        uf.z[IDX(g, fk, j, i)] -= dt*(p[IDX(g, fk, j, i)] - p[IDX(g, fk - 1, j, i)])/dh;
    });
}

/* Proiezione MAC: risolve la Poisson (sum_vicini-6p)/dh^2 = div(uf)/dt con
 * Neumann omogeneo e corregge le facce. sol e' la stima iniziale (p del passo
 * precedente), persistente tra i passi. */
static void project(float* sol, float dt) {
    compute_divergence(dt);
    mg_solve(sol, RHS, RES, POISSON_NIT);
    bc_pressure(DOM, sol);       /* ghost di p allineati tra i rank */
    correct_faces(sol, dt);
}

/* u += dt*g (correzione con gradiente di pressione centrato) */
static void add_gradient(Vec3f u, Vec3f grd, float dt) {
    const Grid g = DOM;
    const int nx = NX, ny = NY, nzl = NZL;
    parallel_for(nx*ny*nzl, [=] MF_DEV (int idx) {
        int i = idx % nx; int t = idx / nx;
        int j = t % ny; int k = t / ny;
        size_t p = IDX(g, k, j, i);
        u.x[p] += dt*grd.x[p];
        u.y[p] += dt*grd.y[p];
        u.z[p] += dt*grd.z[p];
    });
}

/* ============================ passo temporale ============================ */

static double max_abs_field(const float* a) { return (double)mf_max_abs(a, DOM.n); }

static double compute_dt(double dt_prev, double t, double tnext) {
    double umax = max_abs_field(UF.x);
    double m = max_abs_field(UF.y); if (m > umax) umax = m;
    m = max_abs_field(UF.z); if (m > umax) umax = m;
    /* il passo deve essere IDENTICO su tutti i rank: massimo globale (1
     * allreduce per passo, trascurabile rispetto agli scambi degli aloni) */
    double umax_g = umax;
    MPI_Allreduce(&umax, &umax_g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    umax = umax_g;
    double dt;
    if (umax > 1e-12) dt = CFL*DH/umax;
    else dt = TMAX;
    if (dt > 1.1*dt_prev) dt = 1.1*dt_prev;   /* crescita max 10%/passo */
    if (t + dt > tnext) dt = tnext - t;
    if (dt < 1e-12) dt = 1e-12;
    return dt;
}

/* ============================ condizioni iniziali ============================ */

static void init_ic(Vec3f& u) {
    const float s2 = (float)(SIGMA*SIGMA);
    const float slope = (float)(TILT*DD/(0.5*LZ));
    const size_t face = (size_t)NX*NY;
    float* hp = (float*)std::malloc(3*face*sizeof(float));
    if (!hp) { std::fprintf(stderr, "malloc IC fallita\n"); std::exit(2); }
    for (int k = 0; k < NZL; k++) {
        int kg = K0 + k;
        double z = (kg + 0.5)*DH;
        for (int j = 0; j < NY; j++) {
            double y = -0.5 + (j + 0.5)*DH;
            for (int i = 0; i < NX; i++) {
                double x = -0.5 + (i + 0.5)*DH;
                double ux = 0., uy = 0.;
                for (int s = -1; s <= 1; s += 2) {
                    double xc = s*DD*(1. + PERT*cos(2.*PI_D*z/LZ))
                              - s*slope*(z - 0.5*LZ);
                    double dx = x - xc;
                    double r = sqrt(dx*dx + y*y);
                    if (r > 0.) {
                        double ut = 0.5*(double)s2*(1. - exp(-r*r/(double)s2))/r;
                        ux += -ut*y/r;
                        uy +=  ut*dx/r;
                    }
                }
                size_t q = (size_t)j*NX + i;
                hp[q] = (float)ux;
                hp[face + q] = (float)uy;
                hp[2*face + q] = 0.f;
            }
        }
        for (int c = 0; c < 3; c++) {
            float* dst = (c == 0) ? u.x : ((c == 1) ? u.y : u.z);
            mf_h2d_2d(dst + IDX(DOM, k, 0, 0), (size_t)DOM.nxh*sizeof(float),
                      hp + (size_t)c*face, (size_t)NX*sizeof(float),
                      (size_t)NX*sizeof(float), (size_t)NY);
        }
    }
    std::free(hp);
}

/* ============================ diagnostica ============================ */

static void allreduce_sum(double* v, int n) {
    MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}
static void allreduce_max(double* v, int n) {
    MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

#include <sys/stat.h>

/* scarica u sul host (con ghost aggiornati) */
static void download_u(void) {
    mf_d2h(HU0, U.x, DOM.n);
    mf_d2h(HU1, U.y, DOM.n);
    mf_d2h(HU2, U.z, DOM.n);
}

/* calcola omega_z locale (interne) e restituisce norme/quantita' sulla fetta z=ZNODE */
struct Diag {
    double Z, E, Umax, Wmax, G, sep, px, py;
};

static double* WZ_SLICE = NULL;   /* omega_z sulla fetta nodo (nx*ny), allocata una volta */

static Diag diagnostics_compute(void) {
    const Grid& g = DOM;
    const int nx = NX, ny = NY, nzl = NZL;
    const double dh3 = (double)DH*DH*DH;
    const double dh2 = (double)DH*DH;
    const double inv2dh = 1.0/(2.0*DH);
    long long n = (long long)nx*ny*nzl;
    const int k_node_g = 2*NX - 1;                  /* fetta z = 2 - dh/2 */
    Diag d; d.Z = d.E = d.Umax = d.Wmax = d.G = d.sep = d.px = d.py = 0.;

    double sum[3] = {0., 0., 0.};       /* Z, E, G */
    double mx[3] = {0., 0., -1e300};    /* Umax, Wmax, m1 */

    for (long long q = 0; q < n; q++) {
        int i = (int)(q % nx); long long t = q/nx;
        int j = (int)(t % ny); int k = (int)(t/ny);
        size_t p = IDX(g, k, j, i);
        /* differenze in float (come le macro WX/WY/WZ di saetta3d.c), poi double */
        float dxz = HU2[IDX(g, k, j + 1, i)] - HU2[IDX(g, k, j - 1, i)];
        float dyz = HU1[IDX(g, k + 1, j, i)] - HU1[IDX(g, k - 1, j, i)];
        float dzy = HU1[IDX(g, k, j, i + 1)] - HU1[IDX(g, k, j, i - 1)];
        float dxy = HU0[IDX(g, k, j + 1, i)] - HU0[IDX(g, k, j - 1, i)];
        float dxz2 = HU0[IDX(g, k + 1, j, i)] - HU0[IDX(g, k - 1, j, i)];
        float dyx = HU2[IDX(g, k, j, i + 1)] - HU2[IDX(g, k, j, i - 1)];
        double wx = ((double)dxz - (double)dyz)*inv2dh;
        double wy = ((double)dxz2 - (double)dyx)*inv2dh;
        double wz = ((double)dzy - (double)dxy)*inv2dh;
        double ux = HU0[p], uy = HU1[p], uz = HU2[p];
        sum[0] += (wx*wx + wy*wy + wz*wz)*dh3;
        sum[1] += 0.5*(ux*ux + uy*uy + uz*uz)*dh3;
        double um = sqrt(ux*ux + uy*uy + uz*uz);
        if (um > mx[0]) mx[0] = um;
        double wm = sqrt(wx*wx + wy*wy + wz*wz);
        if (wm > mx[1]) mx[1] = wm;
        if (K0 + k == k_node_g) {
            sum[2] += wz*dh2;
            if (wz > mx[2]) mx[2] = wz;
            WZ_SLICE[(size_t)j*nx + i] = wz;
        }
    }
    allreduce_sum(sum, 3);
    allreduce_max(mx, 3);
    d.Z = sum[0]; d.E = sum[1]; d.G = sum[2];
    d.Umax = mx[0]; d.Wmax = mx[1];
    double m1 = mx[2];

    /* secondo nucleo: posizione del massimo di wz sulla fetta nodo */
    double px = 0., py = 0., w1 = 0.;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double wz = WZ_SLICE[(size_t)j*nx + i];
                if (wz == m1) {
                    w1 += 1.; px += -0.5 + (i + 0.5)*(double)DH; py += -0.5 + (j + 0.5)*(double)DH;
                }
            }
    }
    double red[3] = {px, py, w1};
    allreduce_sum(red, 3);
    px = red[0]; py = red[1]; w1 = red[2];
    if (w1 > 0.) { px /= w1; py /= w1; }

    /* secondo nucleo: massimo nella meta' opposta del piano */
    double m2 = 0.;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double x = -0.5 + (i + 0.5)*(double)DH, y = -0.5 + (j + 0.5)*(double)DH;
                if (x*px + y*py < 0.) {
                    double wz = WZ_SLICE[(size_t)j*nx + i];
                    if (wz > m2) m2 = wz;
                }
            }
    }
    allreduce_max(&m2, 1);

    double qx = 0., qy = 0., w2 = 0.;
    if (K0 <= k_node_g && k_node_g < K1) {
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                double x = -0.5 + (i + 0.5)*(double)DH, y = -0.5 + (j + 0.5)*(double)DH;
                if (x*px + y*py < 0. && WZ_SLICE[(size_t)j*nx + i] == m2) {
                    w2 += 1.; qx += x; qy += y;
                }
            }
    }
    double red2[3] = {qx, qy, w2};
    allreduce_sum(red2, 3);
    qx = red2[0]; qy = red2[1]; w2 = red2[2];
    if (w2 > 0.) { qx /= w2; qy /= w2; }
    d.sep = (w1 > 0. && w2 > 0.) ? sqrt((px - qx)*(px - qx) + (py - qy)*(py - qy)) : 0.;
    d.px = px; d.py = py;
    return d;
}

/* ============================ dump del campo ============================ */

/* Scrive omega_z con lo stesso formato di saetta3d.c:
 *   int32 nx,ny,nz ; float32 con i piu' veloce, poi j, poi k.
 * Ogni rank calcola la propria fetta, MPI_Gather la concatena in ordine z. */
static void dump_wz(const char* path) {
    const Grid& g = DOM;
    const int nx = NX, ny = NY, nzl = NZL;
    const double inv2dh = 1.0/(2.0*DH);
    size_t nloc = (size_t)nx*ny*nzl;
    float* loc = (float*)std::malloc(nloc*sizeof(float));
    if (!loc) { std::fprintf(stderr, "malloc dump fallita\n"); std::exit(2); }
    for (int k = 0; k < nzl; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                size_t p = IDX(g, k, j, i);
                float dzy = HU1[IDX(g, k, j, i + 1)] - HU1[IDX(g, k, j, i - 1)];
                float dxy = HU0[IDX(g, k, j + 1, i)] - HU0[IDX(g, k, j - 1, i)];
                loc[((size_t)k*ny + j)*nx + i] = (float)(((double)dzy - (double)dxy)*inv2dh);
            }
    float* all = NULL;
    if (RANK == 0) {
        all = (float*)std::malloc(nloc*(size_t)NRANKS*sizeof(float));
        if (!all) { std::fprintf(stderr, "malloc gather fallita\n"); std::exit(2); }
    }
    MPI_Gather(loc, (int)nloc, MPI_FLOAT, all, (int)nloc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (RANK == 0) {
        FILE* fp = std::fopen(path, "wb");
        if (!fp) { std::fprintf(stderr, "non riesco a scrivere %s\n", path); }
        else {
            int dims[3] = {nx, ny, nzl*(int)NRANKS};
            std::fwrite(dims, sizeof(int), 3, fp);
            std::fwrite(all, sizeof(float), nloc*(size_t)NRANKS, fp);
            std::fclose(fp);
            std::fprintf(stderr, "# dump: scritto %s (%d x %d x %d)\n", path, nx, ny, nzl*(int)NRANKS);
        }
        std::free(all);
    }
    std::free(loc);
}

/* ============================ output diagnostica ============================ */

static FILE* DIAG_FP = NULL;
static FILE* TIME_FP = NULL;
static char DIAG_PATH[1200], TIME_PATH[1200];

static void open_outputs(void) {
    if (RANK != 0) return;
    std::snprintf(DIAG_PATH, sizeof(DIAG_PATH), "%s/diagnostics.dat", OUTDIR);
    std::snprintf(TIME_PATH, sizeof(TIME_PATH), "%s/timing.dat", OUTDIR);
    DIAG_FP = std::fopen(DIAG_PATH, "w");
    if (DIAG_FP) {
        std::fprintf(DIAG_FP, "# t dt it enstrophy energy umax wmax "
                     "circulation_slice separation xc yc cells mgp mgp_resa\n");
        std::fflush(DIAG_FP);
    }
    TIME_FP = std::fopen(TIME_PATH, "w");
    if (TIME_FP) {
        std::fprintf(TIME_FP, "# step t dt wall_s comm_s comm_MB n_exch\n");
        std::fflush(TIME_FP);
    }
}

static void write_diagnostics(double t, double dt, long step, int nit, double resa) {
    bc_u(U, 1);
    download_u();
    Diag d = diagnostics_compute();
    if (RANK == 0) {
        if (DIAG_FP) {
            std::fprintf(DIAG_FP, "%.6g %.6e %ld %.8e %.8e %.8e %.8e %.8e %.6e %.6e %.6e %ld %d %.3e\n",
                         t, dt, step, d.Z, d.E, d.Umax, d.Wmax, d.G, d.sep, d.px, d.py,
                         (long)NX*NY*NZ, nit, resa);
            std::fflush(DIAG_FP);
        }
        std::fprintf(stderr, "%.6g %.6e %.6e %.6e %.6e %.6e %.6e\n",
                     t, d.Z, d.E, d.Umax, d.Wmax, d.G, d.sep);
        std::fflush(stderr);
    }
}

/* ============================ parametri e main ============================ */

static void usage(const char* prog) {
    std::fprintf(stderr,
        "uso: %s [opzioni]\n"
        "  -tmax T        tempo finale (default 10)\n"
        "  -cfl C         CFL (default 0.3)\n"
        "  -nu NU         viscosita' (default 1e-5)\n"
        "  -poisson_nit N V-cycle per proiezione (default 2)\n"
        "  -out DIR       cartella di output (default \"out\")\n"
        "  -dump FILE     file del dump finale (default DIR/omega3d_matrix.bin)\n"
        "  -diag T        passo della diagnostica (default 1)\n"
        "  -verbose       stampa configurazione\n", prog);
}

static void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-tmax") && i + 1 < argc) TMAX = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-cfl") && i + 1 < argc) CFL = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-nu") && i + 1 < argc) NU = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-poisson_nit") && i + 1 < argc) POISSON_NIT = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-out") && i + 1 < argc) std::snprintf(OUTDIR, sizeof(OUTDIR), "%s", argv[++i]);
        else if (!std::strcmp(argv[i], "-dump") && i + 1 < argc) std::snprintf(DUMPFILE, sizeof(DUMPFILE), "%s", argv[++i]);
        else if (!std::strcmp(argv[i], "-diag") && i + 1 < argc) TDIAG = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "-verbose")) VERBOSE = 1;
        else if (!std::strcmp(argv[i], "-h") || !std::strcmp(argv[i], "--help")) {
            if (RANK == 0) usage(argv[0]);
            MPI_Finalize(); std::exit(0);
        }
        else {
            if (RANK == 0) { std::fprintf(stderr, "opzione sconosciuta: %s\n", argv[i]); usage(argv[0]); }
            MPI_Finalize(); std::exit(2);
        }
    }
}

static void *xmalloc(size_t n) {
    void* p = std::malloc(n);
    if (!p) { std::fprintf(stderr, "malloc %zu fallita\n", n); std::exit(2); }
    return p;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &NRANKS);

    NX = MF_NX; NY = NX; NZ = 4*NX;
    DH = 1.0f/(float)NX;
    parse_args(argc, argv);

    if (NZ % NRANKS != 0) {
        if (RANK == 0)
            std::fprintf(stderr, "ERRORE: nz=%d non divisibile per %d rank (usare 1,2,4 rank)\n",
                         NZ, NRANKS);
        MPI_Finalize();
        return 2;
    }
    K0 = RANK*(NZ/NRANKS);
    K1 = K0 + NZ/NRANKS;
    NZL = K1 - K0;
    if (NZL < 2) {
        if (RANK == 0) std::fprintf(stderr, "ERRORE: slab troppo sottile (%d piani)\n", NZL);
        MPI_Finalize();
        return 2;
    }
#if MF_USE_CUDA
    mf_cuda_ok(cudaSetDevice(0), "cudaSetDevice");
#endif
    DOM = mk_grid(NX, NY, NZL, NZ, K0, 0);

    if (RANK == 0 && std::strlen(DUMPFILE) == 0)
        std::snprintf(DUMPFILE, sizeof(DUMPFILE), "%s/omega3d_matrix.bin", OUTDIR);
    if (RANK == 0) mkdir(OUTDIR, 0777);
    MPI_Barrier(MPI_COMM_WORLD);
    if (RANK == 0) open_outputs();

    /* allocazione campi */
    U.x = mf_alloc(DOM.n); U.y = mf_alloc(DOM.n); U.z = mf_alloc(DOM.n);
    UF.x = mf_alloc(DOM.n); UF.y = mf_alloc(DOM.n); UF.z = mf_alloc(DOM.n);
    FLUX.x = mf_alloc(DOM.n); FLUX.y = mf_alloc(DOM.n); FLUX.z = mf_alloc(DOM.n);
    GRD.x = mf_alloc(DOM.n); GRD.y = mf_alloc(DOM.n); GRD.z = mf_alloc(DOM.n);
    P = mf_alloc(DOM.n); PF = mf_alloc(DOM.n); RHS = mf_alloc(DOM.n); RES = mf_alloc(DOM.n);
    {
        float* a[3] = {U.x, U.y, U.z};
        float* b[3] = {UF.x, UF.y, UF.z};
        float* d[3] = {FLUX.x, FLUX.y, FLUX.z};
        float* e[3] = {GRD.x, GRD.y, GRD.z};
        for (int c = 0; c < 3; c++) {
            mf_zero(a[c], DOM.n); mf_zero(b[c], DOM.n);
            mf_zero(d[c], DOM.n); mf_zero(e[c], DOM.n);
        }
    }
    mf_zero(P, DOM.n); mf_zero(PF, DOM.n); mf_zero(RHS, DOM.n); mf_zero(RES, DOM.n);

    mg_init();

    HU0 = (float*)xmalloc(DOM.n*sizeof(float));
    HU1 = (float*)xmalloc(DOM.n*sizeof(float));
    HU2 = (float*)xmalloc(DOM.n*sizeof(float));
    WZ_SLICE = (double*)xmalloc((size_t)NX*NY*sizeof(double));
    ensure_exchange_capacity(12*(size_t)NX*NY);
    if (RANK == 0) {
        std::fprintf(stderr, "# mpi_fissa: %d rank, griglia %d x %d x %d (%.1f M celle), Delta=%.6e\n",
                     NRANKS, NX, NY, NZ, (double)NX*NY*NZ/1e6, (double)DH);
        std::fprintf(stderr, "# slab: z=[%d,%d) (%d piani), livello distribuito max LD=%d\n",
                     K0, K1, NZL, LD);
        double mem = (double)DOM.n*sizeof(float)*(3 + 3 + 3 + 3 + 4)/1e6;
        std::fprintf(stderr, "# memoria campi per rank: ~%.0f MB (+multigrid)\n", mem);
        if (VERBOSE)
            std::fprintf(stderr, "# TMAX=%g CFL=%g nu=%g poisson_nit=%d TDIAG=%g out=%s\n",
                         TMAX, CFL, NU, POISSON_NIT, TDIAG, OUTDIR);
    }

    /* condizioni iniziali */
    init_ic(U);
    bc_u(U, 2);
    acceleration(U, UF);

    double t = 0.;
    long step = 0;
    double dt_prev = 1e30;
    double tnext_diag = TDIAG;
    double dt0 = compute_dt(dt_prev, t, tnext_diag);
    write_diagnostics(t, dt0, step, 0, 0.0);

    double t_start = now_s();
    while (t < TMAX - 1e-12) {
        double tnext = (tnext_diag < TMAX) ? tnext_diag : TMAX;
        if (tnext <= t + 1e-12) tnext = t + TDIAG;
        double comm0 = T_COMM;
        double w0 = now_s();

        bc_u(U, 2);
        bc_u(GRD, 1);
        double dt = compute_dt(dt_prev, t, tnext);
        float dtf = (float)dt;

        predict_uf(U, UF, GRD, dtf);
        project(PF, (float)(0.5*dt));
        advect(U, UF, GRD, FLUX, dtf);
        bc_u(U, 1);
        viscosity(U, dtf);
        bc_u(U, 1);
        acceleration(U, UF);
        project(P, dtf);
        centered_gradient(P, GRD);
        add_gradient(U, GRD, dtf);

        t += dt;
        step++;
        dt_prev = dt;
        double w1 = now_s();
        if (RANK == 0 && TIME_FP) {
            std::fprintf(TIME_FP, "%ld %.6g %.6e %.4f %.4f %.1f %ld\n",
                         step, t, dt, w1 - w0, T_COMM - comm0, B_COMM/1e6, N_EXCH);
            if (step % 100 == 0) std::fflush(TIME_FP);
        }
        if (VERBOSE && RANK == 0 && step % 50 == 0)
            std::fprintf(stderr, "# step %ld t=%.4f dt=%.3e wall=%.2f s comm=%.1f%%\n",
                         step, t, dt, w1 - w0,
                         100.0*(T_COMM - comm0)/((w1 - w0) > 0 ? (w1 - w0) : 1.));

        if (t >= tnext_diag - 1e-9 || t >= TMAX - 1e-12) {
            residual_lvl(LV[0]);
            double resa = (double)mf_max_abs(RES, DOM.n);
            write_diagnostics(t, dt, step, POISSON_NIT, resa);
            if (RANK == 0 && TIME_FP) std::fflush(TIME_FP);
            tnext_diag += TDIAG;
        }
    }
    double t_end = now_s();

    /* dump finale del campo omega_z */
    bc_u(U, 1);
    download_u();
    dump_wz(DUMPFILE);
    if (RANK == 0) {
        std::fprintf(stderr,
                     "#\n# riepilogo (mpi_fissa)\n"
                     "# rank = %d, griglia = %d x %d x %d celle (%.2f M)\n"
                     "# passi = %ld, t = %.6g, wall = %.1f s (%.3f s/passo)\n"
                     "# comunicazione: %.2f s (%.1f%%), %.1f MB totali, %ld scambi\n",
                     NRANKS, NX, NY, NZ, (double)NX*NY*NZ/1e6,
                     step, t, t_end - t_start, (t_end - t_start)/(step > 0 ? step : 1),
                     T_COMM, 100.0*T_COMM/((t_end - t_start) > 0 ? (t_end - t_start) : 1.),
                     B_COMM/1e6, N_EXCH);
        if (DIAG_FP) std::fclose(DIAG_FP);
        if (TIME_FP) std::fclose(TIME_FP);
        char dpath[1300];
        std::snprintf(dpath, sizeof(dpath), "%s/done_0.txt", OUTDIR);
        FILE* fp = std::fopen(dpath, "w");
        if (fp) { std::fprintf(fp, "OK %ld %.6g\n", step, t); std::fclose(fp); }
    }

    /* rilascio (best-effort: il processo termina subito dopo) */
    mf_free(U.x); mf_free(U.y); mf_free(U.z);
    mf_free(UF.x); mf_free(UF.y); mf_free(UF.z);
    mf_free(FLUX.x); mf_free(FLUX.y); mf_free(FLUX.z);
    mf_free(GRD.x); mf_free(GRD.y); mf_free(GRD.z);
    mf_free(P); mf_free(PF); mf_free(RHS); mf_free(RES);
    for (int l = 1; l < NL; l++) {
        mf_free(LV[l].sol); mf_free(LV[l].rhs); mf_free(LV[l].res);
    }
    std::free(HU0); std::free(HU1); std::free(HU2); std::free(WZ_SLICE);
    if (ALLG_DEV) mf_free(ALLG_DEV);
    if (EXSEND) mf_free_pinned(EXSEND);
    if (EXRECV) mf_free_pinned(EXRECV);
    MPI_Finalize();
    return 0;
}
