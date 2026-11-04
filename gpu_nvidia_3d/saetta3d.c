/**
  Vortex coalescing 3D on GPU (CUDA backend, single precision).
  ==============================================================
  Two co-rotating vortex tubes with an axial "varicose"
  perturbation of the separation and INCLINED axes (X configuration in
  the x-z plane, not parallel):

      d(z)   = DD * (1 + PERT*cos(2 pi z / LZ))   varicose separation
      X      = TILT * DD * (z - LZ/2)/(LZ/2)      X shift of the centers

  The minimum separation is at z = LZ/2 (where the tubes merge first);
  at the two ends the tubes are farther apart (X opening ~ 2*atan(TILT*DD/(LZ/2))).
  The 2D case (saetta.c, psi-omega) does not generalize to 3D: here we use
  the incompressible velocity-pressure formulation (centered.h).

  Fixed 3D grid with cubic cells of side Delta = 1/(NDIV0*2^MAXLEVEL):
  box 1 x 1 x LZ (x,y in [-0.5,0.5], z in [0,LZ]).
  Divisions double at each MAXLEVEL: in x,y they are NDIV0*2^MAXLEVEL,
  in z NDIV0*2^MAXLEVEL*(LZ+0.5) (cubic cells: the ratio between the
  z and x,y divisions is fixed by the domain, i.e. by LZ).
  NOTE: z is NOT periodic (free-slip walls as x,y). A straight tilted
  tube cannot be periodic in z: at the boundary its center would jump
  by TILT*DD*LZ/(LZ/2), creating a discontinuity.

  VRAM (measured on RTX 3070 8 GB, NDIV0=1, MAXLEVEL=8, MOVIE=1):
    ~7.2 GB peak with the movie outputs; the movies write WZ
    reusing p (see note below: no dedicated field, -365 MB).
    The rest is solver structure (u, uf, g, p, pf + metrics +
    multigrid hierarchy of the Poisson solve + CUDA context): the only other
    way down is to reduce the cells (MAXLEVEL/NDIV0/LZ).
*/
#include "navier-stokes/centered.h"

// Minimal zlib interface: we do NOT include <zlib.h> because qcc cannot
// parse the system headers it pulls in (unistd.h...).
// Link with -lz (see build.sh).
typedef unsigned char Bytef;
typedef unsigned long uLongf;
extern int compress2 (Bytef * dest, uLongf * destLen,
    const Bytef * source, unsigned long sourceLen, int level);
#define Z_OK 0

// POSIX mkdir(): declared by hand to avoid including <sys/stat.h>.
extern int mkdir (const char * path, unsigned int mode);

// CFL (centered.h defaults to 0.8): same as in the 2D solver.
#ifndef CFL_USER
# define CFL_USER 0.3
#endif

event defaults_cfl (i = 0)
    CFL = CFL_USER;

// Physical parameters (same as the 2D case)
#ifndef DD
# define DD       0.14    // half-separation of the nuclei (2D: d = 0.28)
#endif
#ifndef SIGMA
# define SIGMA    0.09    // width of the nuclei
#endif
#ifndef OMEGA0
# define OMEGA0   1.      // amplitude (omega_z at the center)
#endif
#ifndef NU
# define NU       1e-5    // viscosity
#endif
#ifndef PERT
# define PERT     0.10    // axial perturbation (fraction of DD)
#endif
#ifndef TILT
# define TILT     2.      // X tilt (TILT=2 -> opening ~8 degrees)
#endif

// Geometry and time
#ifndef LZ
# define LZ       4.      // domain length along z (not periodic, see the note above)
#endif
#ifndef MAXLEVEL
# define MAXLEVEL 8       // in x,y: NDIV0*2^MAXLEVEL cells (with LZ along z)
#endif
#ifndef NDIV0
# define NDIV0    1       // base divisions in x,y at MAXLEVEL=0:
                          // they double at each MAXLEVEL (NDIV0*2^MAXLEVEL)
#endif
#ifndef TMAX
# define TMAX     300.
#endif
#ifndef MOVIE
# define MOVIE    1
#endif
#ifndef VTK
# define VTK      0       // 3D .vti dump for ParaView/pyvista (disabled:
                          // MOVIE makes the movies through output_ppm; to
                          // re-enable: -DVTK=1, disk ~360 MB per dump)
#endif
#ifndef VTK_TSTEP
# define VTK_TSTEP 5.     // how often to save the 3D field
#endif
#ifndef VTK_TSTART
# define VTK_TSTART 0.    // first t from which to write/load the .vti
#endif
#ifndef VTK_DIR
# define VTK_DIR  "vtk3d" // folder for the .vti dumps (created automatically)
#endif
#ifndef VTK_UPLOAD
# define VTK_UPLOAD 0     // 1 = upload every .vti to Google Drive as soon as it is written
#endif
#ifndef VTK_UPLOAD_CMD
// --delete: after the upload the local .vti is removed (the files stay on
// Drive); --folder-name separates the different animations on Drive. "--file"
// must stay at the end: the C code appends the path to it at once.
# define VTK_UPLOAD_CMD "python3 carica_vti_gdrive.py --non-interactive --delete --folder-name saetta3d_anim_T1500 --file"
#endif
#ifndef VTK_UPLOAD_LOG
# define VTK_UPLOAD_LOG "carica_vti.log"
#endif
#ifndef VTK_UPLOAD_ABORT
# define VTK_UPLOAD_ABORT 1  // 1 = stop the simulation if the upload fails
#endif
#ifndef RESA_TMIN
# define RESA_TMIN 1.
#endif

#define ZNODE (0.5*LZ)    // slice with the minimum d(z): early merging

// Constant viscosity (face vector)
const face vector muc[] = { NU, NU, NU };

/* ===================== checkpoint / restart =====================
   Scrive ogni CHK_TSTEP (e a t=TMAX) un file binario chk_<t>.bin con
   velocità cell-centered (media delle facce) e pressione.  Con
   -DCHK_START=<t> la run riparte da quel file: la velocità di faccia
   viene ricostruita come media dei due lati e la prima proiezione
   sistema la divergenza.  Non è bit-exact (ricostruzione O(Delta^2)),
   ma permette di continuare l'animazione. */
#ifndef CHECKPOINT
# define CHECKPOINT 0
#endif
#ifndef CHK_TSTEP
# define CHK_TSTEP 1e30
#endif
#ifndef CHK_START
# define CHK_START -1.
#endif

static void open_diagnostics(void) {
    if (pid() == 0) {
        FILE * fp = fopen("diagnostics.dat", "w");
        fprintf(fp, "# t dt it enstrophy energy umax wmax circulation_slice separation xc yc cells mgp mgp_resa\n");
        fclose(fp);
        fprintf(stderr, "# t enstrophy energy umax wmax circulation_slice separation\n");
    }
}

#if CHECKPOINT
scalar CHKX[], CHKY[], CHKZ[];

static void chk_set_time(double tt) { extern double t; t = tt; }

static void chkname(double t, char *buf) {
    snprintf(buf, 256, "chk_%.3f.bin", t);
}

static void write_checkpoint(double t) {
    char name[256]; chkname(t, name);
    FILE *fp = fopen(name, "wb");
    if (!fp) { fprintf(ferr, "# chk: fopen %s fallita\n", name); return; }
    int dims[3] = {NDIV0 << MAXLEVEL, NDIV0 << MAXLEVEL,
                   (NDIV0 << MAXLEVEL)*(int)(LZ + 0.5)};
    fwrite(dims, sizeof(int), 3, fp);
    fwrite(&t, sizeof(double), 1, fp);
    boundary((scalar *){u});
    enum { CH = 65536 };
    float *b = malloc(CH*4*sizeof(float));
    size_t m = 0;
    foreach (serial) {
        b[4*m+0] = u.x[];
        b[4*m+1] = u.y[];
        b[4*m+2] = u.z[];
        b[4*m+3] = p[];
        if (++m == CH) { fwrite(b, sizeof(float), 4*m, fp); m = 0; }
    }
    if (m) fwrite(b, sizeof(float), 4*m, fp);
    free(b); fclose(fp);
    fprintf(ferr, "# chk: scritto %s (t = %.3f)\n", name, t);
}

static int read_checkpoint(double tload) {
    char name[256]; chkname(tload, name);
    FILE *fp = fopen(name, "rb");
    if (!fp) { fprintf(ferr, "# chk: manca %s\n", name); return 0; }
    int dims[3]; double t = 0.;
    if (fread(dims, sizeof(int), 3, fp) != 3 ||
        fread(&t, sizeof(double), 1, fp) != 1) { fclose(fp); return 0; }
    long nx = NDIV0 << MAXLEVEL, ny = nx, nz = nx*(int)(LZ + 0.5);
    if (dims[0] != nx || dims[1] != ny || dims[2] != nz) {
        fprintf(ferr, "# chk: dimensioni %d %d %d diverse da %ld %ld %ld\n",
                dims[0], dims[1], dims[2], nx, ny, nz);
        fclose(fp); return 0;
    }
    size_t n = (size_t)nx*ny*nz;
    float *b = malloc(4*n*sizeof(float));
    if (!b || fread(b, sizeof(float), 4*n, fp) != 4*n) { free(b); fclose(fp); return 0; }
    fclose(fp);
    size_t m = 0;
    foreach (serial) {
        CHKX[] = b[4*m+0]; CHKY[] = b[4*m+1]; CHKZ[] = b[4*m+2]; p[] = b[4*m+3];
        m++;
    }
    free(b);
    foreach (serial) {
        u.x[] = CHKX[]; u.y[] = CHKY[]; u.z[] = CHKZ[];
    }
    boundary((scalar *){u});
    boundary({p});
    fprintf(ferr, "# chk: caricato %s (t = %.3f)\n", name, t);
    return 1;
}
#endif // CHECKPOINT

// Centered components of the vorticity (from the differences of u)
#define WX ((u.z[0,1,0] - u.z[0,-1,0]) - (u.y[0,0,1] - u.y[0,0,-1]))/(2.*Delta)
#define WY ((u.x[0,0,1] - u.x[0,0,-1]) - (u.z[1,0,0] - u.z[-1,0,0]))/(2.*Delta)
#define WZ ((u.y[1,0,0] - u.y[-1,0,0]) - (u.x[0,1,0] - u.x[0,-1,0]))/(2.*Delta)

// VRAM note: no dedicated field is used for the outputs (2D slices of
// omega_z): the movies reuse p (the pressure). Between two steps p only
// serves as the initial guess for the next Poisson solve, so overwriting it
// with WZ for the output does not change the solution (one field less = ~300 MB at L8).

// maximum residual of the pressure Poisson solve (at steady state)
double max_resa = 0., max_resa_t = 0.;

int main()
{
    origin(-0.5, -0.5, 0.);
    Dimensions.z = (int)(LZ + 0.5);
    init_grid(NDIV0 << MAXLEVEL);
    // z is not periodic: free-slip walls (default symmetry) as x,y.

    mu = muc;

    // With -DPOISSON_ITER=n the Poisson solver does exactly n iterations.
#ifdef POISSON_ITER
    NITERMIN = NITERMAX = POISSON_ITER;
    TOLERANCE = 1e30;
#endif

    run();
}

/**
  Initial conditions: superposition of the azimuthal velocity fields
  of two tilted Gaussian tubes (the field is divergence free):

      omega_z(r) = OMEGA0 exp(-r^2/sigma^2)
      u_theta(r) = (sigma^2/(2r))(1 - exp(-r^2/sigma^2))

  with the tube centers following X: xc_s(z) = s d(z) - s X(z).
*/

event init (t = 0)
{
#if CHECKPOINT
    if (CHK_START >= 0.) {
        if (read_checkpoint(CHK_START)) {
            open_diagnostics();
            chk_set_time(CHK_START);
            goto chk_loaded;
        }
        fprintf(ferr, "# chk: checkpoint t=%g non trovato, uso la IC\n", CHK_START);
    }
#endif
    double slope = TILT*DD/(0.5*LZ);   // transverse shift per unit of z
    foreach() {
        u.x[] = u.y[] = u.z[] = 0.;
        for (int s = -1; s <= 1; s += 2) {
            double xc = s*DD*(1. + PERT*cos(2.*pi*z/LZ))
                - s*slope*(z - 0.5*LZ);
            double dx = x - xc;
            double r = sqrt(sq(dx) + sq(y));
            if (r > 0.) {
                double ut = 0.5*sq(SIGMA)*(1. - exp(-sq(r)/sq(SIGMA)))/r;
                u.x[] += -ut*y/r;
                u.y[] +=  ut*dx/r;
            }
        }
    }
    boundary((scalar *){u});

    open_diagnostics();

#if CHECKPOINT
chk_loaded: ;
#endif
}

/**
  Diagnostics once per unit of time.
  Enstrophy Z = int |omega|^2 dV, energy E = 1/2 int |u|^2 dV,
  circulation and separation of the nuclei on the slice z = LZ/2.
*/

event diagnostics (t += 1; t <= TMAX)
{
    boundary((scalar *){u});

    double Z = 0., E = 0., G = 0., Umax = 0., Wmax = 0.;
    foreach(reduction(+:Z) reduction(+:E) reduction(+:G)
        reduction(max:Umax) reduction(max:Wmax)) {
        double dV = cube(Delta);
        Z += (sq(WX) + sq(WY) + sq(WZ))*dV;
        E += 0.5*(sq(u.x[]) + sq(u.y[]) + sq(u.z[]))*dV;
        Umax = max(Umax, sqrt(sq(u.x[]) + sq(u.y[]) + sq(u.z[])));
        Wmax = max(Wmax, sqrt(sq(WX) + sq(WY) + sq(WZ)));
        if (z <= ZNODE && z > ZNODE - Delta)   // "node" slice
            G += WZ*sq(Delta);
    }

    // nuclei on the node slice: first maximum, then the second in the opposite half-plane
    double m1 = -HUGE;
    foreach(reduction(max:m1))
        if (z <= ZNODE && z > ZNODE - Delta) m1 = max(m1, WZ);

    double px = 0., py = 0., w1 = 0.;
    foreach(reduction(+:px) reduction(+:py) reduction(+:w1))
        if (z <= ZNODE && z > ZNODE - Delta && WZ == m1) { w1 += 1.; px += x; py += y; }
    if (w1 > 0.) { px /= w1; py /= w1; }

    double m2 = 0.;
    foreach(reduction(max:m2))
        if (z <= ZNODE && z > ZNODE - Delta && x*px + y*py < 0.) m2 = max(m2, WZ);

    double qx = 0., qy = 0., w2 = 0.;
    foreach(reduction(+:qx) reduction(+:qy) reduction(+:w2))
        if (z <= ZNODE && z > ZNODE - Delta && WZ == m2 && x*px + y*py < 0.) {
            w2 += 1.; qx += x; qy += y;
        }
    if (w2 > 0.) { qx /= w2; qy /= w2; }
    double sep = (w1 > 0. && w2 > 0.) ? sqrt(sq(px - qx) + sq(py - qy)) : 0.;

    int cells = 0;
    foreach(reduction(+:cells)) cells++;

    if (t >= RESA_TMIN && mgp.resa > max_resa) {
        max_resa = mgp.resa;
        max_resa_t = t;
    }

    if (pid() == 0) {
        FILE * fp = fopen("diagnostics.dat", "a");
        fprintf(fp, "%.6g %.6e %d %.8e %.8e %.8e %.8e %.8e %.6e %.6e %.6e %d %d %.3e\n",
            t, dt, i, Z, E, Umax, Wmax, G, sep, px, py, cells, mgp.i, mgp.resa);
        fclose(fp);

        fprintf(stderr, "%.6g %.6e %.6e %.6e %.6e %.6e %.6e\n",
            t, Z, E, Umax, Wmax, G, sep);
    }
}

#if MOVIE
// Movie: x-y slices at z = LZ/2 (node) and z = 0 (antinode).
// Reuses p as buffer (no dedicated field, see the note above).
event movie (t += 0.5; t <= TMAX)
{
    boundary((scalar *){u});
    foreach()
        p[] = WZ;
    boundary({p});
    output_ppm(p, file = "omega3d_node.mp4", n = 1024,
        min = -0.5*OMEGA0, max = OMEGA0, z = ZNODE, linear = true);
    output_ppm(p, file = "omega3d_anti.mp4", n = 1024,
        min = -0.5*OMEGA0, max = OMEGA0, z = 0., linear = true);
}
#endif

// Final frame (node slice).
event snapshot (t = TMAX)
{
    boundary((scalar *){u});
    foreach()
        p[] = WZ;
    boundary({p});
    output_ppm(p, file = "omega3d.ppm", n = 2048,
        min = -0.5*OMEGA0, max = OMEGA0, z = ZNODE, linear = true);
}

/**
  3D dump for visualization: VTK ImageData format (.vti, raw
  little-endian, cell-center node coordinates) with omega_z and
  |omega|, COMPRESSED with zlib (standard VTI format, lossless).
  Written by the CPU (serial foreach) every VTK_TSTEP into the folder
  VTK_DIR (default "vtk3d", created if it does not exist).
  To upload them to Google Drive: ./carica_vti_gdrive.py (see --help).
  With -DVTK_UPLOAD=1 every file is uploaded right after it is written and
  the solver WAITS for the upload to finish before the next timestep.
  Compile:
      EXTRA="-DVTK=1 -DVTK_UPLOAD=1" ./build.sh
  First authorization (once, interactive):
      python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json --auth-only
*/

#if VTK

#define VTK_BLOCK 32768        // zlib block size (as in VTK)
// zlib level: 9 compresses ~3% more but is ~10x slower; on floats
// level 1 is fine (the dumps stay readable by ParaView/pyvista).
#ifndef VTK_ZLEVEL
# define VTK_ZLEVEL 1
#endif
#if VTK_ZLEVEL > 0
# define VTK_COMPRESSOR " compressor=\"vtkZLibDataCompressor\""
#else
# define VTK_COMPRESSOR ""
#endif

typedef struct {
    unsigned char * data;      // [UInt32 header][concatenated zlib blocks]
    size_t size;
} VtkBlocco;

#if VTK_ZLEVEL > 0
// Compresses a float array into the compressed VTK "appended" format:
//   UInt32 nblocks, UInt32 block_size, UInt32 last_partial(0 if exact),
//   UInt32 csize[nblocks], then the zlib blocks.
static VtkBlocco comprimi (const float * a, size_t n)
{
    VtkBlocco b = {NULL, 0};
    size_t bytes = n*sizeof(float);
    size_t nblocks = (bytes + VTK_BLOCK - 1)/VTK_BLOCK;
    size_t last = bytes % VTK_BLOCK;
    size_t hdr = 3 + nblocks;
    size_t cap = bytes + nblocks*64 + 4096;
    b.data = malloc(cap);
    if (!b.data) {
        fprintf(ferr, "vtk: malloc compressione fallito\n");
        return b;
    }
    unsigned int * h = (unsigned int *) b.data;
    h[0] = nblocks;
    h[1] = VTK_BLOCK;
    h[2] = last;
    unsigned char * p = b.data + 4*hdr;
    size_t tot = 4*hdr;
    for (size_t i = 0; i < nblocks; i++) {
        size_t len = (i == nblocks - 1 && last) ? last : VTK_BLOCK;
        uLongf destLen = cap - (p - b.data);
        if (compress2(p, &destLen,
                (const Bytef *)((const char *) a + i*VTK_BLOCK), len, VTK_ZLEVEL) != Z_OK) {
            fprintf(ferr, "vtk: compressione fallita\n");
            free(b.data);
            b.data = NULL;
            return b;
        }
        h[3 + i] = destLen;
        p += destLen;
        tot += destLen;
    }
    b.size = tot;
    return b;
}
#else
// UNcompressed "appended" block (VTK raw): UInt32 nbytes + raw floats.
static VtkBlocco comprimi (const float * a, size_t n)
{
    VtkBlocco b = {NULL, 0};
    size_t bytes = n*sizeof(float);
    if (bytes > 0xffffffffu) {
        fprintf(ferr, "vtk: blocco troppo grande\n");
        return b;
    }
    b.data = malloc(4 + bytes);
    if (!b.data) {
        fprintf(ferr, "vtk: malloc fallito\n");
        return b;
    }
    unsigned int nb = (unsigned int) bytes;
    memcpy(b.data, &nb, 4);
    memcpy(b.data + 4, a, bytes);
    b.size = 4 + bytes;
    return b;
}
#endif

#if VTK_UPLOAD
// Uploads the .vti just written and WAITS for the upload to finish: system()
// is synchronous, so the solver does not move to the next timestep until
// the upload program has terminated.
static void carica_vti (const char * path)
{
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "%s \"%s\" >> %s 2>&1",
        VTK_UPLOAD_CMD, path, VTK_UPLOAD_LOG);
    fprintf(ferr, "# vtk: carico %s su Google Drive (attendo l'upload)...\n",
        path);
    int status = system(cmd);
    if (status != 0) {
        fprintf(ferr, "# vtk: upload FALLITO (status %d): vedi %s\n",
            status, VTK_UPLOAD_LOG);
#if VTK_UPLOAD_ABORT
        exit(1);
#endif
    }
    else
        fprintf(ferr, "# vtk: caricato su Google Drive: %s\n", path);
}
#endif // VTK_UPLOAD

static void write_vti (double t)
{
    mkdir (VTK_DIR, 0777);   // if the folder already exists it fails: ignored

    long nx = NDIV0 << MAXLEVEL, ny = nx, nz = nx*(int)(LZ + 0.5);
    double hdel = L0/(NDIV0 << MAXLEVEL);  // cell Delta
    size_t n = (size_t)nx*ny*nz;
    float * wz = malloc(n*sizeof(float));
    float * wm = malloc(n*sizeof(float));
    if (!wz || !wm) {
        fprintf(ferr, "vtk: malloc fallito (%.0f MB)\n",
            2.*n*sizeof(float)/1e6);
        free(wz); free(wm);
        return;
    }

    boundary((scalar *){u});
    foreach (serial) {
        size_t i = (x - X0)/hdel, j = (y - Y0)/hdel, k = (z - Z0)/hdel;
        size_t idx = (k*ny + j)*nx + i;
        double wx = WX, wy = WY, vwz = WZ;
        wz[idx] = vwz;
        wm[idx] = sqrt(wx*wx + wy*wy + vwz*vwz);
    }

    VtkBlocco b1 = comprimi(wz, n);
    free(wz);
    VtkBlocco b2 = comprimi(wm, n);
    free(wm);
    if (!b1.data || !b2.data) {
        free(b1.data); free(b2.data);
        return;
    }

    // Name with the time at 3 decimals: with VTK_TSTEP < 1 the old
    // "%04d" produced the same name for different dumps (and the previous
    // dump was overwritten). %08.3f also sorts well.
    char name[128];
    sprintf(name, "%s/vtk3d_%08.3f.vti", VTK_DIR, t);
    FILE * fp = fopen(name, "wb");
    if (!fp) {
        fprintf(ferr, "vtk: non riesco ad aprire %s\n", name);
        free(b1.data); free(b2.data);
        return;
    }
    fprintf(fp,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\"%s>\n"
        "  <ImageData WholeExtent=\"0 %ld 0 %ld 0 %ld\" Origin=\"%g %g %g\" Spacing=\"%g %g %g\">\n"
        "    <Piece Extent=\"0 %ld 0 %ld 0 %ld\">\n"
        "      <PointData Scalars=\"omega_mag\">\n"
        "        <DataArray type=\"Float32\" Name=\"omega_z\" format=\"appended\" offset=\"0\"/>\n"
        "        <DataArray type=\"Float32\" Name=\"omega_mag\" format=\"appended\" offset=\"%lu\"/>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n_",
        VTK_COMPRESSOR,
        nx - 1, ny - 1, nz - 1, X0, Y0, Z0, hdel, hdel, hdel,
        nx - 1, ny - 1, nz - 1, (unsigned long) b1.size);
    fwrite(b1.data, 1, b1.size, fp);
    fwrite(b2.data, 1, b2.size, fp);
    fprintf(fp, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(fp);

    free(b1.data); free(b2.data);
    fprintf(stderr, "# vtk: scritto %s (%.0f MB), t = %g\n",
        name, (b1.size + b2.size + 300.)/1e6, t);
#if VTK_UPLOAD
    carica_vti(name);
#endif
}


// t = 0, VTK_TSTEP, 2*VTK_TSTEP, ... up to TMAX
event vtk_dump (t += VTK_TSTEP; t <= TMAX)
    if (t >= VTK_TSTART) write_vti (t);

#endif // VTK

/**
  WZ field on a regular grid at the final time, for the fixed-grid
  convergence study (error_study_fissa_3d_gpu.sh /
  error_norm_3d.py). Binary format:
      int32 nx, ny, nz ; then float32 with i fastest, then j, then k.
  Can be disabled with -DMATRIX=0.
*/
#ifndef MATRIX
# define MATRIX 1
#endif

#if MATRIX
#if CHECKPOINT
// checkpoint finale in t = TMAX (per ripartire con -DCHK_START=<t>)
event chk_final (t = TMAX)
    if (t > 0.) write_checkpoint(t);
#endif

event matrix (t = TMAX)
{
    long nx = NDIV0 << MAXLEVEL, ny = nx, nz = nx*(int)(LZ + 0.5);
    double hdel = L0/(NDIV0 << MAXLEVEL);
    size_t n = (size_t)nx*ny*nz;
    float * w = malloc(n*sizeof(float));
    if (!w)
        fprintf(ferr, "matrix: malloc fallito\n");
    else {
        boundary((scalar *){u});
        foreach (serial)
            w[((size_t)((z - Z0)/hdel)*ny + (size_t)((y - Y0)/hdel))*nx
              + (size_t)((x - X0)/hdel)] = WZ;
        FILE * fp = fopen("omega3d_matrix.bin", "wb");
        if (fp) {
            int dims[3] = {nx, ny, nz};
            fwrite(dims, sizeof(int), 3, fp);
            fwrite(w, sizeof(float), n, fp);
            fclose(fp);
            fprintf(stderr, "# matrix: scritto omega3d_matrix.bin (%ld x %ld x %ld)\n",
                nx, ny, nz);
        }
        free(w);
    }
}
#endif // MATRIX

// At the end of the run: level, maximum Poisson residual (at steady state) and time.
event write_resa (t = TMAX)
{
    if (pid() == 0) {
        FILE * fp = fopen("resa_max.dat", "w");
        fprintf(fp, "%d %.6e %.6g\n", MAXLEVEL, max_resa, max_resa_t);
        fclose(fp);
    }
}

// Final summary in the log.
event check (t = TMAX)
{
    if (pid() == 0)
        fprintf(stderr,
            "#\n"
            "# riepilogo (3D)\n"
            "# MAXLEVEL = %d, griglia %d x %d x %d celle\n"
            "# massimo residuo Poisson pressione = %.3e (a t = %.6g)\n",
            MAXLEVEL, NDIV0 << MAXLEVEL, NDIV0 << MAXLEVEL,
            (NDIV0 << MAXLEVEL)*(int)(LZ + 0.5),
            max_resa, max_resa_t);
}
