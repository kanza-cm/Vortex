
// Local copy of stream.h with the 2nd order psi-u coupling (AB2).
# include "solutori_secondo_ordine/stream_ab2.h"
#include "diffusion.h"

// CFL override (stream.h hardcodes 0.8): -DCFL_USER=<value>.
// Default 0.8 for the 2nd order solver.
#ifndef CFL_USER
# define CFL_USER 0.8
#endif

event defaults_cfl (i = 0)
    CFL = CFL_USER;

// Parameters
#ifndef DD
# define DD        0.14     // half distance between the cores d = 0.28  ->  d/sigma = 3.0  (coalescence)
#endif
#ifndef SIGMA
# define SIGMA     0.09     // core width
#endif
#ifndef OMEGA0
# define OMEGA0    1.       // initial amplitude
#endif
#ifndef NU
# define NU        1e-5     // viscosity
#endif
#ifndef MAXLEVEL
# define MAXLEVEL  10       // fixed grid 2^n x 2^n
#endif
// Resolution of the field saved to omega_matrix.bin (for the error
// study): the native 2^MAXLEVEL grid by default.
#ifndef MATRIX_N
# define MATRIX_N  (1 << MAXLEVEL)
#endif
#ifndef TMAX
# define TMAX      200.     
#endif
#ifndef MOVIE
# define MOVIE     1
#endif
// Time from which the mgpsi.resa maximum is recorded:
// it excludes the initial transient (psi starts at zero).
#ifndef RESA_TMIN
# define RESA_TMIN 1.
#endif

// Region shown in the figure
#define VIEW  {{-0.5,-0.5},{0.5,0.5}}

// Constant viscosity, used by the diffusion solver
const face vector D[] = { NU, NU };

// state for the final checks 
double G0 = 0., I0 = 0.;                      // circulation and angular momentum at t=0
double Glast = 0., Ilast = 0.;                // last computed values
double Zprev = 0., tprev = -1.;               // for dZ/dt and for the Z monotonicity check
int    enstrophy_violations = 0, ncheck = 0;
double max_balance = 0.;                      // max |dZ/dt + 2*NU*P| / (2*NU*P)
double max_resa = 0.;                         // mgpsi.resa maximum over the whole run
double max_resa_t = 0.;                       // time when max_resa occurs

// Basilisk
int main()
{
    origin(-0.5, -0.5);
    init_grid(1 << MAXLEVEL);

    // With -DPOISSON_ITER=n the Poisson solver does exactly n iterations
    // (no tolerance criterion). Used by the convergence studies.
#ifdef POISSON_ITER
    NITERMIN = NITERMAX = POISSON_ITER;
    TOLERANCE = 1e30;
#endif

    run();
}

 // Initial conditions: sum of two co-rotating gaussians 

event init(i = 0)
{
    double dd = DD, sigma = SIGMA, omega0 = OMEGA0;
    foreach()
        omega[] = omega0 * (exp(-(sq(x - dd) + sq(y)) / sq(sigma)) +
            exp(-(sq(x + dd) + sq(y)) / sq(sigma)));

    boundary({ omega }); // b.c.

    if (pid() == 0) { // start writing the diagnostics file
        FILE* fp = fopen("diagnostics.dat", "w");
        fprintf(fp, "# t dt it enstrophy palinstrophy energy circulation "
            "impulse xc yc maxomega separation angle "
            "taylor core_radius aspect maxlevel cells mgpsi mgpsi_resa\n");
        fclose(fp);

        fprintf(stderr, "# t enstrophy palinstrophy energy circulation "
            "impulse maxomega separation taylor core_radius\n");
    }
}

/**
  Viscous diffusion, Crank-Nicolson with Strang splitting:
  first half after dt is computed, second half after advection. */

event tracer_advection(i++)
{
    diffusion(omega, 0.5*dt, D);
    psi_from_omega();
}

event tracer_diffusion(i++, last)
    diffusion(omega, 0.5*dt, D);

/**
  Diagnostics, written once per time unit.

  Physics:
    enstrophy      Z = int omega^2 dA
    palinstrophy   P = int |grad omega|^2 dA        (peak = merger)
    energy         E = 1/2 int |u|^2 dA
    circulation    G = int omega dA
    ang. momentum  I = int (x^2+y^2) omega dA

  Numerical validation:
    circulation    G must not change in time
    enstrophy      Z must never grow (pure diffusion)
    balance        dZ/dt + 2 nu P ~ 0
    centroid       (xc,yc) = int (x,y) omega dA / G
    NOTE: I is NOT conserved in the viscous case; dI/dt = 4 nu G.

  Vortex structure:
    separation     distanza tra i due massimi locali di omega
    angle          orientation of the pair
    taylor         microscale lambda = sqrt(Z/P)
    core_radius    minor semi-axis of the core (covariance of the core cells)
    aspect         elongation of the core
*/

event diagnostics(t += 1; t <= TMAX)
{
    // global integrals, reduction = parallelized over all cells
    double Z = 0., P = 0., E = 0., G = 0., Iang = 0., Mx = 0., My = 0.;
	foreach(reduction(+:Z) reduction(+:P) reduction(+:E) reduction(+:G) // sum over all cells
        reduction(+:Iang) reduction(+:Mx) reduction(+:My)) {
        double w = omega[] * sq(Delta);                                 // vorticity
        Z += sq(omega[]) * sq(Delta);
        G += w;
        Iang += (sq(x) + sq(y)) * w;
        Mx += x * w;
        My += y * w;
        P += (sq((omega[1] - omega[-1]) / (2. * Delta)) +
            sq((omega[0, 1] - omega[0, -1]) / (2. * Delta))) * sq(Delta);

        double ux = (u.x[] + u.x[1, 0]) / 2.;           // ux and uy at the cell center
        double uy = (u.y[] + u.y[0, 1]) / 2.;
        E += 0.5 * (sq(ux) + sq(uy)) * sq(Delta);
    }
    double xc = (G != 0.) ? Mx / G : 0.;                // the ? operator avoids division by 0
    double yc = (G != 0.) ? My / G : 0.;
    double taylor = (P > 0.) ? sqrt(Z / P) : 0.;
    stats s = statsf(omega);

    /** separation and angle of the two cores
        Global maximum and its mean position p; second maximum in the
        half-plane opposite to p. The test omega[] == m is exact: after a
        reduction the maximum is one of the values present on the grid. */ 
	double m1 = -HUGE;                                       // global maximum, first
    foreach(reduction(max:m1))
        m1 = max(m1, omega[]);

    double px = 0., py = 0., w1 = 0.;                         // position
    foreach(reduction(+:px) reduction(+:py) reduction(+:w1))
        if (omega[] == m1) { w1 += 1.; px += x; py += y; }
    if (w1 > 0.) { px /= w1; py /= w1; }

	double m2 = 0.;                                          // global maximum, second               
    foreach(reduction(max:m2))
        if (x * px + y * py < 0.) m2 = max(m2, omega[]);

    double qx = 0., qy = 0., w2 = 0.;                         // position
    foreach(reduction(+:qx) reduction(+:qy) reduction(+:w2)) 
        if (omega[] == m2 && x * px + y * py < 0.) { w2 += 1.; qx += x; qy += y; }
    if (w2 > 0.) { qx /= w2; qy /= w2; }
    // distance between the cores
    double sep = (w1 > 0. && w2 > 0.) ? sqrt(sq(px - qx) + sq(py - qy)) : 0.;

    // angle of the core axis, folded into (-90, 90]
    double angle = atan2(py, px) * 180. / pi;
    if (angle > 90.)  angle -= 180.;
    if (angle <= -90.) angle += 180.;

    /** core shape
        For two symmetric cores at +/-p with width w the covariance
        eigenvalues are p^2+w^2 and w^2: the minor semi-axis is w, so
        sqrt(l1/l2) measures how elongated the core is */
    double thr = 0.5 * s.max;
    double m0 = 0., cx = 0., cy = 0., vxx = 0., vyy = 0., vxy = 0.; // compute weight m0
    foreach(reduction(+:m0) reduction(+:cx) reduction(+:cy)         // and compute moments
        reduction(+:vxx) reduction(+:vyy) reduction(+:vxy)) {       // second moments
        if (omega[] < thr)
            continue;
        double w = omega[] * sq(Delta);
        m0 += w; cx += w * x; cy += w * y;
        vxx += w * sq(x); vyy += w * sq(y); vxy += w * x * y;
    }
    double core_radius = 0., aspect = 0.;                           
    if (m0 > 0.) {// compute covariance eigenvalues
        double cxx = vxx / m0 - sq(cx / m0), cyy = vyy / m0 - sq(cy / m0);
        double cxy = vxy / m0 - (cx / m0) * (cy / m0);
        double trv = cxx + cyy, detv = cxx * cyy - sq(cxy);
        double disc = sqrt(max(0., sq(trv) / 4. - detv));
        double l1 = trv / 2. + disc, l2 = trv / 2. - disc;
        core_radius = sqrt(max(0., l2));
        aspect = (l2 > 0.) ? sqrt(l1 / l2) : 0.; // how elongated/stretched it is
    }

    // conservation checks 
    if (t == 0.) { G0 = G; I0 = Iang; }
    if (tprev >= 0. && t > tprev) {
        double balance = (Z - Zprev) / (t - tprev) + 2. * NU * P;
        if (2. * NU * P > 0.)
            max_balance = max(max_balance, fabs(balance) / (2. * NU * P));
        if (Z > Zprev * (1. + 1e-10))
            enstrophy_violations++;
        ncheck++;
    }
    Zprev = Z; tprev = t;
    Glast = G; Ilast = Iang;

    // Poisson residual maximum at steady state, sampled at the diagnostics
    // times (the same ones written to diagnostics.dat), with its time.
    if (t >= RESA_TMIN && mgpsi.resa > max_resa) {
        max_resa = mgpsi.resa;
        max_resa_t = t;
    }
	// number of cells and maximum level
    int lmax = 0, cells = 0;
    foreach(reduction(max:lmax) reduction(+:cells))
        lmax = max(lmax, level), cells++;

    //  output 
    if (pid() == 0) {
        FILE* fp = fopen("diagnostics.dat", "a");
        fprintf(fp,
            "%.6g %.6e %d %.8e %.8e %.8e %.8e %.8e %.3e %.3e "
            "%.8e %.6e %.4f %.6e %.6e %.4f %d %d %d %.3e\n",
            t, dt, i, Z, P, E, G, Iang, xc, yc,
            s.max, sep, angle, taylor, core_radius, aspect, lmax, cells,
            t > 0. ? mgpsi.i : 0, t > 0. ? mgpsi.resa : 0.);
        fclose(fp);

        fprintf(stderr,
            "%.6g %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
            t, Z, P, E, G, Iang, s.max, sep, taylor, core_radius);
    }
}

#if MOVIE
// Movie: n = pixels along the long side, bilinear interpolation.
event movie(t += 0.5; t <= TMAX)
    output_ppm(omega, file = "omega.mp4", n = 2048, min = 0, max = OMEGA0,
        box = VIEW, linear = true);
#endif

// Final frame.
event snapshot(t = TMAX)
    output_ppm(omega, file = "omega.ppm", n = 4096, min = 0, max = OMEGA0,
        box = VIEW, linear = true);

// Field on a regular grid at TMAX (output_matrix format) for the error
// study: comparison between runs at increasing resolution (error_norm.py).
event matrix(t = TMAX)
    output_matrix(omega, file = "omega_matrix.bin", n = MATRIX_N, box = VIEW);

// For the studies: at the end of the run it writes the level, the Poisson
// residual maximum (at steady state) and the time it occurs.
event write_resa(t = TMAX)
{
    if (pid() == 0) {
        FILE* fp = fopen("resa_max.dat", "w");
        fprintf(fp, "%d %.6e %.6g\n", MAXLEVEL, max_resa, max_resa_t);
        fclose(fp);
    }
}

/**
  Final summary of the checks, printed to the log.
  Expected: constant circulation, enstrophy never growing, balance ~0.
  Angular momentum changes through viscosity (dI/dt = 4 nu G), so
  I(T)/I(0)-1 is not a numerical error. */

event check(t = TMAX)
{
    if (pid() == 0)
        fprintf(stderr,
            "#\n"
            "# controlli di conservazione (validazione) \n"
            "# circolazione    G(T)/G(0) - 1 = %.3e\n"
            "# enstrofia: violazioni di monotonia = %d su %d passi\n"
            "# bilancio |dZ/dt + 2 nu P| / (2 nu P): massimo = %.3e\n"
            "# impulso angolare I(T)/I(0) - 1 = %.3e   (atteso ~ 4 nu G T / I0)\n",
            G0 != 0. ? Glast / G0 - 1. : 0.,
            enstrophy_violations, ncheck, max_balance,
            I0 != 0. ? Ilast / I0 - 1. : 0.);
}
