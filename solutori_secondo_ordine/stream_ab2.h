/**
# A 2nd-order streamfunction--vorticity solver for the Navier--Stokes equations

Second-order accurate in time extension of `navier-stokes/stream.h`, also
with a variable CFL time step.

Exact sequence of events at each step:
1. `velocity` (`stream_ab2.h`): solves $\nabla^2\psi^n = \omega^n$ and computes $\mathbf{u}^n$.
2. `velocity` (`advection.h`): computes the new `dt` ($\Delta t_n$) from $\mathbf{u}^n$.
3. `tracer_advection` (`stream_ab2.h`): moves `u` to the intermediate time
   $t^{n+1/2} = t^n + \Delta t_n/2$:
   - with a midpoint predictor step (RK2) on the first step, or if `dt`
     jumps abruptly, `dt > 3.*dtprev`;
   - with Adams-Bashforth 2 at variable step $r = \frac{\Delta t_n}{2\,\Delta t_{n-1}}$
     on all other steps.
4. `tracer_advection` (`tracer.h`): advects $\omega^n \to \omega^{n+1}$ with `bcg.h`
   using $\mathbf{u}^{n+1/2}$.
*/

#include "advection.h"
#include "poisson.h"

scalar omega[], psi[];
mgstats mgpsi;
scalar* tracers = { omega };

psi[right] = dirichlet(0);
psi[left] = dirichlet(0);
psi[top] = dirichlet(0);
psi[bottom] = dirichlet(0);

/* Velocity history at t^{n-1} and the previous time step */
face vector uprev[];
double dtprev;
bool ab2_first = true;

static void psi_from_omega(void)
{
    mgpsi = poisson(psi, omega);
    trash({ u });
    coord f = { -1.[0], 1.[0] };
    foreach_face()
        u.x[] = f.x * (psi[0, 1] + psi[-1, 1] - psi[0, -1] - psi[-1, -1]) / (4. * Delta);
}

event defaults(i = 0)
{
    CFL = 0.8;
    ab2_first = true;
    display("squares (color = 'omega', spread = -1);");
}

/**
1. Runs BEFORE `dt` is computed in `advection.h` (qcc's inversion rule
   for same-name events): updates $\psi^n$ and $\mathbf{u}^n$. */

event velocity(i++)
{
    psi_from_omega();
}

/**
2. Runs AFTER `dt` is computed in `advection.h` and BEFORE
   `advection (tracers, uf, dt)` in `tracer.h`: computes $\mathbf{u}^{n+1/2}$. */

event tracer_advection(i++)
{
    if (!ab2_first)
        if (dt > 3.[0] * dtprev)
            ab2_first = true;

    if (ab2_first) {
        foreach_face()
            uprev.x[] = u.x[];

        scalar omega0[];
        foreach()
            omega0[] = omega[];
        mgstats mg0 = mgpsi;

        advection({ omega }, u, 0.5[0] * dt);
        psi_from_omega();

        foreach()
            omega[] = omega0[];
        mgpsi = mg0;
        ab2_first = false;
    }
    else {
        double r = 0.5[0] * (dt / dtprev);
        foreach_face() {
            double un = u.x[];
            u.x[] = (1.[0] + r) * un - r * uprev.x[];
            uprev.x[] = un;
        }
    }
    dtprev = dt;
}
