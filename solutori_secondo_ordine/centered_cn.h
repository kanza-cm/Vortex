/**
# A 2nd-order Crank--Nicolson extension for `navier-stokes/centered.h`

Extends the viscous term of `navier-stokes/centered.h` from 1st-order
Backward Euler to 2nd-order Crank--Nicolson in time, with a variable CFL
timestep:
1. `advection_term` (`centered_cn.h`): evaluates the explicit viscous increment
   $\Delta\mathbf{u}_{\text{visc}}^n = \frac{\Delta t_n}{\rho}\nabla\cdot(2\mu\mathbf{D}^n)$
   from $\mathbf{u}^n$ and adds $\Delta\mathbf{u}_{\text{visc}}^n / \Delta t_n$
   to `g` so that the BCG face predictor (`prediction` and `advection`) is
   2nd-order accurate at finite Reynolds numbers.
2. `viscous_term` (`centered_cn.h`): restores `g` and advances `u` by the
   explicit half-step $\frac{1}{2}\Delta\mathbf{u}_{\text{visc}}^n$.
3. `viscous_term` (`centered.h`): applies `correction(dt)`, solves the
   implicit half-step $\Delta t_n / 2$ via `viscosity()`, and applies
   `correction(-dt)`.
*/

#if EMBED
# error "centered_cn.h requires standard viscosity.h (EMBED is not supported)"
#endif

#include "viscosity.h"

#define viscosity(u, mu, rho, dt, nrelax) \
    viscosity(u, mu, rho, 0.5[0]*(dt), nrelax)
#include "navier-stokes/centered.h"
#undef viscosity

vector duvisc[];

event defaults (i = 0)
{
    foreach_dimension()
        duvisc.x.nodump = true;
}

/**
1. Runs BEFORE `advection_term (i++, last)` in `centered.h` (and after
   `stability` and `properties`): computes $\Delta\mathbf{u}_{\text{visc}}^n$
   and includes the viscous acceleration in `g` for the BCG predictor. */

event advection_term (i++)
{
    if (constant(mu.x) != 0.) {
        restriction ({mu, rho});
        struct Viscosity p = { mu, rho, dt };
        residual_viscosity ((scalar *){u}, (scalar *){u},
                            (scalar *){duvisc}, &p);
        if (!stokes)
            foreach()
                foreach_dimension()
                    g.x[] += duvisc.x[]/dt;
    }
}

/**
2. Runs AFTER `advection_term (i++, last)` and BEFORE
   `viscous_term (i++, last)` in `centered.h`: restores `g` and adds the
   explicit half-step $\frac{1}{2}\Delta\mathbf{u}_{\text{visc}}^n$. */

event viscous_term (i++)
{
    if (constant(mu.x) != 0.) {
        foreach()
            foreach_dimension() {
                if (!stokes)
                    g.x[] -= duvisc.x[]/dt;
                u.x[] += 0.5[0]*duvisc.x[];
            }
    }
}
