# Talk script — Coalescence of a Pair of Gaussian Vortex Tubes (English, 10-slide version)

**Duration:** about 10-11 minutes (10 slides).
**Use:** the text is meant to be said almost verbatim; numbers match the slides. Notes in [square brackets] are stage directions.

---

## Slide 1 — Numerical Formulation & Setup (0:00-1:20)

We solve the incompressible Navier-Stokes equations in velocity-pressure form. The scheme is centred: pressure projection with ∇²p = (1/Δt) ∇·u*, correction of the face velocities, second-order BCG advection. The same core is implemented in two ways: fixed grid with Basilisk on CUDA in single precision, and adaptive grid with incflo/AMReX, also on CUDA.

The parameters are the same as in the 2D case: σ = 0.09, separation 2d = 0.28 with D = 0.14, viscosity 10⁻⁵, circulation Reynolds number about 2.5×10^3. Grid levels go from L4 to L8, that is from 64³ up to 256×256×1024, with CFL between 0.3 and 0.5. For AMR: refinement ratio 4 and VORTERR threshold between 0.3 and 10⁻².

## Slide 2 — Fixed-Grid GPU: Cost Model (1:20-2:10)

The fixed-grid cost is the table in front of you. At L6, one million cells, 0.15 GB and about 0.2 seconds per step; at L7, 8.4 million cells, 1.1 GB and 1.6 seconds; at L8, 67 million cells, 7.5 GB and 2.8 seconds per step. The model is simple: about 120 bytes per cell, including fields, ghosts, the multigrid hierarchy and the CUDA context.

Single precision is not a shortcut: with 8 GB we can reach 67 million cells, with 16 GB about 140 million, but the next level, 537 million cells, would need about 58 GB. The limit is memory before it is compute.

Where does 120 bytes per cell come from? In single precision every value costs 4 bytes. The solver allocates 26 variables -- the start-up log prints '26 var': three velocity components, three face velocities, three Crank-Nicolson viscous increments, cell and face pressure, acceleration, mu/rho and the multigrid temporaries. That is 26 x 4 = 104 bytes per cell; adding one ghost layer and the allocator/CUDA-context overhead brings it to about 120 bytes per cell, i.e. about 30 fp32 values per cell. Hence 8 GB / 120 B ~ 67M cells (L8), 16 GB ~ 140M, and L9 (537M cells) would need ~58 GB.

## Slide 3 — Conservation Laws & Dynamics Diagnostics (2:10-3:50)

Now the conservation diagnostics at T = 1600. The circulation is not measured on a single slice: it is the volume average over the whole 3D domain, Gamma = (1/Lz) int_V omega_z dV, compared with the initial analytic value Gamma_0 = 2 pi sigma^2 = 5.089e-2.

It drops by 42.9% in the inclined case and 7.8% in the non-inclined one. This is not a solver error: a fixed volume is not a material volume, and with walls int_V omega_z dV becomes a boundary term, so viscous diffusion, tilting and wall shear all contribute. At T = 50 the same 3D average was only about 1% (inclined) and 0.03% (non-inclined): the decay is a late-time effect.

The angular impulse grows (+3.8% inclined, +134% non-inclined) as vorticity spreads radially, while palinstrophy decays from 424 to 138 (inclined) and from 399 to 363 (non-inclined), consistent with viscous smoothing.

---

## Slide 4 — Grid Convergence & Parameter Sensitivity (3:50-5:20)

Here is the convergence study. Method: same CFL, so Δt ∝ h; the ω_z field of each level is compared with the L8 reference after restriction to the Lf = 5 grid and a Gaussian filter with σ = 3 cells, which removes the phase-sensitive filament content.

Measured orders: inclined p = 1.93 from L5 to L6 and 2.24 from L6 to L7 (fitted 2.09 over L5-L7); non-inclined 2.22 and 2.50 (fitted 2.36). Without the filter the inclined case drops to about 1.4-2.0: the scheme is second order and the apparent loss is a phase error.

The operational message: accuracy is driven by MAXLEVEL; the AMR tolerance compresses the cells, it does not replace resolution.

## Slide 5 — Poisson Solver: Elliptic Residuals (5:20-6:10)

The pressure projection is the elliptic part. The table shows the maximum residual at T = 30: it grows with the level, because there are more cells at the same tolerance, but it stays below 1.3×10⁻² at L8; in the non-inclined case it stays below 2.3×10⁻³ at all levels.

The residual must be compared with the solver tolerance, 10⁻³/Δt², not with 1. So the solve is converged and is not the bottleneck: the dominant error is spatial discretisation. I also checked that with exactly 5 fixed V-cycles for pressure the convergence-study errors change by less than 0.4%: 5 are enough, and the residual stays small.

## Slide 6 — Adaptive Grid: How It Is Built (6:10-7:00)

Let me move to AMR. We start from a uniform base 2^MINLEVEL and every refined level multiplies by 4. Tagging is on vorticity: cells with |ω| above the VORTERR threshold become new boxes, which are rebuilt periodically at regrid, with average down and up between levels. There is a maximum cell budget: if the hierarchy explodes, the thresholds are raised automatically. The figure on the right shows the nested hierarchy: L0, L1, L2.

## Slide 7 — AMR on GPU: Data Structures & Regrid Cycle (7:00-7:50)

On GPU each level is a BoxArray: a list of non-overlapping boxes covering only the regions to be refined. The data are MultiFabs: BoxArray plus distribution mapping plus device arrays, in single precision. Tagging is a GPU kernel that computes |ω| cell by cell; flagged cells are clustered into a few large boxes with a Berger-Oliger-type algorithm, to limit the number of kernel launches. At regrid we build the new hierarchy, average down the fine solution and fill the coarse/fine ghosts.

## Slide 8 — AMR on GPU: Execution Model & Costs (7:50-8:40)

Updating one level is a sequence of GPU kernels: advection and diffusion are one kernel per box, the Poisson solve uses a multigrid across levels. Ghosts are filled by copying from same-level neighbours and interpolating from the coarse level; after regrid we do average down and fill patches. Global reductions, such as norms and the number of tagged cells, must complete before a new regrid.

The cost: about 357 MB per million cells, three times the fixed grid. Measured, 11.7 million cells occupy 6.4 GB; the T = 30 campaign has a cap of 15 million cells to fit in 8 GB.

Where does the AMR cost come from? Each MultiFab holds the same fields with two to three ghost layers, plus BoxArray/distribution-mapping metadata and the temporary multigrid buffers kept in the arena. The design estimate is about 320 bytes per cell, i.e. about 80 fp32 values per cell (26 fields plus ghosts and temporaries), roughly 2.7x the fixed grid; the measured value in the table is 357 MB per million cells. At 11.7M cells that gives 6.4 GB, which is ~550 bytes per cell because the finest levels are duplicated with their coarse levels and the arena reserves buffers; that is why the practical cap is 15M cells in 8 GB (8 GB / 550 B ~ 15M), not the 22M suggested by 357 bytes per cell alone.

## Slide 9 — Adaptive Grid: Error Map at T=30 (8:40-9:30)

This is the error map at T = 30, for both configurations: MAXLEVEL 4-8 on the vertical axis, the threshold on the horizontal one.

At MAXLEVEL 4-5 the error is set by the grid, not by the threshold: it is flat, from +0.14 to -0.07 for the inclined case and from -0.45 to -0.79 for the non-inclined one.

From L6 the threshold matters, and the finest levels give the largest gain; the AMR hierarchy reaches up to 1.5×10⁷ cells at L8.

## Slide 10 — Methods & Software (9:30-10:30)

Before closing, a word on the software. The fixed-grid solver is Basilisk (Popinet, 2015) with the centred scheme and the CUDA backend; the block-structured AMR runs use incflo / AMReX (Zhang et al., 2019). On top of these I implemented the second-order Crank-Nicolson viscous term, the palinstrophy and angular-impulse diagnostics computed on the fly, and the AMR error-map and GPU cost pipelines. All runs are on a single RTX 3070 with 8 GB, in single precision.


---

# Appendix — likely questions

**Why does the circulation on the slice drop by 43% at T = 1600?**
At T = 200 the drop was below 0.5% (tilting and stretching only). At T = 1600 vorticity has diffused and spread, and the flux through the slice also changes because of viscous decay and wall shear: the drop measures the decay of the moment, not a scheme error.

**Why do the inclined cases lose second order at T = 50?**

At T = 50 the flow is dominated by filaments: the L∞ error is essentially a phase error, not a dissipation error. L₂ gives about 1; one would need a finer reference (L9 does not fit in 8 GB) or filtered norms: by coarsening both fields to L5/L6 the order should come back close to 2.

**What are the Poisson residuals at T = 50?**

Maximum values: 7.2e-2 for the inclined case at L8 and 2.2e-3 for the non-inclined one, always below the tolerance 10⁻³/Δt². The inclined L8 is the most demanding case, but it stays at 7% of the tolerance.

**Are 5 pressure iterations enough?**

Yes: with 5 fixed V-cycles only on pressure the T = 30 convergence-study errors change by less than 0.4%, and the final residual at L8 is 8×10⁻⁵ (inclined) and 4×10⁻⁵ (non-inclined).

**Why is there no MPI in the presentation?**

The MPI prototype exists and is validated (z-slab decomposition, distributed V-cycle), but on two PCs the gain is memory, not time (a realistic cap of about 1.5×). I left it out to keep the talk manageable; I can discuss it in questions.

**Why is the circulation on the slice not conserved?**

In 3D only the circulation along a material curve is conserved (Kelvin); the fixed slice z = 2 is not material, and tilting/stretching change the flux through it. That is why I use it as a consistency check, not as a balance law.

**Where are the movies and the data?**

The 3D animation and the slice movies are in gpu_nvidia_3d/risultati/video/; the diagnostics and error studies are in gpu_nvidia_3d/risultati/error_study/ and gpu_nvidia_3d/risultati/run_L7_T200_diag/.
