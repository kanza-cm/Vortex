# Vortex-tube coalescing - solvers, scripts and results

This repository collects four implementations of the same 3D vortex-tube
coalescing case (plus the 2D converging study that motivates it):

| directory | solver |
|---|---|
| `solutori_secondo_ordine/` | 2D CPU Basilisk solvers (fixed grid, grid study, adaptive), with error/heatmap studies |
| `gpu_nvidia/` | 2D GPU (CUDA) Basilisk solver |
| `gpu_nvidia_3d/` | 3D fixed-grid GPU (CUDA) Basilisk solver |
| `gpu_amr_3d/` | 3D adaptive-mesh GPU solver (incflo / AMReX) |
| `mpi_fissa/` | MPI domain-decomposition prototype for the fixed-grid GPU solver (2 GPUs over 2 PCs) |

Root files (`saetta.c`, `build.sh`, `run.sh`, `error_norm.py`) are the 2D
uniform-grid reference case.

## Quick start

```sh
# 2D CPU solver (Basilisk)
./build.sh && ./run.sh

# 3D fixed-grid GPU solver
cd gpu_nvidia_3d && ./build.sh && ./avvia.sh

# 3D AMR GPU solver
cd gpu_amr_3d && ./build.sh && ./avvia.sh 4 6
```

Requirements: Basilisk in `~/basilisk/src`, CUDA 13.1 toolchain in
`~/cuda-env.sh`, incflo/AMReX in `~/src/incflo`. Each solver directory has
its own README or header comments with details.

## Results

Generated data (plotfiles, matrices, movies, heatmaps) lives in the
`risultati/` sub-directory of each solver and is not versioned.
