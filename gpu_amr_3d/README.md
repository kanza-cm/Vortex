# gpu_amr_3d - 3D vortex tubes with AMR on GPU (incflo / AMReX)

Adaptive-mesh 3D version of the vortex-tube case, built on incflo + AMReX
(CUDA backend, single precision). Two co-rotating tubes with varicose
perturbation and inclined axes merge in time; the grid is refined where
|omega| is large.

## Launchers

| script | what it does |
|---|---|
| `build.sh` | builds incflo (CMake superbuild, CUDA) |
| `run.sh` | one run: `MINLEVEL`/`MAXLEVEL` (and `REF_RATIO`) -> AMReX `amr.*` |
| `avvia.sh` | build + run in background + status after 60 s |
| `film.sh` | run up to `TMAX` and then 2D/3D videos of the refinement |
| `livelli.sh` | level convention (sourced by the launchers) |

Level convention (same as the 2D solver): level L -> dx = 2^-L,
`n_cell = 2^MINLEVEL 2^MINLEVEL 4*2^MINLEVEL`, one AMR level refines by
`REF_RATIO` (default 4).

```sh
./run.sh 4 6                          # base 16x16x64, one level x4 -> 64x64x256
VORTERR="0.05" ./run.sh 4 6           # refinement threshold on |omega|
STOP_TIME=20 MAX_STEP=200 ./run.sh    # short run
./avvia.sh 4 6                        # build + launch + status
TMAX=20 ./film.sh 4 6                 # run + videos
```

## Astromenti

- `strumenti/video_amr_plotfile.py` - 2D slice video from AMReX plotfiles
- `strumenti/video_amr_3d.py` - rotating 3D view (AMR boxes + |omega| surface)
- `strumenti/error_norm_amr3d.py` - L_inf/L2 between two AMR runs on a common grid
- `strumenti/heatmap_adattiva_3d.sh` - error heatmap (MAXLEVEL x 1/VORTERR)

```sh
TMAX=10 LMIN=4 LMAX=7 OUT=risultati/heatmap_T10 ./strumenti/heatmap_adattiva_3d.sh
```

## Risultati

`risultati/` (non versionata): run dirs `run_M<min>-<max>_T<TMAX>/` con
plotfile/PNG/MP4, `heatmap_adattiva_3d/` con heatmap e .dat.

`incflo_patches.diff` documenta le modifiche al codice incflo (probtype 1001
"vortex tubes", tagging su |omega|, budget celle).
