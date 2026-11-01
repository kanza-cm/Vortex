#!/usr/bin/env python3
"""3D visualization of the vorticity for the saetta3d.c solver.

Reads the vtk3d_*.vti dumps (VTK ImageData) and shows the field with
volume rendering (default) or isosurfaces.

Usage:
  python3 visualizza3d.py                 # interactive window, last dump
  python3 visualizza3d.py vtk3d/vtk3d_0050.000.vti  # one specific file
  python3 visualizza3d.py --png out.png   # PNG frame (off-screen)
  python3 visualizza3d.py --anim          # omega3d_volume.mp4 from
                                          # all the vtk3d_*.vti

Options:
  --scalare NOME  omega_mag (default) or omega_z
  --iso           isosurfaces instead of volume rendering
  --vista ASSE    camera: x, y, z, oblique (default: oblique)
"""
import argparse
import glob
import os
import sys

import pyvista as pv


def trova_file(nome):
    if nome:
        if os.path.exists(nome):
            return [nome]
        alt = os.path.join("vtk3d", nome)
        return [alt] if os.path.exists(alt) else [nome]
    fs = sorted(glob.glob("vtk3d/vtk3d_*.vti") + glob.glob("vtk3d_*.vti"))
    if not fs:
        sys.exit("nessun vtk3d_*.vti trovato: esegui prima la simulazione "
                 "(saetta3d.c scrive un dump ogni VTK_TSTEP)")
    return fs


def scena(pl, griglia, scalare, iso, t=None):
    vmax = float(griglia[scalare].max())
    if iso:
        # three isosurface levels: the vortex tubes
        for frac, col in [(0.15, "#4c9be8"), (0.45, "#ffb347"),
                          (0.80, "#e84c3d")]:
            sup = griglia.contour(isosurfaces=[frac*vmax], scalars=scalare)
            pl.add_mesh(sup, color=col, opacity=0.65, smooth_shading=True)
    else:
        pl.add_volume(griglia, scalars=scalare, cmap="inferno",
                      clim=(0., 0.5*vmax), opacity="sigmoid",
                      log_scale=False, show_scalar_bar=True,
                      scalar_bar_args={"title": scalare})
    if t is not None:
        pl.add_text("t = %.0f" % t, position="upper_left", font_size=14)


def camera(pl, vista):
    # the tubes are aligned with z: a natural view along y
    if vista == "x":
        pl.view_yz()
    elif vista == "y":
        pl.view_xz()
    elif vista == "z":
        pl.view_xy()
    else:
        pl.view_xz()
        pl.camera.azimuth = 20
        pl.camera.elevation = 10
    pl.reset_camera()
    pl.camera.zoom(0.92)


def main():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("file", nargs="?", default=None)
    p.add_argument("--scalare", default="omega_mag",
                   choices=["omega_mag", "omega_z"])
    p.add_argument("--iso", action="store_true")
    p.add_argument("--anim", action="store_true")
    p.add_argument("--png", default=None)
    p.add_argument("--vista", default="obliqua",
                   choices=["x", "y", "z", "obliqua"])
    a = p.parse_args()

    if a.anim:
        fs = trova_file(None)
        print("creo omega3d_volume.mp4 da %d fotogrammi ..." % len(fs))
        pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
        pl.open_movie("omega3d_volume.mp4", framerate=2)
        for f in fs:
            t = float(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
            g = pv.read(f)
            pl.clear()
            scena(pl, g, a.scalare, a.iso, t)
            camera(pl, a.vista)
            pl.write_frame()
            print("  %s (t=%g) ok" % (f, t))
        pl.close()
        print("fatto: omega3d_volume.mp4")
        return

    f = trova_file(a.file)[-1]
    g = pv.read(f)
    print("%s: %s, %d celle" % (f, g.n_points, g.n_cells))
    pl = pv.Plotter(off_screen=bool(a.png), window_size=(1280, 720))
    scena(pl, g, a.scalare, a.iso)
    camera(pl, a.vista)
    pl.add_axes()
    if a.png:
        pl.screenshot(a.png)
        print("salvato", a.png)
    else:
        pl.show()


if __name__ == "__main__":
    main()
