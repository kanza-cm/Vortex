#!/usr/bin/env python3
"""Volume animation with a retry per frame and a detailed log; it uses the
scene and camera of visualizza3d.py. Usage: python -u anim_volume.py"""
import glob
import os
import sys
import time

import pyvista as pv

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
import visualizza3d as v

FS = sorted(glob.glob("vtk3d/vtk3d_*.vti") + glob.glob("vtk3d_*.vti"))
print("frame totali:", len(FS), flush=True)

pl = pv.Plotter(off_screen=True, window_size=(1280, 720))
pl.open_movie("omega3d_volume.mp4", framerate=2)

nok = 0
for f in FS:
    t = float(os.path.splitext(os.path.basename(f))[0].split("_")[-1])
    g = None
    for att in range(3):
        try:
            g = pv.read(f)
            break
        except Exception as e:
            print("  retry %s: %s" % (f, e), flush=True)
            time.sleep(2)
    if g is None:
        print("  SALTO", f, flush=True)
        continue
    try:
        pl.clear()
        v.scena(pl, g, "omega_mag", False, t)
        v.camera(pl, "obliqua")
        pl.write_frame()
        nok += 1
        print("  %s (t=%g) ok  [%d/%d]" % (f, t, nok, len(FS)), flush=True)
    except Exception as e:
        print("  ERRORE render %s: %s" % (f, e), flush=True)

pl.close()
print("fatto:", nok, "frame", flush=True)
