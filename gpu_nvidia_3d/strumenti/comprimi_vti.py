#!/usr/bin/env python3
"""Repacks the vtk3d_*.vti dumps with zlib compression (standard VTI
format, lossless: the data does not change, only the container).

Usage:
    python3 comprimi_vti.py [file.vti ...]     # default: tutti i vtk3d_*.vti

For each file: read, rewrite compressed into a .tmp, check that the data
is identical bit by bit, then replace the original.
"""
import glob
import os
import sys

import numpy as np
import vtk
from vtk.util import numpy_support


def comprimi(fn, livello=9):
    r = vtk.vtkXMLImageDataReader()
    r.SetFileName(fn)
    r.Update()
    im = r.GetOutput()

    orig = {}
    pd = im.GetPointData()
    for k in range(pd.GetNumberOfArrays()):
        orig[pd.GetArrayName(k)] = numpy_support.vtk_to_numpy(pd.GetArray(k))

    tmp = fn + ".tmp"
    w = vtk.vtkXMLImageDataWriter()
    w.SetFileName(tmp)
    w.SetInputData(im)
    w.SetCompressorTypeToZLib()
    w.SetCompressionLevel(livello)
    w.EncodeAppendedDataOff()
    w.Write()

    r2 = vtk.vtkXMLImageDataReader()
    r2.SetFileName(tmp)
    r2.Update()
    pd2 = r2.GetOutput().GetPointData()
    for name, a in orig.items():
        b = numpy_support.vtk_to_numpy(pd2.GetArray(name))
        if a.shape != b.shape or not np.array_equal(a, b):
            sys.exit("errore: %s non lossless su %s" % (name, fn))

    s0, s1 = os.path.getsize(fn), os.path.getsize(tmp)
    os.replace(tmp, fn)
    print("%s: %.0f -> %.0f MB  (%.2fx)" % (fn, s0/1e6, s1/1e6, s0/s1), flush=True)


if __name__ == "__main__":
    fs = sys.argv[1:] or sorted(glob.glob("vtk3d/vtk3d_*.vti")
                                + glob.glob("vtk3d_*.vti"))
    for f in fs:
        comprimi(f)
