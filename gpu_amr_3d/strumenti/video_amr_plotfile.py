#!/usr/bin/env python3
"""AMR refinement video from incflo plotfiles (AMReX), no dependencies.

Usage:  video_amr_plotfile.py <dir_with_plt*> <out.mp4> [--z 2.0] [--n 512] [--fps 8]

Reads the AMReX plotfiles (Header + VisMF Cell_H/Cell_D_*), draws a slice
(vort = |omega|) with the AMR box outlines, and encodes it with ffmpeg.
"""
import os, re, sys, math, argparse, tempfile, subprocess
from array import array

BOX_RE = re.compile(r"\(\((-?\d+),(-?\d+),(-?\d+)\)\s*\((-?\d+),(-?\d+),(-?\d+)\)"
                    r"\s*\((-?\d+),(-?\d+),(-?\d+)\)\)")

def read_header(pf):
    lines = open(os.path.join(pf, "Header")).read().splitlines()
    pos = [0]
    def nxt():
        v = lines[pos[0]]; pos[0] += 1; return v
    nxt()                                   # version
    ncomp = int(nxt())
    varnames = [nxt() for _ in range(ncomp)]
    spacedim = int(nxt())
    time = float(nxt())
    finest = int(nxt())
    problo = [float(x) for x in nxt().split()]
    probhi = [float(x) for x in nxt().split()]
    nxt()                                   # ref_ratio
    nxt()                                   # domains per level (all on one line)
    nxt()                                   # level_steps
    dx = []
    for lev in range(finest + 1):
        dx.append([float(x) for x in nxt().split()])
    nxt(); nxt()                            # coord sys, "0"
    levels = []
    for lev in range(finest + 1):
        p = nxt().split()
        nboxes = int(p[1])
        nxt()                               # level_steps[lev]
        boxes = []
        for b in range(nboxes):
            lo = []; hi = []
            for d in range(spacedim):
                a, bb = nxt().split()
                lo.append(float(a)); hi.append(float(bb))
            boxes.append((lo, hi))
        mfpath = nxt()
        levels.append(dict(lev=lev, boxes=boxes, dx=dx[lev],
                           path=os.path.join(pf, mfpath)))
    return dict(time=time, finest=finest, problo=problo, probhi=probhi,
                varnames=varnames, ncomp=ncomp, levels=levels)

def read_offsets(cell_h):
    offs = []
    for ln in open(cell_h).read().splitlines():
        m = re.match(r"\s*FabOnDisk:\s*(\S+)\s+(\d+)", ln)
        if m:
            offs.append((m.group(1), int(m.group(2))))
    return offs

def read_fab_comp(path, offset, comp, nx, ny, nz):
    n = nx * ny * nz
    with open(path, "rb") as f:
        f.seek(offset)
        while True:                          # ASCII FAB header up to '\n'
            c = f.read(1)
            if not c:
                raise RuntimeError("plotfile troncato")
            if c == b"\n":
                break
        f.seek(comp * n * 4, 1)              # components are contiguous
        a = array("f")
        a.fromfile(f, n)
    return a

def rect(img, n, x0, y0, x1, y1, color):
    x0 = max(0, min(n - 1, int(x0))); x1 = max(0, min(n - 1, int(x1)))
    y0 = max(0, min(n - 1, int(y0))); y1 = max(0, min(n - 1, int(y1)))
    for px in range(x0, x1 + 1):
        for py in (y0, y1):
            k = 3 * (py * n + px)
            img[k], img[k+1], img[k+2] = color
    for py in range(y0, y1 + 1):
        for px in (x0, x1):
            k = 3 * (py * n + px)
            img[k], img[k+1], img[k+2] = color

def cmap(t):
    t = 0.0 if t < 0 else (1.0 if t > 1 else t)
    if t < 0.25:
        a = t / 0.25; return (0, 0, int(150 * a))
    if t < 0.5:
        a = (t - 0.25) / 0.25; return (0, int(170 * a), int(150 + 105 * a))
    if t < 0.75:
        a = (t - 0.5) / 0.25; return (int(255 * a), int(170 + 85 * a), int(255 - 155 * a))
    a = (t - 0.75) / 0.25; return (255, int(255 - 160 * a), int(100 - 100 * a))

def render_frame(pf, z, n, outppm):
    H = read_header(pf)
    problo, probhi = H["problo"], H["probhi"]
    vname = "vort" if "vort" in H["varnames"] else H["varnames"][0]
    icomp = H["varnames"].index(vname)
    x0, x1 = problo[0], probhi[0]; y0, y1 = problo[1], probhi[1]
    spx = (x1 - x0) / n; spy = (y1 - y0) / n
    vals = [0.0] * (n * n)
    outlines = []
    for L in H["levels"]:
        dx, dy, dz = L["dx"]
        hdr = os.path.join(os.path.dirname(L["path"]), "Cell_H")
        offs = read_offsets(hdr)
        ddir = os.path.dirname(L["path"])
        kz = int(math.floor((z - problo[2]) / dz))
        for (fn, off), (blo, bhi) in zip(offs, L["boxes"]):
            ilo = [int(round((blo[d] - problo[d]) / L["dx"][d])) for d in range(3)]
            ihi = [int(round((bhi[d] - problo[d]) / L["dx"][d])) - 1 for d in range(3)]
            px0 = (problo[0] + ilo[0] * dx - x0) / spx
            px1 = (problo[0] + (ihi[0] + 1) * dx - x0) / spx
            py0 = (problo[1] + ilo[1] * dy - y0) / spy
            py1 = (problo[1] + (ihi[1] + 1) * dy - y0) / spy
            if L["lev"] >= 1:
                outlines.append((L["lev"], px0, py0, px1, py1))
            if not (ilo[2] <= kz <= ihi[2]):
                continue
            nx = ihi[0] - ilo[0] + 1; ny = ihi[1] - ilo[1] + 1; nz = ihi[2] - ilo[2] + 1
            data = read_fab_comp(os.path.join(ddir, fn), off, icomp, nx, ny, nz)
            kk = kz - ilo[2]
            ipx0 = max(0, int(math.ceil(px0 - 0.5))); ipx1 = min(n - 1, int(math.floor(px1 - 0.5)))
            ipy0 = max(0, int(math.ceil(py0 - 0.5))); ipy1 = min(n - 1, int(math.floor(py1 - 0.5)))
            for py in range(ipy0, ipy1 + 1):
                yc = y0 + (py + 0.5) * spy
                j = int(math.floor((yc - problo[1]) / dy)) - ilo[1]
                if j < 0 or j >= ny:
                    continue
                base = nx * ny * kk + nx * j
                row = py * n
                for px in range(ipx0, ipx1 + 1):
                    xc = x0 + (px + 0.5) * spx
                    i = int(math.floor((xc - problo[0]) / dx)) - ilo[0]
                    if 0 <= i < nx:
                        vals[row + px] = data[base + i]
    img = bytearray(n * n * 3)
    lmin = math.log10(1e-4); lmax = math.log10(1.5)
    for py in range(n):
        row = py * n
        for px in range(n):
            v = vals[row + px]
            t = (math.log10(max(v, 1e-4)) - lmin) / (lmax - lmin)
            c = cmap(t)
            k = 3 * (row + px)
            img[k], img[k+1], img[k+2] = c
    colors = {1: (0, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255)}
    for lev, px0, py0, px1, py1 in outlines:
        rect(img, n, px0, py0, px1, py1, colors.get(lev, (255, 255, 255)))
    rect(img, n, 0, 0, n - 1, n - 1, (255, 255, 255))
    with open(outppm, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (n, n))
        f.write(bytes(img))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pldir"); ap.add_argument("out")
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    plts = [os.path.join(a.pldir, d) for d in os.listdir(a.pldir)
            if d.startswith("plt") and os.path.isdir(os.path.join(a.pldir, d))]
    if not plts:
        sys.exit("nessun plotfile in " + a.pldir)
    plts.sort(key=lambda p: read_header(p)["time"])
    print("%d plotfile, t da %.3f a %.3f" %
          (len(plts), read_header(plts[0])["time"], read_header(plts[-1])["time"]))
    tmp = tempfile.mkdtemp(prefix="amrvid_")
    for k, pf in enumerate(plts):
        out = os.path.join(tmp, "f_%05d.ppm" % k)
        render_frame(pf, a.z, a.n, out)
        if k % 10 == 0:
            print("  frame %3d/%3d  t=%.3f  %s" %
                  (k, len(plts), read_header(pf)["time"], os.path.basename(pf)), flush=True)
    cmd = ["ffmpeg", "-y", "-framerate", str(a.fps), "-i", os.path.join(tmp, "f_%05d.ppm"),
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", a.out]
    print("ffmpeg ...", flush=True)
    subprocess.run(cmd, check=True)
    # previews
    outdir = os.path.dirname(a.out) or "."
    stem = os.path.splitext(os.path.basename(a.out))[0].replace("_raw", "").replace("_2d", "")
    for idx in (0, len(plts) // 2, len(plts) - 1):
        src = os.path.join(tmp, "f_%05d.ppm" % idx)
        dst = os.path.join(outdir, "video%s_frame_%05d.png" % (stem, idx))
        subprocess.run(["ffmpeg", "-y", "-i", src, dst], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("video:", a.out)

if __name__ == "__main__":
    main()
