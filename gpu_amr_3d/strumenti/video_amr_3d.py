#!/usr/bin/env python3
"""3D video of the AMR refinement from incflo plotfiles (stdlib + ffmpeg only).

Draws the AMR boxes (wireframe, cyan = level 1), the domain (white), and an
|omega| isosurface approximated by a colored point cloud (orthographic
camera that rotates during the video).

Usage: video_amr_3d.py <dir_with_plt*> <out.mp4> [--n 640] [--thr 0.2] [--fps 5]
"""
import os, sys, math, argparse, tempfile, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_amr_plotfile import read_header, read_offsets, read_fab_comp, cmap

def load_levels(H):
    icomp = H["varnames"].index("vort") if "vort" in H["varnames"] else 0
    out = []
    for L in H["levels"]:
        ddir = os.path.dirname(L["path"])
        offs = read_offsets(os.path.join(ddir, "Cell_H"))
        fabs = []
        for (fn, off), (blo, bhi) in zip(offs, L["boxes"]):
            ilo = [int(round((blo[d] - H["problo"][d]) / L["dx"][d])) for d in range(3)]
            ihi = [int(round((bhi[d] - H["problo"][d]) / L["dx"][d])) - 1 for d in range(3)]
            nx = ihi[0] - ilo[0] + 1; ny = ihi[1] - ilo[1] + 1; nz = ihi[2] - ilo[2] + 1
            data = read_fab_comp(os.path.join(ddir, fn), off, icomp, nx, ny, nz)
            fabs.append((ilo, ihi, nx, ny, nz, data))
        out.append(dict(lev=L["lev"], dx=L["dx"], fabs=fabs, boxes=L["boxes"]))
    return out

def line(img, n, x0, y0, x1, y1, col):
    x0 = int(x0); y0 = int(y0); x1 = int(x1); y1 = int(y1)
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < n and 0 <= y0 < n:
            k = 3 * (y0 * n + x0)
            img[k], img[k+1], img[k+2] = col
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy

def dot(img, n, x, y, col):
    for oy in (0, 1):
        for ox in (0, 1):
            px = int(x) + ox; py = int(y) + oy
            if 0 <= px < n and 0 <= py < n:
                k = 3 * (py * n + px)
                img[k], img[k+1], img[k+2] = col

def render_frame3d(pf, n, thr, phi_deg, elev_deg, outppm):
    H = read_header(pf)
    problo, probhi = H["problo"], H["probhi"]
    center = [(problo[d] + probhi[d]) / 2.0 for d in range(3)]
    phi = math.radians(phi_deg); el = math.radians(elev_deg)
    cp, sp = math.cos(phi), math.sin(phi); ce, se = math.cos(el), math.sin(el)

    def proj(x, y, z):
        rx = x - center[0]; ry = y - center[1]; rz = z - center[2]
        xr = rx * cp - ry * sp
        yr = rx * sp + ry * cp
        yv = yr * ce - rz * se
        zv = yr * se + rz * ce
        return xr, yv, zv

    corners = [(x, y, z) for x in (problo[0], probhi[0])
               for y in (problo[1], probhi[1]) for z in (problo[2], probhi[2])]
    scr = [proj(*c) for c in corners]
    xs = [p[0] for p in scr]; ys = [p[1] for p in scr]; ds = [p[2] for p in scr]
    span = max(max(xs) - min(xs), max(ys) - min(ys))
    margin = 34
    scale = (n - 2 * margin) / span
    ox = n / 2.0 - scale * (max(xs) + min(xs)) / 2.0
    oy = n / 2.0 + scale * (max(ys) + min(ys)) / 2.0
    dmin, dmax = min(ds), max(ds)
    dspan = (dmax - dmin) or 1.0

    def to_px(x, y, z):
        sx, sy, dz = proj(x, y, z)
        return ox + scale * sx, oy - scale * sy, (dz - dmin) / dspan

    img = bytearray(n * n * 3)
    for k in range(0, n * n * 3, 3):
        img[k] = 4; img[k+1] = 4; img[k+2] = 16

    # domain (faint white wireframe)
    def box_edges(lo, hi, col):
        c = [(lo[0], lo[1], lo[2]), (hi[0], lo[1], lo[2]), (hi[0], hi[1], lo[2]), (lo[0], hi[1], lo[2]),
             (lo[0], lo[1], hi[2]), (hi[0], lo[1], hi[2]), (hi[0], hi[1], hi[2]), (lo[0], hi[1], hi[2])]
        p = [to_px(*q) for q in c]
        for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
            line(img, n, p[a][0], p[a][1], p[b][0], p[b][1], col)
    box_edges(problo, probhi, (140, 140, 160))

    # |omega| point cloud (one point per cell, finest available level)
    pts = []
    for L in load_levels(H):
        dx, dy, dz = L["dx"]
        for ilo, ihi, nx, ny, nz, data in L["fabs"]:
            for kk in range(nz):
                zc = problo[2] + (ilo[2] + kk + 0.5) * dz
                base = kk * nx * ny
                for jj in range(ny):
                    yc = problo[1] + (ilo[1] + jj + 0.5) * dy
                    b2 = base + jj * nx
                    for ii in range(nx):
                        v = data[b2 + ii]
                        if v < thr:
                            continue
                        xc = problo[0] + (ilo[0] + ii + 0.5) * dx
                        px, py, dep = to_px(xc, yc, zc)
                        pts.append((dep, px, py, v))
    lmin = math.log10(1e-4); lmax = math.log10(1.5)
    pts.sort(key=lambda p: p[0])
    for dep, px, py, v in pts:
        t = (math.log10(max(v, 1e-4)) - lmin) / (lmax - lmin)
        c = cmap(t)
        shade = 0.45 + 0.55 * dep
        dot(img, n, px, py, (int(c[0]*shade), int(c[1]*shade), int(c[2]*shade)))

    # AMR boxes at level >= 1 (cyan = L1, yellow = L2)
    cmapbox = {1: (0, 255, 255), 2: (255, 255, 0), 3: (255, 0, 255)}
    for L in H["levels"]:
        if L["lev"] < 1:
            continue
        for blo, bhi in L["boxes"]:
            box_edges(blo, bhi, cmapbox.get(L["lev"], (255, 255, 255)))

    with open(outppm, "wb") as f:
        f.write(b"P6\n%d %d\n255\n" % (n, n))
        f.write(bytes(img))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pldir"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=640)
    ap.add_argument("--thr", type=float, default=0.2)
    ap.add_argument("--fps", type=int, default=5)
    a = ap.parse_args()
    plts = [os.path.join(a.pldir, d) for d in os.listdir(a.pldir)
            if d.startswith("plt") and os.path.isdir(os.path.join(a.pldir, d))]
    plts.sort(key=lambda p: read_header(p)["time"])
    print("%d plotfile, t %.3f -> %.3f" % (len(plts), read_header(plts[0])["time"],
                                           read_header(plts[-1])["time"]), flush=True)
    tmp = tempfile.mkdtemp(prefix="amr3d_")
    for k, pf in enumerate(plts):
        frac = k / max(1, len(plts) - 1)
        render_frame3d(pf, a.n, a.thr, 18 + 55 * frac, 24, os.path.join(tmp, "f_%05d.ppm" % k))
        if k % 5 == 0:
            print("  frame %2d/%2d  t=%.2f" % (k, len(plts), read_header(pf)["time"]), flush=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
           "-i", os.path.join(tmp, "f_%05d.ppm"), "-c:v", "libx264", "-crf", "18",
           "-pix_fmt", "yuv420p", "-vf",
           "scale=trunc(iw/2)*2:trunc(ih/2)*2", a.out]
    subprocess.run(cmd, check=True)
    print("video:", a.out)
    print("frames in", tmp)

if __name__ == "__main__":
    main()
