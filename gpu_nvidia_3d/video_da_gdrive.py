#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_da_gdrive.py
==================

Builds the 3D animation (omega3d_volume.mp4) reading the VTI directly
from Google Drive, one file at a time: download the VTI, render a PNG,
delete the local VTI. The original files stay on Drive.

Meant for long runs: locally there are only the VTI in use (~0.5 GB) and
the PNG frames (a few MB each), not hundreds of GB of dumps.

Usage (from the gpu_nvidia_3d folder):
    ~/.venvs/vis/bin/python3 video_da_gdrive.py --watch
    # or only the files already on Drive:
    ~/.venvs/vis/bin/python3 video_da_gdrive.py
    # quality/memory:
    ... --stride 4 --iso --fps 10
    # different folder/run:
    ... --folder-name saetta3d_anim_T1500 --end 1500

Authentication: uses credenziali/token_gdrive.json (created by
carica_vti_gdrive.py --auth-only). If it is missing, run first:
    python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json --auth-only

Output: risultati/video/omega3d_volume.mp4, frame PNG in
risultati/video/frames/.

GPU note: the simulation uses almost all the VRAM. The rendering shrinks the
volume with --stride (default 2); if it runs out of memory use --stride 4 or
--iso, or build the video with the simulation stopped.
"""

import argparse
import glob
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import carica_vti_gdrive as drive
import imageio.v2 as iio
import numpy as np
import pyvista as pv
import visualizza3d as vis

VERBOSE = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def vlog(msg):
    if VERBOSE:
        log(msg)


def download(auth, file_id, dest, size=None):
    """Downloads a Drive file in streaming (no whole file in RAM)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    url = "%s/files/%s?alt=media&supportsAllDrives=true" % (drive.DRIVE, file_id)
    attempt = 0
    while True:
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": "Bearer " + auth.access_token()})
            with urllib.request.urlopen(req, timeout=600) as r, \
                    open(part, "wb") as f:
                while True:
                    chunk = r.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            os.replace(part, dest)
            got = os.path.getsize(dest)
            if size is not None and got != size:
                raise RuntimeError("dimensione %d != %d (Drive)" % (got, size))
            return
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code == 401 and attempt == 0:
                attempt += 1
                log("token scaduto/revocato: provo refresh")
                auth._refresh()
                continue
            if e.code in (429, 500, 502, 503, 504):
                attempt += 1
                wait = min(60, 2 ** min(attempt, 6))
                log("HTTP %d, riprovo tra %d s" % (e.code, wait))
                time.sleep(wait)
                continue
            raise RuntimeError("download HTTP %d: %s"
                               % (e.code, body.decode(errors="replace")[:300]))
        except (urllib.error.URLError, TimeoutError,
                ConnectionError, OSError) as e:
            attempt += 1
            wait = min(60, 2 ** min(attempt, 6))
            log("rete (%s), riprovo tra %d s" % (e, wait))
            time.sleep(wait)
            continue


def render(vti, png, t, args):
    g = pv.read(vti)
    if args.stride > 1:
        # Block averaging done in numpy: vtkImageShrink3D
        # would rename or drop omega_z (it keeps only the first array).
        dims = g.dimensions
        spacing = tuple(g.spacing)
        origin = tuple(g.origin)
        a = np.array(g[args.scalare], copy=True)
        del g
        nx, ny, nz = dims
        s = args.stride
        cx, cy, cz = nx // s, ny // s, nz // s
        a3 = a.reshape((nx, ny, nz), order="F")[:cx*s, :cy*s, :cz*s]
        b = np.zeros((cx, cy, cz), dtype=a3.dtype)
        for dx in range(s):
            for dy in range(s):
                for dz in range(s):
                    b += a3[dx::s, dy::s, dz::s]
        b /= float(s * s * s)
        g = pv.ImageData(dimensions=(cx, cy, cz))
        g.spacing = (spacing[0]*s, spacing[1]*s, spacing[2]*s)
        g.origin = origin
        g[args.scalare] = b.ravel(order="F")
        del a, a3, b
    pl = pv.Plotter(off_screen=True, window_size=args.size)
    try:
        vis.scena(pl, g, args.scalare, args.iso, t)
        vis.camera(pl, args.vista)
        pl.add_axes()
        pl.screenshot(png)
    finally:
        pl.close()
    del g


def frame_name(t):
    return "f_%08.3f.png" % t


def frame_time(path):
    try:
        return float(os.path.basename(path)[2:-4])
    except ValueError:
        return None


def list_vti(auth, folder_id, end_t, shared):
    remote = drive.list_remote(auth, folder_id, shared)
    out = []
    for name, meta in remote.items():
        if not name.endswith(".vti"):
            continue
        try:
            t = float(os.path.splitext(name)[0].split("_")[-1])
        except ValueError:
            continue
        if end_t is not None and t > end_t + 1e-9:
            continue
        out.append((t, name, meta))
    out.sort(key=lambda x: x[0])
    return out


def assemble(args):
    frames = sorted(glob.glob(os.path.join(args.frames, "f_*.png")))
    if not frames:
        log("nessun frame da montare")
        return
    log("monto %s da %d fotogrammi (%.1f fps)..."
        % (args.video, len(frames), args.fps))
    w = iio.get_writer(args.video, fps=args.fps)
    try:
        for i, f in enumerate(frames):
            img = iio.imread(f)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            w.append_data(img)
            if (i + 1) % 100 == 0:
                log("  %d/%d" % (i + 1, len(frames)))
    finally:
        w.close()
    log("video pronto: %s (%.1f MB)"
        % (args.video, os.path.getsize(args.video) / 1e6))
    if args.clean_frames:
        for f in frames:
            os.remove(f)
        log("frame PNG rimossi (--clean-frames)")


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="genera il video 3D scaricando i VTI da Google Drive "
                    "uno alla volta (cancella solo la copia locale)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--folder-name", default="saetta3d_anim_T1500")
    ap.add_argument("--folder-id", default=None)
    ap.add_argument("--token", default=drive.DEFAULT_TOKEN,
                    help="token OAuth "
                         "(default: credenziali/token_gdrive.json)")
    ap.add_argument("--shared-drive", action="store_true")
    ap.add_argument("--end", type=float, default=1500.,
                    help="ultimo tempo da includere (default: 1500)")
    ap.add_argument("--watch", action="store_true",
                    help="aspetta i nuovi VTI mentre la simulazione li carica")
    ap.add_argument("--poll", type=float, default=30.,
                    help="secondi tra un controllo e l'altro con --watch")
    ap.add_argument("--idle-timeout", type=float, default=3600.,
                    help="con --watch, esci dopo N secondi senza file nuovi")
    ap.add_argument("--stride", type=int, default=2,
                    help="riduzione del volume prima del render (default: 2)")
    ap.add_argument("--iso", action="store_true",
                    help="isosuperfici invece del volume rendering")
    ap.add_argument("--scalare", default="omega_mag",
                    choices=["omega_mag", "omega_z"])
    ap.add_argument("--vista", default="obliqua",
                    choices=["x", "y", "z", "obliqua"])
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=float, default=10.)
    ap.add_argument("--frames",
                    default=os.path.join(BASE, "risultati", "video", "frames"))
    ap.add_argument("--tmp",
                    default=os.path.join(BASE, "risultati", "video", "tmp"))
    ap.add_argument("--video",
                    default=os.path.join(BASE, "risultati", "video",
                                         "omega3d_volume.mp4"))
    ap.add_argument("--overwrite", action="store_true",
                    help="ri-renderizza anche i frame gia' presenti")
    ap.add_argument("--no-video", action="store_true",
                    help="non montare il video, solo i PNG")
    ap.add_argument("--clean-frames", action="store_true",
                    help="cancella i PNG dopo aver montato il video")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    VERBOSE = args.verbose
    try:
        w, h = args.size.lower().split("x")
        args.size = (int(w), int(h))
    except ValueError:
        ap.error("--size deve essere LARGHEZZAxALTEZZA, es. 1280x720")
    if args.stride < 1:
        ap.error("--stride deve essere >= 1")
    if not os.path.exists(args.token):
        sys.exit("token OAuth non trovato: %s\n"
                 "Esegui prima: python3 carica_vti_gdrive.py "
                 "--client-secrets client_secret.json --auth-only" % args.token)

    auth = drive.GoogleAuth(None, args.token, non_interactive=True)
    folder_id, folder_name = drive.ensure_folder(
        auth, args.folder_id, args.folder_name, args.shared_drive,
        create=False)
    if not folder_id:
        sys.exit("cartella Drive '%s' non trovata" % args.folder_name)
    log("cartella Drive: %s (id %s)" % (folder_name, folder_id))

    os.makedirs(args.frames, exist_ok=True)
    os.makedirs(args.tmp, exist_ok=True)
    processed = set()
    if not args.overwrite:
        for png in glob.glob(os.path.join(args.frames, "f_*.png")):
            t = frame_time(png)
            if t is not None and os.path.getsize(png) > 0:
                processed.add(t)
        if processed:
            log("%d frame gia' presenti, li salto (--overwrite per rifarli)"
                % len(processed))

    idle = 0.
    while True:
        files = list_vti(auth, folder_id, args.end, args.shared_drive)
        todo = [f for f in files if f[0] not in processed]
        for t, name, meta in todo:
            png = os.path.join(args.frames, frame_name(t))
            if os.path.exists(png) and not args.overwrite:
                processed.add(t)
                continue
            vti = os.path.join(args.tmp, name)
            size = int(meta["size"]) if meta.get("size") else None
            if os.path.exists(vti) and size is not None and \
                    os.path.getsize(vti) == size:
                vlog("uso il VTI gia' scaricato: %s" % name)
            else:
                log("scarico %s (%.1f MB)"
                    % (name, (size or 0) / 1e6))
                download(auth, meta["id"], vti, size)
            log("rendo t=%g -> %s" % (t, os.path.basename(png)))
            try:
                render(vti, png, t, args)
            except Exception as e:
                log("ERRORE render t=%g: %s" % (t, e))
                log("il VTI resta in %s per riprovare" % vti)
                raise
            os.remove(vti)
            processed.add(t)
            log("ok t=%g (%d file pronti)" % (t, len(processed)))

        if not args.watch:
            break
        if args.end is not None and processed and max(processed) >= args.end - 1e-9:
            log("raggiunto T = %g" % args.end)
            break
        if not todo:
            idle += args.poll
            if idle >= args.idle_timeout:
                log("nessun file nuovo da %.0f s: mi fermo" % idle)
                break
            vlog("attendo nuovi VTI (%.0f s)..." % args.poll)
        else:
            idle = 0.
        time.sleep(args.poll)

    if not args.no_video:
        assemble(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrotto", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print("errore: %s" % e, file=sys.stderr)
        sys.exit(1)
