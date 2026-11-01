#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carica_vti_gdrive.py
====================

Uploads the .vti dumps written by saetta3d.c to Google Drive (folder
VTK_DIR, default "vtk3d"). Meant to run next to the simulation: it can
stay listening (--watch) and upload the files as they appear, skipping
the ones already on Drive.

Uses only the Python standard library (no pip): authentication is OAuth 2.0 with a loopback redirect and PKCE.

--------------------------------------------------------------------------
SETUP (once)
--------------------------------------------------------------------------
1. On https://console.cloud.google.com/ create a project and enable
   "Google Drive API" (APIs & Services -> Library).
2. Configure the OAuth consent screen (type "External"; if the app is not   verified, add your account under "Test users").
3. APIs & Services -> Credentials -> Create credentials
   -> OAuth client ID -> Application type: "Desktop app".
4. Download the JSON and save it as credenziali/client_secret.json in the
   project folder. On the first run the program prints a URL: open it in
   browser, authorize, and the token is saved to
   credenziali/token_gdrive.json
   (mode 600). Later runs do not need the browser.

--------------------------------------------------------------------------
EXAMPLES
--------------------------------------------------------------------------
# upload the .vti in ./vtk3d to the Drive folder "saetta3d_vti"
# (created if missing), once
python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json

# stay listening, upload a file once it has been quiet for 10 s, then
# delete it locally (useful to keep the disk from filling)
python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json \\
    --watch 60 --delete

# use an existing Drive folder (ID taken from the folder URL)
python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json \\
    --folder-id 1AbCdEfGhIjKlMnOpQrStUvWxYz

# dry run, nothing is written to Drive
python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json --dry-run

# only authorize and save the token (once, before the solver)
python3 carica_vti_gdrive.py --client-secrets credenziali/client_secret.json --auth-only

# upload a single file (this is how saetta3d.c calls it:
# -DVTK=1 -DVTK_UPLOAD=1; the solver waits for the upload to end)
python3 carica_vti_gdrive.py --non-interactive --file vtk3d/vtk3d_0000.000.vti
"""

import argparse
import base64
import hashlib
import http.server
import json
import os
import secrets
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOKEN = os.path.join(BASE, "credenziali", "token_gdrive.json")

SCOPE = "https://www.googleapis.com/auth/drive"
AUTH_URI = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"
DRIVE = "https://www.googleapis.com/drive/v3"
UPLOAD = "https://www.googleapis.com/upload/drive/v3"
FOLDER_MIME = "application/vnd.google-apps.folder"
CHUNK = 8 * 1024 * 1024  # must be a multiple of 256 KiB

VERBOSE = False


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def vlog(msg):
    if VERBOSE:
        log(msg)


class AuthError(RuntimeError):
    pass


class UploadSessionExpired(RuntimeError):
    pass


class _LoopbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        if "code" in params:
            self.server.auth_code = params["code"][0]
            msg = "Autorizzazione completata: puoi chiudere questa finestra."
        elif "error" in params:
            self.server.auth_error = params["error"][0]
            msg = "Autorizzazione negata: " + params["error"][0]
        else:
            msg = "Risposta inattesa."
        body = ("<html><body><p>%s</p></body></html>" % msg).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.server.done.set()

    def log_message(self, *args):
        pass


class GoogleAuth:
    """OAuth 2.0 (Desktop app) with a cached token and automatic refresh."""

    def __init__(self, client_secrets=None, token_path=DEFAULT_TOKEN,
                 no_browser=False, non_interactive=False, auth_timeout=300):
        self.token_path = token_path
        self.no_browser = no_browser
        self.non_interactive = non_interactive
        self.auth_timeout = auth_timeout
        self.client_id = None
        self.client_secret = None
        self.tokens = None

        if os.path.exists(token_path):
            with open(token_path, "r", encoding="utf-8") as f:
                self.tokens = json.load(f)
            self.client_id = self.tokens.get("client_id")
            self.client_secret = self.tokens.get("client_secret")
            vlog("token OAuth caricato da %s" % token_path)

        if client_secrets:
            self._read_client_secrets(client_secrets)
        elif not (self.client_id and self.client_secret):
            raise AuthError(
                "serve --client-secrets credenziali/client_secret.json (OAuth client "
                "di tipo Desktop app) al primo avvio")

        if not self._token_valid():
            if self.tokens and self.tokens.get("refresh_token"):
                try:
                    self._refresh()
                except Exception as e:
                    log("refresh del token fallito (%s): nuova autorizzazione" % e)
                    self.tokens = None
            if not self._token_valid():
                if self.non_interactive:
                    raise AuthError(
                        "token OAuth assente o scaduto: esegui prima "
                        "'carica_vti_gdrive.py --auth-only' in modo interattivo")
                self._authorize()

    def _read_client_secrets(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        block = data.get("installed") or data.get("web")
        if not block:
            raise AuthError("formato client_secret.json non riconosciuto")
        if "installed" not in data:
            raise AuthError(
                "il client OAuth deve essere di tipo 'Desktop app' "
                "(non 'Web application')")
        self.client_id = block["client_id"]
        self.client_secret = block["client_secret"]

    def _token_valid(self):
        return bool(self.tokens and self.tokens.get("access_token")
                    and time.time() < self.tokens.get("expires_at", 0))

    def _save(self):
        if not self.tokens:
            return
        tmp = self.token_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.tokens, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.token_path)

    def _post_form(self, data):
        body = urllib.parse.urlencode(data).encode()
        attempt = 0
        while True:
            req = urllib.request.Request(
                TOKEN_URI, data=body, method="POST",
                headers={"Content-Type": "application/x-www-form-urlencoded"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                msg = e.read().decode(errors="replace")
                if e.code in (429, 500, 502, 503, 504):
                    attempt += 1
                    wait = min(60, 2 ** min(attempt, 6))
                    log("token endpoint HTTP %d, riprovo tra %d s"
                        % (e.code, wait))
                    time.sleep(wait)
                    continue
                raise AuthError("token endpoint HTTP %d: %s" % (e.code, msg))
            except (urllib.error.URLError, TimeoutError,
                    ConnectionError, OSError) as e:
                attempt += 1
                wait = min(60, 2 ** min(attempt, 6))
                log("rete sul token endpoint (%s), riprovo tra %d s"
                    % (e, wait))
                time.sleep(wait)
                continue

    def _refresh(self):
        vlog("refresh del token OAuth")
        res = self._post_form({
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "refresh_token": self.tokens["refresh_token"],
            "grant_type": "refresh_token",
        })
        self.tokens["access_token"] = res["access_token"]
        self.tokens["expires_at"] = time.time() + res.get("expires_in", 3600) - 60
        if res.get("refresh_token"):
            self.tokens["refresh_token"] = res["refresh_token"]
        self._save()

    def _authorize(self):
        if self.non_interactive:
            raise AuthError(
                "autorizzazione interattiva disabilitata (--non-interactive)")
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")

        server = http.server.HTTPServer(("127.0.0.1", 0), _LoopbackHandler)
        server.auth_code = None
        server.auth_error = None
        server.done = threading.Event()
        port = server.server_address[1]
        redirect_uri = "http://127.0.0.1:%d/" % port
        threading.Thread(target=server.serve_forever, daemon=True).start()

        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        url = AUTH_URI + "?" + urllib.parse.urlencode(params)
        log("apri questo URL nel browser per autorizzare Google Drive:")
        print("\n" + url + "\n", flush=True)
        if not self.no_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass

        log("in attesa dell'autorizzazione (timeout %d s)..."
            % self.auth_timeout)
        if not server.done.wait(self.auth_timeout):
            server.shutdown()
            raise AuthError("timeout: autorizzazione non ricevuta")
        server.shutdown()

        if server.auth_error:
            raise AuthError("autorizzazione negata: " + server.auth_error)
        if not server.auth_code:
            raise AuthError("codice di autorizzazione non ricevuto")

        res = self._post_form({
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": server.auth_code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": redirect_uri,
        })
        self.tokens = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "access_token": res["access_token"],
            "refresh_token": res.get("refresh_token"),
            "expires_at": time.time() + res.get("expires_in", 3600) - 60,
            "scope": res.get("scope", SCOPE),
        }
        self._save()
        log("autorizzazione salvata in %s" % self.token_path)

    def access_token(self):
        if time.time() >= self.tokens.get("expires_at", 0) - 30:
            self._refresh()
        return self.tokens["access_token"]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_no_redirect = urllib.request.build_opener(NoRedirect)


def _http(auth, method, url, data=None, headers=None, opener=None,
          retries=0):
    # retries=0 -> retry forever on network errors/5xx/429
    base = dict(headers or {})
    attempt = 0
    while retries <= 0 or attempt < retries:
        headers = dict(base)
        headers["Authorization"] = "Bearer " + auth.access_token()
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method=method)
        try:
            open_ = urllib.request.urlopen if opener is None else opener.open
            with open_(req, timeout=300) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            body = e.read()
            if e.code == 401 and attempt == 0:
                attempt += 1
                log("HTTP 401: refresh del token")
                auth._refresh()
                continue
            if e.code in (429, 500, 502, 503, 504):
                attempt += 1
                wait = min(60, 2 ** min(attempt, 6))
                log("HTTP %d, riprovo tra %d s" % (e.code, wait))
                time.sleep(wait)
                continue
            raise RuntimeError("HTTP %d su %s: %s" % (
                e.code, url.split("?")[0],
                body.decode(errors="replace")[:500]))
        except (urllib.error.URLError, TimeoutError,
                ConnectionError, OSError) as e:
            attempt += 1
            wait = min(60, 2 ** min(attempt, 6))
            log("errore di rete (%s), riprovo tra %d s" % (e, wait))
            time.sleep(wait)
            continue
    raise RuntimeError("richiesta fallita dopo %d tentativi" % attempt)


def _json(auth, method, url, obj=None, retries=0):
    data = None
    headers = {}
    if obj is not None:
        data = json.dumps(obj).encode()
        headers["Content-Type"] = "application/json; charset=UTF-8"
    status, _, body = _http(auth, method, url, data=data, headers=headers,
                            retries=retries)
    return json.loads(body.decode()) if body else None


def _q(value):
    return value.replace("\\", "\\\\").replace("'", "\\'")


def ensure_folder(auth, folder_id, folder_name, shared_drive=False,
                  create=True):
    if folder_id:
        meta = _json(auth, "GET",
                     "%s/files/%s?fields=id,name,mimeType&supportsAllDrives=true"
                     % (DRIVE, folder_id))
        if meta.get("mimeType") != FOLDER_MIME:
            raise RuntimeError("'%s' non e' una cartella di Drive" % folder_id)
        return folder_id, meta.get("name", folder_id)

    params = {
        "q": "name = '%s' and mimeType = '%s' and trashed = false"
             % (_q(folder_name), FOLDER_MIME),
        "fields": "files(id,name)",
        "pageSize": "10",
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    if shared_drive:
        params["corpora"] = "allDrives"
    res = _json(auth, "GET", "%s/files?%s" % (DRIVE, urllib.parse.urlencode(params)))
    files = res.get("files", []) if res else []
    if files:
        return files[0]["id"], files[0].get("name", folder_name)
    if not create:
        return None, folder_name

    meta = _json(auth, "POST",
                 "%s/files?fields=id,name&supportsAllDrives=true" % DRIVE,
                 {"name": folder_name, "mimeType": FOLDER_MIME})
    return meta["id"], meta.get("name", folder_name)


def list_remote(auth, folder_id, shared_drive=False):
    files = {}
    token = None
    while True:
        params = {
            "q": "'%s' in parents and trashed = false" % _q(folder_id),
            "fields": "nextPageToken,files(id,name,size,mimeType,md5Checksum)",
            "pageSize": "1000",
            "supportsAllDrives": "true",
            "includeItemsFromAllDrives": "true",
        }
        if shared_drive:
            params["corpora"] = "allDrives"
        if token:
            params["pageToken"] = token
        res = _json(auth, "GET", "%s/files?%s" % (DRIVE, urllib.parse.urlencode(params)))
        for f in (res or {}).get("files", []):
            files[f["name"]] = f
        token = (res or {}).get("nextPageToken")
        if not token:
            return files


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def upload_resumable(auth, path, name, folder_id=None, file_id=None):
    size = os.path.getsize(path)
    meta = {"name": name}
    if file_id:
        method = "PATCH"
        url = ("%s/files/%s?uploadType=resumable&supportsAllDrives=true"
               "&fields=id,name,size" % (UPLOAD, file_id))
    else:
        method = "POST"
        url = ("%s/files?uploadType=resumable&supportsAllDrives=true"
               "&fields=id,name,size" % UPLOAD)
        meta["parents"] = [folder_id]

    _, headers, _ = _http(
        auth, method, url, data=json.dumps(meta).encode(),
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Type": "application/octet-stream",
            "X-Upload-Content-Length": str(size),
        })
    session = headers.get("Location")
    if not session:
        raise RuntimeError("il server non ha restituito la sessione di upload")

    sent = 0
    with open(path, "rb") as f:
        while sent < size:
            f.seek(sent)
            chunk = f.read(min(CHUNK, size - sent))
            start = sent
            end = sent + len(chunk) - 1
            attempts = 0
            while True:
                req = urllib.request.Request(
                    session, data=chunk, method="PUT",
                    headers={
                        "Content-Length": str(len(chunk)),
                        "Content-Range": "bytes %d-%d/%d" % (start, end, size),
                    })
                try:
                    with _no_redirect.open(req, timeout=600) as r:
                        body = r.read()
                        if r.status in (200, 201):
                            return json.loads(body.decode())
                except urllib.error.HTTPError as e:
                    body = e.read()
                    if e.code == 308:  # Google: "Resume Incomplete"
                        rng = e.headers.get("Range") if e.headers else None
                        if rng and rng.startswith("bytes="):
                            recv = int(rng.split("=", 1)[1].split("-", 1)[1])
                            sent = recv + 1
                        else:
                            sent = end + 1
                        break
                    if e.code == 401:
                        log("upload HTTP 401: refresh del token")
                        auth._refresh()
                        raise UploadSessionExpired("token rinnovato")
                    if e.code in (404, 410, 416):
                        raise UploadSessionExpired(
                            "sessione non piu' valida (HTTP %d)" % e.code)
                    if e.code in (429, 500, 502, 503, 504):
                        attempts += 1
                        wait = min(60, 2 ** min(attempts, 6))
                        log("upload HTTP %d, riprovo tra %d s" % (e.code, wait))
                        time.sleep(wait)
                        continue
                    raise RuntimeError(
                        "upload HTTP %d: %s" % (
                            e.code, body.decode(errors="replace")[:500]))
                except UploadSessionExpired:
                    raise
                except (urllib.error.URLError, TimeoutError,
                        ConnectionError, OSError) as e:
                    # SSL/EOF, DNS, timeout, dropped connection, etc.:
                    # retry forever (backoff capped at 60 s): the solver
                    # stays waiting until the network is back.
                    attempts += 1
                    wait = min(60, 2 ** min(attempts, 6))
                    log("rete durante l'upload (%s), riprovo tra %d s"
                        % (e, wait))
                    time.sleep(wait)
                    continue
    raise RuntimeError("upload terminato senza risposta finale")


def upload_with_restart(auth, path, name, folder_id=None, file_id=None):
    """Restarts from scratch if the resumable session expires during a long
    network outage (the local file is intact)."""
    for _ in range(20):
        try:
            return upload_resumable(auth, path, name,
                                    folder_id=folder_id, file_id=file_id)
        except UploadSessionExpired as e:
            log("sessione di upload scaduta (%s): riparto da zero" % e)
            time.sleep(2)
    raise RuntimeError("upload: troppe sessioni scadute")


def find_remote_file(auth, folder_id, name, shared_drive=False):
    params = {
        "q": "name = '%s' and '%s' in parents and trashed = false"
             % (_q(name), _q(folder_id)),
        "fields": "files(id,name,size,md5Checksum)",
        "pageSize": "10",
        "supportsAllDrives": "true",
        "includeItemsFromAllDrives": "true",
    }
    if shared_drive:
        params["corpora"] = "allDrives"
    res = _json(auth, "GET",
                "%s/files?%s" % (DRIVE, urllib.parse.urlencode(params)))
    files = (res or {}).get("files", [])
    return files[0] if files else None


def sync_file(auth, args, folder_id, path):
    """Uploads (or updates) a single file, meant to be called by the
    solver right after the .vti is written: blocks until the upload ends."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        log("file non trovato: %s" % path)
        return 1
    size = os.path.getsize(path)
    if size == 0:
        log("file vuoto: %s" % path)
        return 1

    rem = find_remote_file(auth, folder_id, name, args.shared_drive) \
        if folder_id else None
    if rem is not None:
        same = str(rem.get("size", "")) == str(size)
        if same and args.checksum and rem.get("md5Checksum"):
            same = rem["md5Checksum"] == _md5(path)
        if same:
            log("gia' su Drive: %s" % name)
            if args.delete and not args.dry_run:
                try:
                    os.remove(path)
                    log("  rimosso in locale: %s" % name)
                except OSError as e:
                    log("  non riesco a rimuovere %s: %s" % (name, e))
            return 0

    action = "aggiorna" if rem is not None else "carica"
    if args.dry_run:
        log("[dry-run] %s %s (%.1f MB)" % (action, name, size / 1e6))
        return 0

    log("%s %s (%.1f MB)" % (action, name, size / 1e6))
    before = os.stat(path)
    try:
        if rem is not None:
            meta = upload_with_restart(auth, path, name, file_id=rem["id"])
        else:
            meta = upload_with_restart(auth, path, name, folder_id=folder_id)
    except Exception as e:
        log("  ERRORE su %s: %s" % (name, e))
        return 1
    log("  ok: id %s" % meta.get("id", "?"))

    if args.delete:
        after = os.stat(path)
        if (after.st_size, after.st_mtime_ns) != \
           (before.st_size, before.st_mtime_ns):
            log("  non cancello %s: il file e' cambiato durante l'upload"
                % name)
        else:
            os.remove(path)
            log("  rimosso in locale: %s" % name)
    return 0


def file_ready(path, min_age):
    try:
        st = os.stat(path)
    except OSError:
        return False
    if st.st_size == 0:
        return False
    return (time.time() - st.st_mtime) >= min_age


def sync_once(auth, args, folder_id):
    remote = list_remote(auth, folder_id, args.shared_drive) \
        if folder_id else {}
    try:
        names = sorted(n for n in os.listdir(args.dir) if n.endswith(".vti"))
    except FileNotFoundError:
        log("cartella locale '%s' non trovata" % args.dir)
        return 0, 0, 0

    uploaded = skipped = errors = 0
    for name in names:
        path = os.path.join(args.dir, name)
        if not os.path.isfile(path):
            continue
        if not file_ready(path, args.min_age):
            vlog("skip %s (ancora in scrittura o troppo recente)" % name)
            continue

        size = os.path.getsize(path)
        rem = remote.get(name)
        if rem is not None:
            same = str(rem.get("size", "")) == str(size)
            if same and args.checksum and rem.get("md5Checksum"):
                same = rem["md5Checksum"] == _md5(path)
            if same:
                vlog("gia' su Drive: %s" % name)
                skipped += 1
                if args.delete and not args.dry_run:
                    try:
                        os.remove(path)
                        log("rimosso in locale: %s" % name)
                    except OSError as e:
                        log("non riesco a rimuovere %s: %s" % (name, e))
                continue

        action = "aggiorna" if rem is not None else "carica"
        if args.dry_run:
            log("[dry-run] %s %s (%.1f MB)" % (action, name, size / 1e6))
            continue

        log("%s %s (%.1f MB)" % (action, name, size / 1e6))
        before = os.stat(path)
        try:
            if rem is not None:
                meta = upload_with_restart(auth, path, name,
                                           file_id=rem["id"])
            else:
                meta = upload_with_restart(auth, path, name,
                                           folder_id=folder_id)
            uploaded += 1
            log("  ok: id %s" % meta.get("id", "?"))
        except Exception as e:
            errors += 1
            log("  ERRORE su %s: %s" % (name, e))
            continue

        if args.delete:
            after = os.stat(path)
            if (after.st_size, after.st_mtime_ns) != \
               (before.st_size, before.st_mtime_ns):
                log("  non cancello %s: il file e' cambiato durante l'upload"
                    % name)
            else:
                os.remove(path)
                log("  rimosso in locale: %s" % name)

    return uploaded, skipped, errors


def main():
    global VERBOSE
    ap = argparse.ArgumentParser(
        description="carica i dump .vti su Google Drive (solo standard library)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--dir", default="vtk3d",
                    help="cartella locale dei .vti (default: vtk3d)")
    ap.add_argument("--file", default=None,
                    help="carica solo questo file (usato dal solver dopo "
                         "ogni dump); ignora --dir/--min-age/--watch")
    ap.add_argument("--folder-id", default=None,
                    help="ID della cartella Drive di destinazione")
    ap.add_argument("--folder-name", default="saetta3d_vti",
                    help="nome della cartella Drive da creare/trovare "
                         "(default: saetta3d_vti)")
    ap.add_argument("--client-secrets", default=None,
                    help="credenziali/client_secret.json (OAuth Desktop app), "
                         "necessario al primo avvio")
    ap.add_argument("--token", default=DEFAULT_TOKEN,
                    help="file del token OAuth "
                         "(default: credenziali/token_gdrive.json)")
    ap.add_argument("--min-age", type=float, default=10.,
                    help="salta i file modificati da meno di N secondi "
                         "(default: 10)")
    ap.add_argument("--delete", action="store_true",
                    help="cancella il .vti locale dopo l'upload riuscito")
    ap.add_argument("--watch", type=float, default=None, metavar="SEC",
                    help="resta in esecuzione e ricontrolla ogni SEC secondi")
    ap.add_argument("--no-browser", action="store_true",
                    help="non prova ad aprire il browser (stampa l'URL)")
    ap.add_argument("--auth-only", action="store_true",
                    help="esegue solo l'autorizzazione OAuth e salva il token")
    ap.add_argument("--non-interactive", action="store_true",
                    help="non avvia mai il browser: se il token manca o e' "
                         "scaduto esce con errore (usato dal solver)")
    ap.add_argument("--checksum", action="store_true",
                    help="confronta anche l'md5 dei file gia' su Drive")
    ap.add_argument("--shared-drive", action="store_true",
                    help="cerca la cartella anche nei drive condivisi")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra le azioni senza scrivere su Drive")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    VERBOSE = args.verbose
    if args.watch is not None and args.watch < 1:
        ap.error("--watch deve essere >= 1 secondo")
    if args.watch is not None and args.file:
        ap.error("--watch non e' compatibile con --file")

    if (args.dry_run and not args.client_secrets
            and not os.path.exists(args.token)):
        log("dry-run senza credenziali: elenco solo i file locali")
        if args.file:
            if os.path.isfile(args.file):
                log("[dry-run] caricherei %s (%.1f MB)"
                    % (args.file, os.path.getsize(args.file) / 1e6))
            else:
                log("file non trovato: %s" % args.file)
                return 1
            return 0
        try:
            names = sorted(n for n in os.listdir(args.dir)
                           if n.endswith(".vti"))
        except FileNotFoundError:
            log("cartella locale '%s' non trovata" % args.dir)
            return 0
        for name in names:
            path = os.path.join(args.dir, name)
            if os.path.isfile(path) and file_ready(path, args.min_age):
                log("[dry-run] caricherei %s (%.1f MB)"
                    % (name, os.path.getsize(path) / 1e6))
        return 0

    try:
        auth = GoogleAuth(args.client_secrets, args.token,
                          no_browser=args.no_browser,
                          non_interactive=args.non_interactive)
        if args.auth_only:
            log("token OAuth pronto in %s" % args.token)
            return 0
        folder_id, folder_name = ensure_folder(
            auth, args.folder_id, args.folder_name, args.shared_drive,
            create=not args.dry_run)
        if folder_id:
            log("cartella Drive: %s (id %s)" % (folder_name, folder_id))
        else:
            log("cartella Drive '%s' non trovata (dry-run: non la creo)"
                % folder_name)

        if args.file:
            return sync_file(auth, args, folder_id, args.file)

        while True:
            uploaded, skipped, errors = sync_once(auth, args, folder_id)
            log("giro completato: %d caricati, %d gia' presenti, %d errori"
                % (uploaded, skipped, errors))
            if args.watch is None:
                return 1 if errors else 0
            time.sleep(args.watch)
    except KeyboardInterrupt:
        log("interrotto")
        return 130
    except AuthError as e:
        print("errore di autenticazione: %s" % e, file=sys.stderr)
        return 1
    except RuntimeError as e:
        print("errore: %s" % e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
