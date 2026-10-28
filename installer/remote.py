"""Remote access pack: reach Artifex from your phone over Tailscale only.

The danger this exists to prevent is an API pointed at the internet by
accident: a server bound to 0.0.0.0, a router port-forward, or Tailscale
Funnel (which publishes a port to the public internet). So:

- Artifex keeps listening on 127.0.0.1 only. Tailscale forwards a port of
  this PC's tailnet address to it (`tailscale serve --tcp`), reachable from
  your own tailnet devices and nothing else.
- The only Tailscale commands this module ever runs are fixed, read-only
  queries (status, serve status, lock status, version) and one fixed serve
  command with a validated port number. It never runs `tailscale funnel`.
- check_exposure() runs after every setup and on demand
  (`python -m installer remote check`), and fails loudly on anything
  reachable beyond this PC and your tailnet.
- The Tailscale installer is pinned (version, size, SHA-256 in
  catalog.json), and its Windows code signature must be valid and from
  Tailscale Inc. before it is opened. Installing a system service needs
  admin rights, so Windows asks you to approve it (UAC); nothing is
  installed silently.
"""

import ipaddress
import json
import os
import re
import secrets
import shutil
import subprocess
import time

from installer import fetch, hint

WATCHED_PORTS = {8000: "Artifex API", 8080: "web gateway", 8081: "llama.cpp server",
                 8095: "vision helper", 11434: "Ollama", 2222: "Kali SSH"}
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
KEY_FILE = "artifex_api_key.txt"

OK, WARN, FAIL, INFO = "OK", "WARN", "FAIL", "INFO"


def _finding(level, title, detail="", fix=""):
    return {"level": level, "title": title, "detail": detail, "fix": fix}


def tailscale_exe() -> str:
    found = shutil.which("tailscale")
    if found:
        return found
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"),
                 os.environ.get("ProgramFiles(x86)", "")):
        p = os.path.join(base, "Tailscale", "tailscale.exe") if base else ""
        if p and os.path.isfile(p):
            return p
    return ""


def _ts(args, timeout=20, run=subprocess.run):
    exe = tailscale_exe()
    if not exe:
        return None
    try:
        return run([exe] + list(args), capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3]) or (0,)


def classify_address(addr: str) -> str:
    """'loopback', 'tailnet', 'any' (all interfaces) or 'lan' (everything else)."""
    addr = addr.strip("[]").split("%")[0]
    if addr in ("0.0.0.0", "::", "*"):
        return "any"
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return "lan"
    if ip.is_loopback:
        return "loopback"
    if ip in (_TAILNET_V4 if ip.version == 4 else _TAILNET_V6):
        return "tailnet"
    return "lan"


def listening_sockets(run=subprocess.run) -> list:
    """[(address, port, pid)] for listening TCP sockets, from netstat."""
    try:
        out = run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True,
                  timeout=30).stdout
        out += run(["netstat", "-ano", "-p", "TCPv6"], capture_output=True, text=True,
                   timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    socks = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "TCP" and parts[3] == "LISTENING":
            addr, _, port = parts[1].rpartition(":")
            if port.isdigit():
                socks.append((addr, int(port), parts[4]))
    return socks


def check_exposure(root: str, api_port: int = 8000, min_version: str = "",
                   run=subprocess.run) -> list:
    """Everything that decides whether Artifex is reachable beyond this PC
    and your tailnet. Read-only."""
    f = []
    # 1. Ports: anything Artifex-related must be loopback (or tailnet) only.
    exposed = [(a, p, pid) for a, p, pid in listening_sockets(run)
               if p in WATCHED_PORTS and classify_address(a) in ("any", "lan")]
    if exposed:
        for a, p, pid in exposed:
            f.append(_finding(FAIL, f"{WATCHED_PORTS[p]} (port {p}) is listening on {a}",
                              f"process {pid}; other machines on your network can reach it, "
                              "and so can the internet if your router forwards that port",
                              "restart it bound to 127.0.0.1 (the installer's launchers do this)"))
    else:
        f.append(_finding(OK, "No Artifex port is open to your network",
                          "every Artifex-related port listens on 127.0.0.1 or your tailnet only"))

    # 2. Tailscale itself.
    if not tailscale_exe():
        f.append(_finding(INFO, "Tailscale is not installed",
                          "remote access is off; Artifex is reachable from this PC only",
                          hint("remote setup")))
        return f
    ver = _ts(["version"], run=run)
    version = (ver.stdout.splitlines() or [""])[0].strip() if ver else ""
    if min_version and _version_tuple(version) < _version_tuple(min_version):
        f.append(_finding(WARN, f"Tailscale {version} is older than {min_version}",
                          "newer releases fix security issues",
                          "update Tailscale from its tray menu or tailscale.com/download"))
    st = _ts(["status", "--json"], run=run)
    try:
        state = json.loads(st.stdout).get("BackendState", "") if st else ""
    except ValueError:
        state = ""
    if state != "Running":
        f.append(_finding(WARN, f"Tailscale is not connected ({state or 'unknown'})",
                          fix="open Tailscale and sign in"))

    # 3. Funnel = public internet. Never acceptable for Artifex.
    ss = _ts(["serve", "status", "--json"], run=run)
    try:
        serve = json.loads(ss.stdout) if ss and ss.stdout.strip() else {}
    except ValueError:
        serve = {}
    funnel = [k for k, v in (serve.get("AllowFunnel") or {}).items() if v]
    if funnel:
        f.append(_finding(FAIL, "Tailscale Funnel is ON: " + ", ".join(funnel),
                          "Funnel publishes these to the public internet",
                          "tailscale funnel reset   (then re-run this check)"))
    else:
        f.append(_finding(OK, "Tailscale Funnel is off", "nothing is published to the internet"))
    fwd = ((serve.get("TCP") or {}).get(str(api_port)) or {}).get("TCPForward", "")
    if fwd:
        host = fwd.rpartition(":")[0]
        if classify_address(host) == "loopback" or host == "localhost":
            f.append(_finding(OK, f"Phone access: tailnet port {api_port} -> {fwd}",
                              "reachable from your own tailnet devices only"))
        else:
            f.append(_finding(WARN, f"Tailnet port {api_port} forwards to {fwd}",
                              "expected 127.0.0.1", hint("remote setup")))
    else:
        f.append(_finding(INFO, "Phone access is not set up",
                          fix=hint("remote setup")))

    # 4. An API key, since tailnet devices can reach the API.
    key = ""
    try:
        key = open(os.path.join(root, KEY_FILE), encoding="utf-8").read().strip()
    except OSError:
        pass
    if fwd and len(key) < 32:
        f.append(_finding(FAIL, "The API is reachable from your tailnet without a strong key",
                          fix=hint("remote setup") + " (creates one)"))
    elif key:
        f.append(_finding(OK, "API key is set", f"{KEY_FILE}, {len(key)} characters"))

    # 5. Tailnet Lock: new devices need signing by a device you trust.
    lk = _ts(["lock", "status"], run=run)
    if lk and "ENABLED" in (lk.stdout or ""):
        f.append(_finding(OK, "Tailnet Lock is on", "a stolen key can't add a device"))
    else:
        f.append(_finding(WARN, "Tailnet Lock is off",
                          "anyone who gets a login or auth key for your tailnet could add a device",
                          "tailscale lock init (see tailscale.com/kb/1226)"))
    return f


def report(findings: list) -> str:
    marks = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]", INFO: "[info]"}
    lines = []
    for x in findings:
        lines.append(f"{marks[x['level']]} {x['title']}")
        if x["detail"]:
            lines.append(f"         {x['detail']}")
        if x["fix"]:
            lines.append(f"         fix: {x['fix']}")
    return "\n".join(lines)


def ensure_api_key(root: str) -> str:
    """A 256-bit random key in <root>/artifex_api_key.txt (created once)."""
    path = os.path.join(root, KEY_FILE)
    if os.path.isfile(path):
        key = open(path, encoding="utf-8").read().strip()
        if len(key) >= 32:
            return path
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(secrets.token_urlsafe(32) + "\n")
    return path


def verify_signature(path: str, run=subprocess.run) -> str:
    """Windows Authenticode check; returns the signer subject or raises."""
    # The path goes in through the environment, never into the script text,
    # so quotes or $ in a folder name can't become PowerShell code.
    ps = ("$s = Get-AuthenticodeSignature -LiteralPath $env:ARTIFEX_SIG_PATH; "
          "\"$($s.Status)|$($s.SignerCertificate.Subject)\"")
    env = dict(os.environ, ARTIFEX_SIG_PATH=os.path.abspath(path))
    out = run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
              capture_output=True, text=True, timeout=60, env=env).stdout.strip()
    status, _, subject = out.partition("|")
    # Exact organisation field, not a substring ("O=Tailscale Inc.Fake").
    org = re.search(r"(?:^|,\s*)O=([^,]+)", subject)
    if status != "Valid" or not org or org.group(1).strip() != "Tailscale Inc.":
        raise fetch.VerifyError(f"Tailscale installer signature is not valid "
                                f"({status or 'unknown'}: {subject or 'no signer'})")
    return subject


def install_tailscale(catalog: dict, download_dir: str, log=print, run=subprocess.run):
    t = catalog["tailscale"]
    dest = os.path.join(download_dir, t["file"])
    log(f"downloading Tailscale {t['version']} ({t['size'] // (1 << 20)} MB)")
    fetch.download(t["base_url"] + t["file"], dest, t["size"], t["sha256"],
                   catalog["allowed_hosts"])
    log("  checksum OK; signature: " + verify_signature(dest, run=run))
    log("opening the Tailscale installer: approve the Windows prompt, then sign in")
    run(["msiexec", "/i", dest], check=True)


def wait_connected(timeout_s=600, log=print, run=subprocess.run, sleep=time.sleep) -> bool:
    deadline = time.monotonic() + timeout_s
    said = False
    while time.monotonic() < deadline:
        st = _ts(["status", "--json"], run=run)
        try:
            if st and json.loads(st.stdout).get("BackendState") == "Running":
                return True
        except ValueError:
            pass
        if not said:
            log("waiting for you to sign in to Tailscale (tray icon -> Log in)...")
            said = True
        sleep(5)
    return False


def serve_api(api_port: int, run=subprocess.run):
    """The one write this module makes to Tailscale: forward the tailnet port
    to the loopback API. Fixed command, validated port."""
    if not isinstance(api_port, int) or not 1024 <= api_port <= 65535:
        raise ValueError(f"bad port {api_port!r}")
    r = _ts(["serve", "--bg", f"--tcp={api_port}", f"tcp://127.0.0.1:{api_port}"],
            timeout=60, run=run)
    if r is None or r.returncode != 0:
        raise RuntimeError("tailscale serve failed: " + ((r.stderr or r.stdout).strip() if r else "not installed"))
