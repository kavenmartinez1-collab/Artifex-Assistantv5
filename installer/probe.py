"""Take stock of the PC before anything is installed.

Standard library only (plus core.dxgi_vram, itself ctypes-only), because it
runs on a machine that has nothing else yet; the old setup_wizard needed
PyTorch installed just to find the GPU, and only knew NVIDIA. Read-only:
it queries Windows (DXGI, memory status) and, when present, nvidia-smi.
No network access.
"""

import ctypes
import os
import platform
import re
import shutil
import subprocess
import sys

NVIDIA, AMD, INTEL = 0x10DE, 0x1002, 0x8086
_VENDORS = {NVIDIA: "NVIDIA", AMD: "AMD", INTEL: "Intel"}


def _ram_mb() -> int:
    if sys.platform != "win32":
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // (1 << 20)
        except (ValueError, OSError, AttributeError):
            return 0

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        return 0
    return m.ullTotalPhys // (1 << 20)


def _os_name() -> str:
    if sys.platform == "win32":
        build = int((platform.version().split(".") + ["0", "0", "0"])[2] or 0)
        # platform.release() says "10" on Windows 11; the build number tells them apart.
        return f"Windows {'11' if build >= 22000 else '10'} (build {platform.version()})"
    return f"{platform.system()} {platform.release()}"


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def _nvidia_smi():
    """Driver, max CUDA version and per-GPU compute capability, or None."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        head = subprocess.run([exe], capture_output=True, text=True, timeout=15).stdout
        rows = subprocess.run(
            [exe, "--query-gpu=name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # Newer drivers print "CUDA UMD Version: 13.3"; older ones "CUDA Version: 12.4".
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head or "")
    gpus = []
    for line in (rows or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            try:
                gpus.append({"name": parts[0], "vram_mb": int(float(parts[1])),
                             "compute_cap": parts[2], "driver": parts[3]})
            except ValueError:
                continue
    return {"cuda_version": m.group(1) if m else "", "gpus": gpus}


def _gpus():
    found = []
    try:
        from core.dxgi_vram import probe_adapters
        for a in probe_adapters():
            found.append({"name": a["description"].strip(),
                          "vendor": _VENDORS.get(a["vendor_id"], f"0x{a['vendor_id']:04X}"),
                          "vram_mb": int(a["dedicated_mb"]),
                          "free_mb": int(a.get("free_mb") or 0)})
    except Exception:
        pass
    return found


def probe(install_dir: str = ".") -> dict:
    """Everything the installer needs to choose a backend and a model."""
    nv = _nvidia_smi()
    gpus = _gpus()
    if nv:
        # DXGI caps 32-bit fields on some cards; nvidia-smi is exact for NVIDIA.
        by_name = {g["name"]: g for g in nv["gpus"]}
        for g in gpus:
            n = by_name.get(g["name"])
            if n:
                g.update(vram_mb=n["vram_mb"], compute_cap=n["compute_cap"])
        if not gpus:
            gpus = [{"name": g["name"], "vendor": "NVIDIA", "vram_mb": g["vram_mb"],
                     "compute_cap": g["compute_cap"]} for g in nv["gpus"]]
    # The install root may not exist yet: measure the nearest folder that does.
    target = os.path.abspath(install_dir)
    while not os.path.exists(target) and os.path.dirname(target) != target:
        target = os.path.dirname(target)
    try:
        free_disk_mb = shutil.disk_usage(target).free // (1 << 20)
    except OSError:
        free_disk_mb = 0
    return {
        "os": _os_name(),
        "windows": sys.platform == "win32",
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "cpu_threads": os.cpu_count() or 1,
        "ram_mb": _ram_mb(),
        "free_disk_mb": free_disk_mb,
        "gpus": gpus,
        "nvidia": {"driver": (nv["gpus"][0]["driver"] if nv and nv["gpus"] else ""),
                   "cuda_version": nv["cuda_version"] if nv else ""} if nv else None,
        "ollama": bool(shutil.which("ollama")),
        "docker": bool(shutil.which("docker")),
    }


def summary(p: dict) -> str:
    lines = [f"OS:      {p['os']}",
             f"CPU:     {p['cpu']} ({p['cpu_threads']} threads)",
             f"RAM:     {p['ram_mb'] / 1024:.1f} GB",
             f"Disk:    {p['free_disk_mb'] / 1024:.0f} GB free where Artifex installs"]
    if p["gpus"]:
        for g in p["gpus"]:
            extra = f", compute {g['compute_cap']}" if g.get("compute_cap") else ""
            lines.append(f"GPU:     {g['name']} ({g['vendor']}, {g['vram_mb'] / 1024:.1f} GB{extra})")
    else:
        lines.append("GPU:     none usable found (models will run on the CPU)")
    if p.get("nvidia"):
        lines.append(f"NVIDIA:  driver {p['nvidia']['driver']}, supports CUDA up to "
                     f"{p['nvidia']['cuda_version'] or '?'}")
    lines.append(f"Ollama:  {'installed' if p['ollama'] else 'not installed'}   "
                 f"Docker: {'installed' if p['docker'] else 'not installed'}")
    return "\n".join(lines)
