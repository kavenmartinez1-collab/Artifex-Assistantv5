"""
Artifex-Assistant v5 — Hardware telemetry and system sensing.
Merged from chatbot.py get_hardware_manifest() and pentest_tools.py sense_system().
"""

import os
import re
import socket
import shutil
import platform
import subprocess

import torch
import psutil

from core.config import IS_WINDOWS


def get_hardware_manifest():
    """Identify host GPU, VRAM, CPU, RAM, and OS."""
    gpu_name = "NO_GPU_DETECTED"
    vram_stat = "0.00 / 0.00 GB"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).upper()
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        vram_stat = f"{allocated:.2f} / {total:.2f} GB"

    cpu_name = platform.processor() or "UNKNOWN_X86_CORE"
    ram = f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB"

    return {
        "GPU": gpu_name,
        "VRAM": vram_stat,
        "CPU": cpu_name,
        "SYS_RAM": ram,
        "OS": platform.system().upper(),
    }


def sense_system():
    """
    Probe the local system for a rich environment profile.
    Returns dict with OS, shell, networking, runtimes, utilities, VPN.
    """
    info = {}

    # OS basics
    info["os"] = platform.system()
    info["os_version"] = platform.version()
    info["arch"] = platform.machine()
    info["hostname"] = socket.gethostname()

    if IS_WINDOWS:
        try:
            build = int(info["os_version"].split(".")[-1])
            if build >= 22000:
                info["os_display"] = f"Windows 11 (build {build})"
            else:
                info["os_display"] = f"Windows 10 (build {build})"
        except (ValueError, IndexError):
            info["os_display"] = f"Windows ({info['os_version']})"
    else:
        info["os_display"] = f"{info['os']} {info['os_version']}"

    # Shell detection
    if IS_WINDOWS:
        info["shell"] = "cmd.exe"
        if shutil.which("powershell"):
            info["shell"] = "PowerShell"
        if shutil.which("pwsh"):
            info["shell"] = "PowerShell Core (pwsh)"
        if shutil.which("bash"):
            info["has_bash"] = True
        if shutil.which("wsl"):
            info["has_wsl"] = True
    else:
        info["shell"] = os.environ.get("SHELL", "/bin/sh")

    # Local IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["local_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["local_ip"] = "unknown"

    # Language runtimes
    runtimes = {}
    for name, candidates in [
        ("python3", ["python3", "python"]),
        ("python2", ["python2"]),
        ("ruby", ["ruby"]),
        ("perl", ["perl"]),
        ("php", ["php"]),
        ("go", ["go"]),
        ("gcc", ["gcc"]),
        ("javac", ["javac"]),
        ("node", ["node"]),
    ]:
        for c in candidates:
            path = shutil.which(c)
            if path:
                runtimes[name] = path
                break
    info["runtimes"] = runtimes

    # General CLI utilities
    utilities = {}
    for name in [
        "curl", "wget", "ssh", "scp", "nc", "ncat", "socat",
        "git", "tar", "gzip", "unzip", "7z",
        "openssl", "base64", "xxd", "file",
        "ping", "traceroute", "tracert", "dig", "nslookup", "host",
        "awk", "sed", "grep", "find", "xargs",
        "powershell", "certutil", "bitsadmin",
        "docker", "python3", "pip", "pip3",
    ]:
        path = shutil.which(name)
        if path:
            utilities[name] = path
    info["utilities"] = utilities

    # VPN detection (common HTB setup)
    try:
        if IS_WINDOWS:
            result = subprocess.run(
                "ipconfig", capture_output=True, text=True, timeout=5
            )
            if "tun" in result.stdout.lower() or "tap" in result.stdout.lower():
                info["vpn_detected"] = True
        else:
            result = subprocess.run(
                ["ip", "addr"], capture_output=True, text=True, timeout=5
            )
            if "tun" in result.stdout:
                info["vpn_detected"] = True
                for line in result.stdout.split("\n"):
                    if "tun0" in line or ("inet " in line and "tun" in result.stdout):
                        m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
                        if m:
                            info["vpn_ip"] = m.group(1)
    except Exception:
        pass

    return info
