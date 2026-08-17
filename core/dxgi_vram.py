"""
Artifex Assistant V5 — Cross-vendor VRAM probe (Windows).

nvidia-smi only sees NVIDIA cards, which leaves every other adapter in a
multi-GPU split unguarded by the VRAM gate.  Two Windows APIs together
cover all vendors:

  DXGI (EnumAdapters1 + GetDesc1): every hardware adapter's name, PCI
  vendor, total dedicated VRAM, and LUID.

  PDH performance counters ("GPU Adapter Memory(*)\\Dedicated Usage"):
  live SYSTEM-WIDE dedicated VRAM usage per adapter, keyed by the same
  LUID.  This is the number that matters for gating — measured here,
  IDXGIAdapter3::QueryVideoMemoryInfo's Budget stays optimistic in the
  querying process even while another process holds 12 GB, so Budget
  CANNOT be used as "free".  free = dedicated_total − pdh_usage.

Pure ctypes; no packages, no subprocesses.  Every entry point degrades
to an empty result off Windows or on any API failure.
"""

import ctypes
import logging
import re
import sys

_log = logging.getLogger(__name__)

_MB = 1024 * 1024

# DXGI_ADAPTER_FLAG_SOFTWARE — the "Microsoft Basic Render Driver"
_ADAPTER_FLAG_SOFTWARE = 2
# DXGI_MEMORY_SEGMENT_GROUP_LOCAL — dedicated VRAM (not shared sysmem)
_SEGMENT_LOCAL = 0
_DXGI_ERROR_NOT_FOUND = -2005270526  # 0x887A0002 as signed int32


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]

    @classmethod
    def make(cls, d1, d2, d3, *d4):
        return cls(d1, d2, d3, (ctypes.c_ubyte * 8)(*d4))


_IID_IDXGIFactory1 = _GUID.make(
    0x770AAE78, 0xF26F, 0x4DBA,
    0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
_IID_IDXGIAdapter3 = _GUID.make(
    0x645967A4, 0x1392, 0x4310,
    0xA7, 0x98, 0x80, 0x53, 0xCE, 0x3E, 0x93, 0xFD)


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", ctypes.c_uint),
    ]


class _DXGI_QUERY_VIDEO_MEMORY_INFO(ctypes.Structure):
    _fields_ = [
        ("Budget", ctypes.c_uint64),
        ("CurrentUsage", ctypes.c_uint64),
        ("AvailableForReservation", ctypes.c_uint64),
        ("CurrentReservation", ctypes.c_uint64),
    ]


def _method(obj, index, restype, *argtypes):
    """Resolve COM vtable slot `index` on interface pointer `obj`."""
    vtable = ctypes.cast(
        obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    proto = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return proto(vtable[index])


def _release(obj):
    if obj:
        try:
            _method(obj, 2, ctypes.c_ulong)(obj)  # IUnknown::Release
        except Exception:
            pass


# ── PDH: system-wide dedicated usage per adapter LUID ──────────────────

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
_LUID_RE = re.compile(r"luid_0x([0-9A-Fa-f]{8})_0x([0-9A-Fa-f]{8})")


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_uint32),
                ("largeValue", ctypes.c_int64)]   # union collapsed to LONGLONG


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p),
                ("FmtValue", _PDH_FMT_COUNTERVALUE)]


def pdh_dedicated_usage_mb():
    """System-wide dedicated VRAM usage per adapter, keyed by LUID string.

    Returns {"high:low": usage_mb} (decimal LUID parts, matching
    probe_adapters()' `luid` field), summed across phys_N nodes.
    Empty dict off Windows or on any PDH failure.
    """
    if sys.platform != "win32":
        return {}
    query = ctypes.c_void_p()
    try:
        pdh = ctypes.WinDLL("pdh")
        if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) != 0:
            return {}
        counter = ctypes.c_void_p()
        rc = pdh.PdhAddEnglishCounterW(
            query, r"\GPU Adapter Memory(*)\Dedicated Usage", 0,
            ctypes.byref(counter))
        if rc != 0:
            return {}
        if pdh.PdhCollectQueryData(query) != 0:
            return {}
        buf_size = ctypes.c_uint32(0)
        item_count = ctypes.c_uint32(0)
        rc = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(buf_size),
            ctypes.byref(item_count), None)
        if (rc & 0xFFFFFFFF) != _PDH_MORE_DATA:   # expect "more data" probe
            return {}
        buf = (ctypes.c_byte * buf_size.value)()
        rc = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(buf_size),
            ctypes.byref(item_count),
            ctypes.cast(buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W)))
        if rc != 0:
            return {}
        items = ctypes.cast(
            buf, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        usage: dict[str, float] = {}
        for i in range(item_count.value):
            item = items[i]
            name = item.szName or ""
            m = _LUID_RE.search(name)
            if not m:
                continue
            high = int(m.group(1), 16)
            low = int(m.group(2), 16)
            key = f"{high}:{low}"
            usage[key] = usage.get(key, 0.0) + item.FmtValue.largeValue / _MB
        return usage
    except Exception as e:
        _log.warning("PDH GPU memory query failed: %s", e)
        return {}
    finally:
        if query:
            try:
                ctypes.WinDLL("pdh").PdhCloseQuery(query)
            except Exception:
                pass


def probe_adapters():
    """Enumerate hardware display adapters with live memory info.

    Returns a list of dicts (possibly empty), one per hardware adapter:
      description     adapter marketing name (e.g. "AMD Radeon RX 6700 XT")
      vendor_id       PCI vendor (0x10DE NVIDIA, 0x1002 AMD, 0x8086 Intel)
      dedicated_mb    total dedicated VRAM
      budget_mb       OS budget hint (per-process — informational ONLY)
      usage_mb        system-wide dedicated usage (PDH; all processes)
      free_mb         dedicated_mb − usage_mb — free for a new engine
      usage_source    "pdh" (authoritative) or "budget" (weak fallback)
      luid            adapter LUID as "high:low" string (stable per boot)
      software        True for software rasterizers (already filtered out)

    Empty list off Windows or when DXGI is unavailable/fails.
    """
    if sys.platform != "win32":
        return []
    factory = ctypes.c_void_p()
    adapters = []
    try:
        dxgi = ctypes.WinDLL("dxgi")
        hr = dxgi.CreateDXGIFactory1(
            ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
        if hr != 0 or not factory:
            _log.warning("CreateDXGIFactory1 failed (hr=0x%08X)", hr & 0xFFFFFFFF)
            return []

        enum_adapters1 = _method(
            factory, 12, ctypes.c_int32,
            ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))

        index = 0
        while True:
            adapter1 = ctypes.c_void_p()
            hr = enum_adapters1(factory, index, ctypes.byref(adapter1))
            if hr == _DXGI_ERROR_NOT_FOUND:
                break
            if hr != 0 or not adapter1:
                _log.warning("EnumAdapters1(%d) failed (hr=0x%08X)",
                             index, hr & 0xFFFFFFFF)
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                hr = _method(adapter1, 10, ctypes.c_int32,
                             ctypes.POINTER(_DXGI_ADAPTER_DESC1))(
                    adapter1, ctypes.byref(desc))
                if hr != 0:
                    continue
                if desc.Flags & _ADAPTER_FLAG_SOFTWARE:
                    continue
                if desc.DedicatedVideoMemory < 256 * _MB:
                    continue  # headless stubs / display-only adapters

                entry = {
                    "description": desc.Description,
                    "vendor_id": desc.VendorId,
                    "dedicated_mb": desc.DedicatedVideoMemory / _MB,
                    "budget_mb": 0.0,
                    "usage_mb": 0.0,
                    "free_mb": 0.0,
                    "usage_source": "budget",
                    "luid": f"{desc.AdapterLuid.HighPart}:{desc.AdapterLuid.LowPart}",
                    "software": False,
                }

                # IDXGIAdapter3 for QueryVideoMemoryInfo (Win10+)
                adapter3 = ctypes.c_void_p()
                hr = _method(adapter1, 0, ctypes.c_int32,
                             ctypes.POINTER(_GUID),
                             ctypes.POINTER(ctypes.c_void_p))(
                    adapter1, ctypes.byref(_IID_IDXGIAdapter3),
                    ctypes.byref(adapter3))
                if hr == 0 and adapter3:
                    try:
                        info = _DXGI_QUERY_VIDEO_MEMORY_INFO()
                        hr = _method(
                            adapter3, 14, ctypes.c_int32,
                            ctypes.c_uint, ctypes.c_int,
                            ctypes.POINTER(_DXGI_QUERY_VIDEO_MEMORY_INFO))(
                            adapter3, 0, _SEGMENT_LOCAL, ctypes.byref(info))
                        if hr == 0:
                            entry["budget_mb"] = info.Budget / _MB
                            entry["usage_mb"] = info.CurrentUsage / _MB
                            entry["free_mb"] = max(
                                0.0, (info.Budget - info.CurrentUsage) / _MB)
                    finally:
                        _release(adapter3)

                adapters.append(entry)
            finally:
                _release(adapter1)
                index += 1
    except Exception as e:
        _log.warning("DXGI probe failed: %s", e)
        return []
    finally:
        _release(factory)

    # System-wide usage overrides the per-process view: the DXGI numbers
    # above can't see other processes (measured: Budget stayed at 11.4 GB
    # while llama-server held 11.9 GB of the same card).
    pdh_usage = pdh_dedicated_usage_mb()
    for a in adapters:
        used = pdh_usage.get(a["luid"])
        if used is not None:
            a["usage_mb"] = used
            a["free_mb"] = max(0.0, a["dedicated_mb"] - used)
            a["usage_source"] = "pdh"
    return adapters


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for a in probe_adapters():
        print(f"{a['description']}  (vendor 0x{a['vendor_id']:04X}, luid {a['luid']})")
        print(f"  dedicated {a['dedicated_mb']:.0f} MB | usage {a['usage_mb']:.0f} MB"
              f" | free {a['free_mb']:.0f} MB [{a['usage_source']}]"
              f" | budget {a['budget_mb']:.0f} MB")
