"""Cross-platform peak-RSS (memory dimension M) measurement.

EXPERIMENT_PLAN.md §6.4:
  * Termux/Linux: /proc/<pid>/status VmHWM
  * Windows: PeakWorkingSet64 (via ctypes GetProcessMemoryInfo)
  * Unix generally: resource.getrusage(...).ru_maxrss

All return peak RSS in MiB for the current process.
"""
from __future__ import annotations

import os
import sys


def external_process_rss_mb(name_substr: str) -> float:
    """Current working-set (MiB) summed over processes whose image name
    contains `name_substr`. Used to fold the resident llama-server (model
    weights + KV cache) into the memory dimension when generation runs in a
    separate process; on a single-process deployment (phone) this is 0.
    """
    if sys.platform != "win32":
        total = 0.0
        try:
            import subprocess
            out = subprocess.run(["ps", "-eo", "rss,comm"], capture_output=True,
                                 text=True, timeout=5).stdout.splitlines()
            for line in out[1:]:
                parts = line.split(None, 1)
                if len(parts) == 2 and name_substr in parts[1]:
                    total += float(parts[0]) / 1024.0  # kB -> MiB
        except Exception:
            pass
        return total
    # Windows: tasklist reports Mem Usage per PID.
    try:
        import subprocess
        out = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {name_substr}*", "/FO", "CSV",
             "/NH"], capture_output=True, text=True, timeout=8).stdout
        total = 0.0
        for line in out.splitlines():
            cells = [c.strip('"') for c in line.split('","')]
            if len(cells) >= 5 and name_substr.lower() in cells[0].lower():
                kb = cells[4].replace(",", "").replace(" K", "").strip()
                try:
                    total += float(kb) / 1024.0  # kB -> MiB
                except ValueError:
                    pass
        return total
    except Exception:
        return 0.0


def peak_rss_mb() -> float:
    # Linux / Android (Termux): /proc/self/status VmHWM is the high-water mark.
    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{os.getpid()}/status") as fh:
                for line in fh:
                    if line.startswith("VmHWM:"):
                        return float(line.split()[1]) / 1024.0  # kB -> MiB
        except OSError:
            pass
        return _rusage_mb()

    if sys.platform == "win32":
        return _windows_peak_mb()

    return _rusage_mb()  # macOS / other Unix


def _rusage_mb() -> float:
    try:
        import resource
    except ImportError:
        return 0.0
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: kB; macOS: bytes.
    return (ru / 1024.0) if sys.platform.startswith("linux") else (ru / 1024.0 / 1024.0)


def _windows_peak_mb() -> float:
    import ctypes
    from ctypes import wintypes

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(
        handle, ctypes.byref(counters), counters.cb)
    if not ok:
        return 0.0
    return counters.PeakWorkingSetSize / 1024.0 / 1024.0


if __name__ == "__main__":
    print(f"peak_rss = {peak_rss_mb():.1f} MiB")
