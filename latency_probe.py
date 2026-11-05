#!/usr/bin/env python3
"""
LATENCY-PROBE — find out why a GPU server is slow when nothing looks wrong.

What this measures
------------------
The cost of a CUDA API call, from the CPU side, on work that has already
completed. If an event is complete, `cudaEventQuery` waits for nothing on the
GPU: it only asks the driver for a status. So every microsecond above normal is
the cost of the path CPU -> driver -> device, not compute.

Why this matters more than raw throughput numbers
-------------------------------------------------
A GPU can look perfectly healthy in nvidia-smi — normal clock, power,
temperature, no errors — and still make every call from your process cost 200 us
instead of 2 us. That is a 100x penalty on everything the CPU does with the
device: launching kernels, polling, copying, synchronising. Frameworks then
report "slow GPU" while the GPU itself is idle.

Two effects make this invisible to casual testing, and this tool measures both:

1. IDLE DECAY. Back-to-back calls keep the driver path warm and cost a few
   microseconds. Introduce gaps between calls — which real inference always has
   between steps — and each call has to wake the path again. On affected cards
   that wake-up costs ~200 us; on healthy cards ~2 us.

2. CONCURRENCY COLLAPSE. With a single thread most cards look fine. Add a few
   threads touching the same device and affected cards degrade by orders of
   magnitude — into seconds — because calls serialize on a driver lock. A
   healthy card pays a moderate price (a few hundred microseconds); a broken one
   can stall for seconds or hang a process in uninterruptible sleep.

Real incident that produced this tool
-------------------------------------
8x A100-SXM4-40GB, vLLM, tensor parallel 4 + expert parallel 4: 1.9 tok/s
instead of 200+. Profilers showed multi-second stalls on two of four ranks.
The same machine ran a different engine with pipeline/tensor parallel 8 (one
process per card) at full speed. Root cause: four of the cards degrade under
concurrent driver access. Single-threaded they are indistinguishable from good
ones, which is why the problem survived months of investigation.

Usage
-----
    python3 latency_probe.py                  # prompts for a choice
    python3 latency_probe.py --all            # every card
    python3 latency_probe.py -d 0 -d 3        # specific cards
    python3 latency_probe.py --list           # list cards only
    python3 latency_probe.py --gaps 0.01      # gap for the spaced test, seconds
    python3 latency_probe.py --threads 1,2,4  # thread counts to sweep

No dependencies: the Python standard library plus libcudart, which ships with
the NVIDIA driver. Nothing to download, no models, no weights.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Thresholds, microseconds. Derived from live A100 measurements:
#   healthy card: tight loop ~2 us, spaced 14 us, 4 loader threads ~200 us
#   affected card: spaced 200 us, 1 loader thread -> milliseconds to seconds
# ---------------------------------------------------------------------------
HEALTHY_TIGHT_US = 20.0     # back-to-back calls on a good card
HEALTHY_SPACED_US = 50.0    # 10 ms between calls on a good card
BAD_SPACED_US = 120.0       # spaced calls clearly degraded
BAD_LOADED_US = 300.0       # threads make the call cost hundreds of microseconds
SAMPLES = 100


# ---------------------------------------------------------------------------
# Environment. If something is missing, say what, do not crash.
# ---------------------------------------------------------------------------
def _candidate_libs() -> list[str]:
    found: list[str] = []
    guess = ctypes.util.find_library("cudart")
    if guess:
        found.append(guess)
    for path in (
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.13",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/lib/x86_64-linux-gnu/libcudart.so",
        "/usr/lib/wsl/lib/libcudart.so",
    ):
        if os.path.exists(path):
            found.append(path)
    return found


def load_runtime():
    """Load libcudart and declare every signature we use."""
    problems: list[str] = []
    libs = _candidate_libs()
    if not libs:
        problems.append(
            "libcudart not found\n"
            "  Linux: it ships with the driver and with the CUDA toolkit\n"
            "    check: ls /usr/local/cuda/lib64/libcudart.so*\n"
            "    or install the toolkit: https://developer.nvidia.com/cuda-downloads"
        )
        return None, problems

    lib = None
    for name in libs:
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError as exc:
            problems.append(f"failed to load {name}: {exc}")
    if lib is None:
        return None, problems

    # Explicit signatures: without them ctypes passes a Python int where a
    # pointer is expected and we corrupt memory instead of getting an answer.
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDeviceCount.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetLastError.restype = ctypes.c_int

    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cudaStreamCreate.restype = ctypes.c_int
    lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaStreamDestroy.restype = ctypes.c_int
    lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    lib.cudaStreamSynchronize.restype = ctypes.c_int

    lib.cudaEventCreateWithFlags.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_uint]
    lib.cudaEventCreateWithFlags.restype = ctypes.c_int
    lib.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    lib.cudaEventDestroy.restype = ctypes.c_int
    lib.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cudaEventRecord.restype = ctypes.c_int
    lib.cudaEventQuery.argtypes = [ctypes.c_void_p]
    lib.cudaEventQuery.restype = ctypes.c_int

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int
    lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_size_t, ctypes.c_void_p]
    lib.cudaMemsetAsync.restype = ctypes.c_int

    return lib, problems


def check_environment(lib, problems: list[str]) -> tuple[bool, int]:
    if lib is None:
        print("CANNOT RUN: missing components\n")
        for p in problems:
            print(f"  - {p}")
        print("\nInstall the NVIDIA driver; it provides libcudart and the devices.")
        return False, 0

    count = ctypes.c_int(0)
    rc = lib.cudaGetDeviceCount(ctypes.byref(count))
    if rc != 0:
        msg = lib.cudaGetErrorString(rc).decode(errors="replace")
        print("CANNOT RUN: CUDA sees no devices\n")
        print(f"  - cudaGetDeviceCount returned: {msg}")
        print("  - check: nvidia-smi, access to /dev/nvidia*,")
        print("    the CUDA_VISIBLE_DEVICES variable")
        return False, 0

    if count.value == 0:
        print("CANNOT RUN: no usable GPU\n")
        print("  - nvidia-smi must list the cards")
        print("  - inside a container you need --gpus all")
        return False, 0

    return True, count.value


def device_names(count: int) -> list[str]:
    names = [f"GPU {i}" for i in range(count)]
    smi = _find_nvidia_smi()
    if not smi:
        return names
    try:
        import subprocess
        out = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            line = line.strip()
            # format: "GPU 0: NVIDIA A100-SXM4-40GB (UUID: GPU-...)"
            if not line.startswith("GPU ") or ": " not in line:
                continue
            idx_part, rest = line.split(": ", 1)
            try:
                idx = int(idx_part.replace("GPU", "").strip())
            except ValueError:
                continue
            model = rest.split(" (")[0].strip()
            if 0 <= idx < count and model:
                names[idx] = model
    except Exception:
        pass
    return names


def _find_nvidia_smi() -> str | None:
    for path in ("/usr/bin/nvidia-smi", "/usr/local/nvidia/bin/nvidia-smi"):
        if os.path.exists(path):
            return path
    from shutil import which
    return which("nvidia-smi")


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
class Device:
    """One card inside the calling process."""

    def __init__(self, lib, index: int):
        self.lib = lib
        self.index = index
        self.stream = ctypes.c_void_p()
        self.event = ctypes.c_void_p()
        self.buffer = ctypes.c_void_p()

    def open(self) -> None:
        """Create the stream and the event. No work yet.

        cudaMalloc is deliberately not called here: the allocation itself goes
        through the driver path and would warm it, and the first (idle) sample
        would no longer measure a cold path.
        """
        lib = self.lib
        _check(lib, lib.cudaSetDevice(self.index), "cudaSetDevice")
        _check(lib, lib.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        _check(lib, lib.cudaEventCreateWithFlags(ctypes.byref(self.event), 0),
               "cudaEventCreate")

    def prime(self) -> None:
        """First real work: allocate, memset, record and wait for the event.

        After this the event stays complete, so every later query returns at
        once and we time the call, not the work.
        """
        lib = self.lib
        # 1 MiB, large enough that the driver does not collapse it into a stub
        _check(lib, lib.cudaMalloc(ctypes.byref(self.buffer), 1 << 20), "cudaMalloc")
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaMemsetAsync(self.buffer, 0, 4096, self.stream),
               "cudaMemsetAsync")
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def time_call(self, samples: int, gap: float = 0.0,
                  budget: float = 60.0) -> dict:
        """Median/max of a single cudaEventQuery, microseconds.

        gap=0 gives back-to-back calls, which keep the driver path warm.
        gap>0 inserts an idle pause, which is what real inference sees between
        steps, and is where degraded cards fall apart.

        budget is a wall-clock ceiling in seconds. A stalled card can take
        seconds per call, so without a ceiling one card can hold the tool for
        minutes. The sample count reached is reported as `n`: a value far below
        the requested count is itself a signal that the card stalled.
        """
        lib = self.lib
        query = lib.cudaEventQuery
        ev = self.event
        xs: list[float] = []
        started = time.time()
        for _ in range(samples):
            t0 = time.perf_counter_ns()
            rc = query(ev)
            dt = (time.perf_counter_ns() - t0) / 1000.0
            if rc not in (0, 1):  # 0 = success, 1 = not ready
                raise RuntimeError(
                    f"cudaEventQuery returned {rc}: "
                    f"{lib.cudaGetErrorString(rc).decode(errors='replace')}")
            xs.append(dt)
            if gap:
                time.sleep(gap)
            if time.time() - started > budget:
                break
        xs.sort()
        return {"p50": xs[len(xs) // 2], "max": xs[-1], "n": len(xs),
                "stalled": len(xs) < samples // 2}

    def busy_worker(self, stop: threading.Event) -> None:
        """Keep this card busy from a background thread.

        cudaSetDevice binds the device to the calling thread, so each loader
        names the card itself.
        """
        lib = self.lib
        lib.cudaSetDevice(self.index)
        stream = ctypes.c_void_p()
        buf = ctypes.c_void_p()
        if lib.cudaMalloc(ctypes.byref(buf), 1 << 20) != 0:
            return
        if lib.cudaStreamCreate(ctypes.byref(stream)) != 0:
            return
        try:
            while not stop.is_set():
                lib.cudaMemsetAsync(buf, 0, 1 << 20, stream)
                lib.cudaStreamSynchronize(stream)
        finally:
            lib.cudaStreamDestroy(stream)
            lib.cudaFree(buf)

    def close(self) -> None:
        lib = self.lib
        if self.event:
            lib.cudaEventDestroy(self.event)
        if self.stream:
            lib.cudaStreamDestroy(self.stream)
        if self.buffer:
            lib.cudaFree(self.buffer)


def _check(lib, rc: int, what: str) -> None:
    if rc != 0:
        msg = lib.cudaGetErrorString(rc).decode(errors="replace")
        raise RuntimeError(f"{what} failed: {rc} ({msg})")


def probe_device(lib, index: int, thread_counts: list[int], gap: float) -> dict:
    """Full sweep on one card: tight, spaced, and thread scaling."""
    dev = Device(lib, index)
    dev.open()
    try:
        dev.prime()
        tight = dev.time_call(SAMPLES)
        spaced = dev.time_call(60, gap=gap)

        loaded: dict[int, dict] = {}
        for threads in thread_counts:
            stop = threading.Event()
            workers = [threading.Thread(target=dev.busy_worker, args=(stop,),
                                        daemon=True) for _ in range(threads)]
            for w in workers:
                w.start()
            time.sleep(1.0)
            try:
                loaded[threads] = dev.time_call(40, gap=0.0)
            finally:
                stop.set()
                for w in workers:
                    w.join(timeout=15)
                lib.cudaDeviceSynchronize()
            time.sleep(0.5)

        return {"index": index, "tight": tight, "spaced": spaced, "loaded": loaded}
    finally:
        dev.close()


def verdict(row: dict, baseline: float | None = None) -> tuple[str, str]:
    """Classify a card.

    Three independent signals:
      tight      back-to-back calls, the driver path stays warm
      spaced     calls with gaps, each pays the wake-up cost
      loaded[n]  calls while n threads hammer the same device

    `baseline` is the best loaded number seen across all probed cards. When a
    bad card is present, comparing against it is far more reliable than any
    absolute threshold: card-to-card spread on good hardware is a few
    microseconds, while a bad card is hundreds of times worse.
    """
    tight = row["tight"]["p50"]
    spaced = row["spaced"]["p50"]
    loaded = [v["p50"] for v in row["loaded"].values() if v]
    worst_loaded = max(loaded, default=0.0)
    stalled = any(v.get("stalled") for v in row["loaded"].values()) or row["spaced"].get("stalled")

    if stalled:
        return "STALLED", ("calls took so long that the sample budget ran out; "
                           "the card effectively froze under this pattern")
    if spaced > BAD_SPACED_US:
        return "DEGRADED", ("an idle gap makes the next call cost ~"
                            f"{spaced:.0f} us; the driver path does not stay warm")
    if baseline and worst_loaded > max(baseline * 10.0, BAD_LOADED_US):
        return "CONTENTION", (f"under threads the call costs {worst_loaded / 1000:.1f} ms, "
                              f"{worst_loaded / baseline:.0f}x the best card")
    if worst_loaded > BAD_LOADED_US:
        return "CONTENTION", (f"under threads the call costs {worst_loaded / 1000:.1f} ms "
                              "- calls serialize on a driver lock")
    if tight > HEALTHY_TIGHT_US:
        return "SLOW", "even back-to-back calls are expensive"
    if spaced > HEALTHY_SPACED_US:
        return "WARN", "wake-up after idle is slower than expected"
    return "OK", "calls are cheap at rest, after idle, and under threads"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def choose_devices(count: int, args) -> list[int]:
    if args.devices:
        bad = [d for d in args.devices if not 0 <= d < count]
        if bad:
            print(f"No such cards: {bad} (available 0..{count - 1})")
            sys.exit(2)
        return sorted(set(args.devices))
    if args.all:
        return list(range(count))
    if count == 1:
        return [0]

    names = device_names(count)
    print(f"Found cards: {count}\n")
    for i in range(count):
        print(f"  [{i}] {names[i]}")
    print("\nWhich cards?")
    print("  all             - every card")
    print("  0,2             - a list of indices")
    print("  0               - a single card")
    print("  q               - quit")
    try:
        answer = input("\n> ").strip().lower()
    except EOFError:
        answer = "all"
    if answer in ("q", "quit", "exit", ""):
        print("Cancelled.")
        sys.exit(0)
    if answer == "all":
        return list(range(count))
    picked = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and int(part) < count:
            picked.append(int(part))
    return sorted(set(picked)) or list(range(count))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Measure CUDA call latency and find cards that degrade "
                    "under idle gaps or concurrent access.")
    ap.add_argument("-d", "--device", dest="devices", type=int, action="append",
                    metavar="N", help="device index; may be given several times")
    ap.add_argument("--all", action="store_true", help="check every card")
    ap.add_argument("--list", action="store_true", help="list cards only")
    ap.add_argument("--gap", type=float, default=0.01,
                    help="idle gap for the spaced test, seconds (default 0.01)")
    ap.add_argument("--threads", default="1,2,4",
                    help="thread counts to sweep, comma separated (default 1,2,4)")
    args = ap.parse_args()

    thread_counts = [int(t) for t in args.threads.split(",") if t.strip()]

    lib, problems = load_runtime()
    ok, count = check_environment(lib, problems)
    if not ok:
        return 2

    if args.list:
        names = device_names(count)
        for i in range(count):
            print(f"  [{i}] {names[i]}")
        return 0

    picked = choose_devices(count, args)
    names = device_names(count)

    print(f"\nProbing {len(picked)} card(s)  "
          f"(gap {args.gap * 1000:.0f} ms, threads {args.threads})")
    print("Values: median cost of one cudaEventQuery, microseconds.\n")

    header = f"{'#':>3} {'model':<24} {'tight':>8} {'spaced':>8}"
    for t in thread_counts:
        header += f" {str(t) + 'thr':>9}"
    header += "  verdict"
    print(header)
    print("-" * len(header))

    rows = []
    for idx in picked:
        try:
            row = probe_device(lib, idx, thread_counts, args.gap)
        except Exception as exc:  # noqa: BLE001
            print(f"{idx:>3} {names[idx]:<24} ERROR: {exc}")
            continue
        row["name"] = names[idx]
        rows.append(row)

    if not rows:
        print("\nNot a single card could be probed.")
        return 1

    # Baseline for the relative test: the cheapest loaded number seen among all
    # probed cards. Comparing cards against each other is far more reliable than
    # any absolute threshold, because a bad card is hundreds of times worse.
    baseline = min(
        (max(v["p50"] for v in r["loaded"].values() if v) for r in rows
         if any(r["loaded"].values())),
        default=None,
    )

    for row in rows:
        status, why = verdict(row, baseline)
        row["status"], row["why"] = status, why

        line = (f"{row['index']:>3} {row['name']:<24} {row['tight']['p50']:>8.1f} "
                f"{row['spaced']['p50']:>8.1f}")
        for t in thread_counts:
            v = row["loaded"].get(t)
            line += f" {v['p50']:>9.1f}" if v else f" {'-':>9}"
        print(line + f"  {status}")

    print("-" * len(header))

    bad = [r for r in rows
           if r["status"] in ("DEGRADED", "CONTENTION", "SLOW", "STALLED")]
    warn = [r for r in rows if r["status"] == "WARN"]
    good = [r for r in rows if r["status"] == "OK"]

    print()
    if good:
        print(f"Healthy ({len(good)}): " + ", ".join(f"GPU {r['index']}" for r in good))
    if warn:
        print(f"Warning ({len(warn)}): " + ", ".join(f"GPU {r['index']}" for r in warn))
    if bad:
        print(f"Problem ({len(bad)}): " + ", ".join(f"GPU {r['index']}" for r in bad))

    print()
    if bad:
        print("RESULT: some cards are slow on the driver side while the GPU itself")
        print("        is idle. This shows up as low throughput and multi-second")
        print("        stalls in frameworks, with no errors in nvidia-smi.")
        print()
        print("        Practical consequences:")
        print("          - one process per card (TP/PP with separate workers) is")
        print("            usually fine, because each worker keeps one channel busy")
        print("          - several contexts or many threads on one card is not:")
        print("            calls serialize and card time collapses")
        print()
        print("        What to check next:")
        print("          1. group the cards by Board Part Number and VBIOS:")
        print("             nvidia-smi -i N -q | grep -E 'VBIOS|Board Part'")
        print("          2. look for GSP timeouts in the kernel log:")
        print("             sudo dmesg -T | grep -E 'Xid|GSP'")
        print("          3. restructure the workload to one worker per card if the")
        print("             affected cards cannot be replaced or reflashed")
        return 1

    if warn:
        print("RESULT: no collapse, but wake-up after idle is slower than expected.")
        print("        Re-run on an idle machine; another job may hold the card.")
        return 1

    print("RESULT: every card is cheap at rest, after idle, and under threads.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
