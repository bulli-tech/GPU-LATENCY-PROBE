#!/usr/bin/env python3
"""
GSP-CHECKUP — quick check whether the GSP processor on an NVIDIA GPU is healthy.

What this is and why
--------------------
On modern NVIDIA GPUs the driver talks to the GSP (GPU System Processor), a small
processor on the card itself, over RPC. When that channel is healthy, any CUDA API
call (cudaEventQuery, cudaStreamQuery, cudaMemcpyAsync, ...) costs a few
microseconds. When the channel has degraded, the same calls cost hundreds of
microseconds, and under load they stretch into seconds because requests queue up
in front of the GSP.

This is what a "slow GPU" looks like in practice: training and inference get
many times slower, while nvidia-smi reports normal temperature, power and
utilization, and the logs may contain no errors at all.

How we measure
--------------
We time a single cudaEventQuery call on an event that has already completed.
The call waits for nothing on the GPU: it only asks for a status. So every
microsecond above normal is the cost of the path CPU -> driver -> GSP.

Three modes, because they produce different numbers:
  cold   — right after context creation (some healthy cards show 150-200 us here)
  warm   — after the card has done real work
  loaded — while several threads keep the card busy

Usage
-----
    python3 GSP-CHECKUP.py              # prompts for a choice when several cards exist
    python3 GSP-CHECKUP.py --all        # every card
    python3 GSP-CHECKUP.py --list       # list cards only
    python3 GSP-CHECKUP.py -d 0 -d 2    # specific cards
    python3 GSP-CHECKUP.py --threads 8  # loader threads for the loaded mode

No dependencies: Python standard library plus libcudart, which ships with the
NVIDIA driver. No weights, no models, nothing to download.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import sys
import threading
import time

# ---------------------------------------------------------------------------
# Thresholds, in microseconds. Derived from live measurements: a healthy A100
# answers in ~2 us, a card with a degraded GSP in 150-180 us, i.e. ~70x worse.
# ---------------------------------------------------------------------------
HEALTHY_US = 30.0        # warm/loaded p50 below this: card is healthy
BAD_US = 100.0           # above this: GSP has degraded
SAMPLES = 100            # measurements per median


# ---------------------------------------------------------------------------
# Environment checks. Without these the script cannot run, and it is better to
# say so plainly than to fail with an obscure error.
# ---------------------------------------------------------------------------
def _candidate_libs() -> list[str]:
    """Where to look for libcudart. Order matters: the system linker first."""
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
    """Load libcudart and declare the signatures. Returns (lib or None, problems)."""
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

    # Declare signatures explicitly. Without this ctypes passes a Python int
    # where a pointer is expected, and we corrupt memory instead of getting
    # an answer.
    lib.cudaSetDevice.argtypes = [ctypes.c_int]
    lib.cudaSetDevice.restype = ctypes.c_int
    lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    lib.cudaGetDeviceCount.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]
    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetLastError.restype = ctypes.c_int

    lib.cudaStreamCreate.restype = ctypes.c_int
    lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
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
    """Verify that any card is visible at all. Returns (ok, device count)."""
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
        print("CANNOT RUN: no usable GPU in this system\n")
        print("  - nvidia-smi must list the cards")
        print("  - inside a container you need --gpus all")
        return False, 0

    return True, count.value


# ---------------------------------------------------------------------------
# Card names. The runtime API does not expose them, so we ask nvidia-smi.
# Without it we simply show the index.
# ---------------------------------------------------------------------------
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
    """One card: stream, event, buffer."""

    def __init__(self, lib, index: int):
        self.lib = lib
        self.index = index
        self.stream = ctypes.c_void_p()
        self.event = ctypes.c_void_p()
        self.buffer = ctypes.c_void_p()

    def open(self) -> None:
        """Create the context, a stream and an event. No work yet.

        Note that cudaMalloc is deliberately NOT called here. cudaMalloc itself
        goes to the GSP, so making it the first call would warm the channel and
        the "cold" measurement would lose its meaning. So we create only the
        stream and the event (neither touches the GSP), measure cold, and only
        then touch memory in warm_up().
        """
        lib = self.lib
        _check(lib, lib.cudaSetDevice(self.index), "cudaSetDevice")

        _check(lib, lib.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        # flag 0: no cudaEventBlockingSync, we want a plain poll
        _check(lib, lib.cudaEventCreateWithFlags(ctypes.byref(self.event), 0),
               "cudaEventCreate")

    def warm_up(self) -> None:
        """First real work on the card: allocation, memset, event record."""
        lib = self.lib
        # 1 MiB is enough that the driver does not collapse it into a stub
        # allocation, which would not exercise the GSP path.
        _check(lib, lib.cudaMalloc(ctypes.byref(self.buffer), 1 << 20), "cudaMalloc")
        self._record_and_wait()

    def _record_and_wait(self) -> None:
        """Record the event on the stream and wait for the work to finish.

        After this, cudaEventQuery reports "complete" and waits for nothing on
        the GPU. That is exactly what lets us time the call itself.
        """
        lib = self.lib
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaMemsetAsync(self.buffer, 0, 4096, self.stream),
               "cudaMemsetAsync")
        _check(lib, lib.cudaEventRecord(self.event, self.stream), "cudaEventRecord")
        _check(lib, lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")

    def sample(self, n: int = SAMPLES) -> dict:
        """Median cudaEventQuery time over n measurements, in microseconds."""
        lib = self.lib
        query = lib.cudaEventQuery
        ev = self.event
        xs: list[float] = []
        for _ in range(n):
            t0 = time.perf_counter_ns()
            rc = query(ev)
            dt = (time.perf_counter_ns() - t0) / 1000.0
            if rc not in (0, 1):  # 0 = success, 1 = not ready; anything else is an error
                raise RuntimeError(f"cudaEventQuery returned {rc}: "
                                   f"{lib.cudaGetErrorString(rc).decode(errors='replace')}")
            xs.append(dt)
        xs.sort()
        return {
            "p50": xs[len(xs) // 2],
            "p90": xs[int(len(xs) * 0.9)],
            "max": xs[-1],
        }

    def busy_worker(self, stop: threading.Event) -> None:
        """Keep the card busy until asked to stop.

        Each worker needs its own device selection: cudaSetDevice binds the
        device to the calling thread, so every thread picks the card itself.
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


def probe_device(lib, index: int, threads: int) -> dict:
    """Full run on one card: cold, warm, loaded."""
    dev = Device(lib, index)
    dev.open()
    try:
        # cold is taken before any memory is touched: the GSP channel is not
        # warmed up by work yet
        cold = dev.sample(n=30)
        dev.warm_up()
        warm = dev.sample()

        stop = threading.Event()
        workers = [threading.Thread(target=dev.busy_worker, args=(stop,), daemon=True)
                   for _ in range(threads)]
        for w in workers:
            w.start()
        time.sleep(1.0)  # let the workers reach steady state
        try:
            loaded = dev.sample(n=50)
        finally:
            stop.set()
            for w in workers:
                w.join(timeout=15)
            lib.cudaDeviceSynchronize()

        return {"index": index, "cold": cold, "warm": warm,
                "loaded": loaded, "threads": threads}
    finally:
        dev.close()


def verdict(row: dict) -> tuple[str, str]:
    """Return (short status, explanation).

    We look at warm and loaded. warm is the clean path to the GSP with no load
    from us. loaded is the same under thread contention, and that is the number
    that goes into seconds once requests start queueing in front of the GSP.
    """
    warm = row["warm"]["p50"]
    loaded = row["loaded"]["p50"]

    if warm <= HEALTHY_US and loaded <= HEALTHY_US:
        return "OK", "GSP answers quickly, including under load"
    if warm > BAD_US:
        return "BAD", ("every CUDA call costs hundreds of microseconds, "
                       "which points to a degraded GSP channel")
    if loaded > BAD_US:
        return "BAD", ("fine at rest, but under load the delay grows hundreds "
                       "of times over: requests queue in front of the GSP")
    return "WARN", "delay is elevated but not by orders of magnitude; re-check"


# ---------------------------------------------------------------------------
# Card selection
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
    print("\nWhat do you want to check?")
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
    picked: list[int] = []
    for part in answer.replace(" ", "").split(","):
        if part.isdigit() and int(part) < count:
            picked.append(int(part))
    if not picked:
        print("Could not parse the choice, using every card.")
        return list(range(count))
    return sorted(set(picked))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check whether the GSP processor on an NVIDIA GPU is healthy.")
    ap.add_argument("-d", "--device", dest="devices", type=int, action="append",
                    metavar="N", help="device index; may be given several times")
    ap.add_argument("--all", action="store_true", help="check every card")
    ap.add_argument("--list", action="store_true", help="list cards only")
    ap.add_argument("--threads", type=int, default=4,
                    help="loader threads in the loaded mode (default 4)")
    args = ap.parse_args()

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

    print(f"\nChecking {len(picked)} card(s)  (loader threads: {args.threads})")
    print("Delay is the median of a single cudaEventQuery call, microseconds.\n")
    print(f"{'#':>3} {'model':<28} {'cold':>8} {'warm':>8} {'loaded':>10}  verdict")
    print("-" * 78)

    rows = []
    for idx in picked:
        try:
            row = probe_device(lib, idx, args.threads)
        except Exception as exc:  # noqa: BLE001
            print(f"{idx:>3} {names[idx]:<28} {'-':>8} {'-':>8} {'-':>10}  "
                  f"ERROR: {exc}")
            continue
        row["name"] = names[idx]
        status, why = verdict(row)
        row["status"] = status
        row["why"] = why
        rows.append(row)
        print(f"{idx:>3} {names[idx]:<28} {row['cold']['p50']:>8.1f} "
              f"{row['warm']['p50']:>8.1f} {row['loaded']['p50']:>10.1f}  {status}")

    print("-" * 78)

    if not rows:
        print("\nNot a single card could be checked.")
        return 1

    bad = [r for r in rows if r["status"] == "BAD"]
    warn = [r for r in rows if r["status"] == "WARN"]
    good = [r for r in rows if r["status"] == "OK"]

    print()
    if good:
        print(f"Healthy ({len(good)}): "
              + ", ".join(f"GPU {r['index']}" for r in good))
    if warn:
        print(f"Questionable ({len(warn)}): "
              + ", ".join(f"GPU {r['index']}" for r in warn))
    if bad:
        print(f"Degraded GSP ({len(bad)}): "
              + ", ".join(f"GPU {r['index']}" for r in bad))

    print()
    if bad:
        print("RESULT: some cards have a GSP channel that is orders of magnitude")
        print("        slower than normal. Training and inference will be slow on")
        print("        them even when nvidia-smi reports healthy utilisation.")
        print("\n        What to do next:")
        print("          1. compare VBIOS and Board Part Number across healthy")
        print("             and affected cards:")
        print("             nvidia-smi -i N -q | grep -E 'VBIOS|Board Part'")
        print("          2. if the revisions differ, update the VBIOS on the older ones")
        print("          3. keep affected cards out of production until resolved")
        return 1

    if warn:
        print("RESULT: no clear degradation, but the delay is elevated. Re-check on")
        print("        an idle system: another job may be holding the card.")
        return 1

    print("RESULT: every card checked is healthy, the GSP answers quickly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
