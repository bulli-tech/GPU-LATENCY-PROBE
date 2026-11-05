# GPU-CHECKUP

Detect one of the most deceptive GPU server problems: a **degraded GSP
channel** — the path to the management processor that runs inside every
modern NVIDIA GPU.

The symptom looks like this: `nvidia-smi` reports normal temperature, power and
utilization, there are no errors in the logs, yet training and inference are
several times slower than they should be. The culprit is neither your code nor
your framework nor the driver version — it is the card.

`GSP-CHECKUP.py` finds this in a single run and tells you plainly which cards
are healthy and which are not.

## What we ran into

A server with 8×A100-SXM4-40GB ran a DeepSeek-V4.1 model under vLLM at 1.9 tok/s
instead of 200+. The profiler showed stalls of 0.65–2.85 seconds on ranks 1 and
2, while ranks 0 and 3 showed 0.2 ms. It looked like "NCCL is waiting", "the
host callback is slow", "the allocator is the problem".

What we tried, and what did not help:

- found and fixed real bugs in the expert-offload kernels — throughput unchanged;
- disabled the background thread of the pinned allocator — the stall moved but
  the speed did not improve;
- isolated NCCL tests: a small AllGather took 55 µs, perfectly normal;
- isolated empty host callbacks: 0.07–0.5 ms, far from seconds;
- switched the driver 610.57.04 → 580.196.07 (CUDA 13.3 → 13.0) — not a single
  number changed.

What it actually was: **the machine was populated from two different batches of
the same GPU model**. Three cards were fine, five were not.

| GPUs | Board Part Number | VBIOS | State |
|------|-------------------|-------|-------|
| 0, 3, 7 | 692-2G506-0200-**002** | 92.00.**36**.00.04 | healthy |
| 1, 2, 4, 5, 6 | 692-2G506-0200-**000** | 92.00.**0C**.00.05 | degraded |

The split matched the measurements 8 out of 8, with no exceptions. On the
degraded cards every CUDA API call cost ~160 µs instead of ~2 µs, and under
load it stretched into seconds. The kernel log showed `Xid 119` — a timeout
while waiting for a reply from the GSP.

## Quick start

```bash
# no arguments: prompts for a choice when several cards are present
python3 GSP-CHECKUP.py

# every card
python3 GSP-CHECKUP.py --all

# list the cards only
python3 GSP-CHECKUP.py --list

# specific cards
python3 GSP-CHECKUP.py -d 0 -d 3

# more loader threads for the loaded phase (default 4)
python3 GSP-CHECKUP.py --all --threads 8
```

The script is dependency-free: only the Python standard library plus
`libcudart`, which ships with the driver. No weights, no models, nothing to
download.

## What it prints

```
Checking 8 card(s)  (loader threads: 4)
Delay is the median of a single cudaEventQuery call, microseconds.

  # model                           cold     warm     loaded  verdict
------------------------------------------------------------------------------
  0 NVIDIA A100-SXM4-40GB             2.5      2.5       22.8  OK
  1 NVIDIA A100-SXM4-40GB             2.5      2.2      172.0  BAD
  2 NVIDIA A100-SXM4-40GB             2.3      2.2      172.2  BAD
  3 NVIDIA A100-SXM4-40GB             2.2      2.2        9.0  OK
  4 NVIDIA A100-SXM4-40GB             2.1      2.1      156.7  BAD
  5 NVIDIA A100-SXM4-40GB             2.5      2.0      172.6  BAD
  6 NVIDIA A100-SXM4-40GB             2.1      2.1      158.6  BAD
  7 NVIDIA A100-SXM4-40GB             2.1      2.7        7.7  OK
------------------------------------------------------------------------------

Healthy (3): GPU 0, GPU 3, GPU 7
Degraded GSP (5): GPU 1, GPU 2, GPU 4, GPU 5, GPU 6
```

Exit code is `0` when every card is healthy, `1` when any card is degraded,
`2` when the environment is not usable (no driver, no GPUs).

If something is missing, the script says exactly what instead of failing with a
stack trace:

```
CANNOT RUN: missing components

  - libcudart not found
  Linux: it ships with the driver and with the CUDA toolkit
    check: ls /usr/local/cuda/lib64/libcudart.so*
```

## How the measurement works

1. **Environment check.** The script verifies it can run at all: `libcudart`
   is resolvable, `cudaGetDeviceCount` returns at least one device. Otherwise
   it explains what is missing and exits `2`.

2. **Cold pass.** For each device it creates a stream and an event, then times
   `cudaEventQuery` on that event in a tight loop, 30 samples. No memory has
   been touched yet, so the GSP channel is cold.

3. **Warm pass.** `cudaMalloc` of 1 MiB plus a `cudaMemsetAsync` and an event
   record. The event is now complete and stays complete, so every later query
   returns at once — the numbers reflect the cost of the call, not the work.
   100 samples.

4. **Loaded pass.** `--threads` worker threads each own their stream and keep
   the card busy with `cudaMemsetAsync` + `cudaStreamSynchronize`, while the
   main thread keeps querying. Under this contention the driver must serialize
   calls onto the device channel; a degraded GSP cannot keep up and the median
   climbs by orders of magnitude. 50 samples.

5. **Verdict.** `OK` if both `warm` and `loaded` are below `HEALTHY_US`
   (30 µs). `BAD` if `warm` or `loaded` exceeds `BAD_US` (100 µs). Otherwise
   `WARN`. The ratio between a healthy and a degraded card is roughly 70×,
   so the thresholds are not sensitive to minor noise.

All numbers are CPU-side wall clock (`perf_counter_ns`), median per phase.

### Why a completed event

`cudaEventQuery` on an already-complete event waits for nothing on the GPU.
It still has to reach the driver and, on the paths that involve it, the GSP.
That is precisely the cost we want to isolate: if the call takes 160 µs instead
of 2 µs while doing no work, the time is spent in the driver-to-GSP path.

### Why `cudaMalloc` is not the first call

`cudaMalloc` itself goes through the GSP, so calling it before the cold pass
would warm the channel and the cold number would lose its meaning. We measured
this the hard way: the first version reported a healthy 2 µs on a degraded card
because the allocation had already primed it.

## Interpretation

| Observation | Meaning |
|---|---|
| `warm` ≈ 1–3 µs on every card | Normal. The GSP answers in microseconds. |
| `warm` > 100 µs consistently | The channel is degraded even when idle. |
| `warm` normal, `loaded` > 100 µs | The channel collapses under concurrency — the throughput killer. |
| `cold` high, `warm` low | Cold start of the channel, not a defect. |
| Wide spread across cards | Mixed batches or revisions. Compare `Board Part Number` and `VBIOS`. |

## What to check when the script reports BAD

```bash
# compare revisions across cards
for i in 0 1 2 3 4 5 6 7; do
  echo "=== GPU$i ==="
  nvidia-smi -i $i -q | grep -E "Serial Number|VBIOS Version|Board Part Number"
done

# look for GSP timeouts in the kernel log
sudo dmesg -T | grep -E "Xid|GSP" | tail -20
```

What the findings mean:

- **different `VBIOS Version` or `Board Part Number` between healthy and
  affected cards** — two batches; update the VBIOS on the older ones or keep
  them out of production;
- **`Xid 119` in dmesg** — the driver timed out waiting for a GSP reply. This is
  a consequence of the same degradation, not a separate failure;
- **identical revisions but different behaviour** — then look at power and
  cooling for that specific slot; that is a different investigation.

## Should I trust the number?

The script measures what the driver actually does, not what vendor tools report.
`nvidia-smi` samples utilization over time windows and will happily show a card
as busy while every call takes 150 µs to submit.

For a second opinion, compare against a known-good card in the same machine:

```bash
python3 GSP-CHECKUP.py -d 0   # known-good
python3 GSP-CHECKUP.py -d 1   # suspect
```

A factor of two is noise. A factor of fifty is a defective card or batch.

## Inside a container

Mount the script and run it with all GPUs visible:

```bash
docker run --rm --gpus all -v "$PWD/GSP-CHECKUP.py:/tmp/g.py:ro" \
  <image with python and cudart> -I /tmp/g.py --all
```

Note that `--gpus device=N` renumbers devices: inside the container the card
becomes index 0. Use `--gpus all` so the indices match the host.

## Requirements

- Linux, Python 3.8+.
- NVIDIA driver providing `libcudart`.
- No root required for the measurement itself.

## Limitations

- The script measures the symptom, not the root cause. `BAD` means "the path to
  the GSP has degraded"; why it degraded is answered by comparing revisions,
  reading the kernel log, and talking to the vendor.
- Thresholds (`HEALTHY_US = 30`, `BAD_US = 100`) were tuned on A100. On other
  architectures, run it once on a card you know is good and look at `warm`.
- If the card is busy with another job, `loaded` will be inflated. Measure on an
  idle system.
- A card in `EXCLUSIVE_PROCESS` compute mode may refuse a second context; the
  script reports the error, which is expected.

## License

MIT. See `LICENSE`.
