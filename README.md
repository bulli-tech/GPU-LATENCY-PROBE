# GPU-LATENCY-PROBE

Find out why a GPU server is slow when nothing in `nvidia-smi` looks wrong.

This tool measures the **cost of a CUDA API call** and how that cost behaves
under the two conditions every real workload has: idle gaps between calls, and
several threads touching the same card. On a healthy GPU the cost is a few
microseconds in every case. On an affected GPU it is fine in a tight loop and
falls apart everywhere else — which is exactly why the problem survives months
of investigation.

## The short version

- A GPU can report normal clocks, power, temperature and zero errors while every
  call from your process costs hundreds of microseconds instead of two.
- The penalty is almost invisible to a tight-loop benchmark. It appears when
  calls are **spaced out**, as they are between inference steps.
- It gets dramatically worse under **concurrent access** to the same card: a
  couple of threads turn microseconds into milliseconds or seconds, and can park
  a process in uninterruptible sleep.
- **A layout with one worker process per card often runs fine anyway.** That is
  why `pipeline_parallel=8` looked healthy on the same machine where tensor
  parallel 4 + expert parallel 4 crawled.

## The incident behind this tool

8× A100-SXM4-40GB, vLLM, TP=4 + EP=4, DeepSeek V4.1: **1.9 tok/s instead of
200+**. Profiler traces showed multi-second stalls on two of four ranks while
the other two were fine. Every layer of the stack was suspected in turn:
expert-offload kernels, the pinned allocator, the engram host callback, NCCL.

All of it was downstream of one fact: **some of the cards degrade when the driver
is accessed concurrently**. The same machine ran a different engine with
pipeline parallel 8 — one worker process, and therefore one driver channel, per
card — at full speed. That layout never triggers the collapse.

Measured on that machine:

| Card group | tight loop | spaced 10 ms | 1 loader thread | 4 loader threads |
|---|---|---|---|---|
| healthy (0, 3, 7) | 2.1–2.5 µs | 13.5 µs | 3.7–4.9 µs | 8.3–18.8 µs |
| affected (1, 2, 4, 5, 6) | 2.0–2.5 µs | 13.4 µs | 1347–1377 µs | 170–173 µs |

Note the first two columns: at rest and even after idle gaps **every card looks
identical**. The split appears only under concurrency. That is the trap — and it
is why a single-threaded benchmark proves nothing.

The affected cards also grouped by hardware revision, 8 of 8:

| GPUs | Board Part Number | VBIOS |
|---|---|---|
| 0, 3, 7 | 692-2G506-0200-**002** | 92.00.**36**.00.04 |
| 1, 2, 4, 5, 6 | 692-2G506-0200-**000** | 92.00.**0C**.00.05 |

The kernel log contained `Xid 119` (GSP RPC timeout) on an affected card — but
only after a stress test. It is a consequence, not the cause.

## Usage

```bash
python3 latency_probe.py                  # prompts for a choice when several cards exist
python3 latency_probe.py --all            # every card
python3 latency_probe.py -d 0 -d 3        # specific cards
python3 latency_probe.py --list           # list cards only
python3 latency_probe.py --gap 0.01       # idle gap for the spaced test, seconds
python3 latency_probe.py --threads 1,2,4  # thread counts to sweep
```

No dependencies: Python standard library plus `libcudart`, which ships with the
driver.

### Output

```
Probing 8 card(s)  (gap 10 ms, threads 1,2,4)
Values: median cost of one cudaEventQuery, microseconds.

  # model                       tight   spaced      1thr      4thr  verdict
---------------------------------------------------------------------------
  0 NVIDIA A100-SXM4-40GB         2.4     13.7       4.9      18.8  OK
  1 NVIDIA A100-SXM4-40GB         2.1     13.5    1347.7     170.0  CONTENTION
  3 NVIDIA A100-SXM4-40GB         2.1     13.5       4.5       9.8  OK
  7 NVIDIA A100-SXM4-40GB         2.5     13.6       3.7       8.3  OK
---------------------------------------------------------------------------

Healthy (3): GPU 0, GPU 3, GPU 7
Problem (5): GPU 1, GPU 2, GPU 4, GPU 5, GPU 6
```

Exit code: `0` all healthy, `1` problems found, `2` cannot run (no driver, no
GPUs). If something is missing the tool says exactly what:

```
CANNOT RUN: missing components

  - libcudart not found
  Linux: it ships with the driver and with the CUDA toolkit
    check: ls /usr/local/cuda/lib64/libcudart.so*
```

## What is measured, exactly

A CUDA event is recorded on a stream and that stream is synchronized. The event
is now **complete and stays complete**. Every later `cudaEventQuery` on it waits
for nothing on the GPU — it only asks the driver for a status. The elapsed CPU
wall time is therefore the cost of the call itself, not of any work.

Three measurements per card:

1. **tight** — 100 back-to-back calls. Keeps the driver path warm, so this is
   the best case and is nearly always fine, even on bad cards.
2. **spaced** — 60 calls with `--gap` seconds between them (default 10 ms). Each
   call pays the wake-up cost of the path, which is what inference sees between
   steps.
3. **loaded** — the tight loop repeated while `--threads` loader threads keep
   the same card busy on their own streams. Calls serialize on a driver lock;
   this is the column that separates good cards from bad ones.

A wall-clock budget stops any phase, so a frozen card cannot hang the tool. When
the budget runs out the card is reported `STALLED`, which is itself the finding.

The baseline for the relative test is the cheapest `loaded` number among the
probed cards. Comparing cards against each other is far more reliable than an
absolute threshold: spread between good cards is a few microseconds while a bad
card is hundreds of times worse.

### Note on cudaMalloc

Allocation happens **after** the first timing, never before. Allocation itself
goes through the driver path; doing it first warms the channel and hides the
effect. This was found the hard way: the first version of this tool reported a
healthy 2 µs on a card that costs 200 µs when idle.

## Verdicts

| Verdict | Meaning |
|---|---|
| `OK` | Cheap back-to-back, after idle gaps, and under threads. |
| `WARN` | Wake-up after idle is elevated but nothing collapses. Re-check on an idle machine. |
| `SLOW` | Even back-to-back calls are expensive. Every call pays a tax. |
| `DEGRADED` | An idle gap makes the next call cost hundreds of microseconds. |
| `CONTENTION` | A few threads turn the call into milliseconds — calls serialize on a driver lock. |
| `STALLED` | Calls were so slow the sample budget ran out; the card effectively froze. |

## What to do with the result

If some cards are `DEGRADED`, `CONTENTION` or `STALLED`:

1. **Group by hardware revision.** This is the fastest confirmation of a batch
   problem:

   ```bash
   for i in 0 1 2 3 4 5 6 7; do
     echo "=== GPU$i ==="
     nvidia-smi -i $i -q | grep -E "Serial Number|VBIOS Version|Board Part Number"
   done
   ```

2. **Check the kernel log** for GSP timeouts:

   ```bash
   sudo dmesg -T | grep -E "Xid|GSP" | tail -20
   ```

   `Xid 119` means the driver timed out waiting for the GSP. It confirms the
   driver path is in trouble but does not say why.

3. **Restructure the workload: one worker process per card.** A layout that
   gives each card its own process and its own driver channel (pipeline
   parallel, or tensor parallel with separate workers) keeps one channel busy
   and usually avoids the collapse entirely, even on affected cards. This is the
   only free workaround, and on our machine it was the difference between
   1.9 tok/s and full speed for another engine.

4. **Do not assume a driver upgrade fixes it.** Switching 610.57.04 → 580.196.07
   changed nothing in our case. Disabling the GSP path with
   `NVreg_EnableGpuFirmware=0` is not honoured on this hardware either: the
   module loads, but GSP stays active.

5. **Be careful with stress tests on affected cards.** Pushing several threads
   at them can drive the driver into a lock held by a thread stuck waiting for
   GSP; `nvidia-smi` then blocks too and processes sit in `D` state. If that
   happens, unload and reload the modules rather than killing processes.

## Caveats

- The measurement is relative, not calibrated. Compare cards within one machine,
  not against absolute numbers from another machine.
- Run it on a quiet machine. Load on the same card inflates the `loaded` column
  (intended); load on the CPU inflates everything.
- Thresholds were tuned on A100. On other architectures, run it on a card you
  trust and read `tight` and `spaced` directly instead of trusting the verdict.
- A card in `EXCLUSIVE_PROCESS` compute mode may refuse a second context; the
  tool reports the error, which is expected.
- The tool measures a symptom, not a root cause. It tells you **which cards**
  and **which access pattern** to avoid. Why a card behaves this way is a
  hardware/vendor question.

## License

MIT. See `LICENSE`.
