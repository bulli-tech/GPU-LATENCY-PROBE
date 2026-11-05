# The GSP trap: how an "open" kernel module silently made 5 of 8 GPUs slow

A complete record of one investigation: what looked like faulty hardware, what
actually was the problem, and how to detect and fix it.

Short version: **the GPU driver was installed as the open kernel modules
(`Dual MIT/GPL`), NVIDIA's GSP firmware was active, and on five of eight A100
cards the GSP path collapses under concurrent access.** Loading the
**proprietary** modules with `NVreg_EnableGpuFirmware=0` fixed all of them
completely — 300x faster on the affected cards.

---

## 1. The symptom

Server: 8× A100-SXM4-40GB, NVSwitch, Debian 13, kernel 6.12.105.

A DeepSeek V4.1 model under vLLM with TP=4 + EP=4 ran at **1.9 tok/s instead of
200+**. Profiler traces showed 0.65–2.85 s stalls on two of four ranks while the
other two showed 0.2 ms.

The same machine ran a different engine with **pipeline parallel 8** — eight
processes, one per card — at full speed. That was the first real clue and it was
almost missed.

## 2. What was suspected and ruled out

Every layer of the stack was investigated. None of it was the cause:

| Suspect | Finding |
|---|---|
| Expert-offload kernels | Real bugs found and fixed (uninitialised resident slots, a block race in `plan_kernel`, a wrong overflow count). Throughput unchanged. |
| Pinned allocator | Moving allocator events to a background thread removed the visible stall, speed did not improve. |
| Engram host callback | Isolated empty callbacks cost 0.07–0.5 ms, not seconds. |
| NCCL | A small AllGather measured 55 µs. AllGathers in the trace were *waiting* for the slow ranks, not causing them. |
| Driver version | 610.57.04 → 580.196.07 (CUDA 13.3 → 13.0): not a single number changed. |
| `NVreg_EnableGpuFirmware=0` on open modules | Module loads, GSP stays active, no effect — **because the open modules do not honour it.** This was the hidden trap. |

## 3. The measurement that isolated the problem

Measure the cost of a CUDA call from the CPU side, on work that has already
completed. An event that is complete makes `cudaEventQuery` wait for nothing on
the GPU, so the elapsed time is the cost of the path CPU → driver → device.

Three conditions matter, and only the last two reveal anything:

| Condition | Healthy card | Affected card |
|---|---|---|
| **tight** — 100 back-to-back calls | 2.1–2.5 µs | 2.0–2.5 µs |
| **spaced** — calls 10 ms apart | 13.5 µs | 13.4 µs |
| **loaded** — 1 loader thread | 3.7–4.9 µs | **1347–1377 µs** |
| **loaded** — 4 loader threads | 8.3–18.8 µs | 170–173 µs |

Two lessons worth more than the numbers:

- **A tight-loop benchmark proves nothing.** At rest and with idle gaps every
  card looks identical. The 300x difference appears only under concurrency.
- **`cudaMalloc` must not be the first call** in such a test. Allocation itself
  goes through the driver path and warms it; doing it first hides the effect
  entirely.

### Why concurrency is the trigger

Several threads or contexts contend for the NVRM driver lock. On the affected
cards the lock holder blocks waiting for a GSP RPC reply; other processes end up
in **D state** (uninterruptible sleep). The kernel log shows:

```
NVRM: Xid (PCI:0000:10:00): 119, pid=3728, name=python3,
      Timeout after 45s of waiting for RPC response from GPU1 GSP!
      Expected function 76 (GSP_RM_CONTROL) sequence 1436
```

`Xid 119` is a **consequence**, not the cause. It only appears once a stress test
has pushed the driver into the lock.

**Warning:** a stress test with 4–8 threads on an affected card can wedge the
driver completely — `nvidia-smi` stops responding too and processes hang in `D`.
Killing them does not help; modules must be unloaded and reloaded, or the box
rebooted.

## 4. The actual root cause

```
$ modinfo nvidia | grep -i license
license:        Dual MIT/GPL
```

`Dual MIT/GPL` means **open kernel modules**. The GSP firmware is mandatory
there: `NVreg_EnableGpuFirmware=0` is accepted, the module loads, and GSP stays
active anyway. Every earlier attempt to disable it failed for exactly this
reason and was wrongly recorded as "not supported on A100".

With the **proprietary** modules the parameter is honoured:

```
license:        NVIDIA
GSP Firmware Version : N/A      (all 8 cards)
```

And the collapse disappears:

| Card group | open modules, GSP on | proprietary, GSP off |
|---|---|---|
| healthy (0, 3, 7) | 3.7–4.9 µs | 4.7–4.9 µs |
| affected (1, 2, 4, 5, 6) | **1347–1377 µs** | **4.5–5.4 µs** |

## 5. The hardware detail that is real but secondary

The cards group perfectly by production batch, 8 of 8:

| GPUs | Board Part Number | Board ID | VBIOS |
|---|---|---|---|
| 0, 3, 7 | 692-2G506-0200-**002** | 0xb00 | 92.00.**36**.00.04 |
| 1, 2, 4, 5, 6 | 692-2G506-0200-**000** | 0x1000 | 92.00.**0C**.00.05 |

Both groups use GPU Part Number `20B0-884-A1` and identical Inforom `92.00.36.00.04`.
The difference is real: on the `92.00.0C.00.05` cards the GSP path does not
survive contention. But it is not a broken card, and it does not require new
hardware — it requires the driver to stop using GSP.

This also means: **two batches are fine as long as GSP is off.** No RMA needed.

## 6. The fix, step by step

### 6.1 Check which module type you have

```bash
modinfo nvidia | grep -i license
# NVIDIA          -> proprietary, good
# Dual MIT/GPL    -> open, must be replaced
```

### 6.2 Install the proprietary modules

**`.run` installer:**

```bash
sh NVIDIA-Linux-x86_64-580.196.07.run --kernel-module-type=proprietary
```

Verify the flag exists on your version: `nvidia-installer -A | grep kernel-module-type`.

Note that `--print-recommended-kernel-module-type` may answer `open` even for
A100 — the recommendation is about features, not about this problem.

**Ubuntu:** `nvidia-driver-580` instead of `nvidia-driver-580-open`
**RHEL and clones:** `nvidia-driver:580-dkms` instead of `580-open`

Fabric Manager must be the **same version** as the driver, otherwise NVSwitch
will not come up.

### 6.3 Disable GSP

```bash
echo "options nvidia NVreg_EnableGpuFirmware=0" | sudo tee /etc/modprobe.d/nvidia-gsp.conf
sudo update-initramfs -u     # Debian/Ubuntu
# sudo dracut -f             # RHEL and clones
sudo reboot
```

The initramfs step matters: without it the option is lost on the next boot and
GSP comes back. Container OS images (Talos, Flatcar) need the kernel argument
instead: `nvidia.NVreg_EnableGpuFirmware=0`.

For the current boot without a reboot, unload and reload the modules:

```bash
systemctl stop nvidia-fabricmanager nvidia-persistenced
rmmod nvidia_drm nvidia_peermem nvidia_modeset nvidia_uvm nvidia
modprobe nvidia NVreg_EnableGpuFirmware=0
modprobe nvidia_uvm
systemctl start nvidia-persistenced nvidia-fabricmanager
```

### 6.4 Verify

```bash
modinfo nvidia | grep -i license            # must say NVIDIA
nvidia-smi -q | grep -i "GSP Firmware"      # must say N/A on every card
```

If it still reports a version like `580.196.07`, GSP did not turn off — almost
always because the modules are still the open ones, or the option never made it
into the initramfs.

---

## 7. Building modules on this server (gotchas)

This machine has a specific trap worth recording.

The kernel headers are split across two trees:

```
/lib/modules/6.12.105+deb13-amd64/build  -> linux-headers-6.12.105+deb13-amd64   (arch, config, generated)
/lib/modules/6.12.105+deb13-amd64/source -> linux-headers-6.12.105+deb13-common  (include/linux and the rest)
```

The installer passes `SYSSRC` and `SYSOUT` pointing at the **same** directory
`build`. NVIDIA's `conftest.sh` then looks for `include/linux/...` in the arch
tree, does not find it, every probe silently fails, and you get this:

```
nv-caps.c: error: passing argument 4 of 'proc_create_data' from incompatible pointer type
nvswitch_event.h: error: macro "__assign_str" passed 2 arguments, but takes just 1
```

That looks like "the driver is incompatible with kernel 6.12". It is not. The
probes simply returned the defaults for old kernels. Building with the two trees
passed explicitly fixes it:

```bash
make -C kernel-open \
  NV_KERNEL_SOURCE=/usr/src/linux-headers-6.12.105+deb13-common \
  NV_KERNEL_OUTPUT=/usr/src/linux-headers-6.12.105+deb13-amd64 \
  modules modules_install
```

Confirm the probes ran correctly:

```bash
grep -E "NV_PROC_OPS_PRESENT|NV_ASSIGN_STR_ARGUMENT_COUNT" conftest/types.h conftest/functions.h
# expect: #define NV_PROC_OPS_PRESENT and #define NV_ASSIGN_STR_ARGUMENT_COUNT 1
```

Also worth knowing: this host had missing symlinks in
`/usr/src/linux-headers-6.12.105+deb13-amd64/include` pointing at the `-common`
tree. Without them no NVIDIA module builds at all. Do not remove them.

---

## 8. Paths and layout on our server

| Item | Host path | Container path |
|---|---|---|
| Model | `/home/srv02/models/ds41-unc-exl3-2.0bpw-k2-calibrated` | `/models/ds41-unc-exl3-2.0bpw-k2-calibrated` |
| Engram tables | `/home/srv02/models/deepseek-v4.1-1017-exl3-engram` | `/models/deepseek-v4.1-1017-exl3-engram` |
| vLLM sources | `/home/srv02/vllm-src` | `/opt/vllm-src` |
| Candidates, traces | `/home/srv02/cache/dsv41-host-candidate-20261104/` | `/cache/dsv41-host-candidate-20261104/` |
| Driver installers, backups | `/home/srv02/backup/rollback-13.0/` | — |

Kernel modules land in `/lib/modules/6.12.105+deb13-amd64/kernel/drivers/video/`.

The driver here is installed by the `.run` installer, not by a package, so
`dpkg` knows nothing about it. Backups that exist:

```
/home/srv02/backup/rollback-13.0/NVIDIA-Linux-x86_64-580.196.07.run   (sha256 verified)
/home/srv02/backup/rollback-13.0/NVIDIA-Linux-x86_64-610.57.04.run   (sha256 verified)
/home/srv02/backup/rollback-13.0/fm-580.196.07.tar.xz                (matching Fabric Manager)
/home/srv02/backup/pre-580/nv-fabricmanager                           (previous FM binary)
```

---

## 9. Checklist for the same problem elsewhere

1. `python3 latency_probe.py --all --threads 1,4` — does the `loaded` column
   explode on some cards while `tight` and `spaced` look fine?
2. `modinfo nvidia | grep -i license` — is it `Dual MIT/GPL`?
3. If yes, that alone may be the whole story. Switch to proprietary modules and
   disable GSP before investigating anything else.
4. `dmesg | grep Xid` — `Xid 119` confirms the GSP path is timing out.
5. Group cards by `VBIOS` and `Board Part Number` only *after* step 3. In our
   case the grouping matched perfectly and still pointed at the wrong
   conclusion, because the batch difference only matters while GSP is enabled.
6. Remember the fast workaround that needs no driver change: **one process per
   card** (pipeline parallel, or tensor parallel with separate workers) keeps one
   driver channel per device and avoids the contention entirely.

## 10. Timeline, honestly

| Step | Conclusion at the time | Status now |
|---|---|---|
| Profiler shows stalls on 2 ranks | "NCCL is waiting" | wrong: ranks wait for the driver lock |
| Engram callback interval has a 2 s gap | "the host callback is slow" | wrong: the callback waits for the driver |
| `cudaEventQuery` costs 158 µs on 5 cards | "those cards are faulty" | wrong: GSP path, not hardware |
| Cards group by VBIOS 8 of 8 | "two batches, -000 is defective" | partly right: different GSP behaviour, not a defect |
| Driver downgrade changes nothing | "not a driver version issue" | right, but for the wrong reason |
| `NVreg_EnableGpuFirmware=0` does nothing | "GSP cannot be disabled on A100" | wrong: open modules ignore it |
| Proprietary modules + GSP off | — | **all 8 cards healthy** |

The single most useful lesson: when a parameter seems ignored, check **which
kernel module variant is loaded** before concluding that the hardware does not
support it.

## License

MIT. See `LICENSE`.
