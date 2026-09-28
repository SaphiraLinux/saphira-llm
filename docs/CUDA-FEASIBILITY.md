# CUDA feasibility on Saphira Linux

Probed on homer, 2026-09-28. This is a recorded measurement, not an opinion.

## Verdict

**CUDA cannot run on a musl Saphira Linux container with the NVIDIA software as
shipped.** Both the driver library and the CUDA runtime are glibc-linked, and
NVIDIA publishes no musl build of either.

This is a hard blocker for the original framing of "NVIDIA support running on
Saphira Linux inside systemd-nspawn". It is not a packaging problem; it is a
libc problem in NVIDIA's proprietary userspace, and we do not control it.

## What was measured

| Item | Finding |
| --- | --- |
| `systemd-nspawn` on homer | present, `/usr/bin/systemd-nspawn`, systemd 261 |
| Saphira Linux rootfs for nspawn | **does not exist.** `/var/lib/machines` is empty; no rootfs tarball or squashfs found |
| GPU | NVIDIA RTX 3060, driver 610.57.04 (Open Kernel Module) |
| CUDA toolkit | 13.4.92 at `/opt/cuda` |
| `nvidia-container-toolkit` | **absent** — no `nvidia-container-runtime`, no `nvidia-ctk` |
| musl cross toolchain on homer | absent (homer is Arch/glibc) |

Linkage, which is what actually decides the question:

```
libcudart.so.13.4.92   NEEDED  libdl.so.2 libpthread.so.0 librt.so.1
                               libstdc++.so.6 libm.so.6
libcuda.so.1            NEEDED  libpthread.so.0 libm.so.6 libc.so.6
                               libdl.so.2 librt.so.1
```

`libcuda.so.1` carries symbol versions up to `GLIBC_2.9`. `libcuda.so.1` is
required even by a pure driver-API program, so there is no subset of CUDA that
avoids the glibc dependency. A search of `/opt/cuda` and `/usr/lib` for musl
variants found none.

## What this rules in and out

**Ruled out**

* Loading NVIDIA userspace into a musl Saphira Linux container as shipped.
* A musl build of `libcuda` or `libcudart` from NVIDIA. It does not exist.
* Building one ourselves in any maintainable sense. That means forking the
  proprietary driver userspace, and it would be a permanent maintenance burden
  tracking every 610.x driver release. Not a viable engineering answer.

**Still available, in rough order of sanity**

1. **Keep Saphira Linux CPU-only and host CUDA elsewhere on homer.** The
   runtime's premise is "mostly with no nvidia", and x86-64-v3 is the Saphira
   baseline, so CPU-only is the actual deliverable. CUDA becomes an out-of-tree
   vehicle on homer for two purposes only: producing a GPU reference, and
   measuring whether any model genuinely needs a GPU. This keeps the nspawn
   rootfs work entirely off the critical path.
2. **Run CUDA in a glibc container on homer** (Debian/Ubuntu) via
   `nvidia-container-toolkit`, which is not currently installed. Works, but
   means CUDA does not run on Saphira Linux, which was the stated aim. Honest
   to say plainly: this achieves GPU testing, not Saphira GPU support.
3. **Revisit if NVIDIA ever ships musl support.** Nothing to do but watch.

## Consequences for the plan

* Phase 9 as originally written is not executable as described. There is no
  Saphira Linux CUDA path to build against.
* There is no ready ggml CUDA ternary kernel to port regardless: upstream
  BitNet's GPU path is a separate PyTorch implementation under `gpu/`. So even
  a working CUDA environment would mean writing a new kernel, not porting one.
* The CUDA feasibility spike has done its job. The question was asked early
  precisely so it would be answered before effort was sunk into it, and the
  answer is no for the Saphira-hosted form.

The CPU work is unaffected and remains the critical path. Nothing in Phases 0
through 8 depends on this outcome.
