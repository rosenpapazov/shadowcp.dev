+++
title = "Running Local AI on Intel Lunar Lake — Part 1: Hardware, Drivers, and the Intel Compute Stack"
date = 2026-04-06T12:00:00+03:00
draft = false
description = "Setting up the Intel compute stack on Fedora 43 for local AI inference — GPU and NPU drivers, memory budgeting, and choosing the right models for a Lunar Lake laptop."
images = ["/img/image.jpg"]
series = ["Running Local AI on Intel Lunar Lake"]

[params]
  seriesPart = 1
  seriesTotalParts = 4
+++

*Part 1 of 4 in a series on building a fully local AI development environment on an Intel Core Ultra 7 268V laptop running Fedora 43.*

---

## Why local AI?

Cloud-hosted AI is convenient until it isn't. Latency spikes, privacy concerns with proprietary code, rate limits during crunch time, and the monthly bill that creeps up. Running models locally solves all of these — if your hardware can handle it.

Intel's Lunar Lake processors are interesting for this. They ship with three compute engines: a CPU, an integrated Arc GPU, and a dedicated NPU (Neural Processing Unit). The question is whether the Linux software stack is mature enough to actually use them for LLM inference.

This series documents my experience setting it up on Fedora 43. Spoiler: it works, but the path has sharp edges.

## The hardware

The Intel Core Ultra 7 268V (Lunar Lake) packs:

- **CPU:** 8 cores (4P + 4E), up to 5.0 GHz
- **GPU:** Intel Arc 130V/140V — 8 Xe2 cores at up to 2.0 GHz
- **NPU:** Intel AI Boost — 48 TOPS at INT8
- **RAM:** 30 GB LPDDR5x-8533, on-package

The critical detail is **shared memory**. There is no dedicated VRAM. The GPU, NPU, and CPU all draw from the same 30 GB pool. Every gigabyte allocated to a model is a gigabyte taken from the system. This shapes every decision that follows.

## What we're building

Two inference stacks, each serving a different purpose:

1. **Code autocompletion** — a small, fast model (Qwen2.5-Coder-1.5B, INT4 quantized) served via OpenVINO on the integrated GPU, consumed by VS Code through the Continue.dev extension.
2. **IT reasoning and debugging** — a larger chain-of-thought model (DeepSeek-R1-Distill-Qwen-7B, Q4_K_M) served via llama.cpp with the SYCL backend, accessible through Open WebUI in a browser.

The architecture:

```
VS Code (Continue.dev)
  └─ Tab autocomplete ──▶ completion-server :8081  (OpenVINO GenAI)

Browser (Open WebUI :3000)
  └─ Chat / reasoning ──▶ llama-gpu :8080  (llama.cpp SYCL)
```

Both run as systemd user services. The completion server auto-starts on login; the reasoning stack starts on demand.

## Installing the Intel compute stack

Fedora 43 ships kernel 6.19, which has the `intel_vpu` (NPU) and `xe` (GPU) drivers built in. The kernel side works out of the box — `/dev/dri/renderD128` for the GPU and `/dev/accel/accel0` for the NPU are both present at boot. What's missing is the userspace.

### GPU compute runtime

The GPU needs OpenCL and Level Zero libraries for compute workloads. Fedora has these packaged:

```bash
sudo dnf install intel-compute-runtime intel-opencl intel-level-zero oneapi-level-zero
```

A subtle point: `intel-level-zero` is the GPU-specific Level Zero *backend*. `oneapi-level-zero` is the Level Zero *loader* (`libze_loader.so.1`). You need both. Without the loader, nothing can discover the GPU backend — OpenVINO will see `CPU` only.

I spent time debugging this because the package names suggest they're the same thing. They aren't.

### NPU userspace driver — building from source

This is where Fedora's packaging falls short. The `intel-npu-driver` package exists in Fedora Rawhide but isn't available in Fedora 43's repos. Intel's official releases on GitHub only provide `.deb` packages for Ubuntu. The community COPR that used to provide RPMs was archived in October 2025.

That leaves building from source:

```bash
git clone --depth 1 https://github.com/intel/linux-npu-driver.git /tmp/linux-npu-driver
cd /tmp/linux-npu-driver
git submodule update --init --recursive
cmake -B build -S . \
  -DCMAKE_CXX_FLAGS="-fcf-protection=none" \
  -DCMAKE_C_FLAGS="-fcf-protection=none"
cmake --build build --parallel $(nproc)
sudo cmake --install build
```

The `-fcf-protection=none` flag is needed because the driver's build system doesn't account for Fedora's default control-flow enforcement settings.

The resulting `libze_intel_npu.so` installs to `/usr/local/lib64/`, which isn't in Fedora's default library search path:

```bash
echo /usr/local/lib64 | sudo tee /etc/ld.so.conf.d/local-lib64.conf
sudo ldconfig
```

### Verification

After installation and a re-login (to pick up group membership changes):

```bash
$ python3 -c "from openvino import Core; print(Core().available_devices)"
['CPU', 'GPU', 'NPU']
```

All three devices visible. The foundation is in place.

## The memory budget

Before choosing models, I mapped out the memory constraints. With 30 GB total and ~13 GB typically consumed by the OS, desktop, and VS Code, I have roughly 17 GB for AI workloads. But I want headroom for KV cache, context windows, and not swapping under load. My practical budget is ~8 GB for models.

| Component | RAM |
|---|---|
| 7B model (Q4_K_M) + KV cache | ~5.5 GB |
| 1.5B model (INT4) + overhead | ~1.2 GB |
| Open WebUI container | ~0.5 GB |
| **AI total** | **~7.2 GB** |
| OS + desktop + VS Code | ~13 GB |
| **System total** | **~20 GB** |
| **Remaining** | **~10 GB** |

This rules out 14B+ models for the reasoning stack (Q4_K_M would be ~8.5 GB, leaving almost nothing for context) and confirms that two models can coexist comfortably.

## Why these specific models?

### Qwen2.5-Coder-1.5B for completion

For inline code completion, latency matters more than capability. The model needs to respond in under a second to feel useful. Qwen2.5-Coder-1.5B hits the sweet spot:

- Outperforms StarCoder2-3B on code completion benchmarks despite being half the size
- At INT4 quantization, the entire model fits in under 1 GB
- Fast enough for real-time autocomplete on integrated GPU

### DeepSeek-R1-Distill-Qwen-7B for reasoning

For debugging and IT problem-solving, I wanted chain-of-thought reasoning — the model should show its work, not just give an answer. DeepSeek-R1's distilled variants inherit this behavior from the full R1 model.

The 7B variant on the Qwen 2.5 base is the largest that fits comfortably in the memory budget. At Q4_K_M quantization (4.4 GB file), it leaves room for a 4096-token context window.

Observed throughput: **~16.5 tokens/sec** on the Arc integrated GPU. Not fast, but usable for interactive chat where you're reading the response as it generates.

## What's next

In [Part 2]({{< ref "local-ai-intel-lunar-lake-part2-models" >}}), I cover the model conversion pipeline, the two inference engines (llama.cpp SYCL and OpenVINO GenAI), writing a custom API server, and the pitfalls I hit along the way — including a Python version incompatibility, a dependency matrix from hell, and the NPU that almost worked.

In [Part 3]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}}), I wire everything into VS Code and Open WebUI, create systemd services for lifecycle management, and share the operational lessons learned.

---

*Series: Running Local AI on Intel Lunar Lake*
- **Part 1: Hardware, Drivers, and the Intel Compute Stack** (you are here)
- [Part 2: Models, Inference Engines, and the NPU That Almost Worked]({{< ref "local-ai-intel-lunar-lake-part2-models" >}})
- [Part 3: VS Code, Open WebUI, and Running It All as Services]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}})
- Part 4: Consolidating on Ollama with IPEX-LLM *(coming soon)*
