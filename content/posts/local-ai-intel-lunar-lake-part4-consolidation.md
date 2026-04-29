+++
title = "Running Local AI on Intel Lunar Lake — Part 4: Consolidating on Ollama with IPEX-LLM"
date = 2026-04-29T12:00:00+03:00
draft = false
description = "Collapsing two inference engines into a single Intel IPEX-LLM Ollama container — fewer moving parts, native FIM support, and the trade-offs of dropping the OpenVINO stack."
images = ["/img/image.jpg"]
series = ["Running Local AI on Intel Lunar Lake"]

[params]
  seriesPart = 4
  seriesTotalParts = 4
+++

*Part 4 of 4 in a series on building a fully local AI development environment on an Intel Core Ultra 7 268V laptop running Fedora 43.*

---

In Parts [1]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}})–[3]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}}), I built a working local AI stack with two separate inference engines: llama.cpp SYCL for the reasoning model and a custom OpenVINO GenAI server for code completion. It worked, but it was two containers, a Python venv, a handwritten FastAPI server, and a fragile dependency matrix — all to serve two models on the same GPU.

This part consolidates everything into a single Ollama instance running on Intel's IPEX-LLM container. Fewer moving parts, less to maintain, same result.

## Why consolidate?

The original two-engine architecture existed for good reasons: OpenVINO's calibration-aware INT4 quantization produces better quality at small model sizes, and it kept the NPU path open. But after living with the setup, the practical downsides outweighed the theoretical benefits:

- **The OpenVINO completion server was fragile.** The dependency matrix between `openvino`, `optimum-intel`, `nncf`, and `torch` broke twice during routine updates. The custom FastAPI wrapper needed a threading lock to avoid crash-on-concurrent-requests — a bug that only surfaced under real VS Code usage.
- **The NPU path is still blocked.** Nothing changed on the driver front. Keeping the OpenVINO stack around "just in case" was costing maintenance effort for zero benefit.
- **Two engines mean two failure modes.** When something went wrong, I had to diagnose whether it was llama.cpp, OpenVINO, the container, the venv, or the model. With one engine, the blast radius shrinks.

Intel's [IPEX-LLM](https://github.com/intel-analytics/ipex-llm) project provides a patched Ollama build that runs on Intel GPUs via SYCL. It ships as a container image with all the oneAPI dependencies bundled. One container, both models, native Ollama API.

## Setting up the IPEX-LLM Ollama container

### Running the container

```bash
podman run -d --name ollama-gpu \
  --device /dev/dri:/dev/dri \
  -v ~/models/ollama:/root/.ollama:Z \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0 \
  intelanalytics/ipex-llm-inference-cpp-xpu:latest \
  bash -c "source /llm/scripts/start-ollama.sh && sleep infinity"
```

A few things that aren't obvious:

1. **You can't just run `ollama serve`.** The `ollama` binary isn't in the container's `$PATH` — it lives at `/usr/local/lib/python3.11/dist-packages/bigdl/cpp/libs/ollama/ollama`. The container ships a startup script at `/llm/scripts/start-ollama.sh` that initializes the environment (`init-ollama`), sets `OLLAMA_NUM_GPU=999` and `ZES_ENABLE_SYSMAN=1`, then launches Ollama in the background. The `sleep infinity` keeps the container alive after the background process starts.

2. **Only `/dev/dri` is needed.** Unlike the llama.cpp container, which I passed both `/dev/dri` (GPU) and `/dev/accel` (NPU) into, Ollama has no NPU support, so only the GPU device matters.

3. **The volume mount** maps `~/models/ollama` to Ollama's model storage at `/root/.ollama` inside the container. Models persist across container restarts.

### Pulling the models

```bash
podman exec ollama-gpu bash -c "cd /llm/ollama && ./ollama pull qwen2.5-coder:1.5b-base"
podman exec ollama-gpu bash -c "cd /llm/ollama && ./ollama pull deepseek-r1:7b"
```

The `exec` commands need the full path dance (`cd /llm/ollama && ./ollama`) because the binary isn't in `$PATH`.

**Why `qwen2.5-coder:1.5b-base` and not `qwen2.5-coder:1.5b`?** The `-base` variant is a pure completion model — no chat template, no instruction tuning. This is what Fill-in-the-Middle (FIM) autocomplete needs. The instruct variant would try to chat with you instead of completing your code. Ollama natively supports the FIM endpoint that Continue.dev uses, so there's no need for the custom server wrapper I wrote in Part 2.

The DeepSeek-R1 7B model from Ollama's registry ships as Q4_K_M — the same quantization level as the GGUF file I was using with llama.cpp. No quality loss from the switch.

### Verifying

```bash
$ curl -s http://127.0.0.1:11434/v1/models | python3 -m json.tool
{
    "object": "list",
    "data": [
        {
            "id": "deepseek-r1:7b",
            "object": "model",
            ...
        },
        {
            "id": "qwen2.5-coder:1.5b-base",
            "object": "model",
            ...
        }
    ]
}
```

Both models visible through Ollama's OpenAI-compatible API on port 11434.

## Updating Continue.dev

The Continue config switches from `openai` provider to `ollama`:

```yaml
name: Local AI
version: "0.0.1"
models:
  - name: Qwen2.5-Coder 1.5B (GPU)
    provider: ollama
    model: qwen2.5-coder:1.5b-base
    apiBase: http://127.0.0.1:11434
    roles:
      - autocomplete
```

Key differences from the previous OpenVINO-backed config:

- **`provider: ollama`** instead of `openai` — Continue knows how to talk to Ollama's FIM endpoint natively, which is cleaner than routing through a generic OpenAI-compatible shim.
- **`apiBase`** points to `127.0.0.1:11434` (Ollama's port) instead of `:8081` (the old FastAPI server). Still using `127.0.0.1`, not `localhost` — the Podman IPv4-only port mapping issue from Part 3 hasn't gone anywhere.
- **No `apiKey`** — Ollama doesn't need one, and the `ollama` provider in Continue doesn't expect one.

## Updating Open WebUI

Open WebUI just needs to point at the new backend:

```bash
podman run -d --name open-webui --network host \
  -v ~/open-webui-data:/app/backend/data:Z \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:11434/v1 \
  -e OPENAI_API_KEY=none \
  -e WEBUI_AUTH=false \
  -e PORT=3000 \
  ghcr.io/open-webui/open-webui:main
```

The only change from Part 3 is `OPENAI_API_BASE_URL` — from `http://127.0.0.1:8080/v1` (llama.cpp) to `http://127.0.0.1:11434/v1` (Ollama). Open WebUI discovers both models automatically.

## Replacing the systemd services

The three old services (llama-gpu, completion-server, open-webui) collapse into two.

### The Ollama service (auto-start)

```ini
[Unit]
Description=IPEX-LLM Ollama server (Qwen2.5-Coder 1.5B on Intel GPU)
After=default.target

[Service]
Type=simple
ExecStartPre=-/usr/bin/podman rm -f ollama-gpu
ExecStart=/usr/bin/podman run --rm --name ollama-gpu \
  --device /dev/dri:/dev/dri \
  -v %h/models/ollama:/root/.ollama:Z \
  -p 11434:11434 \
  -e OLLAMA_HOST=0.0.0.0 \
  intelanalytics/ipex-llm-inference-cpp-xpu:latest \
  bash -c "source /llm/scripts/start-ollama.sh && sleep infinity"
ExecStop=/usr/bin/podman stop ollama-gpu
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

### Open WebUI (on-demand)

```ini
[Unit]
Description=Open WebUI chat interface
After=ollama-gpu.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/podman rm -f open-webui
ExecStart=/usr/bin/podman run --rm --name open-webui --network host \
  -v %h/open-webui-data:/app/backend/data:Z \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:11434/v1 \
  -e OPENAI_API_KEY=none \
  -e WEBUI_AUTH=false \
  -e PORT=3000 \
  ghcr.io/open-webui/open-webui:main
ExecStop=/usr/bin/podman stop open-webui
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

The `After=ollama-gpu.service` ordering ensures the Ollama backend is up before Open WebUI tries to query it.

### Cutting over

```bash
# Stop and disable the old services
systemctl --user stop completion-server llama-gpu
systemctl --user disable completion-server llama-gpu

# Remove the old llama-gpu service file (no longer needed)
rm ~/.config/systemd/user/llama-gpu.service

# Enable the new services
systemctl --user daemon-reload
systemctl --user enable --now ollama-gpu.service
```

The completion-server service file can stay on disk (disabled) as a fallback if you ever need to revert to the OpenVINO setup.

### The new daily workflow

```bash
# Ollama (with autocomplete model) is already running at boot

# Need the chat UI?
systemctl --user start open-webui

# Done chatting:
systemctl --user stop open-webui

# Check status:
systemctl --user status ollama-gpu open-webui

# Ollama logs:
journalctl --user -u ollama-gpu -f
```

## What was removed

| Component | Status |
|---|---|
| `llama-gpu.service` | Deleted — llama.cpp SYCL container replaced by Ollama |
| `completion-server.service` | Disabled — custom FastAPI server no longer needed |
| Python venv (`~/venvs/openvino-npu`) | Can be removed — no longer in the serving path |
| `npu_server.py` | Unused — kept as reference |

The OpenVINO venv and conversion scripts still exist on disk. They're useful if you ever need to re-export models with calibration-aware quantization, but they're no longer part of the running system.

## Trade-offs

Nothing is free. Here's what changed:

| | IPEX-LLM Ollama (new) | OpenVINO + llama.cpp (old) |
|---|---|---|
| **Quantization** | GGUF Q4_K_M (post-training) | NNCF INT4 (calibration-aware) for completion; GGUF Q4_K_M for reasoning |
| **Quality at 1.5B** | Slightly lower for completion | Better — calibration matters more at small sizes |
| **Setup complexity** | One container, zero custom code | Two containers, a venv, and a FastAPI server |
| **NPU future path** | None — Ollama has no NPU support | One-line device change when driver matures |
| **FIM support** | Native in Ollama | Had to implement manually |
| **Failure modes** | One | Three |

The quantization quality difference at 1.5B is real but marginal in practice — the model is small enough that neither quantization method produces stellar completions. The simplicity gain is worth more than the theoretical quality edge.

The NPU path is the real loss. If Intel's NPU driver matures and you want to offload the completion model, you'd need to go back to the OpenVINO stack. But that's a future problem for a future driver release.

## Final architecture

```
VS Code (Continue.dev)
  └─ Tab autocomplete ──▶ ollama-gpu :11434  (qwen2.5-coder:1.5b-base)

Browser (Open WebUI :3000)
  └─ Chat / reasoning ──▶ ollama-gpu :11434  (deepseek-r1:7b)
```

One container. Two models. Both on the GPU. Everything behind systemd.

---

*Series: Running Local AI on Intel Lunar Lake*
- [Part 1: Hardware, Drivers, and the Intel Compute Stack]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}})
- [Part 2: Models, Inference Engines, and the NPU That Almost Worked]({{< ref "local-ai-intel-lunar-lake-part2-models" >}})
- [Part 3: VS Code, Open WebUI, and Running It All as Services]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}})
- **Part 4: Consolidating on Ollama with IPEX-LLM** (you are here)
