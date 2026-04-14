+++
title = "Running Local AI on Intel Lunar Lake — Part 3: VS Code, Open WebUI, and Running It All as Services"
date = 2026-04-14T12:00:00+03:00
draft = false
description = "Wiring local AI models into VS Code and Open WebUI, creating systemd user services for lifecycle management, and the operational lessons from running it all daily."
images = ["/img/image.jpg"]
series = ["Running Local AI on Intel Lunar Lake"]

[params]
  seriesPart = 3
  seriesTotalParts = 4
+++

*Part 3 of 4 in a series on building a fully local AI development environment on an Intel Core Ultra 7 268V laptop running Fedora 43.*

---

In [Part 1]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}}), I set up the Intel compute stack. In [Part 2]({{< ref "local-ai-intel-lunar-lake-part2-models" >}}), I got two models running — a 7B reasoning model via llama.cpp SYCL and a 1.5B completion model via OpenVINO GenAI. Now it's time to wire them into the tools I actually use.

## VS Code: Continue.dev for autocomplete

The [Continue](https://continue.dev/) extension connects VS Code to local (or remote) AI models. It supports tab autocomplete, inline chat, and more. I only use it for autocomplete — the rest is noise for my workflow.

### Installing and configuring

Install from the VS Code marketplace (`continue.continue`). Continue v1.2+ uses a YAML config file at `~/.continue/config.yaml`. Here's the gotcha: the config **requires** top-level `name` and `version` fields, or it silently rejects the entire configuration and shows a persistent error badge in the status bar.

```yaml
name: Local AI
version: "0.0.1"
models:
  - name: Qwen2.5-Coder 1.5B (GPU)
    provider: openai
    model: qwen2.5-coder-1.5b
    apiBase: http://127.0.0.1:8081/v1
    apiKey: none
    roles:
      - autocomplete
```

Key points:

- **`provider: openai`** — Continue treats this as "any OpenAI-compatible API," which is exactly what the custom FastAPI server exposes.
- **`apiBase`** — Must use `127.0.0.1`, not `localhost`. On Fedora, `localhost` resolves to `::1` (IPv6) first, and the server binds IPv4 only. This caused intermittent failures that were maddening to debug.
- **`apiKey: none`** — The server doesn't check keys, but Continue requires the field to be present.
- **`roles: [autocomplete]`** — Restricts this model to tab completion only. Without this, Continue would also try to use it for chat, which a 1.5B model handles poorly.

### FIM vs. next-edit predictions

Continue supports two autocomplete modes: Fill-in-the-Middle (FIM) and Next Edit Prediction (NEP). FIM inserts code at the cursor position using surrounding context. NEP predicts what you'll edit next based on recent changes — it's more aggressive but requires a model trained for it.

Qwen2.5-Coder supports FIM natively. I didn't enable NEP because the 1.5B model isn't large enough to predict edits reliably, and the latency penalty of wrong predictions is worse than no prediction at all.

## Open WebUI for chat

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-like interface for local models. It connects to any OpenAI-compatible API.

### Deployment

```bash
podman run -d --name open-webui --network host \
  -v ~/open-webui-data:/app/backend/data:Z \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
  -e OPENAI_API_KEY=none \
  -e WEBUI_AUTH=false \
  -e PORT=3000 \
  ghcr.io/open-webui/open-webui:main
```

Three things to note:

1. **`-e PORT=3000`** — Open WebUI defaults to port 8080, which collides with the llama.cpp server. I wasted time wondering why the reasoning model was returning HTML before realizing both services were fighting over the same port.

2. **`--network host`** — Required so that Open WebUI can reach `127.0.0.1:8080` (the llama.cpp server). With Podman's default bridge networking, `127.0.0.1` inside the container refers to the container itself, not the host.

3. **`WEBUI_AUTH=false`** — Disables the login screen. This is a single-user laptop; authentication adds friction for no benefit.

Access at `http://127.0.0.1:3000`. The DeepSeek-R1 model appears automatically once Open WebUI queries the llama.cpp `/v1/models` endpoint.

## Systemd user services

I want the completion server to start on login (autocomplete should always be there) and the reasoning stack to start on demand (I don't need a 5.5 GB model loaded unless I'm actively debugging something).

All services live in `~/.config/systemd/user/`.

### The completion server (auto-start)

```ini
[Unit]
Description=Code completion server (Qwen2.5-Coder-1.5B via OpenVINO)
After=default.target

[Service]
Type=simple
WorkingDirectory=%h/tmp/local-ai
ExecStart=%h/venvs/openvino-npu/bin/uvicorn npu_server:app --host 127.0.0.1 --port 8081
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- **`%h`** expands to the user's home directory in systemd user units.
- The `ExecStart` points directly to the venv's `uvicorn` binary — no `source activate` needed.
- `Restart=on-failure` with a 5-second backoff handles occasional OpenVINO crashes during high load.

Enable it:

```bash
systemctl --user enable --now completion-server.service
```

### The reasoning model (on-demand)

```ini
[Unit]
Description=llama.cpp GPU server (DeepSeek-R1 7B)
After=default.target

[Service]
Type=simple
ExecStartPre=-/usr/bin/podman rm -f llama-gpu
ExecStart=/usr/bin/podman run --rm --name llama-gpu \
  --device /dev/dri:/dev/dri --device /dev/accel:/dev/accel \
  -v %h/models:/models:Z -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-intel \
  -m /models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  -c 4096 -ngl 99 --host 0.0.0.0 --port 8080
ExecStop=/usr/bin/podman stop llama-gpu
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

- **`ExecStartPre=-/usr/bin/podman rm -f llama-gpu`** — The `-` prefix means "don't fail if this errors." It cleans up any leftover container from a previous crash. Without this, `podman run --name llama-gpu` fails if a stopped container with that name still exists.
- **`--device /dev/dri:/dev/dri --device /dev/accel:/dev/accel`** — Passes both GPU and NPU devices into the container. The NPU device isn't used yet but costs nothing to include.
- The `:Z` volume flag triggers SELinux relabeling — mandatory on Fedora or the container can't read the model files.

This service is installed but **not** enabled:

```bash
systemctl --user enable llama-gpu.service  # install, but don't start at boot
```

Start it when needed:

```bash
systemctl --user start llama-gpu.service
```

### Open WebUI (on-demand, follows the reasoning model)

```ini
[Unit]
Description=Open WebUI chat interface
After=llama-gpu.service

[Service]
Type=simple
ExecStartPre=-/usr/bin/podman rm -f open-webui
ExecStart=/usr/bin/podman run --rm --name open-webui --network host \
  -v %h/open-webui-data:/app/backend/data:Z \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8080/v1 \
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

- **`After=llama-gpu.service`** — Ensures the reasoning model is started first when both are started together. Note that `After` only orders startup; it doesn't create a hard dependency. Open WebUI will still start even if the reasoning model isn't running — it just won't find any models until the backend comes up.
- **`RestartSec=10`** — Longer backoff than the other services. Open WebUI takes a few seconds to initialize, and rapid restarts tend to produce port conflicts.

### The daily workflow

```bash
# Morning: completion server is already running (auto-start)
# Need the reasoning stack? One command:
systemctl --user start llama-gpu.service open-webui.service

# Done for now:
systemctl --user stop llama-gpu.service open-webui.service

# Check what's running:
systemctl --user status completion-server llama-gpu open-webui

# View logs:
journalctl --user -u completion-server -f
journalctl --user -u llama-gpu -f
```

### Surviving reboots

For systemd user services to start at boot (before the user logs in), you need lingering enabled:

```bash
loginctl enable-linger $USER
```

Without this, user services only start when you log in — which is actually fine for a laptop. I enabled lingering anyway so the completion server is ready by the time VS Code starts.

## Memory in practice

With everything running, here's what the memory landscape looks like:

| Component | RAM |
|---|---|
| OS + desktop + VS Code | ~13 GB |
| Completion server (OpenVINO, Qwen 1.5B INT4) | ~1.2 GB |
| Reasoning model (llama.cpp, DeepSeek 7B Q4_K_M) | ~5.5 GB |
| Open WebUI | ~0.5 GB |
| **Total** | **~20.2 GB** |
| **Free (of 30 GB)** | **~9.8 GB** |

When the reasoning stack is stopped, system memory use drops to ~14 GB, leaving plenty of headroom for other work. The completion server's 1.2 GB footprint is small enough to leave running permanently without noticing it.

## Things I'd do differently

1. **Start with the GPU for everything.** I spent days chasing NPU support that isn't ready yet. The GPU handles both models fine. Ship the working setup first, optimize later.

2. **Pin OpenVINO versions from day one.** The dependency matrix between `openvino`, `optimum-intel`, `nncf`, and `torch` is fragile. I should have recorded the working combination immediately instead of discovering it through trial and error a second time after a venv got corrupted during debugging.

3. **Use `127.0.0.1` everywhere from the start.** The IPv4/IPv6 issue with Podman and Fedora's `localhost` resolution cost me 20 minutes of confusion. Just default to explicit IPv4 addresses.

4. **Don't underestimate tokenizer generation.** The OpenVINO model conversion and tokenizer XML generation are separate steps, and the XMLs must match the runtime version. This tripped me up twice.

## Final state

The setup delivers what I wanted: fully local AI assistance that works offline, has zero latency variance, costs nothing per token, and keeps proprietary code off third-party servers. The Arc iGPU won't win any benchmarks, but ~16.5 tokens/sec for reasoning and sub-second autocomplete completions are usable for daily work.

The NPU sits idle for now. When the Linux driver catches up — specifically, when it supports the `NPU_MAX_TILES` property and ships with a working model compiler — switching the completion model over will be a one-line config change.

Until then, the GPU does the job.

---

*Series: Running Local AI on Intel Lunar Lake*
- [Part 1: Hardware, Drivers, and the Intel Compute Stack]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}})
- [Part 2: Models, Inference Engines, and the NPU That Almost Worked]({{< ref "local-ai-intel-lunar-lake-part2-models" >}})
- **Part 3: VS Code, Open WebUI, and Running It All as Services** (you are here)
- Part 4: Consolidating on Ollama with IPEX-LLM *(coming soon)*
