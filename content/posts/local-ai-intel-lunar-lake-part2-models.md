+++
title = "Running Local AI on Intel Lunar Lake — Part 2: Models, Inference Engines, and the NPU That Almost Worked"
date = 2026-04-08T12:00:00+03:00
draft = false
description = "Model conversion pipelines, llama.cpp SYCL and OpenVINO GenAI inference engines, dependency pinning, and why the Intel NPU doesn't work for LLM inference yet."
images = ["/img/image.jpg"]
series = ["Running Local AI on Intel Lunar Lake"]

[params]
  seriesPart = 2
  seriesTotalParts = 4
+++

*Part 2 of 4 in a series on building a fully local AI development environment on an Intel Core Ultra 7 268V laptop running Fedora 43.*

---

In [Part 1]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}}), I set up the Intel compute stack — GPU runtime, Level Zero, and NPU driver — on Fedora 43. All three devices show up in OpenVINO. Now it's time to get models running on them.

This part covers two inference engines, a model conversion pipeline with sharp edges, and a deep dive into why the NPU didn't work out (yet).

## Two engines for two models

I chose different inference engines for the two models, and not arbitrarily.

### llama.cpp SYCL for the reasoning model

The reasoning model (DeepSeek-R1 7B) ships as a GGUF file — the standard format for llama.cpp. The question was how to run llama.cpp on an Intel GPU.

**Why not Ollama?** Ollama is the obvious choice for local LLMs, but it has no native Intel GPU support. The IPEX-LLM project provides a patched Ollama that works with Intel GPUs, but it's a wrapper around a wrapper — another version to track, another thing to break.

**llama.cpp's SYCL backend** is the more direct path. It's maintained upstream in the llama.cpp project and compiled into official container images. Using the container means I don't need the Intel oneAPI toolkit installed on my host — all the SYCL runtime dependencies are bundled.

```bash
podman pull ghcr.io/ggml-org/llama.cpp:server-intel
```

A note on the image tag: the documentation and various guides reference `server--intel-sycl`, but the actual tag on GHCR is `server-intel`. I had to query the container registry to find it:

```bash
skopeo list-tags docker://ghcr.io/ggml-org/llama.cpp | grep intel
```

### OpenVINO GenAI for the completion model

The completion model (Qwen2.5-Coder-1.5B) uses OpenVINO instead of llama.cpp for two reasons:

1. **Better quantization quality.** OpenVINO uses NNCF for calibration-aware INT4 weight compression, which produces higher quality quantized models than GGUF's post-training quantization — especially at small model sizes where every bit of precision matters.

2. **NPU path.** OpenVINO is the only inference framework that supports the Intel NPU on Linux. Even though NPU inference didn't work out (more on that below), using OpenVINO means switching from GPU to NPU is a one-line config change when driver support matures.

## Downloading the reasoning model

The GGUF download is straightforward:

```bash
pip install huggingface-hub
huggingface-cli download bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF \
  DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf --local-dir ~/models/
```

One gotcha on Fedora 43: the default `python3` is 3.14, but `huggingface-cli` installs its entry point under `~/.local/bin/` which may not be in your PATH, and the shell's `python3` may resolve to a Linuxbrew installation rather than the system one. I ended up invoking it explicitly:

```bash
/usr/bin/python3.14 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF',
                'DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf',
                local_dir='/home/user/models/')
"
```

The file is 4.4 GB.

## Converting the completion model — the dependency maze

This is where things got painful. Converting a Hugging Face model to OpenVINO IR with INT4 quantization requires `optimum-intel`, which depends on `openvino`, `nncf`, `transformers`, and `torch`. These packages have a fragile version matrix.

### Python 3.14 doesn't work (yet)

Fedora 43 ships Python 3.14 as the default. OpenVINO 2025.x doesn't have Python 3.14 wheels — `numpy` fails to build from source. The fix is to use Python 3.12, which is also installed:

```bash
/usr/bin/python3.12 -m venv ~/venvs/openvino-npu
```

**Update:** OpenVINO 2026.0 does ship `cp314` wheels, so this is resolved if you're willing to use the latest OpenVINO release. However, as I'll explain below, OpenVINO 2026.0 introduces a different NPU compatibility issue.

### The version pin dance

After multiple rounds of import errors, I landed on this combination that actually works:

```bash
source ~/venvs/openvino-npu/bin/activate
pip install "openvino==2025.4.1" "openvino-genai==2025.4.1"
pip install "torch>=2.5,<2.7" --index-url https://download.pytorch.org/whl/cpu
pip install "optimum-intel[openvino]==1.22.0" Pillow
```

Key constraints:
- `optimum-intel 1.22` requires OpenVINO >= 2025.4
- The `torch.onnx` API used by optimum-intel broke in torch 2.7+, so torch must be < 2.7
- Install torch from the CPU-only index to avoid pulling 5 GB of CUDA dependencies you'll never use
- `nncf` (pulled by optimum-intel) must match the OpenVINO version, or you get `AttributeError: module 'openvino' has no attribute 'Node'`
- `Pillow` is an undeclared dependency that crashes the import chain if missing

### The actual conversion

With the right versions in place:

```python
from optimum.intel import OVModelForCausalLM
from transformers import AutoTokenizer

model = OVModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    export=True,
    trust_remote_code=True,
    quantization_config={
        "bits": 4, "sym": True,
        "group_size": 128, "ratio": 1.0,
    },
)
model.save_pretrained("~/models/qwen2.5-coder-1.5b-npu")

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    trust_remote_code=True,
)
tokenizer.save_pretrained("~/models/qwen2.5-coder-1.5b-npu")
```

This produces an OpenVINO IR model with 100% of ratio-defining layers in `int4_sym` (group size 128). The model files total about 870 MB on disk.

### The missing tokenizer XMLs

There's one more step that `optimum-intel 1.22` doesn't do for you: generating OpenVINO tokenizer and detokenizer models. The `openvino_genai.LLMPipeline` needs these to handle tokenization end-to-end without falling back to Python.

```python
from openvino_tokenizers import convert_tokenizer
from openvino import save_model
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("~/models/qwen2.5-coder-1.5b-npu")
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
save_model(ov_tokenizer, "~/models/qwen2.5-coder-1.5b-npu/openvino_tokenizer.xml")
save_model(ov_detokenizer, "~/models/qwen2.5-coder-1.5b-npu/openvino_detokenizer.xml")
```

Without these files, the pipeline throws `Neither tokenizer nor detokenizer models were provided`.

**Important:** The tokenizer XMLs must be generated with the same OpenVINO version that will load them. I learned this the hard way when I downgraded OpenVINO during debugging and the server refused to start with a cryptic `Charsmap normalizer accepts precompiled mapping and it should be of type u8 tensor` error.

## Running the reasoning model

Starting the llama.cpp server is a single Podman command:

```bash
podman run -d --name llama-gpu \
  --device /dev/dri:/dev/dri \
  --device /dev/accel:/dev/accel \
  -v ~/models:/models:Z \
  -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-intel \
  -m /models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf \
  -c 4096 -ngl 99 --host 0.0.0.0 --port 8080
```

- `-ngl 99` offloads all model layers to the GPU
- `-c 4096` sets the context window (limited to save shared memory)
- `--device /dev/dri` and `--device /dev/accel` pass GPU and NPU devices into the container
- The `:Z` on the volume mount is SELinux relabeling — required on Fedora

The server exposes an OpenAI-compatible API. Testing it:

```bash
$ curl -s http://127.0.0.1:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek","messages":[{"role":"user","content":"What is a kernel module?"}],"max_tokens":200}'
```

The response includes both `content` (the answer) and `reasoning_content` (the chain-of-thought). At ~16.5 tokens/sec on the Arc iGPU, it takes about 12 seconds to generate a 200-token response.

## Writing the completion server

The completion model doesn't have a built-in server — OpenVINO GenAI provides a Python API but no HTTP endpoint. I wrote a FastAPI wrapper that exposes OpenAI-compatible `/v1/completions` and `/v1/chat/completions` endpoints:

```python
"""OpenAI-compatible API server for code completion using OpenVINO GenAI."""

import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import openvino_genai as ov_genai
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = "/home/user/models/qwen2.5-coder-1.5b-npu"
DEVICE = "GPU"  # Change to "NPU" when driver support matures

pipe: ov_genai.LLMPipeline
generate_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    pipe = ov_genai.LLMPipeline(MODEL_PATH, DEVICE)
    yield


app = FastAPI(lifespan=lifespan)


def generate_text(prompt: str, max_tokens: int, temperature: float) -> str:
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_tokens
    config.temperature = max(temperature, 0.01)
    config.do_sample = temperature >= 0.01
    with generate_lock:
        return pipe.generate(prompt, config)
```

The `generate_lock` is critical. VS Code's Continue extension fires multiple completion requests in quick succession as you type. Without the lock, concurrent calls to `pipe.generate()` crash with `Generate cannot be called while ContinuousBatchingPipeline is already in running state`. The mutex serializes requests so only one generation runs at a time.

The server runs via uvicorn:

```bash
source ~/venvs/openvino-npu/bin/activate
uvicorn npu_server:app --host 127.0.0.1 --port 8081
```

## The NPU that almost worked

The original plan was to run the completion model on the NPU, freeing the GPU entirely for the reasoning model. The NPU is detected, the driver is loaded, OpenVINO sees it. But when you try to actually use it:

```python
pipe = ov_genai.LLMPipeline(model_path, "NPU")
# RuntimeError: Unsupported configuration key: NPU_MAX_TILES
```

### What's happening

The OpenVINO NPU plugin (shipped inside the pip package) tries to set a property called `NPU_MAX_TILES` during initialization. This property controls tile-based parallelism on the NPU hardware. But the NPU userspace driver (v1.32, built from source) doesn't recognize this property — it's not in the driver's supported property list.

I tested every OpenVINO version available:
- **2025.1.0** — different error (model format incompatibility), but no `NPU_MAX_TILES` crash. However, the model fails to compile on NPU with `Failed to compile Model0_FCEW000__0 for all devices in [NPU]`.
- **2025.4.1** — `NPU_MAX_TILES` error
- **2026.0.0** — same `NPU_MAX_TILES` error

The 2025.1 path revealed a second problem: the NPU driver was built *without the compiler* (`ENABLE_NPU_COMPILER_BUILD=OFF` by default). `NPU_COMPILER_VERSION` reports `0`. Even when I set `NPU_COMPILER_TYPE` to `MLIR` (OpenVINO's built-in compiler), the model subgraphs still fail to compile for the NPU target.

### The diagnosis

There are two independent issues:

1. **Property mismatch:** OpenVINO 2025.4+ and 2026.0 set `NPU_MAX_TILES`, which the linux-npu-driver v1.32 doesn't support. This is likely a version synchronization gap — the driver and the OpenVINO plugin are developed on different release cadences.

2. **Missing NPU compiler:** The driver's `ENABLE_NPU_COMPILER_BUILD` option requires OpenVINO as a build dependency, creating a circular dependency. Without it, `NPU_COMPILER_VERSION` is 0 and no model can be compiled for the NPU hardware.

### The pragmatic fallback

Both models run on the GPU. At ~1 GB + ~5.5 GB, they fit comfortably in the 30 GB shared memory pool. The GPU handles both workloads without contention issues because the completion model generates short responses (50-100 tokens) while the reasoning model produces longer ones — they rarely overlap in practice.

The completion server has a single config variable:

```python
DEVICE = "GPU"  # Change to "NPU" when driver support matures
```

When a future NPU driver adds `NPU_MAX_TILES` support, switching is a one-line change and a service restart.

## A note on Podman and IPv6

This cost me 20 minutes of debugging. Podman's port mapping (`-p 8080:8080`) creates an IPv4-only listener on the host. On Fedora, `localhost` resolves to `::1` (IPv6) first:

```bash
$ curl http://localhost:8080/health
curl: (56) Recv failure: Connection reset by peer

$ curl http://127.0.0.1:8080/health
{"status":"ok"}
```

**Always use `127.0.0.1`** in configs that point to Podman containers. This applies to Continue.dev settings, Open WebUI environment variables, and any scripts that hit the APIs.

## What's next

In [Part 3]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}}), I connect both models to VS Code and Open WebUI, create systemd user services so everything survives reboots, and cover the operational details of living with local AI day-to-day.

---

*Series: Running Local AI on Intel Lunar Lake*
- [Part 1: Hardware, Drivers, and the Intel Compute Stack]({{< ref "local-ai-intel-lunar-lake-part1-foundation" >}})
- **Part 2: Models, Inference Engines, and the NPU That Almost Worked** (you are here)
- [Part 3: VS Code, Open WebUI, and Running It All as Services]({{< ref "local-ai-intel-lunar-lake-part3-integration" >}})
- Part 4: Consolidating on Ollama with IPEX-LLM *(coming soon)*
