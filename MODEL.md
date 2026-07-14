# Obtaining the models

To keep this repository lightweight and license-compliant, **model weights are
not distributed here.** The generator weights are gated under Google's Gemma
license, and the quantized GGUF (~1.6 GB) exceeds GitHub's file-size limits.
Reproduce them locally as follows.

## Generator — Gemma-2-2B-it, 4-bit `Q4_K_M` GGUF

The study uses `gemma-2-2b-it` quantized to `Q4_K_M` with
[`llama.cpp`](https://github.com/ggml-org/llama.cpp) at pinned commit:

```
665892536dfb1b7532161e3182304bd35c33e768
```

Steps:

1. **Accept the license & download the base model** from Hugging Face:
   [`google/gemma-2-2b-it`](https://huggingface.co/google/gemma-2-2b-it)
   (requires accepting Google's Gemma Terms of Use).

2. **Build llama.cpp** at the pinned commit:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp
   cd llama.cpp && git checkout 665892536dfb1b7532161e3182304bd35c33e768
   cmake -B build && cmake --build build --config Release
   ```

3. **Convert to GGUF and quantize to `Q4_K_M`:**
   ```bash
   python convert_hf_to_gguf.py /path/to/gemma-2-2b-it --outfile gemma-2-2b-it-f16.gguf
   ./build/bin/llama-quantize gemma-2-2b-it-f16.gguf gemma-2-2b-it-Q4_K_M.gguf Q4_K_M
   ```

4. **Place the result** at `models/gemma-2-2b-it-Q4_K_M.gguf`, or point the
   pipeline at it via environment variable:
   ```bash
   export ALEMC_GGUF=/abs/path/to/gemma-2-2b-it-Q4_K_M.gguf
   ```
   (See `config.py: GEN_MODEL_GGUF`.)

## Query embedder — `bge-small-en-v1.5` (no manual step)

`BAAI/bge-small-en-v1.5` (384-dim, MIT-licensed) is fetched automatically by
`fastembed` (ONNX runtime, no PyTorch) the first time the pipeline runs. No
manual download is required. See `config.py: EMBED_MODEL_NAME`.

## Note on the fallback

Every backend degrades to a deterministic `mock` implementation when native
libraries or weights are absent, so the harness and downstream tables/figures
can be exercised without any model present. Mock runs are flagged `is_mock=1`
in the output CSVs and must never be reported as real measurements.
