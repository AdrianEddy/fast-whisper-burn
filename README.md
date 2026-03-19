# Fast Whisper Burn

Highly optimized Rust implementation of OpenAI's Whisper speech recognition model, built on the latest [Burn](https://github.com/tracel-ai/burn) deep learning framework.

This started as a fork of [Gadersd/whisper-burn](https://github.com/Gadersd/whisper-burn), but is essentially rewritten from the ground up.

## Highlights

- **Up-to-date Burn framework** — uses up-to-date Burn's version (v0.21).
- **f32 and f16 precision** — the convert tool produces both full-precision (`.bpk`) and mixed-precision (`-f16.bpk`) weight files. The `--f16` flag at runtime loads the smaller f16 weights while keeping numerically sensitive layers (LayerNorm, embeddings, convolutions) in f32.
- **Native Burn Attention** — uses Burn's built-in `MultiHeadAttention` modules with KV-cache support, opening the path to flash attention when Burn adds it.
- **Silero VAD** — includes an embedded Silero v6 voice-activity-detection model that segments audio into speech regions before transcription, skipping silence and reducing wasted compute.
- **Extensive transcription config** — timestamps, 100 languages, beam search / greedy decoding, temperature fallback, token-level timing, initial prompt carry-over, and more (see `WhisperParams`).

## Performance Optimizations

The f16 inference path includes several custom optimizations beyond Burn's defaults, focused on eliminating GPU kernel-launch overhead and reducing redundant computation:

### Custom CubeCL Kernels (`custom_kernels.rs`)

- **Fused LayerNorm with mixed precision** — a single CubeCL kernel that reads f16 weights, accumulates in f32 for numerical stability, and writes f16 output, avoiding separate cast→norm→cast kernel launches.
- **Fused softmax with mixed precision** — similarly fused softmax that operates on f16 tensors with f32 accumulation internally.
- **Fused single-query attention** — a custom attention kernel for the autoregressive decode step (seq_len=1). Computes QK dot products, online softmax, and V weighting in a single kernel launch, eliminating the ~6 intermediate kernels a standard matmul→softmax→matmul decomposition would require. Uses online softmax (single-pass, no intermediate score tensor).
- **Fused LSTM cell for VAD** — a single CubeCL kernel that performs the entire LSTM cell computation (matmul + bias + gate activations + cell/hidden state update) used in the Silero VAD model. Replaces ~12 separate kernel launches (matmul, add, split, 3×sigmoid, 2×tanh, 3×mul, add) per LSTM step with one fused kernel. Each of the 128 threads computes 4 dot products (one per gate) over shared-memory-cached hidden state.

### Fused Decoder Weights

- **Pre-computed QKV projection** — before inference, the separate Q/K/V weight matrices for self-attention are concatenated into a single `[d_model, 3*d_model]` matrix. Each decode step does one matmul instead of three, cutting kernel launches per layer from 3 to 1.
- **Pre-computed logit embedding** — the `[vocab_size, d_model]` token embedding is transposed and cast to f16 once before inference, avoiding a per-step transpose + cast of a 51864×512 matrix.

### Region-Level Batched Inference

The biggest single optimization. Instead of processing each VAD speech region sequentially (30 regions = 30 encoder passes + 30 decode loops), all regions are processed simultaneously in a single GPU batch:

- **Batched encoder** — all regions' mel spectrograms are stacked into `[N, 80, 3000]` and encoded in one forward pass.
- **Batched greedy decode** — all N regions are decoded in parallel (batch=N). Each decode step is a single GPU forward pass serving all regions, with lightweight per-region token selection on the CPU. Finished regions are padded with EOT tokens.
- **Batched beam search** — all N regions × beam_size beams run as a single batch (e.g., 30 regions × 4 beams = batch 120). Per-region beam management (candidate ranking, reordering) happens on the CPU between batched GPU forward passes.
- **Quality fallback** — if any region's batched output is degenerate (low entropy / repetitive text, failed decode), that specific region is re-decoded sequentially with temperature fallback, paying the retry cost only for the few problematic regions.
- **Memory-conscious chunking** — regions are processed in configurable chunks (default 10) so that 10-hour recordings with hundreds of regions don't exhaust GPU memory.

### Net Impact

On a 10-minute test file with 30 speech regions (base.en model, f16, GPU):

| Decode strategy | Sequential | Batched | Speedup |
|-----------------|-----------|---------|---------|
| Beam search (default) | ~15s | ~8s | ~2× |
| Greedy | ~12s | ~2s | ~6× |

## Supported Models

Any standard OpenAI Whisper `.pt` checkpoint works: `tiny`, `base`, `small`, `medium`, `large-v1`/`v2`/`v3`, `large-v3-turbo`, and their `.en` English-only variants.

## Quick Start

### 1. Convert a `.pt` checkpoint

Download (or train) a PyTorch checkpoint, then convert it to Burn's format:

```bash
# Converts base.en.pt → base.en.bpk + base.en-f16.bpk + base.en.cfg
cargo run --release --bin convert base.en.pt
```

The converter reads the `.pt` file directly via `burn-store`'s PyTorch reader — no Python scripts required. Model architecture config is inferred automatically from tensor shapes inside the checkpoint.

To convert a HuggingFace-hosted model, first export it to `.pt`:

```bash
python3 python/convert_huggingface_model.py openai/whisper-base.en base.en.pt
cargo run --release --bin convert base.en.pt
```

### 2. Prepare audio

The input must be 16 kHz mono WAV:

```bash
sox audio.wav -r 16000 -c 1 audio16k.wav
# or
ffmpeg -i audio.wav -ar 16000 -ac 1 audio16k.wav
```

### 3. Transcribe

```bash
# f32 precision, beam decoding (default)
cargo run --release --bin transcribe base.en audio16k.wav en transcription.txt

# f16 mixed-precision
cargo run --release --bin transcribe base.en audio16k.wav en transcription.txt --f16

# Greedy search
cargo run --release --bin transcribe base.en audio16k.wav en transcription.txt --greedy
```

The transcriber outputs both a plain-text `.txt` and a timestamped `.srt` file.

## License

MIT
