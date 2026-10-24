//! Pacing the GPU work a transcription queues.
//!
//! A transcription's GPU work is a sequence of bounded **steps**: a block of VAD windows, the
//! encoder's convolution stem, each encoder block, the decoder's start (the cross-attention keys
//! and values, then the prompt) and each decoder step. A step asks the [installed](install)
//! [`Gate`] before it builds its operations, and reports its [`Slot`] [queued](Slot::queued) once
//! they are built and before anything of them is read back, so a slot is never held across a
//! readback or any other wait. Where the code chooses a step's size (the windows of an encoder
//! batch, the windows of a VAD block), it sizes the step to [`Gate::step_flops`].
//!
//! **Matmul-equivalent FLOPs:** the FLOPs a step would take at the device's measured dense matmul
//! speed, so that `seconds = flops / speed`: the GPU's time running it. A step's cost is its
//! analytic FLOPs over its kind's efficiency relative to a big matmul, plus, for the kinds bound by
//! their dispatches (an encoder block, a decoder pass), a fixed part; both measured on an RTX 5090.
//!
//! A refusal ([`Gate::enter`] → `None`) ends the transcription with [`Stopped`] before that step
//! builds anything, so nothing is left half-queued.
//!
//! With no gate installed, a step costs one uncontended read of the installed gate and runs
//! exactly as it would without pacing.

use std::{
    error::Error,
    fmt,
    sync::{Arc, PoisonError, RwLock},
};

/// Paces the GPU work a transcription queues: each bounded step asks before it is queued.
pub trait Gate: Send + Sync {
    /// The most work one step should queue, in matmul-equivalent FLOPs ([module docs](self)), or
    /// `None` for no bound. Steps whose size the code chooses (an encoder batch, a VAD batch) size
    /// themselves to it.
    fn step_flops(&self) -> Option<f64>;
    /// Wait until a step of about `flops` may be queued. `None`: the transcription is to stop.
    fn enter(&self, flops: f64) -> Option<Box<dyn Slot>>;
}

pub trait Slot {
    /// The step's operations are built and nothing of them has been read back yet: the host
    /// flushes them to the GPU and retires the slot once the GPU has run them.
    fn queued(self: Box<Self>);
}

static GATE: RwLock<Option<Arc<dyn Gate>>> = RwLock::new(None);

/// The gate every transcription in this process asks; `None` removes it. Replaceable (the host
/// reopens its device after a device loss).
pub fn install(gate: Option<Arc<dyn Gate>>) {
    *GATE.write().unwrap_or_else(PoisonError::into_inner) = gate;
}

/// A transcription the installed [`Gate`] stopped ([`Gate::enter`] returned `None`). The error of
/// every entry point that runs on the GPU when it is stopped: `err.is::<Stopped>()`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stopped;

impl fmt::Display for Stopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("the GPU pacing gate stopped the transcription")
    }
}

impl Error for Stopped {}

/// The installed gate, cloned out so that no lock is held while it waits.
fn installed() -> Option<Arc<dyn Gate>> {
    GATE.read().unwrap_or_else(PoisonError::into_inner).clone()
}

/// One step of `flops` (matmul-equivalent): `build` builds its operations and reads none of them
/// back. Its result is read back, if at all, after this returns.
pub(crate) fn step<R>(flops: f64, build: impl FnOnce() -> R) -> Result<R, Stopped> {
    let Some(gate) = installed() else {
        return Ok(build());
    };
    let slot = gate.enter(flops).ok_or(Stopped)?;
    drop(gate);
    let built = build();
    slot.queued();
    Ok(built)
}

/// The installed gate's [`Gate::step_flops`]; `None` without a gate.
pub(crate) fn step_budget() -> Option<f64> {
    installed().and_then(|gate| gate.step_flops())
}

/// Items of `per_item` FLOPs each that one step of `budget` may take on top of its `fixed` FLOPs:
/// `max` without a budget, else as many as fit, at least one (an item isn't split), at most `max`.
pub(crate) fn fit(budget: Option<f64>, fixed: f64, per_item: f64, max: usize) -> usize {
    match budget {
        Some(budget) => (((budget - fixed) / per_item) as usize).clamp(1, max.max(1)),
        None => max,
    }
}

// Each kind's efficiency relative to Diem's reference matmul (`burn-models` `speed::measure`:
// 1024² f16, 16 queued, best of three; 25.3 TFLOP/s here, the fastest of five), and the fixed part
// of the kinds bound by their dispatches, in matmul-equivalent FLOPs. Fitted by
// `pacing_tests::calibrate` on an RTX 5090 (Vulkan) as Diem builds this crate (burn f2b659ec,
// cubecl 1c5c8cbc, extensive autotune), 2026-10-24: each step timed alone by the GPU's time running
// it (behind a blocker, so that its build's launches don't stretch it), over base.en (beam 3) and
// large-v3-turbo (greedy) batches of 1, 4 and 10 regions and VAD blocks of 64, 256 and 1024
// windows; the fastest of two runs after a warm-up, as other processes shared the GPU. The fits
// are within 30% of every step measured.
//
// A VAD block's LSTM runs its windows one after another: ~12 µs a window.
const VAD_EFFICIENCY: f64 = 4.7e-3;
// The f32 convolutions: 2.7 ms a window for base.en, 21 ms for large-v3-turbo.
const ENCODER_STEM_EFFICIENCY: f64 = 3.4e-2;
const ENCODER_BLOCK_EFFICIENCY: f64 = 3.1;
const ENCODER_BLOCK_FIXED: f64 = 9.1e9; // 0.36 ms
// The f32 cross-attention key and value projections, with the beams' encoder output copied.
const CROSS_CACHE_EFFICIENCY: f64 = 1.0;
// A decoder pass is its dispatches: 1.5 ms for one sequence, 2.0–2.6 ms for 10–30.
const DECODER_PASS_EFFICIENCY: f64 = 0.15;
const DECODER_PASS_FIXED: f64 = 3.5e10; // 1.4 ms

/// The analytic FLOPs of one Silero VAD window (576 samples with its context, reflect-padded to
/// 640): the STFT conv (258 × 256 taps × 4 frames), the four encoder convs, the LSTM's input and
/// recurrent projections (128 → 512 each), and the output head.
const VAD_WINDOW_ANALYTIC: f64 = 2.0
    * (258.0 * 256.0 * 4.0
        + 128.0 * 129.0 * 3.0 * 4.0
        + 64.0 * 128.0 * 3.0 * 2.0
        + 64.0 * 64.0 * 3.0
        + 128.0 * 64.0 * 3.0
        + 2.0 * 128.0 * 512.0
        + 128.0);

/// A VAD step of `windows` windows of 512 samples.
pub(crate) fn vad(windows: usize) -> f64 {
    windows as f64 * VAD_WINDOW_ANALYTIC / VAD_EFFICIENCY
}

/// What a Whisper model's steps cost, in matmul-equivalent FLOPs (and, `*_flops`, their analytic
/// FLOPs). `ctx` is the encoder's positions (1500 for a 30 s window), `batch` the windows (or
/// beams) a step runs at once.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Cost {
    /// The model's width (`n_audio_state` = `n_text_state`).
    pub(crate) d: f64,
    pub(crate) n_mels: f64,
    pub(crate) decoder_layers: f64,
    pub(crate) vocab: f64,
}

impl Cost {
    /// The encoder's stem.
    pub(crate) fn encoder_stem(&self, batch: usize, ctx: usize) -> f64 {
        self.stem_flops(batch, ctx) / ENCODER_STEM_EFFICIENCY
    }

    /// One encoder block.
    pub(crate) fn encoder_block(&self, batch: usize, ctx: usize) -> f64 {
        ENCODER_BLOCK_FIXED + self.block_flops(batch, ctx) / ENCODER_BLOCK_EFFICIENCY
    }

    /// The decoder's start for `batch` sequences: every layer's cross-attention keys and values,
    /// then a pass over the prompt's `tokens`.
    pub(crate) fn decoder_start(&self, batch: usize, tokens: usize, ctx: usize) -> f64 {
        self.cross_flops(batch, ctx) / CROSS_CACHE_EFFICIENCY
            + self.decoder_pass(batch * tokens, ctx)
    }

    /// A decoder step: one token for each of `batch` sequences.
    pub(crate) fn decoder_step(&self, batch: usize, ctx: usize) -> f64 {
        self.decoder_pass(batch, ctx)
    }

    fn decoder_pass(&self, tokens: usize, ctx: usize) -> f64 {
        DECODER_PASS_FIXED + self.pass_flops(tokens, ctx) / DECODER_PASS_EFFICIENCY
    }

    /// conv1 (`n_mels` → d, 3 taps) over 2·ctx frames, then conv2 (d → d, 3 taps, stride 2).
    pub(crate) fn stem_flops(&self, batch: usize, ctx: usize) -> f64 {
        let (d, t) = (self.d, ctx as f64);
        batch as f64 * (12.0 * t * self.n_mels * d + 6.0 * t * d * d)
    }

    /// The Q, K, V and output projections (8·t·d²), the MLP (16·t·d²), and the attention's scores
    /// and weighted sum (4·t²·d).
    pub(crate) fn block_flops(&self, batch: usize, ctx: usize) -> f64 {
        let (d, t) = (self.d, ctx as f64);
        batch as f64 * (24.0 * t * d * d + 4.0 * t * t * d)
    }

    /// Every decoder layer's cross-attention keys and values: 4·ctx·d² a layer.
    pub(crate) fn cross_flops(&self, batch: usize, ctx: usize) -> f64 {
        batch as f64 * self.decoder_layers * 4.0 * ctx as f64 * self.d * self.d
    }

    /// A decoder pass over `tokens` tokens: per layer, the self- and cross-attention projections
    /// and the MLP (28·d²) and the cross-attention over `ctx` positions (4·ctx·d), then the logits
    /// (2·d·vocab). The self-attention over the past tokens (4·past·d a layer, at most 4% of a
    /// step at Whisper's widths) is left to the efficiency.
    pub(crate) fn pass_flops(&self, tokens: usize, ctx: usize) -> f64 {
        let d = self.d;
        let per_token =
            self.decoder_layers * (28.0 * d * d + 4.0 * ctx as f64 * d) + 2.0 * d * self.vocab;
        tokens as f64 * per_token
    }

    /// Windows a batched transcription runs at once, `beams` decoder sequences each with a
    /// `prompt`, so that each of its encoder blocks and decoder steps fits the installed gate's
    /// step: `max` without a bound, at least one. (The encoder's stem sizes its own steps.)
    pub(crate) fn windows_per_batch(
        &self,
        beams: usize,
        prompt: usize,
        ctx: usize,
        max: usize,
    ) -> usize {
        let Some(budget) = step_budget() else {
            return max;
        };
        // Each step is affine in its windows: what it takes for none, and for each.
        let fits = |step: &dyn Fn(usize) -> f64| {
            let fixed = step(0);
            fit(Some(budget), fixed, step(1) - fixed, max)
        };
        fits(&|n| self.encoder_block(n, ctx))
            .min(fits(&|n| self.decoder_start(n * beams, prompt, ctx)))
            .min(fits(&|n| self.decoder_step(n * beams, ctx)))
    }
}

#[cfg(test)]
#[path = "pacing_tests.rs"]
mod tests;
