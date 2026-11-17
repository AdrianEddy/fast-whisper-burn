//! The encoder's MLPs shared with a second engine beside the GPU.
//!
//! An encoder block is its attention, with the projections around it, and its MLP. The MLP is the
//! same computation for each position by itself and its weights never change: rows that an engine
//! made for fixed-weight layers — an Apple silicon Mac's Neural Engine — runs about as fast as
//! that Mac's GPU does, while the GPU works on something else. A host that has such an engine
//! hands it to the model as an [`MlpEngine`]
//! ([`Whisper::offload_encoder_mlps`](crate::model::Whisper::offload_encoder_mlps)), built from
//! the weights the model reads out for it ([`MlpWeights`]). This crate knows nothing of what the
//! engine is.
//!
//! With one, a forward's windows go through the blocks as two lanes half a block apart: while the
//! GPU runs one lane's attention half, the engine runs the other lane's MLP. Of a lane's rows
//! the engine takes a share and the GPU the rest, so that neither waits for the other: on a GPU
//! as fast as the engine, all of them; on a faster one, fewer. A forward of one window has one
//! lane, whose MLP rows the two engines share while the attention waits for both.
//!
//! The share is worked out from what the first forward measures ([`Rates`]): two blocks run a
//! half at a time, each half timed alone, their MLP on both engines — which also tells an engine
//! that answers something else than the GPU does, and is then left alone.
//!
//! Whether to share at all is not worked out but tried, for forwards of two lanes and of one
//! each: a forward runs two blocks the GPU's way and two shared, each from a standstill to one,
//! and after three such runs its kind is shared only if the fastest shared run took a fifth
//! less than the fastest of the GPU's. The halves' times cannot tell: each was waited for, and
//! what a shared forward loses is the waiting — as much as the work itself on a small model,
//! where sharing is slower than the GPU alone. The fastest of several, because what disturbs a
//! run only lengthens it: a kernel compiled on a shape's first forward, other work on either
//! engine. And a trial is run again every so often, whatever it found, less often as trials
//! agree: an engine busy with someone else's work when it is asked is not so for good.
//!
//! Every piece of GPU work is a step of its own ([`crate::pacing`]): a lane's attention half, and
//! the GPU's share of its MLP. What is read back — the engine's rows — is read after the step
//! that built it was reported queued.

use std::{
    fmt,
    ops::Range,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, PoisonError,
        atomic::{AtomicBool, AtomicU32, Ordering::Relaxed},
    },
    time::Instant,
};

use burn::tensor::{DType, Device, Tensor, TensorData};
use half::f16;

use crate::model::ResidualEncoderAttentionBlock;
use crate::pacing::{self, Shape, Stopped};

/// An engine beside the GPU that runs the encoder's MLPs.
pub trait MlpEngine: Send + Sync {
    /// The rows one call of the engine takes: what it is handed is a whole number of them.
    fn chunk(&self) -> usize;

    /// The MLP of encoder block `layer` over `input`'s rows — a row the model's width in values,
    /// row-major — as many rows, each the MLP of its own.
    fn run(&self, layer: usize, input: &[f16]) -> Result<Vec<f16>, String>;
}

/// One encoder block's MLP, as a host needs it to build an [`MlpEngine`]
/// ([`Whisper::encoder_mlp_weights`](crate::model::Whisper::encoder_mlp_weights)):
/// `gelu(x · upᵀ + up_bias) · downᵀ + down_bias`, the GELU the exact one (`x · Φ(x)`).
pub struct MlpWeights {
    /// `[4 · width, width]`, row-major: a row an output.
    pub up: Vec<f16>,
    pub up_bias: Vec<f16>,
    /// `[width, 4 · width]`, row-major.
    pub down: Vec<f16>,
    pub down_bias: Vec<f16>,
}

/// What a block's halves take on this machine when each is waited for, in seconds: its attention
/// half a window and its MLP a row on the GPU, its MLP a row on the engine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rates {
    pub(crate) attend: f64,
    pub(crate) mlp: f64,
    pub(crate) engine: f64,
}

/// What sharing may take of the GPU's alone to be worth it: a clear gain, for two engines'
/// power and a second copy of the weights. Where the two ways come out alike (Whisper base on an
/// M4: 0.85 to 1.0), run after run would otherwise find now one faster and now the other.
const WORTH: f64 = 0.8;

/// The blocks whose halves the first forward times: the first to have every kernel compiled and
/// the engine awake, the second for the [`Rates`].
const MEASURED: usize = 2;

/// The blocks a run of a trial takes each way, and the runs of a trial.
const TRIED: usize = 2;
const RUNS: u32 = 3;

/// The forwards from a trial to the next: twice as many each time the next finds the same, up to
/// the longest.
const GAP: u32 = 8;
const LONGEST_GAP: u32 = 1024;

/// How far the engine's rows may be from the GPU's, relative to their size, for it to be used:
/// float16 on either is a thousandth off; an engine built for other weights is off by its all.
const AGREES: f64 = 0.05;

impl Rates {
    /// Of the `rows` of a lane of `windows` windows, those the engine takes: a whole number of
    /// its `chunk`s, such that the two engines finish as nearly together as that allows — the
    /// GPU with the rest of the rows and, in a forward of two lanes (`paired`), with the other
    /// lane's attention half. None where even by these times sharing gains too little
    /// ([`WORTH`]); where it gains, a trial has the last word.
    pub(crate) fn share(&self, windows: usize, rows: usize, chunk: usize, paired: bool) -> usize {
        let attend = self.attend * windows as f64;
        let beside = if paired { attend } else { 0.0 };
        let both = |taken: usize| {
            (beside + (rows - taken) as f64 * self.mlp).max(taken as f64 * self.engine)
        };
        let taken = (0..=rows / chunk)
            .map(|chunks| chunks * chunk)
            .min_by(|&a, &b| both(a).total_cmp(&both(b)))
            .unwrap_or(0);
        let alone = attend + rows as f64 * self.mlp;
        let shared = attend - beside + both(taken);
        if shared <= WORTH * alone { taken } else { 0 }
    }
}

/// The forwards to let by before something found in vain is tried again: twice as many each
/// time.
#[derive(Default)]
struct Patience {
    in_vain: AtomicU32,
    wait: AtomicU32,
}

impl Patience {
    /// Whether this forward is one to let by.
    fn waiting(&self) -> bool {
        let wait = self.wait.load(Relaxed);
        if wait > 0 {
            self.wait.store(wait - 1, Relaxed);
        }
        wait > 0
    }

    fn in_vain(&self) {
        let in_vain = self.in_vain.fetch_add(1, Relaxed) + 1;
        self.wait.store(1 << in_vain.min(10), Relaxed);
    }
}

/// Whether sharing one way — a forward's windows as two lanes, or as one — is faster here than
/// the GPU alone, as its trials find.
#[derive(Default)]
struct Trial(Mutex<Tried>);

#[derive(Debug, Default)]
struct Tried {
    /// What the last trial found; nothing before the first.
    faster: Option<bool>,
    /// The forwards to go before the next trial, and the trials in a row that found the same.
    wait: u32,
    agreed: u32,
    /// The trial under way: its runs so far, and the fastest of them each way, in seconds.
    runs: u32,
    fastest: (f64, f64),
}

impl Trial {
    fn tried(&self) -> MutexGuard<'_, Tried> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Whether the last trial found sharing faster.
    fn faster(&self) -> bool {
        self.tried().faster == Some(true)
    }

    /// Whether the last trial found sharing in vain, and the next is yet to come.
    fn in_vain(&self) -> bool {
        let tried = self.tried();
        tried.faster == Some(false) && tried.wait > 0
    }

    /// A forward begins: one fewer to go before the next trial.
    fn forward(&self) {
        let mut tried = self.tried();
        tried.wait = tried.wait.saturating_sub(1);
    }

    /// Whether a forward of this kind is to run blocks both ways.
    fn due(&self) -> bool {
        self.tried().wait == 0
    }

    /// A run of the trial: as many blocks took `alone` seconds the GPU's way and `shared`
    /// shared. What the trial finds, with its last run.
    fn ran(&self, alone: f64, shared: f64) -> Option<bool> {
        let mut tried = self.tried();
        tried.fastest = match tried.runs {
            0 => (alone, shared),
            _ => (tried.fastest.0.min(alone), tried.fastest.1.min(shared)),
        };
        tried.runs += 1;
        if tried.runs < RUNS {
            return None;
        }
        let faster = tried.fastest.1 <= WORTH * tried.fastest.0;
        tried.agreed = if tried.faster == Some(faster) {
            tried.agreed + 1
        } else {
            0
        };
        tried.faster = Some(faster);
        tried.wait = (GAP << tried.agreed.min(16)).min(LONGEST_GAP);
        tried.runs = 0;
        Some(faster)
    }

    /// As if a trial had found `faster`, with none to follow.
    #[cfg(test)]
    fn settled(&self, faster: bool) {
        let mut tried = self.tried();
        tried.faster = Some(faster);
        tried.wait = u32::MAX;
    }
}

/// An encoder's [`MlpEngine`], and what its forwards found of it.
pub(crate) struct Offload {
    engine: Arc<dyn MlpEngine>,
    rates: OnceLock<Rates>,
    /// For rates by which there was nothing to gain.
    rating: Patience,
    /// Sharing in two lanes, and in one.
    paired: Trial,
    lone: Trial,
    /// The engine failed, or disagreed with the GPU: the blocks are the GPU's again, for good.
    failed: AtomicBool,
    /// The engine has answered with values that are not finite ([`Self::joined`]): said once.
    overflowed: AtomicBool,
}

impl fmt::Debug for Offload {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Offload")
            .field("chunk", &self.engine.chunk())
            .field("rates", &self.rates.get())
            .field("paired", &*self.paired.tried())
            .field("lone", &*self.lone.tried())
            .field("failed", &self.failed.load(Relaxed))
            .finish()
    }
}

/// Some of a forward's windows on their way through the blocks.
struct Lane {
    /// The residual, `[windows, ctx, width]`.
    x: Tensor<3>,
    /// The rows of each MLP the engine takes.
    share: usize,
}

/// A lane's MLP on its way: its rows' halves, the engine's first.
struct Pending {
    lane: usize,
    layer: usize,
    /// The engine's rows (`[1, rows, width]`): on the GPU, and as read back from it.
    engine: Option<(Tensor<3>, Result<TensorData, String>)>,
    /// The GPU's rows, their MLP queued.
    gpu: Option<Tensor<3>>,
}

impl Offload {
    pub(crate) fn new(engine: Arc<dyn MlpEngine>) -> Self {
        Self {
            engine,
            rates: OnceLock::new(),
            rating: Patience::default(),
            paired: Trial::default(),
            lone: Trial::default(),
            failed: AtomicBool::new(false),
            overflowed: AtomicBool::new(false),
        }
    }

    /// [`Self::new`] with the rates a first forward would measure, and no trial yet.
    #[cfg(test)]
    pub(crate) fn rated(engine: Arc<dyn MlpEngine>, rates: Rates) -> Self {
        let offload = Self::new(engine);
        let _ = offload.rates.set(rates);
        offload
    }

    /// [`Self::rated`] with sharing found faster both ways, and no trial to follow.
    #[cfg(test)]
    pub(crate) fn measured(engine: Arc<dyn MlpEngine>, rates: Rates) -> Self {
        let offload = Self::rated(engine, rates);
        offload.paired.settled(true);
        offload.lone.settled(true);
        offload
    }

    /// Whether the engine still runs MLPs.
    pub(crate) fn working(&self) -> bool {
        !self.failed.load(Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn rates(&self) -> Option<Rates> {
        self.rates.get().copied()
    }

    /// The lanes a forward of windows of `ctx` positions goes through the blocks in, were it of
    /// two windows or more: two where by the rates that gains, unless a trial found otherwise
    /// and the next is yet to come; else one.
    pub(crate) fn lanes(&self, ctx: usize) -> usize {
        let paired = self.working()
            && !self.paired.in_vain()
            && self
                .rates
                .get()
                .is_some_and(|rates| rates.share(1, ctx, self.engine.chunk(), true) > 0);
        if paired { 2 } else { 1 }
    }

    fn fail(&self, why: &str) {
        if !self.failed.swap(true, Relaxed) {
            log::warn!(
                "Whisper's encoder keeps its MLPs on the GPU: the engine beside it failed ({why})"
            );
        }
    }

    /// The first of `blocks` over `x` (`[windows, ctx, width]`, f16), and how many those were:
    /// all of them, their MLPs shared with the engine, or as many as it took to find that
    /// sharing gains nothing here — the rest are their caller's.
    pub(crate) fn forward(
        &self,
        blocks: &[ResidualEncoderAttentionBlock],
        x: Tensor<3>,
        shape: &Shape,
    ) -> Result<(Tensor<3>, usize), Stopped> {
        let [windows, ctx, width] = x.dims();
        let chunk = self.engine.chunk();
        // Blocks enough for the rates and a trial, and rows enough for a call.
        if blocks.len() < MEASURED + 2 * TRIED || windows * ctx < chunk {
            return Ok((x, 0));
        }
        self.paired.forward();
        self.lone.forward();
        let (mut x, mut done) = (x, 0);
        let rates = match self.rates.get() {
            Some(rates) => *rates,
            None => {
                if self.rating.waiting() {
                    return Ok((x, 0));
                }
                let mut measured = None;
                for (layer, block) in blocks[..MEASURED].iter().enumerate() {
                    (x, measured) = self.timed(layer, block, x, shape)?;
                    done += 1;
                    if measured.is_none() {
                        return Ok((x, done));
                    }
                }
                let Some(rates) = measured else {
                    return Ok((x, done));
                };
                let gains = [true, false]
                    .iter()
                    .any(|&paired| rates.share(1, ctx, chunk, paired) > 0);
                log::info!(
                    "Whisper's encoder block, {width} wide, each half waited for: its attention half {:.2} ms a window on the GPU, its MLP {:.2} ms on the GPU and {:.2} ms on the engine beside it{}",
                    rates.attend * 1e3,
                    rates.mlp * ctx as f64 * 1e3,
                    rates.engine * ctx as f64 * 1e3,
                    if gains { "" } else { ": nothing to gain" },
                );
                if !gains {
                    self.rating.in_vain();
                    return Ok((x, done));
                }
                *self.rates.get_or_init(|| rates)
            }
        };

        let first = windows.div_ceil(2);
        let paired = windows >= 2 && rates.share(first, first * ctx, chunk, true) > 0;
        let lone = rates.share(windows, windows * ctx, chunk, false);
        if !paired && lone == 0 {
            return Ok((x, done));
        }
        let trial = if paired { &self.paired } else { &self.lone };
        let trying = trial.due();
        if !trying && !trial.faster() {
            return Ok((x, done));
        }

        // A run of a trial: blocks the GPU's way, from a standstill to one, then as many shared.
        let device = x.device();
        let mut alone = None;
        if trying {
            finished(&device, &x);
            let started = Instant::now();
            let cost = shape.encoder_block(windows, ctx);
            for block in &blocks[done..done + TRIED] {
                x = pacing::step(cost, || block.forward(x, true))?;
            }
            finished(&device, &x);
            alone = Some(started.elapsed().as_secs_f64());
            done += TRIED;
        }
        let mut lanes: Vec<Lane> = if paired {
            let lane = |x: Tensor<3>| {
                let windows = x.dims()[0];
                Lane {
                    share: rates.share(windows, windows * ctx, chunk, true),
                    x,
                }
            };
            vec![
                lane(x.clone().slice([0..first])),
                lane(x.slice([first..windows])),
            ]
        } else {
            vec![Lane { x, share: lone }]
        };
        if let Some(alone) = alone {
            let started = Instant::now();
            self.shared(blocks, done..done + TRIED, &mut lanes, shape)?;
            // The lanes were closed in order: the last one's residual is the last of the work.
            finished(&device, &lanes[lanes.len() - 1].x);
            let shared = started.elapsed().as_secs_f64();
            done += TRIED;
            if let Some(faster) = trial.ran(alone, shared) {
                let tried = trial.tried();
                log::info!(
                    "Whisper's encoder, {windows} window(s) in {}: {TRIED} blocks take the GPU alone {:.1} ms and {:.1} ms shared with the engine beside it, the fastest of {RUNS} runs: {} for the next {} forwards",
                    if paired { "two lanes" } else { "one lane" },
                    tried.fastest.0 * 1e3,
                    tried.fastest.1 * 1e3,
                    if faster { "shared" } else { "the GPU's" },
                    tried.wait,
                );
            }
            if !trial.faster() {
                return Ok((merged(lanes), done));
            }
        }
        self.shared(blocks, done..blocks.len(), &mut lanes, shape)?;
        Ok((merged(lanes), blocks.len()))
    }

    /// `lanes` through the blocks `layers` of `blocks`, their MLPs shared with the engine: each
    /// lane's residual as those blocks leave it.
    fn shared(
        &self,
        blocks: &[ResidualEncoderAttentionBlock],
        layers: Range<usize>,
        lanes: &mut [Lane],
        shape: &Shape,
    ) -> Result<(), Stopped> {
        let device = lanes[0].x.device();
        let count = lanes.len();
        let mut pending: Option<Pending> = None;
        let mut outputs: Vec<Option<Tensor<3>>> = vec![None; count];
        for step in layers.start * count..layers.end * count {
            let (layer, at) = (step / count, step % count);
            let block = &blocks[layer];
            // A lone lane's next block needs its own MLP back first; the GPU has its share of it
            // to run meanwhile.
            if pending.as_ref().is_some_and(|waiting| waiting.lane == at) {
                let waiting = pending.take().expect("checked");
                outputs[at] = Some(self.joined(blocks, waiting, shape)?);
            }
            let lane = &mut lanes[at];
            let [windows, ctx, width] = lane.x.dims();
            let rows = windows * ctx;
            let (residual, output) = (lane.x.clone(), outputs[at].take());
            let (residual, input) = pacing::step(shape.encoder_attention(windows, ctx), || {
                let residual = match output {
                    Some(output) => residual + output.reshape([windows, ctx, width]),
                    None => residual,
                };
                block.attend(residual, true)
            })?;
            lane.x = residual;
            // The GPU starts on this lane's block, and the other lane's MLP runs beside it.
            flush(&device);
            if let Some(waiting) = pending.take() {
                let other = waiting.lane;
                outputs[other] = Some(self.joined(blocks, waiting, shape)?);
            }
            let share = if self.working() { lane.share } else { 0 };
            let input = input.reshape([1, rows, width]);
            // Read with this block the only work out, so that the wait is for it alone.
            let engine = (share > 0).then(|| {
                let rows = input.clone().slice([0..1, 0..share]);
                let read = read(rows.clone());
                (rows, read)
            });
            let gpu = match rows - share {
                0 => None,
                rest => {
                    let input = input.slice([0..1, share..rows]);
                    let output = pacing::step(shape.encoder_mlp(rest), || block.mlp(input))?;
                    flush(&device);
                    Some(output)
                }
            };
            pending = Some(Pending {
                lane: at,
                layer,
                engine,
                gpu,
            });
        }
        if let Some(waiting) = pending.take() {
            let last = waiting.lane;
            outputs[last] = Some(self.joined(blocks, waiting, shape)?);
        }
        for (lane, output) in lanes.iter_mut().zip(outputs) {
            if let Some(output) = output {
                let dims = lane.x.dims();
                lane.x = lane.x.clone() + output.reshape(dims);
            }
        }
        Ok(())
    }

    /// A lane's MLP output (`[1, rows, width]`): the engine's rows, run now, then the GPU's. The
    /// engine's on the GPU after all should the engine fail, as from then on — or, this once,
    /// should it answer with a value that is not finite: an engine is free to hold its sums
    /// scaled, and so to run out of float16 where the GPU does not.
    fn joined(
        &self,
        blocks: &[ResidualEncoderAttentionBlock],
        pending: Pending,
        shape: &Shape,
    ) -> Result<Tensor<3>, Stopped> {
        let Pending {
            layer, engine, gpu, ..
        } = pending;
        let engine = match engine {
            Some((input, read)) => {
                let dims = input.dims();
                let ran = read.and_then(|data| {
                    let rows = data.as_slice::<f16>().map_err(|e| format!("{e:?}"))?;
                    let output = self.engine.run(layer, rows)?;
                    if output.len() == rows.len() {
                        Ok(output)
                    } else {
                        Err(format!(
                            "{} values for {} of its rows'",
                            output.len(),
                            rows.len()
                        ))
                    }
                });
                let on_gpu =
                    |input| pacing::step(shape.encoder_mlp(dims[1]), || blocks[layer].mlp(input));
                Some(match ran {
                    Ok(output) if output.iter().all(|value| value.is_finite()) => {
                        Tensor::from_data(
                            TensorData::new(output, dims),
                            (&input.device(), DType::F16),
                        )
                    }
                    Ok(_) => {
                        if !self.overflowed.swap(true, Relaxed) {
                            log::warn!(
                                "Whisper's encoder: the engine beside the GPU ran out of float16 in block {layer}'s MLP; such rows are the GPU's"
                            );
                        }
                        on_gpu(input)?
                    }
                    Err(why) => {
                        self.fail(&why);
                        on_gpu(input)?
                    }
                })
            }
            None => None,
        };
        Ok(match (engine, gpu) {
            (Some(engine), Some(gpu)) => Tensor::cat(vec![engine, gpu], 1),
            (Some(rows), None) | (None, Some(rows)) => rows,
            (None, None) => unreachable!("a lane has rows"),
        })
    }

    /// Block `layer` over `x`, a half at a time and each timed alone, its MLP run on the engine
    /// as well: the block's output, and what its halves took — `None`, and the engine given up
    /// ([`Self::fail`]), where it failed or its rows are not the GPU's.
    fn timed(
        &self,
        layer: usize,
        block: &ResidualEncoderAttentionBlock,
        x: Tensor<3>,
        shape: &Shape,
    ) -> Result<(Tensor<3>, Option<Rates>), Stopped> {
        let [windows, ctx, width] = x.dims();
        let rows = windows * ctx;
        let device = x.device();
        let (x, input) = pacing::step(shape.encoder_attention(windows, ctx), || {
            block.attend(x, true)
        })?;
        let attend = finished(&device, &input);
        let output = pacing::step(shape.encoder_mlp(rows), || block.mlp(input.clone()))?;
        let mlp = finished(&device, &output);

        // As many of the rows as a window's worth of calls takes: the engine's time is its rows'.
        let chunk = self.engine.chunk();
        let taken = (rows / chunk).min(ctx.div_ceil(chunk)) * chunk;
        let first = |rows: &Tensor<3>| {
            let rows = rows.clone().reshape([1, windows * ctx, width]);
            read(rows.slice([0..1, 0..taken]))
        };
        let engine = first(&input).and_then(|input| {
            let input = input.as_slice::<f16>().map_err(|e| format!("{e:?}"))?;
            // Twice: a program's first call has more to do than its later ones.
            self.engine.run(layer, input)?;
            let started = Instant::now();
            let engine = self.engine.run(layer, input)?;
            let took = started.elapsed().as_secs_f64();
            let gpu = first(&output)?;
            let gpu = gpu.as_slice::<f16>().map_err(|e| format!("{e:?}"))?;
            if engine.len() != gpu.len() {
                return Err(format!(
                    "{} values for {} of its rows'",
                    engine.len(),
                    gpu.len()
                ));
            }
            let (mut off, mut size) = (0f64, 0f64);
            for (engine, gpu) in engine.iter().zip(gpu) {
                off += f64::from(engine.to_f32() - gpu.to_f32()).powi(2);
                size += f64::from(gpu.to_f32()).powi(2);
            }
            let apart = (off / size).sqrt();
            if apart.is_nan() || apart > AGREES {
                return Err(format!(
                    "its rows of block {layer}'s MLP are {apart:.3} of their size off the GPU's"
                ));
            }
            Ok(took / taken as f64)
        });
        let rates = match engine {
            Ok(engine) => Some(Rates {
                attend: attend / windows as f64,
                mlp: mlp / rows as f64,
                engine,
            }),
            Err(why) => {
                self.fail(&why);
                None
            }
        };
        Ok((x + output, rates))
    }
}

/// The lanes' residuals as one tensor again, in their order.
fn merged(lanes: Vec<Lane>) -> Tensor<3> {
    let mut residuals: Vec<Tensor<3>> = lanes.into_iter().map(|lane| lane.x).collect();
    if residuals.len() == 1 {
        residuals.remove(0)
    } else {
        Tensor::cat(residuals, 0)
    }
}

/// Has the GPU start on what is queued for it.
fn flush(device: &Device) {
    // A flush that fails is a device that is gone: what reads from it next says so.
    let _ = device.flush();
}

/// The seconds until the GPU has run everything queued for it, `last` being the last of it.
fn finished(device: &Device, last: &Tensor<3>) -> f64 {
    let started = Instant::now();
    flush(device);
    // A read that fails is a device that is gone: the forward's own reads say so.
    let _ = read(last.clone().slice([0..1, 0..1, 0..1]));
    started.elapsed().as_secs_f64()
}

/// `rows`' values, waited for: burn fetches what it hands back when it is first looked at.
fn read(rows: Tensor<3>) -> Result<TensorData, String> {
    let data = rows.try_into_data().map_err(|e| format!("{e:?}"))?;
    data.as_slice::<f16>().map_err(|e| format!("{e:?}"))?;
    Ok(data)
}

#[cfg(test)]
#[path = "offload_tests.rs"]
mod tests;
