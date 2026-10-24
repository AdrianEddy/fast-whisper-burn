//! The pacing seam ([`crate::pacing`]): what a step asks the gate and when it reports, how a
//! refusal ends a transcription, and how steps size themselves to the gate's step.
//!
//! ```text
//! cargo test --release --no-default-features --features vulkan --lib pacing
//! cargo test --release --no-default-features --features vulkan --lib pacing -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The ignored tests run Whisper on the GPU. They read `base.en`, `base` and `large-v3-turbo`
//! (`{name}.cfg`, `{name}-tokenizer.json`, `{name}-f16.bpk`) in place from `FWB_MODELS`, or else
//! from this crate's `models/` like the `transcribe` binary, and transcribe `audio.wav`.

use super::*;
use crate::model::{Whisper, WhisperConfig};
use crate::token::Gpt2Tokenizer;
use crate::transcribe::{
    SamplingStrategy, TranscriptionResult, WhisperParams, detect_language, transcribe,
    transcribe_regions_batched,
};
use crate::vad::{SileroVAD6Model, SpeechRegion, detect_speech_regions};
use burn::backend::Dispatch;
use burn::config::Config;
use burn::tensor::{DType, Device, DeviceKind, Distribution, Tensor};
use burn_backend::{Backend, ProfileOptions, ProfileToken};
use burn_store::ModuleSnapshot;
use std::{
    path::PathBuf,
    sync::{Mutex, MutexGuard},
    thread::{self, ThreadId},
    time::{Duration, Instant},
};

// ═══════════════════════════════════════════
// Gates
// ═══════════════════════════════════════════

/// The gate is process-wide, so the tests that install one take turns; the turn removes it.
static SERIAL: Mutex<()> = Mutex::new(());

struct Turn(#[allow(dead_code)] MutexGuard<'static, ()>);

fn turn() -> Turn {
    Turn(SERIAL.lock().unwrap_or_else(PoisonError::into_inner))
}

impl Drop for Turn {
    fn drop(&mut self) {
        install(None);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Event {
    Enter(f64),
    Refused(f64),
    Queued,
    /// A slot dropped without being reported queued.
    Dropped,
}

type Log = Arc<Mutex<Vec<(ThreadId, Event)>>>;

/// Records every step, refuses the `refuse_at`-th `enter` (counting from 0), and answers
/// `step_flops` with `budget`.
#[derive(Default)]
struct Recorder {
    budget: Option<f64>,
    refuse_at: Option<usize>,
    log: Log,
}

impl Recorder {
    fn installed(budget: Option<f64>, refuse_at: Option<usize>) -> Log {
        let recorder = Recorder {
            budget,
            refuse_at,
            log: Log::default(),
        };
        let log = recorder.log.clone();
        install(Some(Arc::new(recorder)));
        log
    }
}

impl Gate for Recorder {
    fn step_flops(&self) -> Option<f64> {
        self.budget
    }

    fn enter(&self, flops: f64) -> Option<Box<dyn Slot>> {
        let mut log = self.log.lock().unwrap();
        let entered = log
            .iter()
            .filter(|(_, event)| matches!(event, Event::Enter(_) | Event::Refused(_)))
            .count();
        let me = thread::current().id();
        if Some(entered) == self.refuse_at {
            log.push((me, Event::Refused(flops)));
            return None;
        }
        log.push((me, Event::Enter(flops)));
        Some(Box::new(RecordedSlot {
            log: self.log.clone(),
            queued: false,
        }))
    }
}

struct RecordedSlot {
    log: Log,
    queued: bool,
}

impl Slot for RecordedSlot {
    fn queued(mut self: Box<Self>) {
        self.queued = true;
        let me = thread::current().id();
        self.log.lock().unwrap().push((me, Event::Queued));
    }
}

impl Drop for RecordedSlot {
    fn drop(&mut self) {
        if !self.queued {
            let me = thread::current().id();
            self.log.lock().unwrap().push((me, Event::Dropped));
        }
    }
}

/// The steps `log` saw, in order, after checking that on each thread every `enter` was reported
/// queued before the next one, on the thread that entered, and that no slot was dropped.
fn steps(log: &Log) -> Vec<f64> {
    let log = log.lock().unwrap();
    let mut open: Vec<ThreadId> = Vec::new();
    let mut steps = Vec::new();
    for (index, &(thread, event)) in log.iter().enumerate() {
        match event {
            Event::Enter(flops) | Event::Refused(flops) => {
                assert!(
                    !open.contains(&thread),
                    "event {index}: entered while this thread's previous slot was not queued"
                );
                assert!(
                    flops.is_finite() && flops > 0.0,
                    "event {index}: {flops} FLOPs"
                );
                if matches!(event, Event::Enter(_)) {
                    open.push(thread);
                    steps.push(flops);
                }
            }
            Event::Queued => {
                let at = open.iter().position(|&t| t == thread);
                assert!(
                    at.is_some(),
                    "event {index}: queued on a thread with no slot"
                );
                open.swap_remove(at.unwrap());
            }
            Event::Dropped => panic!("event {index}: a slot was dropped, not queued"),
        }
    }
    assert!(open.is_empty(), "slots never queued: {open:?}");
    steps
}

fn refusals(log: &Log) -> usize {
    let log = log.lock().unwrap();
    log.iter()
        .filter(|(_, event)| matches!(event, Event::Refused(_)))
        .count()
}

// ═══════════════════════════════════════════
// The seam, without a GPU
// ═══════════════════════════════════════════

#[test]
fn without_a_gate_a_step_just_runs() {
    let _turn = turn();
    assert_eq!(step(1e9, || 7), Ok(7));
    assert_eq!(step_budget(), None);
}

#[test]
fn a_step_enters_builds_then_reports_queued() {
    let _turn = turn();
    let log = Recorder::installed(None, None);
    let built = step(3e9, || {
        let log = log.lock().unwrap();
        assert!(
            matches!(log.last(), Some((_, Event::Enter(f))) if *f == 3e9),
            "built before entering: {log:?}"
        );
        "ops"
    });
    assert_eq!(built, Ok("ops"));
    assert!(matches!(
        log.lock().unwrap().last(),
        Some((_, Event::Queued))
    ));
    assert_eq!(steps(&log), vec![3e9]);
}

#[test]
fn a_refused_step_builds_nothing() {
    let _turn = turn();
    let log = Recorder::installed(None, Some(1));
    assert_eq!(step(1.0, || ()), Ok(()));
    let mut built = false;
    assert_eq!(step(2.0, || built = true), Err(Stopped));
    assert!(!built);
    assert_eq!(steps(&log), vec![1.0]);
    assert_eq!(refusals(&log), 1);

    let error: Box<dyn Error + Send + Sync> = Stopped.into();
    assert!(
        error.is::<Stopped>(),
        "callers recognise it through the crate's boxed error"
    );
}

#[test]
fn a_gate_is_replaced_and_removed() {
    let _turn = turn();
    let first = Recorder::installed(Some(1.0), None);
    assert_eq!(step_budget(), Some(1.0));
    let second = Recorder::installed(Some(2.0), None);
    assert_eq!(step_budget(), Some(2.0));
    step(5.0, || ()).unwrap();
    assert_eq!(steps(&first), Vec::<f64>::new());
    assert_eq!(steps(&second), vec![5.0]);
    install(None);
    assert_eq!(step_budget(), None);
    step(5.0, || ()).unwrap();
    assert_eq!(steps(&second), vec![5.0]);
}

/// Without a budget, a step takes what its cap allows; with one, what fits beside its fixed part:
/// never none, since an item isn't split, and never past the cap.
#[test]
fn a_step_takes_what_fits_the_budget() {
    assert_eq!(fit(None, 0.0, 1e12, 10), 10, "no budget: the cap");
    assert_eq!(fit(Some(10e9), 0.0, 3e9, 10), 3);
    assert_eq!(fit(Some(10e9), 4e9, 3e9, 10), 2, "beside the fixed part");
    assert_eq!(fit(Some(1e15), 0.0, 3e9, 10), 10, "capped");
    assert_eq!(
        fit(Some(1e9), 0.0, 3e9, 10),
        1,
        "too heavy for any step: alone"
    );
    assert_eq!(
        fit(Some(1e9), 5e9, 3e9, 10),
        1,
        "the fixed part alone overflows: alone"
    );
    assert_eq!(fit(Some(1e9), 0.0, 3e9, 0), 1);
}

/// base.en's shape.
const BASE_EN: Cost = Cost {
    d: 512.0,
    n_mels: 80.0,
    decoder_layers: 6.0,
    vocab: 51864.0,
};

/// Each step's cost is affine in its batch: what it takes for none, plus the same for each item.
#[test]
fn step_costs_are_affine_in_their_batch() {
    let c = BASE_EN;
    let steps: [(&str, &dyn Fn(usize) -> f64); 5] = [
        ("stem", &|n| c.encoder_stem(n, 1500)),
        ("block", &|n| c.encoder_block(n, 1500)),
        ("start", &|n| c.decoder_start(n, 1, 1500)),
        ("step", &|n| c.decoder_step(n, 1500)),
        ("vad", &vad),
    ];
    for (name, step) in steps {
        let (fixed, each) = (step(0), step(1) - step(0));
        assert!(fixed >= 0.0 && each > 0.0, "{name}");
        assert!(
            (step(30) - (fixed + 30.0 * each)).abs() <= 1e-9 * step(30),
            "{name}"
        );
    }
    assert_eq!(c.encoder_stem(0, 1500), 0.0);
    assert_eq!(vad(0), 0.0);
    // One base.en encoder block over a 30 s window: 24·t·d² + 4·t²·d.
    assert_eq!(
        c.block_flops(1, 1500),
        24.0 * 1500.0 * 512.0 * 512.0 + 4.0 * 1500.0 * 1500.0 * 512.0
    );
}

/// The windows of a batched transcription: as many as let its every block and decoder step fit
/// the gate's step, and no more; at least one; all of them without a bound.
#[test]
fn a_batch_of_windows_fits_the_gate_step() {
    let _turn = turn();
    let c = BASE_EN;
    let (beams, prompt, ctx, max) = (3, 1, 1500, 10);
    let largest = |n: usize| {
        c.encoder_block(n, ctx)
            .max(c.decoder_start(n * beams, prompt, ctx))
            .max(c.decoder_step(n * beams, ctx))
    };
    assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), max, "no gate");
    Recorder::installed(None, None);
    assert_eq!(
        c.windows_per_batch(beams, prompt, ctx, max),
        max,
        "a gate without a bound"
    );
    for n in 1..max {
        // A hair over, so that rounding cannot make it n − 1.
        Recorder::installed(Some(largest(n) * (1.0 + 1e-9)), None);
        assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), n);
        Recorder::installed(Some(largest(n + 1) * 0.999), None);
        assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), n);
    }
    Recorder::installed(Some(largest(1) * 0.5), None);
    assert_eq!(
        c.windows_per_batch(beams, prompt, ctx, max),
        1,
        "nothing fits: alone"
    );
    Recorder::installed(Some(1e30), None);
    assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), max, "capped");
}

// ═══════════════════════════════════════════
// Whisper on the GPU
// ═══════════════════════════════════════════

fn device() -> Device {
    #[cfg(feature = "vulkan")]
    {
        Device::vulkan(DeviceKind::DefaultDevice)
    }
    #[cfg(all(not(feature = "vulkan"), feature = "metal"))]
    {
        Device::metal(DeviceKind::DefaultDevice)
    }
    #[cfg(all(not(feature = "vulkan"), not(feature = "metal")))]
    {
        Device::wgpu(DeviceKind::DefaultDevice)
    }
}

fn models_dir() -> PathBuf {
    std::env::var_os("FWB_MODELS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models"))
}

/// `name`'s f16 weights in mixed precision, as Diem loads them.
fn load(name: &str, device: &Device) -> (Whisper, Gpt2Tokenizer) {
    let dir = models_dir();
    let bpe = Gpt2Tokenizer::new(dir.join(format!("{name}-tokenizer.json")).to_str().unwrap())
        .unwrap_or_else(|e| panic!("{name}'s tokenizer in {}: {e}", dir.display()));
    let config = WhisperConfig::load(dir.join(format!("{name}.cfg"))).unwrap();
    let mut store = burn_store::BurnpackStore::from_file(dir.join(format!("{name}-f16.bpk")))
        .with_from_adapter(crate::MixedPrecisionAdapter(DType::F16));
    let mut whisper = config.init(device);
    whisper.load_from(&mut store).unwrap();
    (whisper, bpe)
}

fn audio() -> Vec<f32> {
    let reader = hound::WavReader::open(concat!(env!("CARGO_MANIFEST_DIR"), "/audio.wav")).unwrap();
    assert_eq!(reader.spec().sample_rate, 16_000);
    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect()
}

/// Diem's parameters for a model of `beam_size` (1: greedy, as for large-v3-turbo).
fn params(beam_size: i32) -> WhisperParams {
    WhisperParams {
        print_special: false,
        language: "en".to_string(),
        strategy: if beam_size > 1 {
            SamplingStrategy::BeamSearch {
                beam_size,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: -1 }
        },
        use_f16_compute: true,
        ..WhisperParams::default()
    }
}

/// Overlapping pieces of `audio`, so a batch has several different regions.
fn regions(audio: &[f32]) -> Vec<&[f32]> {
    let s = 16_000;
    vec![
        &audio[..3 * s],
        &audio[2 * s..5 * s],
        &audio[4 * s..],
        audio,
        &audio[s..6 * s],
    ]
}

fn texts(results: &[TranscriptionResult]) -> Vec<String> {
    results.iter().map(|r| r.text.clone()).collect()
}

fn spans(regions: &[SpeechRegion]) -> Vec<(usize, usize)> {
    regions
        .iter()
        .map(|r| (r.start_sample, r.end_sample))
        .collect()
}

fn no_progress(_: usize, _: usize) -> bool {
    true
}

/// Everything Diem runs, once: its outputs, to compare across gates.
#[derive(Debug, PartialEq)]
struct Outputs {
    vad: Vec<(usize, usize)>,
    batched_beam: Vec<String>,
    batched_greedy: Vec<String>,
    sequential_beam: String,
    sequential_greedy: String,
    language: String,
}

struct Models {
    device: Device,
    base_en: (Whisper, Gpt2Tokenizer),
    base: (Whisper, Gpt2Tokenizer),
    vad: SileroVAD6Model,
    audio: Vec<f32>,
}

impl Models {
    /// The models, run once: the kernels are tuned on a shape's first run, and a tuned kernel may
    /// round differently.
    fn load() -> Self {
        let device = device();
        let models = Self {
            base_en: load("base.en", &device),
            base: load("base", &device),
            vad: SileroVAD6Model::new(&device, false).unwrap(),
            audio: audio(),
            device,
        };
        models.run().unwrap();
        models
    }

    fn run(&self) -> crate::token::Result<Outputs> {
        let (whisper, bpe) = &self.base_en;
        let vad = detect_speech_regions(&self.vad, &self.device, &self.audio, Some(no_progress))?;
        let regions = regions(&self.audio);
        let batched = |beam| {
            transcribe_regions_batched(
                whisper,
                bpe,
                &regions,
                16_000,
                &params(beam),
                Some(10),
                Some(no_progress),
            )
        };
        let sequential = |beam| {
            transcribe(
                whisper,
                bpe,
                &self.audio,
                16_000,
                &params(beam),
                Some(no_progress),
            )
        };
        let (base, base_bpe) = &self.base;
        Ok(Outputs {
            vad: spans(&vad),
            batched_beam: texts(&batched(3)?),
            batched_greedy: texts(&batched(1)?),
            sequential_beam: sequential(3)?.text,
            sequential_greedy: sequential(1)?.text,
            language: detect_language(base, base_bpe, &self.audio, 16_000, true)?
                .language
                .as_str()
                .to_string(),
        })
    }
}

/// Every step of everything Diem runs asks, builds, then reports queued, one at a time; and with
/// a gate that only lets steps through, the outputs are those of no gate.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn every_step_asks_then_reports_queued() {
    let _turn = turn();
    let models = Models::load();
    let ungated = models.run().unwrap();
    assert!(
        !ungated.batched_beam.iter().all(|t| t.trim().is_empty()),
        "{ungated:?}"
    );

    let log = Recorder::installed(None, None);
    let gated = models.run().unwrap();
    let steps = steps(&log);
    assert_eq!(gated, ungated);
    println!(
        "{} steps, {:.3e} FLOPs at most",
        steps.len(),
        steps.iter().cloned().fold(0.0, f64::max)
    );
    // At least: a VAD block; per batched run a stem, 6 blocks, a start and a step.
    assert!(steps.len() > 1 + 2 * 9, "{} steps", steps.len());
}

/// A refusal anywhere (the VAD, the encoder's stem or a block, the decoder's start or a step)
/// ends the transcription with `Stopped`, and once the gate is gone the same transcription
/// returns what it did before.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn a_refusal_stops_and_the_next_transcription_runs() {
    let _turn = turn();
    let models = Models::load();
    let expected = models.run().unwrap();
    let (whisper, bpe) = &models.base_en;
    let regions = regions(&models.audio);

    let log = Recorder::installed(None, Some(0));
    let error = detect_speech_regions(
        &models.vad,
        &models.device,
        &models.audio,
        Some(no_progress),
    )
    .err()
    .expect("the VAD stopped");
    assert!(error.is::<Stopped>(), "{error}");
    assert_eq!(refusals(&log), 1);

    // base.en: the stem is step 0, its six blocks 1..=6, the decoder's start 7, then steps.
    for refuse_at in [0, 3, 7, 8, 20] {
        for beam in [3, 1] {
            let log = Recorder::installed(None, Some(refuse_at));
            let error = transcribe_regions_batched(
                whisper,
                bpe,
                &regions,
                16_000,
                &params(beam),
                Some(10),
                Some(no_progress),
            )
            .err()
            .unwrap_or_else(|| panic!("beam {beam}: step {refuse_at} refused, yet it finished"));
            assert!(error.is::<Stopped>(), "{error}");
            assert_eq!(steps(&log).len(), refuse_at);
            assert_eq!(refusals(&log), 1);

            let log = Recorder::installed(None, Some(refuse_at));
            let error = transcribe(
                whisper,
                bpe,
                &models.audio,
                16_000,
                &params(beam),
                Some(no_progress),
            )
            .err()
            .expect("stopped");
            assert!(error.is::<Stopped>(), "{error}");
            assert_eq!(steps(&log).len(), refuse_at);
        }
    }
    let log = Recorder::installed(None, Some(7));
    let (base, base_bpe) = &models.base;
    let error = detect_language(base, base_bpe, &models.audio, 16_000, true)
        .err()
        .unwrap();
    assert!(error.is::<Stopped>(), "{error}");
    assert_eq!(steps(&log).len(), 7);

    install(None);
    assert_eq!(models.run().unwrap(), expected);
}

/// With a bound, a VAD block and a batch of regions are as large as fit it (at least one item);
/// the VAD's regions do not depend on its blocks, nor a region's text on its batch.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn steps_fit_the_gate_step() {
    let _turn = turn();
    let models = Models::load();
    let (whisper, bpe) = &models.base_en;
    let regions = regions(&models.audio);
    let vad = || {
        detect_speech_regions(
            &models.vad,
            &models.device,
            &models.audio,
            Some(no_progress),
        )
    };
    let batched = || {
        transcribe_regions_batched(
            whisper,
            bpe,
            &regions,
            16_000,
            &params(3),
            Some(10),
            Some(no_progress),
        )
    };
    let expected_vad = spans(&vad().unwrap());
    let expected = texts(&batched().unwrap());

    // A hair over 100 windows, so that rounding cannot make it 99.
    let budget = super::vad(100) * (1.0 + 1e-9);
    let log = Recorder::installed(Some(budget), None);
    assert_eq!(spans(&vad().unwrap()), expected_vad);
    let steps_ = steps(&log);
    let windows = models.audio.len().div_ceil(512);
    assert_eq!(steps_.len(), windows.div_ceil(100));
    assert!(
        steps_.iter().all(|&f| f <= budget),
        "{steps_:?} > {budget:e}"
    );

    // Two regions a batch: the largest step of a batch of two, exactly. Every step fits it, and
    // the stem's steps are as many windows as fit it.
    let cost = whisper.cost();
    let layers = whisper.encoder_layer_count();
    let budget = cost
        .encoder_block(2, 1500)
        .max(cost.decoder_start(2 * 3, 1, 1500))
        .max(cost.decoder_step(2 * 3, 1500))
        * (1.0 + 1e-9);
    assert!(
        cost.encoder_stem(1, 1500) <= budget,
        "base.en's stem fits a window a step"
    );
    let log = Recorder::installed(Some(budget), None);
    assert_eq!(texts(&batched().unwrap()), expected);
    let steps_ = steps(&log);
    assert!(
        steps_.iter().all(|&f| f <= budget),
        "{:e} > {budget:e}",
        steps_.iter().cloned().fold(0.0, f64::max)
    );
    // The batches, by their blocks: two of two and one of one. A region whose text fails the
    // quality checks goes through `transcribe` alone, so more blocks of one may follow.
    let blocks = |steps: &[f64], batch: usize| {
        steps
            .iter()
            .filter(|&&f| f == cost.encoder_block(batch, 1500))
            .count()
    };
    assert_eq!(
        blocks(&steps_, 2),
        regions.len() / 2 * layers,
        "batches of two"
    );
    assert!(
        (3..=regions.len()).all(|batch| blocks(&steps_, batch) == 0),
        "no larger batch"
    );
    assert!(blocks(&steps_, 1) >= regions.len() % 2 * layers);
    let stem_windows = (budget / cost.encoder_stem(1, 1500)) as usize;
    assert!(
        steps_
            .iter()
            .any(|&f| f == cost.encoder_stem(stem_windows.min(2), 1500)),
        "the stem's steps, sized apart"
    );

    // Nothing fits: one item a step, and still the same result.
    let log = Recorder::installed(Some(1.0), None);
    assert_eq!(spans(&vad().unwrap()), expected_vad);
    assert_eq!(steps(&log).len(), windows);
    let log = Recorder::installed(Some(1.0), None);
    assert_eq!(texts(&batched().unwrap()), expected);
    let steps_ = steps(&log);
    assert!(
        (2..=regions.len()).all(|batch| blocks(&steps_, batch) == 0),
        "one region a batch"
    );
    assert!(blocks(&steps_, 1) >= regions.len() * layers);
    let stems = steps_
        .iter()
        .filter(|&&f| f == cost.encoder_stem(1, 1500))
        .count();
    assert!(stems >= regions.len(), "a window a stem step");
}

// ═══════════════════════════════════════════
// Calibration and overhead (measurements, not checks)
// ═══════════════════════════════════════════

/// The speed Diem measures (`burn-models` `speed::measure`): FLOP/s of a 1024² by 1024² f16
/// matmul, tuned first, best of three runs of 16 queued at once.
fn matmul_speed(device: &Device) -> f64 {
    const SIDE: usize = 1024;
    const QUEUED: usize = 16;
    let operand = || {
        Tensor::<2>::random(
            [SIDE, SIDE],
            Distribution::Uniform(-1.0, 1.0),
            (device, DType::F16),
        )
    };
    let (a, b) = (operand(), operand());
    drop(a.clone().matmul(b.clone()));
    device.sync().unwrap();
    let mut fastest = f64::INFINITY;
    for _ in 0..3 {
        let start = Instant::now();
        let products: Vec<Tensor<2>> = (0..QUEUED).map(|_| a.clone().matmul(b.clone())).collect();
        device.sync().unwrap();
        fastest = fastest.min(start.elapsed().as_secs_f64());
        drop(products);
    }
    2.0 * (SIDE * SIDE * SIDE * QUEUED) as f64 / fastest
}

/// A step's times, in seconds: building it (`enter` to `queued`), the GPU's time running it, and
/// in all, from `enter` until the GPU has run it with the GPU idle at `enter`.
#[derive(Clone, Copy, Debug)]
struct Times {
    build: f64,
    busy: f64,
    total: f64,
}

/// Times each step alone. Building a step launches some of its kernels and the CPU launches them
/// slower than the GPU runs them, so the GPU's time is measured behind a **blocker**: a chain of
/// matmuls longer than any build, queued at `enter`. The step's kernels then wait behind it and
/// run back to back; the GPU's time is a device-timed window over blocker and step, less one over
/// the blocker alone taken just before.
struct Timer {
    device: Device,
    budget: Option<f64>,
    /// The blocker's operands, and its length in matmuls.
    blocker: Mutex<(Tensor<2>, Tensor<2>, usize)>,
    log: Arc<Mutex<Vec<Times>>>,
}

struct TimedSlot {
    device: Device,
    start: Instant,
    window: ProfileToken,
    blocker: f64,
    /// The blocker's result, alive until its window has closed.
    _blocked: Tensor<2>,
    log: Arc<Mutex<Vec<Times>>>,
}

impl Timer {
    /// A blocker of about `seconds` at `speed`.
    fn new(device: &Device, budget: Option<f64>, speed: f64, seconds: f64) -> Self {
        const SIDE: usize = 2048;
        let operand = || {
            Tensor::<2>::random(
                [SIDE, SIDE],
                Distribution::Uniform(-1.0, 1.0),
                (device, DType::F16),
            )
        };
        let length = (seconds * speed / (2.0 * (SIDE * SIDE * SIDE) as f64)).ceil() as usize;
        Self {
            device: device.clone(),
            budget,
            blocker: Mutex::new((operand(), operand(), length)),
            log: Arc::default(),
        }
    }

    /// Queues the blocker and launches it.
    fn block(&self) -> Tensor<2> {
        let (a, b, length) = &*self.blocker.lock().unwrap();
        let mut x = a.clone();
        for _ in 0..*length {
            x = x.matmul(b.clone());
        }
        self.device.flush().unwrap();
        x
    }

    fn open(&self) -> ProfileToken {
        Dispatch::profile_start(self.device.as_dispatch())
            .unwrap()
            .expect("the device times windows")
    }
}

fn close(device: &Device, window: ProfileToken) -> f64 {
    let duration = Dispatch::profile_end(
        device.as_dispatch(),
        window,
        ProfileOptions::default().flush(),
    )
    .unwrap();
    device.sync().unwrap();
    cubecl::future::block_on(duration.resolve())
        .expect("the window ran")
        .duration()
        .as_secs_f64()
}

impl Gate for Timer {
    fn step_flops(&self) -> Option<f64> {
        self.budget
    }

    fn enter(&self, _flops: f64) -> Option<Box<dyn Slot>> {
        self.device.sync().unwrap();
        let window = self.open();
        let blocked = self.block();
        let blocker = close(&self.device, window);
        drop(blocked);
        let window = self.open();
        let blocked = self.block();
        Some(Box::new(TimedSlot {
            device: self.device.clone(),
            start: Instant::now(),
            window,
            blocker,
            _blocked: blocked,
            log: self.log.clone(),
        }))
    }
}

impl Slot for TimedSlot {
    fn queued(self: Box<Self>) {
        let build = self.start.elapsed().as_secs_f64();
        let busy = close(&self.device, self.window) - self.blocker;
        self.log.lock().unwrap().push(Times {
            build,
            busy,
            total: f64::NAN,
        });
    }
}

/// Times each step alone, its whole time: from `enter`, the GPU idle, until the GPU has run it.
struct Stopwatch {
    device: Device,
    budget: Option<f64>,
    log: Arc<Mutex<Vec<Times>>>,
}

struct StopwatchSlot {
    device: Device,
    start: Instant,
    log: Arc<Mutex<Vec<Times>>>,
}

impl Gate for Stopwatch {
    fn step_flops(&self) -> Option<f64> {
        self.budget
    }

    fn enter(&self, _flops: f64) -> Option<Box<dyn Slot>> {
        self.device.sync().unwrap();
        Some(Box::new(StopwatchSlot {
            device: self.device.clone(),
            start: Instant::now(),
            log: self.log.clone(),
        }))
    }
}

impl Slot for StopwatchSlot {
    fn queued(self: Box<Self>) {
        let build = self.start.elapsed().as_secs_f64();
        self.device.sync().unwrap();
        let total = self.start.elapsed().as_secs_f64();
        self.log.lock().unwrap().push(Times {
            build,
            busy: f64::NAN,
            total,
        });
    }
}

/// `run`'s steps' times: the GPU's from `timer`, the whole from a [`Stopwatch`]; each the
/// smallest over `runs` runs, after one to warm up.
fn step_times(timer: Timer, runs: usize, mut run: impl FnMut()) -> Vec<Times> {
    let device = timer.device.clone();
    let stopwatch = Stopwatch {
        device: device.clone(),
        budget: timer.budget,
        log: Arc::default(),
    };
    let (busy_log, total_log) = (timer.log.clone(), stopwatch.log.clone());
    let gates: [Arc<dyn Gate>; 2] = [Arc::new(timer), Arc::new(stopwatch)];
    let mut best: Vec<Times> = Vec::new();
    for round in 0..=runs {
        for (gate, log) in gates.iter().zip([&busy_log, &total_log]) {
            log.lock().unwrap().clear();
            install(Some(gate.clone()));
            let started = Instant::now();
            run();
            install(None);
            // Rest as long as it ran: these runs keep the GPU at its limit.
            thread::sleep(started.elapsed());
            if round == 0 {
                continue;
            }
            let times = log.lock().unwrap().clone();
            if best.is_empty() {
                best = times
                    .iter()
                    .map(|_| Times {
                        build: f64::INFINITY,
                        busy: f64::INFINITY,
                        total: f64::INFINITY,
                    })
                    .collect();
            }
            assert_eq!(best.len(), times.len(), "the same steps each run");
            for (b, t) in best.iter_mut().zip(times) {
                b.build = b.build.min(t.build);
                if t.busy.is_finite() {
                    b.busy = b.busy.min(t.busy);
                }
                if t.total.is_finite() {
                    b.total = b.total.min(t.total);
                }
            }
        }
    }
    best
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

/// Each field's median.
fn medians(times: &[Times]) -> Times {
    Times {
        build: median(times.iter().map(|t| t.build).collect()),
        busy: median(times.iter().map(|t| t.busy).collect()),
        total: median(times.iter().map(|t| t.total).collect()),
    }
}

/// `t = a + b·f` by least squares over `(f, t)`.
fn affine(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len() as f64;
    let (mf, mt) = points
        .iter()
        .fold((0.0, 0.0), |(f, t), p| (f + p.0 / n, t + p.1 / n));
    let (cov, var) = points.iter().fold((0.0, 0.0), |(c, v), p| {
        (c + (p.0 - mf) * (p.1 - mt), v + (p.0 - mf).powi(2))
    });
    let b = cov / var;
    (mt - b * mf, b)
}

/// Each step kind's efficiency relative to Diem's matmul speed on this machine, and the fixed
/// cost of the kinds that have one, fitted over the models and batches Diem runs: base.en (beam 3)
/// and large-v3-turbo (greedy) over batches of 1, 4 and 10 regions, and VAD blocks of 64, 256 and
/// 1024 windows. Fitted to the GPU's time of each step (see [`Timer`]); the whole time, the GPU
/// idle at the start, is printed beside it. The `*_EFFICIENCY` and `*_FIXED` constants come from
/// it, run as Diem builds this crate (its burn and cubecl, `CUBECL_AUTOTUNE_LEVEL=extensive`).
#[test]
#[ignore = "a measurement: needs a GPU and Whisper weights"]
fn calibrate() {
    let _turn = turn();
    let device = device();
    // The fastest of a few, as another process on the GPU slows it.
    let speed = (0..5).map(|_| matmul_speed(&device)).fold(0.0, f64::max);
    println!("matmul speed: {:.1} TFLOP/s", speed / 1e12);
    let timer = |budget| Timer::new(&device, budget, speed, 0.04);
    let show = |t: Times| {
        format!(
            "build {:6.3} ms, GPU {:7.3} ms, whole {:7.3} ms",
            t.build * 1e3,
            t.busy * 1e3,
            t.total * 1e3
        )
    };

    let audio = audio();
    let vad_model = SileroVAD6Model::new(&device, false).unwrap();
    // Two minutes of speech: the clip, repeated.
    let long: Vec<f32> = audio
        .iter()
        .cycle()
        .take(audio.len() * 16)
        .cloned()
        .collect();
    let mut vad_points = Vec::new();
    for windows in [64usize, 256, 1024] {
        let times = step_times(timer(Some(super::vad(windows) * (1.0 + 1e-9))), 2, || {
            detect_speech_regions(&vad_model, &device, &long, Some(no_progress)).unwrap();
        });
        // The last block is short.
        let t = medians(&times[..times.len() - 1]);
        println!("vad {windows:>4} windows: {}", show(t));
        vad_points.push((windows as f64 * VAD_WINDOW_ANALYTIC, t.busy));
    }

    let (mut stem_points, mut block_points, mut step_points, mut starts) =
        (vec![], vec![], vec![], vec![]);
    for (name, beam) in [("base.en", 3usize), ("large-v3-turbo", 1)] {
        let (whisper, bpe) = load(name, &device);
        let cost = whisper.cost();
        let encoder_layers = whisper.encoder_layer_count();
        let mut params = params(beam as i32);
        // No fallback to the sequential path: one batch, every step in order.
        params.temperature_inc = 0.0;
        let prompt = if bpe.is_multilingual() { 3 } else { 1 };
        for batch in [1usize, 4, 10] {
            let regions: Vec<&[f32]> = (0..batch).map(|_| audio.as_slice()).collect();
            let times = step_times(timer(None), 2, || {
                transcribe_regions_batched(
                    &whisper,
                    &bpe,
                    &regions,
                    16_000,
                    &params,
                    Some(batch),
                    Some(no_progress),
                )
                .unwrap();
            });
            let (stem, rest) = times.split_first().unwrap();
            let (blocks, rest) = rest.split_at(encoder_layers);
            let (start, decode) = rest.split_first().unwrap();
            let sequences = batch * beam;
            let (block, step) = (medians(blocks), medians(decode));
            println!(
                "{name} ×{batch}: {sequences} sequences, {} steps",
                decode.len()
            );
            println!("    stem  {}", show(*stem));
            println!("    block {}", show(block));
            println!("    start {}", show(*start));
            println!("    step  {}", show(step));
            stem_points.push((cost.stem_flops(batch, 1500), stem.busy));
            block_points.push((cost.block_flops(batch, 1500), block.busy));
            step_points.push((cost.pass_flops(sequences, 1500), step.busy));
            starts.push((
                cost.cross_flops(sequences, 1500),
                cost.pass_flops(sequences * prompt, 1500),
                start.busy,
            ));
        }
    }

    // Through the origin, by the totals: the long blocks weigh most, as they should.
    let ratio = |points: &[(f64, f64)]| {
        points.iter().map(|p| p.0).sum::<f64>() / points.iter().map(|p| p.1).sum::<f64>() / speed
    };
    let (step_fixed, step_slope) = affine(&step_points);
    let (block_fixed, block_slope) = affine(&block_points);
    // The start is the cross-attention cache and a pass over the prompt, which the steps price.
    let cross: Vec<(f64, f64)> = starts
        .iter()
        .map(|&(cross, pass, t)| (cross, t - step_fixed - pass * step_slope))
        .collect();
    println!("const VAD_EFFICIENCY: f64 = {:.3e};", ratio(&vad_points));
    println!(
        "const ENCODER_STEM_EFFICIENCY: f64 = {:.3e};",
        ratio(&stem_points)
    );
    println!(
        "const ENCODER_BLOCK_EFFICIENCY: f64 = {:.3e};",
        1.0 / (block_slope * speed)
    );
    println!(
        "const ENCODER_BLOCK_FIXED: f64 = {:.3e}; // {:.3} ms",
        block_fixed * speed,
        block_fixed * 1e3
    );
    println!("const CROSS_CACHE_EFFICIENCY: f64 = {:.3e};", ratio(&cross));
    println!(
        "const DECODER_PASS_EFFICIENCY: f64 = {:.3e};",
        1.0 / (step_slope * speed)
    );
    println!(
        "const DECODER_PASS_FIXED: f64 = {:.3e}; // {:.3} ms",
        step_fixed * speed,
        step_fixed * 1e3
    );
}

/// What a step costs the CPU with no gate and with a pass-through one; then a full Diem
/// transcription of `audio.wav` (VAD, then base.en beam 3 or large-v3-turbo greedy, batched) with
/// no gate, a pass-through gate, and a gate whose slot flushes the device as Diem's does: 20 of
/// each, interleaved.
#[test]
#[ignore = "a measurement: needs a GPU and Whisper weights"]
fn overhead() {
    struct PassThrough(Option<Device>);
    struct NoOp;
    struct Flush(Device);
    impl Slot for NoOp {
        fn queued(self: Box<Self>) {}
    }
    impl Slot for Flush {
        fn queued(self: Box<Self>) {
            self.0.flush().unwrap();
        }
    }
    impl Gate for PassThrough {
        fn step_flops(&self) -> Option<f64> {
            None
        }
        fn enter(&self, _flops: f64) -> Option<Box<dyn Slot>> {
            Some(match &self.0 {
                None => Box::new(NoOp),
                Some(device) => Box::new(Flush(device.clone())),
            })
        }
    }

    let _turn = turn();
    // The seam alone: a step that builds nothing, a million times.
    for (label, gate) in [
        ("no gate", None),
        (
            "pass-through",
            Some(Arc::new(PassThrough(None)) as Arc<dyn Gate>),
        ),
    ] {
        install(gate);
        let start = Instant::now();
        for i in 0..1_000_000u64 {
            std::hint::black_box(step(1e9, || std::hint::black_box(i)).unwrap());
        }
        install(None);
        println!(
            "seam, {label:>12}: {:.1} ns a step",
            start.elapsed().as_secs_f64() * 1e3
        );
    }

    let device = device();
    let audio = audio();
    let vad_model = SileroVAD6Model::new(&device, false).unwrap();
    for (name, beam) in [("base.en", 3), ("large-v3-turbo", 1)] {
        let (whisper, bpe) = load(name, &device);
        let params = params(beam);
        let run = || {
            let regions =
                detect_speech_regions(&vad_model, &device, &audio, Some(no_progress)).unwrap();
            let waveforms: Vec<&[f32]> = regions
                .iter()
                .map(|r| &audio[r.start_sample..r.end_sample])
                .collect();
            let start = Instant::now();
            transcribe_regions_batched(
                &whisper,
                &bpe,
                &waveforms,
                16_000,
                &params,
                Some(10),
                Some(no_progress),
            )
            .unwrap();
            start.elapsed().as_secs_f64()
        };
        let gates: [(&str, Option<Arc<dyn Gate>>); 3] = [
            ("no gate", None),
            (
                "pass-through",
                Some(Arc::new(PassThrough(None)) as Arc<dyn Gate>),
            ),
            (
                "flushing",
                Some(Arc::new(PassThrough(Some(device.clone()))) as Arc<dyn Gate>),
            ),
        ];
        for _ in 0..2 {
            run();
        }
        let mut times = vec![Vec::new(); gates.len()];
        for round in 0..20 {
            // Each gate in each place in turn, so that none gains from where it runs in a round.
            for i in (0..gates.len()).map(|i| (i + round) % gates.len()) {
                let gate = &gates[i].1;
                install(gate.clone());
                let start = Instant::now();
                let transcription = run();
                let total = start.elapsed().as_secs_f64();
                install(None);
                times[i].push((total, transcription));
                thread::sleep(Duration::from_secs_f64(total));
            }
        }
        for ((label, _), times) in gates.iter().zip(times) {
            let fastest = |field: fn(&(f64, f64)) -> f64| {
                times.iter().map(field).fold(f64::INFINITY, f64::min) * 1e3
            };
            let middle =
                |field: fn(&(f64, f64)) -> f64| median(times.iter().map(field).collect()) * 1e3;
            println!(
                "{name} {label:>12}: VAD + transcription median {:.1} ms (fastest {:.1}), transcription median {:.1} ms (fastest {:.1})",
                middle(|t| t.0),
                fastest(|t| t.0),
                middle(|t| t.1),
                fastest(|t| t.1),
            );
        }
    }
}
