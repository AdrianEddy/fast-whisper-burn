//! The pacing seam ([`crate::pacing`]): what a step asks the gate and when it reports, how a
//! refusal ends a transcription, and how steps size themselves to the gate's budget.
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
    time::Instant,
};

/// A matmul speed for the tests that size steps: about the RTX 5090's.
const SPEED: f64 = 150e12;

/// What `cost` takes at `speed`, in seconds.
fn seconds(cost: Cost, speed: f64) -> f64 {
    cost.fixed.as_secs_f64() + cost.flops / speed
}

/// A budget of `seconds` at [`SPEED`], a hair over so that rounding cannot cut what fits.
fn budget_of(seconds: f64) -> Option<(Duration, f64)> {
    Some((
        Duration::from_secs_f64(seconds) + Duration::from_nanos(2),
        SPEED,
    ))
}

fn flops(flops: f64) -> Cost {
    Cost {
        flops,
        fixed: Duration::ZERO,
    }
}

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
    Enter(Cost),
    Refused(Cost),
    Queued,
    /// A slot dropped without being reported queued.
    Dropped,
}

type Log = Arc<Mutex<Vec<(ThreadId, Event)>>>;

/// Records every step, refuses the `refuse_at`-th `enter` (counting from 0), and answers
/// `budget` with `budget`.
#[derive(Default)]
struct Recorder {
    budget: Option<(Duration, f64)>,
    refuse_at: Option<usize>,
    log: Log,
}

impl Recorder {
    fn installed(budget: Option<(Duration, f64)>, refuse_at: Option<usize>) -> Log {
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
    fn budget(&self) -> Option<(Duration, f64)> {
        self.budget
    }

    fn enter(&self, cost: Cost) -> Option<Box<dyn Slot>> {
        let mut log = self.log.lock().unwrap();
        let entered = log
            .iter()
            .filter(|(_, event)| matches!(event, Event::Enter(_) | Event::Refused(_)))
            .count();
        let me = thread::current().id();
        if Some(entered) == self.refuse_at {
            log.push((me, Event::Refused(cost)));
            return None;
        }
        log.push((me, Event::Enter(cost)));
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
fn steps(log: &Log) -> Vec<Cost> {
    let log = log.lock().unwrap();
    let mut open: Vec<ThreadId> = Vec::new();
    let mut steps = Vec::new();
    for (index, &(thread, event)) in log.iter().enumerate() {
        match event {
            Event::Enter(cost) | Event::Refused(cost) => {
                assert!(
                    !open.contains(&thread),
                    "event {index}: entered while this thread's previous slot was not queued"
                );
                assert!(
                    cost.flops.is_finite() && cost.flops > 0.0,
                    "event {index}: {cost:?}"
                );
                if matches!(event, Event::Enter(_)) {
                    open.push(thread);
                    steps.push(cost);
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
    assert_eq!(step(flops(1e9), || 7), Ok(7));
    assert_eq!(budget(), None);
}

#[test]
fn a_step_enters_builds_then_reports_queued() {
    let _turn = turn();
    let log = Recorder::installed(None, None);
    let cost = Cost {
        flops: 3e9,
        fixed: Duration::from_micros(1400),
    };
    let built = step(cost, || {
        let log = log.lock().unwrap();
        assert!(
            matches!(log.last(), Some((_, Event::Enter(c))) if *c == cost),
            "built before entering: {log:?}"
        );
        "ops"
    });
    assert_eq!(built, Ok("ops"));
    assert!(matches!(
        log.lock().unwrap().last(),
        Some((_, Event::Queued))
    ));
    assert_eq!(steps(&log), vec![cost]);
}

#[test]
fn a_refused_step_builds_nothing() {
    let _turn = turn();
    let log = Recorder::installed(None, Some(1));
    assert_eq!(step(flops(1.0), || ()), Ok(()));
    let mut built = false;
    assert_eq!(step(flops(2.0), || built = true), Err(Stopped));
    assert!(!built);
    assert_eq!(steps(&log), vec![flops(1.0)]);
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
    let (one, two) = (
        Some((Duration::from_millis(15), 1e12)),
        Some((Duration::from_millis(50), 2e12)),
    );
    let first = Recorder::installed(one, None);
    assert_eq!(budget(), one);
    let second = Recorder::installed(two, None);
    assert_eq!(budget(), two);
    step(flops(5.0), || ()).unwrap();
    assert_eq!(steps(&first), Vec::<Cost>::new());
    assert_eq!(steps(&second), vec![flops(5.0)]);
    install(None);
    assert_eq!(budget(), None);
    step(flops(5.0), || ()).unwrap();
    assert_eq!(steps(&second), vec![flops(5.0)]);
}

/// Without a budget, a step takes what its cap allows; with one, the most whose fixed time and
/// FLOPs fit it: never none, since an item isn't split, and never past the cap.
#[test]
fn a_step_takes_what_fits_the_budget() {
    let ms = Duration::from_millis;
    // 3 ms of FLOPs an item at 1 TFLOP/s, and a fixed time.
    let step = |fixed: u64| {
        move |n: usize| Cost {
            flops: n as f64 * 3e9,
            fixed: ms(fixed),
        }
    };
    let budget = |time: u64| Some((ms(time), 1e12));
    assert_eq!(fit(None, step(0), 10), 10, "no budget: the cap");
    assert_eq!(fit(budget(10), step(0), 10), 3);
    assert_eq!(fit(budget(10), step(4), 10), 2, "beside the fixed time");
    assert_eq!(
        fit(Some((ms(10), 2e12)), step(4), 10),
        4,
        "at twice the speed"
    );
    assert_eq!(fit(budget(100_000), step(0), 10), 10, "capped");
    assert_eq!(
        fit(budget(1), step(0), 10),
        1,
        "too heavy for any step: alone"
    );
    assert_eq!(
        fit(budget(1), step(5), 10),
        1,
        "the fixed time alone overflows: alone"
    );
    assert_eq!(fit(budget(1), step(0), 0), 1);
}

/// base.en's shape.
const BASE_EN: Shape = Shape {
    d: 512.0,
    n_mels: 80.0,
    decoder_layers: 6.0,
    vocab: 51864.0,
};

/// Each step's cost is its batch's FLOPs, the same for each item, and a fixed time whatever the
/// batch.
#[test]
fn step_costs_are_affine_in_their_batch() {
    let c = BASE_EN;
    let steps: [(&str, &dyn Fn(usize) -> Cost); 5] = [
        ("stem", &|n| c.encoder_stem(n, 1500)),
        ("block", &|n| c.encoder_block(n, 1500)),
        ("start", &|n| c.decoder_start(n, 1, 1500)),
        ("step", &|n| c.decoder_step(n, 1500)),
        ("vad", &vad),
    ];
    for (name, step) in steps {
        assert_eq!(step(0).flops, 0.0, "{name}");
        assert!(step(1).flops > 0.0, "{name}");
        assert!(
            (step(30).flops - 30.0 * step(1).flops).abs() <= 1e-9 * step(30).flops,
            "{name}"
        );
        assert_eq!(step(0).fixed, step(30).fixed, "{name}");
    }
    assert_eq!(c.encoder_stem(1, 1500).fixed, Duration::ZERO);
    assert_eq!(vad(1).fixed, Duration::ZERO);
    assert!(
        c.decoder_step(1, 1500).fixed > Duration::ZERO,
        "a pass is bound by its dispatches"
    );
    // One base.en encoder block over a 30 s window: 24·t·d² + 4·t²·d.
    assert_eq!(
        c.block_flops(1, 1500),
        24.0 * 1500.0 * 512.0 * 512.0 + 4.0 * 1500.0 * 1500.0 * 512.0
    );
}

/// The windows of a batched transcription: as many as let its every block and decoder step fit
/// the gate's budget, and no more; at least one; all of them without a bound.
#[test]
fn a_batch_of_windows_fits_the_gate_budget() {
    let _turn = turn();
    let c = BASE_EN;
    let (beams, prompt, ctx, max) = (3, 1, 1500, 10);
    let largest = |n: usize| {
        [
            c.encoder_block(n, ctx),
            c.decoder_start(n * beams, prompt, ctx),
            c.decoder_step(n * beams, ctx),
        ]
        .into_iter()
        .map(|cost| seconds(cost, SPEED))
        .fold(0.0, f64::max)
    };
    assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), max, "no gate");
    Recorder::installed(None, None);
    assert_eq!(
        c.windows_per_batch(beams, prompt, ctx, max),
        max,
        "a gate without a bound"
    );
    for n in 1..max {
        Recorder::installed(budget_of(largest(n)), None);
        assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), n);
        Recorder::installed(budget_of(largest(n + 1) * 0.999), None);
        assert_eq!(c.windows_per_batch(beams, prompt, ctx, max), n);
    }
    Recorder::installed(budget_of(largest(1) * 0.5), None);
    assert_eq!(
        c.windows_per_batch(beams, prompt, ctx, max),
        1,
        "nothing fits: alone"
    );
    Recorder::installed(budget_of(1e6), None);
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
        "{} steps, {:.3} ms at most at {:.0} TFLOP/s",
        steps.len(),
        steps.iter().map(|&c| seconds(c, SPEED)).fold(0.0, f64::max) * 1e3,
        SPEED / 1e12
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
fn steps_fit_the_gate_budget() {
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
    let fits = |steps: &[Cost], budget: Option<(Duration, f64)>| {
        let (time, speed) = budget.unwrap();
        let over: Vec<&Cost> = steps
            .iter()
            .filter(|&&c| seconds(c, speed) > time.as_secs_f64())
            .collect();
        assert!(over.is_empty(), "over {time:?}: {over:?}");
    };

    // 100 windows a block.
    let budget = budget_of(seconds(super::vad(100), SPEED));
    let log = Recorder::installed(budget, None);
    assert_eq!(spans(&vad().unwrap()), expected_vad);
    let steps_ = steps(&log);
    let windows = models.audio.len().div_ceil(512);
    assert_eq!(steps_.len(), windows.div_ceil(100));
    fits(&steps_, budget);

    // Two regions a batch: the longest step of a batch of two, exactly. Every step fits it, and
    // the stem's steps are as many windows as fit it.
    let shape = whisper.shape();
    let layers = whisper.encoder_layer_count();
    let budget = budget_of(
        [
            shape.encoder_block(2, 1500),
            shape.decoder_start(2 * 3, 1, 1500),
            shape.decoder_step(2 * 3, 1500),
        ]
        .into_iter()
        .map(|c| seconds(c, SPEED))
        .fold(0.0, f64::max),
    );
    let stem_windows = fit(budget, |n| shape.encoder_stem(n, 1500), 10);
    let log = Recorder::installed(budget, None);
    assert_eq!(texts(&batched().unwrap()), expected);
    let steps_ = steps(&log);
    if seconds(shape.encoder_stem(1, 1500), SPEED) <= budget.unwrap().0.as_secs_f64() {
        fits(&steps_, budget);
    }
    // The batches, by their blocks: two of two and one of one. A region whose text fails the
    // quality checks goes through `transcribe` alone, so more blocks of one may follow.
    let blocks = |steps: &[Cost], batch: usize| {
        steps
            .iter()
            .filter(|&&c| c == shape.encoder_block(batch, 1500))
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
    assert!(
        steps_
            .iter()
            .any(|&c| c == shape.encoder_stem(stem_windows.min(2), 1500)),
        "the stem's steps, sized apart"
    );

    // Nothing fits: one item a step, and still the same result.
    let nothing = Some((Duration::from_nanos(1), SPEED));
    let log = Recorder::installed(nothing, None);
    assert_eq!(spans(&vad().unwrap()), expected_vad);
    assert_eq!(steps(&log).len(), windows);
    let log = Recorder::installed(nothing, None);
    assert_eq!(texts(&batched().unwrap()), expected);
    let steps_ = steps(&log);
    assert!(
        (2..=regions.len()).all(|batch| blocks(&steps_, batch) == 0),
        "one region a batch"
    );
    assert!(blocks(&steps_, 1) >= regions.len() * layers);
    let stems = steps_
        .iter()
        .filter(|&&c| c == shape.encoder_stem(1, 1500))
        .count();
    assert!(stems >= regions.len(), "a window a stem step");
}

// ═══════════════════════════════════════════
// Calibration and overhead (measurements, not checks)
// ═══════════════════════════════════════════

/// The speed Diem measures (`burn-models` `speed::measure`, 1a0304d): FLOP/s of a square f16
/// matmul, its kernel tuned first, the best of three runs of 16 queued at once; 1024² by 1024², its
/// side doubled up to 4096 while a run takes under 5 ms.
fn matmul_speed(device: &Device) -> f64 {
    const QUEUED: usize = 16;
    let mut side = 1024;
    loop {
        let operand = || {
            Tensor::<2>::random(
                [side, side],
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
            let products: Vec<Tensor<2>> =
                (0..QUEUED).map(|_| a.clone().matmul(b.clone())).collect();
            device.sync().unwrap();
            fastest = fastest.min(start.elapsed().as_secs_f64());
            drop(products);
        }
        if fastest >= 0.005 || side == 4096 {
            return 2.0 * (side * side * side * QUEUED) as f64 / fastest;
        }
        side *= 2;
    }
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
/// run back to back; the GPU's time is a device-timed window over blocker and step, less the mean
/// of two over the blocker alone, one just before and one just after, as the GPU's clock drifts.
struct Timer {
    device: Device,
    budget: Option<(Duration, f64)>,
    blocker: Blocker,
    log: Arc<Mutex<Vec<Times>>>,
}

/// The blocker's operands, and its length in matmuls.
type Blocker = Arc<Mutex<(Tensor<2>, Tensor<2>, usize)>>;

struct TimedSlot {
    device: Device,
    start: Instant,
    window: ProfileToken,
    blocker: Blocker,
    before: f64,
    /// The blocker's result, alive until its window has closed.
    blocked: Tensor<2>,
    log: Arc<Mutex<Vec<Times>>>,
}

/// Queues the blocker and launches it.
fn block(device: &Device, blocker: &Blocker) -> Tensor<2> {
    let (a, b, length) = &*blocker.lock().unwrap();
    let mut x = a.clone();
    for _ in 0..*length {
        x = x.matmul(b.clone());
    }
    device.flush().unwrap();
    x
}

fn open(device: &Device) -> ProfileToken {
    Dispatch::profile_start(device.as_dispatch())
        .unwrap()
        .expect("the device times windows")
}

/// The blocker alone, its time on the device.
fn blocker_alone(device: &Device, blocker: &Blocker) -> f64 {
    let window = open(device);
    let blocked = block(device, blocker);
    let time = close(device, window);
    drop(blocked);
    time
}

impl Timer {
    /// With a blocker of about `seconds`, as long as its matmuls measure now.
    fn new(device: &Device, budget: Option<(Duration, f64)>, seconds: f64) -> Self {
        const SIDE: usize = 2048;
        let operand = || {
            Tensor::<2>::random(
                [SIDE, SIDE],
                Distribution::Uniform(-1.0, 1.0),
                (device, DType::F16),
            )
        };
        let blocker = Arc::new(Mutex::new((operand(), operand(), 16)));
        drop(block(device, &blocker));
        device.sync().unwrap();
        let each = blocker_alone(device, &blocker) / 16.0;
        blocker.lock().unwrap().2 = (seconds / each).ceil() as usize;
        Self {
            device: device.clone(),
            budget,
            blocker,
            log: Arc::default(),
        }
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
    fn budget(&self) -> Option<(Duration, f64)> {
        self.budget
    }

    fn enter(&self, _cost: Cost) -> Option<Box<dyn Slot>> {
        self.device.sync().unwrap();
        let before = blocker_alone(&self.device, &self.blocker);
        let window = open(&self.device);
        let blocked = block(&self.device, &self.blocker);
        Some(Box::new(TimedSlot {
            device: self.device.clone(),
            start: Instant::now(),
            window,
            blocker: self.blocker.clone(),
            before,
            blocked,
            log: self.log.clone(),
        }))
    }
}

impl Slot for TimedSlot {
    fn queued(self: Box<Self>) {
        let build = self.start.elapsed().as_secs_f64();
        let both = close(&self.device, self.window);
        drop(self.blocked);
        let after = blocker_alone(&self.device, &self.blocker);
        let busy = both - (self.before + after) / 2.0;
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
    budget: Option<(Duration, f64)>,
    log: Arc<Mutex<Vec<Times>>>,
}

struct StopwatchSlot {
    device: Device,
    start: Instant,
    log: Arc<Mutex<Vec<Times>>>,
}

impl Gate for Stopwatch {
    fn budget(&self) -> Option<(Duration, f64)> {
        self.budget
    }

    fn enter(&self, _cost: Cost) -> Option<Box<dyn Slot>> {
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

/// `run`'s steps' times over `runs` runs, after one to warm up: the GPU's from `timer`, the median
/// (a difference of two windows, so noise either way); the build and the whole from a
/// [`Stopwatch`], the smallest.
fn step_times(timer: Timer, runs: usize, mut run: impl FnMut()) -> Vec<Times> {
    let device = timer.device.clone();
    let stopwatch = Stopwatch {
        device: device.clone(),
        budget: timer.budget,
        log: Arc::default(),
    };
    let (busy_log, total_log) = (timer.log.clone(), stopwatch.log.clone());
    let gates: [Arc<dyn Gate>; 2] = [Arc::new(timer), Arc::new(stopwatch)];
    let (mut busy, mut whole): (Vec<Vec<Times>>, Vec<Vec<Times>>) = (vec![], vec![]);
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
            let into = if Arc::ptr_eq(log, &busy_log) {
                &mut busy
            } else {
                &mut whole
            };
            into.push(times);
        }
    }
    let steps = whole[0].len();
    assert!(
        busy.iter().chain(&whole).all(|run| run.len() == steps),
        "the same steps each run"
    );
    (0..steps)
        .map(|i| Times {
            build: whole
                .iter()
                .map(|run| run[i].build)
                .fold(f64::INFINITY, f64::min),
            busy: median(busy.iter().map(|run| run[i].busy).collect()),
            total: whole
                .iter()
                .map(|run| run[i].total)
                .fold(f64::INFINITY, f64::min),
        })
        .collect()
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

/// `t = b·f` through the origin, by the totals: the long steps weigh most, as they should.
fn proportional(points: &[(f64, f64)]) -> f64 {
    points.iter().map(|p| p.1).sum::<f64>() / points.iter().map(|p| p.0).sum::<f64>()
}

/// Each step kind's efficiency relative to Diem's matmul speed on this machine ([`matmul_speed`])
/// and the fixed time of the kinds bound by their dispatches, fitted over the models and batches
/// Diem runs: base.en (beam 3) and large-v3-turbo (greedy) over batches of 1, 4 and 10 regions,
/// and VAD blocks of 64, 256 and 1024 windows. Fitted to the GPU's time of each step (see
/// [`Timer`]); the whole time, the GPU idle at the start, is printed beside it, and each fit beside
/// what it was fitted to. The `*_EFFICIENCY` and `*_FIXED` constants come from it, run as Diem
/// builds this crate (its burn and cubecl, `CUBECL_AUTOTUNE_LEVEL=extensive`).
#[test]
#[ignore = "a measurement: needs a GPU and Whisper weights"]
fn calibrate() {
    let _turn = turn();
    let device = device();
    // The fastest of a few, before and after the steps, as another process on the GPU slows it.
    let fastest = || (0..5).map(|_| matmul_speed(&device)).fold(0.0, f64::max);
    let before = fastest();
    let timer = |budget| Timer::new(&device, budget, 0.02);
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
        let block = super::vad(windows);
        let budget = Some((
            Duration::from_secs_f64(seconds(block, before)) + Duration::from_nanos(2),
            before,
        ));
        let times = step_times(timer(budget), 3, || {
            detect_speech_regions(&vad_model, &device, &long, Some(no_progress)).unwrap();
        });
        // The last block is short.
        let t = medians(&times[..times.len() - 1]);
        println!("vad {windows:>4} windows: {}", show(t));
        vad_points.push((windows as f64 * VAD_WINDOW_ANALYTIC, t.busy));
    }

    let (mut stem_points, mut block_points, mut step_points, mut starts) =
        (vec![], vec![], vec![], vec![]);
    let mut labels = vec![];
    for (name, beam) in [("base.en", 3usize), ("large-v3-turbo", 1)] {
        let (whisper, bpe) = load(name, &device);
        let shape = whisper.shape();
        let encoder_layers = whisper.encoder_layer_count();
        let mut params = params(beam as i32);
        // No fallback to the sequential path: one batch, every step in order.
        params.temperature_inc = 0.0;
        let prompt = if bpe.is_multilingual() { 3 } else { 1 };
        for batch in [1usize, 4, 10] {
            let regions: Vec<&[f32]> = (0..batch).map(|_| audio.as_slice()).collect();
            let times = step_times(timer(None), 3, || {
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
            labels.push(format!("{name} ×{batch}"));
            stem_points.push((shape.stem_flops(batch, 1500), stem.busy));
            block_points.push((shape.block_flops(batch, 1500), block.busy));
            step_points.push((shape.pass_flops(sequences, 1500), step.busy));
            starts.push((
                shape.cross_flops(sequences, 1500),
                shape.pass_flops(sequences * prompt, 1500),
                start.busy,
            ));
        }
    }

    let after = fastest();
    let speed = before.max(after);
    println!(
        "matmul speed: {:.1} TFLOP/s ({:.1} before, {:.1} after)",
        speed / 1e12,
        before / 1e12,
        after / 1e12
    );

    let vad_slope = proportional(&vad_points);
    let stem_slope = proportional(&stem_points);
    let (block_fixed, block_slope) = affine(&block_points);
    let (step_fixed, step_slope) = affine(&step_points);
    // The start is the cross-attention cache and a pass over the prompt, which the steps price;
    // the median of the starts, as the cache's share of a small one is noise.
    let cross_slope = median(
        starts
            .iter()
            .map(|&(cross, pass, t)| (t - step_fixed - pass * step_slope) / cross)
            .collect(),
    );

    // Each fit beside what it was fitted to.
    let mut worst = 0.0f64;
    let mut compare = |kind: &str, label: &str, model: f64, measured: f64| {
        let error = model / measured - 1.0;
        worst = worst.max(error.abs());
        println!(
            "    {kind:<5} {label:<18} fit {:8.3} ms, measured {:8.3} ms, {:+5.0}%",
            model * 1e3,
            measured * 1e3,
            error * 1e2
        );
    };
    println!("fits:");
    for (&(f, t), windows) in vad_points.iter().zip([64, 256, 1024]) {
        compare("vad", &format!("{windows} windows"), f * vad_slope, t);
    }
    for (i, label) in labels.iter().enumerate() {
        compare(
            "stem",
            label,
            stem_points[i].0 * stem_slope,
            stem_points[i].1,
        );
        compare(
            "block",
            label,
            block_fixed + block_points[i].0 * block_slope,
            block_points[i].1,
        );
        let (c, p, t) = starts[i];
        compare(
            "start",
            label,
            c * cross_slope + step_fixed + p * step_slope,
            t,
        );
        compare(
            "step",
            label,
            step_fixed + step_points[i].0 * step_slope,
            step_points[i].1,
        );
    }
    println!("worst: {:.0}%", worst * 1e2);

    let efficiency = |slope: f64| 1.0 / (slope * speed);
    println!("const VAD_EFFICIENCY: f64 = {:.3e};", efficiency(vad_slope));
    println!(
        "const ENCODER_STEM_EFFICIENCY: f64 = {:.3e};",
        efficiency(stem_slope)
    );
    println!(
        "const ENCODER_BLOCK_EFFICIENCY: f64 = {:.3e};",
        efficiency(block_slope)
    );
    println!(
        "const ENCODER_BLOCK_FIXED: Duration = Duration::from_micros({:.0});",
        block_fixed * 1e6
    );
    println!(
        "const CROSS_CACHE_EFFICIENCY: f64 = {:.3e};",
        efficiency(cross_slope)
    );
    println!(
        "const DECODER_PASS_EFFICIENCY: f64 = {:.3e};",
        efficiency(step_slope)
    );
    println!(
        "const DECODER_PASS_FIXED: Duration = Duration::from_micros({:.0});",
        step_fixed * 1e6
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
        fn budget(&self) -> Option<(Duration, f64)> {
            None
        }
        fn enter(&self, _cost: Cost) -> Option<Box<dyn Slot>> {
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
            std::hint::black_box(step(flops(1e9), || std::hint::black_box(i)).unwrap());
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
