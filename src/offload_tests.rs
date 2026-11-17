//! The encoder's MLPs shared with an engine beside the GPU ([`crate::offload`]): which rows the
//! engine takes, that a shared forward gives the GPU's output however its windows fall into
//! lanes, what becomes of an engine that fails, computes something else or is no faster, and that
//! every piece of the GPU's work is a step with nothing read back in it.
//!
//! ```text
//! cargo test --release --no-default-features --features vulkan --lib offload
//! cargo test --release --no-default-features --features vulkan --lib offload -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! The engine here is the CPU, running a small random model's MLPs in f32 from the weights the
//! model hands out ([`Whisper::encoder_mlp_weights`]), as a host's engine would. The ignored test
//! transcribes `audio.wav` with base.en, read like the pacing tests' models, its engine the
//! model's own MLPs on the GPU.

use super::*;
use crate::model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};
use crate::pacing::Cost;
use crate::pacing::tests::{Event, Log, Recorder, audio, device, load, params, steps, turn};
use crate::transcribe::transcribe_regions_batched;
use burn::module::Module;
use burn::tensor::Distribution;
use burn_store::{BurnpackStore, FloatCastAdapter, ModuleSnapshot};
use std::sync::Mutex;

const N_MELS: usize = 80;
/// The positions of one of the small model's windows.
const CTX: usize = 24;
const STATE: usize = 64;
/// Blocks enough for the rates and a trial.
const LAYERS: usize = 6;
/// The rows a call of the small model's engine takes: a third of a window's.
const CHUNK: usize = 8;

/// An engine far faster than the GPU: every row is its.
const EAGER: Rates = Rates {
    attend: 1.0,
    mlp: 1.0,
    engine: 1e-3,
};
/// An engine half as fast as the GPU's MLP: some of the rows are its, the rest the GPU's.
const SLOW: Rates = Rates {
    attend: 1.0,
    mlp: 0.1,
    engine: 0.2,
};

/// A small random model in mixed precision, as Diem loads Whisper's.
fn small(device: &Device) -> Whisper {
    let config = WhisperConfig::new(
        AudioEncoderConfig::new(N_MELS, CTX, STATE, 1, LAYERS),
        TextDecoderConfig::new(32, 8, STATE, 1, 1),
    );
    let mut store =
        BurnpackStore::from_bytes(None).with_to_adapter(FloatCastAdapter::to(DType::F16));
    config.init(device).save_into(&mut store).unwrap();
    let mut store = BurnpackStore::from_bytes(Some(store.get_bytes().unwrap()))
        .with_from_adapter(crate::MixedPrecisionAdapter(DType::F16));
    let mut whisper = config.init(device);
    whisper.load_from(&mut store).unwrap();
    whisper
}

/// `windows` windows of noise for the small model.
fn mel(windows: usize, device: &Device) -> Tensor<3> {
    Tensor::random(
        [windows, N_MELS, 2 * CTX],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn encoded(whisper: &Whisper, mel: &Tensor<3>) -> Vec<f32> {
    whisper
        .forward_encoder_f16(mel.clone())
        .unwrap()
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

/// How far `values` are from `base`, relative to its size.
fn apart(values: &[f32], base: &[f32]) -> f64 {
    assert_eq!(values.len(), base.len());
    let sum = |f: &dyn Fn(f64, f64) -> f64| {
        values
            .iter()
            .zip(base)
            .map(|(a, b)| f(f64::from(*a), f64::from(*b)))
            .sum::<f64>()
    };
    (sum(&|a, b| (a - b).powi(2)) / sum(&|_, b| b * b)).sqrt()
}

/// What float16 leaves of the small model's output: a row that went to the wrong place is off by
/// its all.
const CLOSE: f64 = 2e-2;

/// erf by Abramowitz and Stegun's 7.1.26: within 1.5e-7.
fn erf(x: f32) -> f32 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_6
            + t * (-0.284_496_73 + t * (1.421_413_8 + t * (-1.453_152_1 + t * 1.061_405_4))));
    (1.0 - poly * (-x * x).exp()).copysign(x)
}

/// The small model's MLPs on the CPU, in f32.
struct Cpu {
    mlps: Vec<MlpWeights>,
    /// The layer and the rows of every call.
    calls: Mutex<Vec<(usize, usize)>>,
    /// The call, counting from 0, from which every call fails.
    fails_from: Option<usize>,
    /// Answers zeros: an engine built for other weights.
    wrong: bool,
    /// The calls, from the first, that take a twentieth of a second each: an engine busy with
    /// someone else's work.
    busy_for: usize,
    /// The call, counting from 0, answered with an infinity among its values.
    overflows_at: Option<usize>,
    /// The installed gate's log: no step may be open while the engine runs.
    log: Option<Log>,
}

impl Cpu {
    fn of(whisper: &Whisper) -> Self {
        Self {
            mlps: (0..LAYERS)
                .map(|layer| whisper.encoder_mlp_weights(layer))
                .collect(),
            calls: Mutex::default(),
            fails_from: None,
            wrong: false,
            busy_for: 0,
            overflows_at: None,
            log: None,
        }
    }

    /// The calls so far, sorted, and none left.
    fn calls(&self) -> Vec<(usize, usize)> {
        let mut calls = std::mem::take(&mut *self.calls.lock().unwrap());
        calls.sort_unstable();
        calls
    }
}

impl MlpEngine for Cpu {
    fn chunk(&self) -> usize {
        CHUNK
    }

    fn run(&self, layer: usize, input: &[f16]) -> Result<Vec<f16>, String> {
        let call = {
            let mut calls = self.calls.lock().unwrap();
            calls.push((layer, input.len() / STATE));
            calls.len() - 1
        };
        if let Some(log) = &self.log {
            let log = log.lock().unwrap();
            assert!(
                matches!(log.last(), Some((_, Event::Queued))),
                "the engine ran inside a step: {:?}",
                log.last()
            );
        }
        if self.fails_from.is_some_and(|from| call >= from) {
            return Err("refused".to_string());
        }
        if call < self.busy_for {
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        if self.wrong {
            return Ok(vec![f16::ZERO; input.len()]);
        }
        let mlp = &self.mlps[layer];
        let (width, inter) = (STATE, 4 * STATE);
        let mut output = Vec::with_capacity(input.len());
        for row in input.chunks_exact(width) {
            let hidden: Vec<f32> = (0..inter)
                .map(|j| {
                    let up = &mlp.up[j * width..][..width];
                    let sum = row
                        .iter()
                        .zip(up)
                        .map(|(x, w)| x.to_f32() * w.to_f32())
                        .sum::<f32>()
                        + mlp.up_bias[j].to_f32();
                    0.5 * sum * (1.0 + erf(sum * std::f32::consts::FRAC_1_SQRT_2))
                })
                .collect();
            output.extend((0..width).map(|o| {
                let down = &mlp.down[o * inter..][..inter];
                let sum = hidden
                    .iter()
                    .zip(down)
                    .map(|(h, w)| h * w.to_f32())
                    .sum::<f32>()
                    + mlp.down_bias[o].to_f32();
                f16::from_f32(sum)
            }));
        }
        if self.overflows_at == Some(call) {
            output[STATE + 3] = f16::INFINITY;
        }
        Ok(output)
    }
}

/// The lanes a forward of `windows` windows goes in at `rates`, as the windows and the engine's
/// rows of each.
fn lanes(rates: &Rates, windows: usize) -> Vec<(usize, usize)> {
    let lane = |windows: usize, paired: bool| {
        (windows, rates.share(windows, windows * CTX, CHUNK, paired))
    };
    match windows {
        1 => vec![lane(1, false)],
        _ => vec![lane(windows.div_ceil(2), true), lane(windows / 2, true)],
    }
}

/// `whisper` sharing its MLPs with `engine`: at `rates`, as if a trial had found that faster, or
/// by what its first forward measures.
fn sharing(whisper: &Whisper, engine: Arc<Cpu>, rates: Option<Rates>) -> (Whisper, Arc<Offload>) {
    with(
        whisper,
        match rates {
            Some(rates) => Offload::measured(engine, rates),
            None => Offload::new(engine),
        },
    )
}

/// `whisper` with `offload`.
fn with(whisper: &Whisper, offload: Offload) -> (Whisper, Arc<Offload>) {
    let offload = Arc::new(offload);
    let mut shared = whisper.clone();
    shared.set_encoder_offload(Some(offload.clone()));
    (shared, offload)
}

/// The blocks' MLP inputs handed out for a host to check its engine on are those of the forward:
/// an engine fed them a block at a time answers what the GPU's own MLPs do.
#[test]
fn each_blocks_mlp_input_is_handed_out() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    let engine = Cpu::of(&whisper);
    let inputs = whisper.encoder_mlp_inputs(mel(2, &device));
    assert_eq!(inputs.len(), LAYERS);
    for (layer, input) in inputs.into_iter().enumerate() {
        assert_eq!(input.dims(), [2, CTX, STATE]);
        assert_eq!(input.dtype(), DType::F16);
        let on_gpu = whisper.encoder_mlp(layer, input.clone());
        let rows = input.into_data().try_into_vec::<f16>().unwrap();
        let on_engine: Vec<f32> = engine
            .run(layer, &rows)
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        let on_gpu = on_gpu
            .into_data()
            .convert::<f32>()
            .try_into_vec::<f32>()
            .unwrap();
        let off = apart(&on_engine, &on_gpu);
        assert!(off < CLOSE, "block {layer}: {off} off");
    }
}

/// The engine takes whole calls of a lane's rows, as many as leave the two engines finishing
/// together — beside the other lane's attention half, where there is one — and none where that
/// gains too little.
#[test]
fn the_engine_takes_the_rows_that_keep_both_busy() {
    let rows = 1500;
    // An M4, about: the attention half 12 ms, the MLP 15 ms on the GPU and 7.5 on the engine.
    let m4 = Rates {
        attend: 12e-3,
        mlp: 10e-6,
        engine: 5e-6,
    };
    assert_eq!(
        m4.share(1, rows, 500, true),
        1500,
        "beside an attention half: all"
    );
    assert_eq!(m4.share(1, rows, 500, false), 1000, "alone: two thirds");
    assert_eq!(m4.share(2, 2 * rows, 500, true), 3000);
    assert_eq!(m4.share(1, rows, 750, false), 750, "whole calls");
    assert_eq!(
        m4.share(1, rows, 2000, false),
        0,
        "a call larger than the lane"
    );

    // An engine a tenth as fast as the GPU gains nothing, alone or beside an attention half.
    let slow = Rates {
        engine: 100e-6,
        ..m4
    };
    assert_eq!(slow.share(1, rows, 500, false), 0);
    assert_eq!(slow.share(1, rows, 500, true), 0);
    // A GPU that takes a twentieth of the engine's time: the engine's call would be the wait.
    let fast = Rates {
        attend: 0.5e-3,
        mlp: 0.25e-6,
        engine: 5e-6,
    };
    assert_eq!(fast.share(1, rows, 500, true), 0);
    assert_eq!(fast.share(1, rows, 500, false), 0);
    // Less than a fifth gained is not worth two engines at work.
    let even = Rates {
        attend: 12e-3,
        mlp: 1e-6,
        engine: 1e-6,
    };
    assert_eq!(even.share(1, rows, 500, false), 0, "13.5 ms to 13");

    assert_eq!(lanes(&EAGER, 1), [(1, 24)]);
    assert_eq!(lanes(&EAGER, 5), [(3, 72), (2, 48)]);
    assert_eq!(lanes(&SLOW, 1), [(1, 8)]);
    assert_eq!(lanes(&SLOW, 3), [(2, 24), (1, 8)]);
}

/// However a forward's windows fall into lanes and whichever rows the engine takes, the encoder's
/// output is the GPU's, and the engine was handed each lane's share of each block's rows.
#[test]
fn a_shared_forward_gives_the_gpus_output() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    for rates in [EAGER, SLOW] {
        let engine = Arc::new(Cpu::of(&whisper));
        let (shared, _) = sharing(&whisper, engine.clone(), Some(rates));
        for windows in [1, 2, 3, 5] {
            let mel = mel(windows, &device);
            let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
            assert!(off < CLOSE, "{windows} windows at {rates:?}: {off} off");
            let mut expected: Vec<(usize, usize)> = (0..LAYERS)
                .flat_map(|layer| {
                    lanes(&rates, windows)
                        .into_iter()
                        .map(move |(_, share)| (layer, share))
                })
                .collect();
            expected.sort_unstable();
            assert_eq!(engine.calls(), expected, "{windows} windows at {rates:?}");
        }
    }
}

/// A model's first forward times its first blocks' halves on both engines, and what follows —
/// trials, sharing, or neither, as this machine's GPU and the CPU that stands for an engine here
/// compare — leaves the output the GPU's.
#[test]
fn the_first_forward_times_the_halves() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    let engine = Arc::new(Cpu::of(&whisper));
    let (shared, offload) = sharing(&whisper, engine.clone(), None);
    assert_eq!(offload.rates(), None);
    for (forward, windows) in [2, 1, 3, 2, 2, 2].into_iter().enumerate() {
        let mel = mel(windows, &device);
        let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
        assert!(off < CLOSE, "{windows} windows: {off} off");
        let calls = engine.calls();
        if forward == 0 {
            // Each of the two blocks timed ran a window's rows twice: once to wake the engine.
            let timing: Vec<_> = calls.into_iter().filter(|&(layer, _)| layer < 2).collect();
            assert_eq!(timing, [(0, CTX), (0, CTX), (1, CTX), (1, CTX)]);
        }
        // Kept where by them there is something to gain.
        if let Some(rates) = offload.rates() {
            assert!(
                rates.attend > 0.0 && rates.mlp > 0.0 && rates.engine > 0.0,
                "{rates:?}"
            );
        }
    }
    assert!(offload.working());
}

/// An engine too slow to gain anything when it is first timed is timed again: after two
/// forwards, which are the GPU's alone and ask nothing of it. The outputs are the GPU's
/// throughout.
#[test]
fn an_engine_found_busy_is_timed_again_later() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    // Busy for the four calls of the first timing: each of two blocks' rows, twice.
    let engine = Arc::new(Cpu {
        busy_for: 4,
        ..Cpu::of(&whisper)
    });
    let (shared, offload) = sharing(&whisper, engine.clone(), None);
    let mel = mel(2, &device);
    let alone = encoded(&whisper, &mel);
    let timing = [(0, CTX), (0, CTX), (1, CTX), (1, CTX)];
    let forward = |what: &str| {
        let off = apart(&encoded(&shared, &mel), &alone);
        assert!(off < CLOSE, "{what}: {off} off");
        engine.calls()
    };
    assert_eq!(forward("the timing in vain"), timing);
    assert_eq!(offload.rates(), None);
    assert_eq!(shared.encoder_lanes(), 1);
    assert_eq!(forward("the first forward let by"), []);
    assert_eq!(forward("the second"), []);
    let calls = forward("the second timing");
    let again: Vec<_> = calls.into_iter().filter(|&(layer, _)| layer < 2).collect();
    assert_eq!(again, timing);
    assert!(offload.working());
}

/// A trial is three forwards that run two blocks the GPU's way and two shared. Sharing it finds
/// slower than the GPU alone is let be: those forwards' other blocks are the GPU's, the forwards
/// after them ask nothing of the engine and are not paired, and the trial is run again on the
/// eighth. The outputs are the GPU's throughout.
#[test]
fn sharing_found_slower_is_let_be_and_tried_again() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    for windows in [2, 1] {
        // Every call takes a twentieth of a second, far longer than the small model's blocks.
        let engine = Arc::new(Cpu {
            busy_for: usize::MAX,
            ..Cpu::of(&whisper)
        });
        let (shared, offload) = with(&whisper, Offload::rated(engine.clone(), EAGER));
        assert_eq!(shared.encoder_lanes(), 2, "by the rates, until a trial");
        let mel = mel(windows, &device);
        let alone = encoded(&whisper, &mel);
        let forward = |what: &str| {
            let off = apart(&encoded(&shared, &mel), &alone);
            assert!(off < CLOSE, "{windows} windows, {what}: {off} off");
            engine.calls()
        };
        // Two blocks the GPU's way, then two shared, a call a lane each.
        let mut run: Vec<(usize, usize)> = (2..4)
            .flat_map(|layer| {
                lanes(&EAGER, windows)
                    .into_iter()
                    .map(move |(_, share)| (layer, share))
            })
            .collect();
        run.sort_unstable();
        for _ in 0..3 {
            assert_eq!(forward("a run of the trial"), run);
        }
        assert!(offload.working());
        if windows == 2 {
            assert_eq!(shared.encoder_lanes(), 1);
        }
        for _ in 0..7 {
            assert_eq!(forward("let by"), []);
        }
        for _ in 0..3 {
            assert_eq!(forward("a run of the next trial"), run);
        }
        // Twice as many forwards before the third, the second having found the same.
        for _ in 0..15 {
            assert_eq!(forward("let by"), []);
        }
        assert_eq!(forward("a run of the third trial"), run);
    }
}

/// An engine that fails in the middle of a forward has its rows run on the GPU, there and from
/// then on: the output is the GPU's, and the engine is not asked again.
#[test]
fn an_engine_that_fails_hands_its_rows_back() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    for (windows, fails_from) in [(1, 0), (1, 2), (3, 3), (4, 7)] {
        let engine = Arc::new(Cpu {
            fails_from: Some(fails_from),
            ..Cpu::of(&whisper)
        });
        let (shared, offload) = sharing(&whisper, engine.clone(), Some(EAGER));
        assert_eq!(shared.encoder_lanes(), 2);
        let mel = mel(windows, &device);
        let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
        assert!(
            off < CLOSE,
            "{windows} windows, failing from call {fails_from}: {off} off"
        );
        assert!(!offload.working());
        assert_eq!(
            engine.calls().len(),
            fails_from + 1,
            "asked once more, then never"
        );
        assert_eq!(shared.encoder_lanes(), 1);
        let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
        assert!(off < CLOSE, "the forward after: {off} off");
        assert_eq!(engine.calls(), [], "the forward after is the GPU's alone");
    }
}

/// An engine that answers with a value that is not finite has that call's rows run on the GPU:
/// the output is the GPU's, and the engine goes on being asked.
#[test]
fn rows_the_engine_runs_out_of_float16_on_are_the_gpus() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    for windows in [1, 4] {
        let engine = Arc::new(Cpu {
            overflows_at: Some(2),
            ..Cpu::of(&whisper)
        });
        let (shared, offload) = sharing(&whisper, engine.clone(), Some(EAGER));
        let mel = mel(windows, &device);
        let got = encoded(&shared, &mel);
        assert!(got.iter().all(|value| value.is_finite()));
        let off = apart(&got, &encoded(&whisper, &mel));
        assert!(off < CLOSE, "{windows} windows: {off} off");
        assert!(offload.working());
        let calls = LAYERS * lanes(&EAGER, windows).len();
        assert_eq!(
            engine.calls().len(),
            calls,
            "every block's rows still went to the engine"
        );
    }
}

/// An engine whose rows are not the GPU's — one built from other weights — is found out by the
/// first block measured, and never used: the output is the GPU's.
#[test]
fn an_engine_of_other_weights_is_left_alone() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    let engine = Arc::new(Cpu {
        wrong: true,
        ..Cpu::of(&whisper)
    });
    let (shared, offload) = sharing(&whisper, engine.clone(), None);
    let mel = mel(2, &device);
    let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
    assert!(off < CLOSE, "{off} off");
    assert!(!offload.working());
    assert_eq!(offload.rates(), None);
    assert_eq!(engine.calls(), [(0, CTX), (0, CTX)]);
    assert_eq!(shared.encoder_lanes(), 1);
}

/// Two lanes where the engine gains from them, as measured; one before that is known, without an
/// engine, and after one failed.
#[test]
fn a_forward_is_two_lanes_where_two_gain() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    assert_eq!(whisper.encoder_lanes(), 1);
    let engine = || Arc::new(Cpu::of(&whisper));
    assert_eq!(sharing(&whisper, engine(), None).0.encoder_lanes(), 1);
    assert_eq!(
        sharing(&whisper, engine(), Some(EAGER)).0.encoder_lanes(),
        2
    );
    assert_eq!(sharing(&whisper, engine(), Some(SLOW)).0.encoder_lanes(), 2);
    let useless = Rates {
        attend: 1e-3,
        mlp: 1e-6,
        engine: 1.0,
    };
    let (shared, _) = sharing(&whisper, engine(), Some(useless));
    assert_eq!(shared.encoder_lanes(), 1);
    // And its forwards are the GPU's: nothing is handed to an engine that gains nothing.
    let engine = engine();
    let (shared, _) = sharing(&whisper, engine.clone(), Some(useless));
    for windows in [1, 3] {
        let mel = mel(windows, &device);
        let off = apart(&encoded(&shared, &mel), &encoded(&whisper, &mel));
        assert!(off < CLOSE, "{off} off");
    }
    assert_eq!(engine.calls(), []);
}

/// A shared forward's GPU work is steps: the stem, then for each lane and block its attention
/// half and, where the GPU keeps some of the MLP's rows, those. The engine runs between steps,
/// never in one, and so does every read of its rows.
#[test]
fn a_shared_forward_is_steps_with_the_engine_between_them() {
    let _turn = turn();
    let device = device();
    let whisper = small(&device);
    let shape = whisper.shape();
    let by_flops = |mut costs: Vec<Cost>| {
        costs.sort_unstable_by(|a, b| a.flops.total_cmp(&b.flops));
        costs
    };
    for (rates, windows) in [(EAGER, 1), (EAGER, 4), (SLOW, 1), (SLOW, 3)] {
        let log = Recorder::installed(None, None);
        let engine = Arc::new(Cpu {
            log: Some(log.clone()),
            ..Cpu::of(&whisper)
        });
        let (shared, _) = sharing(&whisper, engine, Some(rates));
        encoded(&shared, &mel(windows, &device));
        let mut expected = vec![shape.encoder_stem(windows, CTX)];
        for (lane, share) in lanes(&rates, windows) {
            for _ in 0..LAYERS {
                expected.push(shape.encoder_attention(lane, CTX));
                if share < lane * CTX {
                    expected.push(shape.encoder_mlp(lane * CTX - share));
                }
            }
        }
        assert_eq!(
            by_flops(steps(&log)),
            by_flops(expected),
            "{windows} windows at {rates:?}"
        );
    }
    // The halves' costs are the block's.
    let (attention, mlp, block) = (
        shape.encoder_attention(3, CTX),
        shape.encoder_mlp(3 * CTX),
        shape.encoder_block(3, CTX),
    );
    assert!((attention.flops + mlp.flops - block.flops).abs() < 1e-9 * block.flops);
    assert_eq!(attention.fixed + mlp.fixed, block.fixed);
}

/// The share of the words of `texts` that `others` have too, counted without case or
/// punctuation: two routes to the same sums can differ in a rounding — a kernel is tuned on a
/// shape's first runs, and a tuned kernel may round differently — and a letter's case or a word
/// at a region's edge with it.
fn alike(texts: &[String], others: &[String]) -> f64 {
    let words = |texts: &[String]| -> Vec<String> {
        let text = texts.join(" ").to_lowercase();
        let mut words: Vec<String> = text
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .map(str::to_string)
            .collect();
        words.sort_unstable();
        words
    };
    let (ours, mut theirs) = (words(texts), words(others));
    let total = ours.len().max(theirs.len());
    let shared = ours
        .iter()
        .filter(|word| {
            let at = theirs.iter().position(|other| other == *word);
            at.map(|at| theirs.swap_remove(at)).is_some()
        })
        .count();
    shared as f64 / total as f64
}

/// The model's own MLPs on the GPU, as an engine: its rows are the GPU's own.
struct OwnMlps {
    whisper: Whisper,
    width: usize,
}

impl MlpEngine for OwnMlps {
    fn chunk(&self) -> usize {
        500
    }

    fn run(&self, layer: usize, input: &[f16]) -> Result<Vec<f16>, String> {
        let device = self.whisper.devices()[0].clone();
        let rows = Tensor::<3>::from_data(
            TensorData::new(input.to_vec(), [1, input.len() / self.width, self.width]),
            (&device, DType::F16),
        );
        let output = self.whisper.encoder_mlp(layer, rows).into_data();
        output
            .try_into_vec::<f16>()
            .map_err(|error| format!("{error:?}"))
    }
}

/// Where a chunk of a batched transcription is one window, as a pacer leaves it on a slow GPU,
/// and the encoder gains from two lanes, a chunk is encoded with the window after it: half the
/// forwards, and the transcripts of the GPU alone, give or take a rounding ([`alike`]). Chunks of
/// several windows stay as they are.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn two_chunks_are_encoded_at_once() {
    let _turn = turn();
    let device = device();
    let (mut whisper, bpe) = load("base.en", &device);
    let audio = audio();
    let s = 16_000;
    let regions: Vec<&[f32]> = vec![
        &audio[..3 * s],
        &audio[2 * s..5 * s],
        &audio[4 * s..],
        &audio[..],
        &audio[s..6 * s],
    ];
    let transcribe_in = |whisper: &Whisper, chunk: usize| -> Vec<String> {
        let results = transcribe_regions_batched(
            whisper,
            &bpe,
            &regions,
            s,
            &params(1),
            Some(chunk),
            None::<fn(usize, usize) -> bool>,
        )
        .unwrap();
        results.into_iter().map(|result| result.text).collect()
    };
    let transcribe = |whisper: &Whisper| transcribe_in(whisper, 1);
    let shape = whisper.shape();
    let forwards = |log: &Log, windows: usize| {
        let stem = shape.encoder_stem(windows, 1500);
        steps(log).iter().filter(|step| **step == stem).count()
    };

    // Twice: a kernel is tuned on its shape's first run, and a tuned kernel may round differently.
    transcribe(&whisper);
    let log = Recorder::installed(None, None);
    let alone = transcribe(&whisper);
    assert_eq!((forwards(&log, 1), forwards(&log, 2)), (5, 0));
    assert!(alone.iter().all(|text| !text.is_empty()), "{alone:?}");

    let engine = Arc::new(OwnMlps {
        width: whisper.encoder_mlp_weights(0).down_bias.len(),
        whisper: whisper.clone(),
    });
    whisper.set_encoder_offload(Some(Arc::new(Offload::measured(engine, EAGER))));
    transcribe(&whisper);
    let log = Recorder::installed(None, None);
    let paired = transcribe(&whisper);
    assert!(alike(&paired, &alone) > 0.9, "{paired:?}\n{alone:?}");
    assert_eq!((forwards(&log, 1), forwards(&log, 2)), (1, 2));

    transcribe_in(&whisper, 2);
    let log = Recorder::installed(None, None);
    let in_twos = transcribe_in(&whisper, 2);
    assert!(alike(&in_twos, &alone) > 0.9, "{in_twos:?}\n{alone:?}");
    assert_eq!(
        (forwards(&log, 1), forwards(&log, 2), forwards(&log, 4)),
        (1, 2, 0),
        "chunks of two windows are not paired into four"
    );
}
