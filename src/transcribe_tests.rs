//! What a transcription reads back from the GPU ([`host_values`]).
//!
//! ```text
//! cargo test --release --no-default-features --features vulkan --lib transcribe::tests
//! cargo test --release --no-default-features --features vulkan --lib transcribe::tests -- --ignored --nocapture
//! ```
//!
//! The ignored test runs base.en on the GPU, read like the pacing tests' models
//! (`FWB_MODELS`, or else this crate's `models/`), on `audio.wav`.

use super::{NonFinite, host_values, transcribe, transcribe_regions_batched};
use crate::pacing::tests::{audio, device, load, params};
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Device, Tensor};

fn values(values: &[f32]) -> Tensor<1> {
    Tensor::from_floats(values, &Device::flex())
}

#[test]
fn a_nan_read_back_is_refused() {
    let error = host_values(values(&[0.5, f32::NAN, 1.0]), "logits").unwrap_err();
    assert_eq!(
        error.downcast_ref::<NonFinite>(),
        Some(&NonFinite("logits"))
    );
    assert_eq!(
        error.to_string(),
        "Whisper's logits came back from the GPU as NaN"
    );
}

/// A suppressed token's logit is `-inf` (`mask_fill`): no NaN, and every comparison orders it.
#[test]
fn masked_logits_are_read_back() {
    let read = host_values(values(&[0.5, f32::NEG_INFINITY, 1.0]), "logits").unwrap();
    assert_eq!(read, [0.5, f32::NEG_INFINITY, 1.0]);
}

/// Every weight NaN, as a GPU that breaks leaves NaN in Whisper's logits: each decoding path fails
/// with [`NonFinite`]. They panicked comparing NaN logits before.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn nan_logits_fail_every_decoding_path() {
    struct Poison;
    impl ModuleMapper for Poison {
        fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
            param.map(|tensor| tensor.mul_scalar(f32::NAN))
        }
    }
    let device = device();
    let audio = audio();
    let no_progress = None::<fn(usize, usize) -> bool>;
    let assert_non_finite = |what: &str, error: crate::token::Result<()>| {
        let error = error.expect_err(what);
        assert!(error.is::<NonFinite>(), "{what}: {error}");
    };

    let (whisper, bpe) = load("base.en", &device);
    let whisper = whisper.map(&mut Poison);
    for beam in [1, 3] {
        let params = params(beam);
        assert_non_finite(
            &format!("one region, beam {beam}"),
            transcribe(&whisper, &bpe, &audio, 16_000, &params, no_progress).map(drop),
        );
        let regions = [&audio[..16_000 * 3], &audio[..]];
        assert_non_finite(
            &format!("batched regions, beam {beam}"),
            transcribe_regions_batched(
                &whisper,
                &bpe,
                &regions,
                16_000,
                &params,
                None,
                no_progress,
            )
            .map(drop),
        );
    }
}
