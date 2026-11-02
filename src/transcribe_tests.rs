//! What a transcription reads back from the GPU ([`host_values`]), and the probability language
//! detection reports.
//!
//! ```text
//! cargo test --release --no-default-features --features vulkan --lib transcribe::tests
//! cargo test --release --no-default-features --features vulkan --lib transcribe::tests -- --ignored --nocapture
//! ```
//!
//! The ignored test runs large-v3-turbo on the GPU, read like the pacing tests' models
//! (`FWB_MODELS`, or else this crate's `models/`), on `audio.wav`.

use super::{
    NonFinite, detect_language, detect_language_among_encoded, encode_first_window, host_values,
    transcribe, transcribe_regions_batched,
};
use crate::pacing::tests::{audio, device, load, params};
use crate::token::Language;
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

/// Every weight NaN, as a GPU that breaks leaves NaN in Whisper's logits: each decoding path, and
/// language detection, fails with [`NonFinite`]. They panicked comparing NaN logits before.
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

    let (whisper, bpe) = load("base", &device);
    let whisper = whisper.map(&mut Poison);
    assert_non_finite(
        "language detection",
        detect_language(&whisper, &bpe, &audio, 16_000, true).map(drop),
    );
}

/// After SOT, large-v3-turbo puts about half its mass on `<|transcribe|>`, not on a language:
/// over the whole vocabulary the spoken language had 0.496, over the languages alone 0.9999.
#[test]
#[ignore = "needs a GPU and Whisper weights"]
fn the_language_probability_is_its_share_of_the_languages() {
    let device = device();
    let (whisper, bpe) = load("large-v3-turbo", &device);
    let detected = detect_language(&whisper, &bpe, &audio(), 16_000, true).unwrap();
    assert_eq!(detected.language.as_str(), "en");
    assert!(detected.probability > 0.99, "{}", detected.probability);

    let encoded = encode_first_window(&whisper, &audio(), 16_000, true).unwrap();
    let among = detect_language_among_encoded(
        &whisper,
        &bpe,
        encoded,
        true,
        &[Language::Polish, Language::German],
    )
    .unwrap();
    assert!(
        matches!(among.language, Language::Polish | Language::German),
        "{}",
        among.language.as_str()
    );
    assert!(
        among.probability < 0.01,
        "a language the audio is not still has its share among all: {}",
        among.probability
    );
}
