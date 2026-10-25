//! [`MixedPrecisionAdapter`] against checkpoints that store every weight in f16, as OpenAI's
//! official `.pt` files do.
//!
//! The model reads its precision-critical weights (LayerNorm, the convolutions, cross-attention
//! and the embeddings) as f32 in both modes: the stem's convolutions see the f32 mel, the fused
//! f16 LayerNorm takes its gamma and beta as f32, and cross-attention projects in f32. So the
//! adapter has to hand those over as f32 whatever the source stored them as.
//!
//! ```text
//! cargo test --release --test mixed_precision
//! ```

use burn::tensor::{DType, Device, DeviceKind, Distribution, Int, Tensor, TensorData};
use burn_store::{
    BurnpackStore, FloatCastAdapter, ModuleAdapter, ModuleContext, ModuleSnapshot, StoreTensor,
    bridge,
};
use fast_whisper_burn::MixedPrecisionAdapter;
use fast_whisper_burn::model::{AudioEncoderConfig, TextDecoderConfig, Whisper, WhisperConfig};

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

// ---------------------------------------------------------------------------
// The adapter alone
// ---------------------------------------------------------------------------

/// Paths of the Whisper module, as burn names them, that run in f32 in both modes.
const CRITICAL: [&str; 8] = [
    "encoder.conv1.weight",
    "encoder.conv2.bias",
    "encoder.positional_embedding",
    "encoder.blocks.0.attn_ln.gamma",
    "encoder.ln_post.beta",
    "decoder.token_embedding",
    "decoder.blocks.3.cross_attn.key.weight",
    "decoder.blocks.3.cross_attn_ln.gamma",
];

/// Paths that run in the target precision.
const OTHERS: [&str; 4] = [
    "encoder.blocks.0.attn.query.weight",
    "encoder.blocks.0.mlp.lin1.weight",
    "decoder.blocks.3.attn.value.bias",
    "decoder.blocks.3.mlp.lin2.weight",
];

fn stored(name: &str, dtype: DType) -> StoreTensor {
    let data = match dtype {
        DType::F16 => TensorData::new(vec![half::f16::from_f32(0.75); 6], [2, 3]),
        DType::F32 => TensorData::new(vec![0.75f32; 6], [2, 3]),
        _ => unreachable!(),
    };
    bridge::from_data(data, name.to_string(), None).into()
}

/// What `name`, stored as `source`, loads as when the model runs in `target`.
fn adapted(name: &str, source: DType, target: DType) -> DType {
    let tensor = MixedPrecisionAdapter(target).adapt(stored(name, source), ModuleContext::new(&[]));
    let dtype = tensor.dtype();
    let values = tensor
        .into_data()
        .unwrap()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap();
    assert_eq!(
        values, [0.75; 6],
        "{name}: {source:?} -> {dtype:?} changed the values"
    );
    dtype
}

#[test]
fn critical_weights_load_as_f32_from_any_source() {
    for name in CRITICAL {
        for source in [DType::F16, DType::F32] {
            for target in [DType::F16, DType::F32] {
                assert_eq!(
                    adapted(name, source, target),
                    DType::F32,
                    "{name}: {source:?} into a {target:?} model"
                );
            }
        }
    }
}

#[test]
fn other_weights_load_as_the_target() {
    for name in OTHERS {
        for source in [DType::F16, DType::F32] {
            for target in [DType::F16, DType::F32] {
                assert_eq!(
                    adapted(name, source, target),
                    target,
                    "{name}: {source:?} into a {target:?} model"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// A whole model from an all-f16 checkpoint
// ---------------------------------------------------------------------------

const N_MELS: usize = 80;
const AUDIO_CTX: usize = 8;
const TEXT_CTX: usize = 8;
const VOCAB: usize = 32;
/// One head of 64, the width the fused attention kernel is built for.
const STATE: usize = 64;

fn tiny() -> WhisperConfig {
    WhisperConfig::new(
        AudioEncoderConfig::new(N_MELS, AUDIO_CTX, STATE, 1, 1),
        TextDecoderConfig::new(VOCAB, TEXT_CTX, STATE, 1, 1),
    )
}

/// A random model saved the way an official checkpoint stores it: every float tensor f16.
fn fp16_checkpoint(device: &Device) -> BurnpackStore {
    let mut store =
        BurnpackStore::from_bytes(None).with_to_adapter(FloatCastAdapter::to(DType::F16));
    tiny().init(device).save_into(&mut store).unwrap();
    BurnpackStore::from_bytes(Some(store.get_bytes().unwrap()))
}

fn load(target: DType, device: &Device) -> Whisper {
    let mut store = fp16_checkpoint(device).with_from_adapter(MixedPrecisionAdapter(target));
    let mut whisper = tiny().init(device);
    let result = whisper.load_from(&mut store).unwrap();
    assert!(
        result.errors.is_empty() && result.missing.is_empty(),
        "{result}"
    );
    whisper
}

/// The encoder over random mel, then one cached decoder step: every precision-critical weight
/// gets read. Returns the step's logits.
fn decode_one(whisper: &Whisper, use_f16: bool, device: &Device) -> Vec<f32> {
    let mel = Tensor::<3>::random(
        [1, N_MELS, 2 * AUDIO_CTX],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let token = Tensor::<2, Int>::zeros([1, 1], device);
    let logits = if use_f16 {
        // As `transcribe_regions_batched` runs it: fused decoder weights only in f16.
        let cache = whisper.create_decoder_cache_f16(whisper.forward_encoder_f16(mel).unwrap());
        let fused = whisper.build_fused_decoder_weights();
        whisper
            .forward_decoder_cached_with_cross_attention_fused(token, cache, &fused, true)
            .logits
    } else {
        let cache = whisper.create_decoder_cache(whisper.forward_encoder(mel).unwrap());
        whisper
            .forward_decoder_cached_with_cross_attention(token, cache)
            .logits
    };
    assert_eq!(logits.dims(), [1, 1, VOCAB]);
    logits
        .into_data()
        .convert::<f32>()
        .try_into_vec::<f32>()
        .unwrap()
}

#[test]
fn fp16_checkpoint_runs_in_f32() {
    let device = device();
    let whisper = load(DType::F32, &device);
    for tensor in whisper.collect(None, None, false) {
        assert_eq!(tensor.dtype, DType::F32, "{}", tensor.name);
    }
    assert!(
        decode_one(&whisper, false, &device)
            .iter()
            .all(|v| v.is_finite())
    );
}

#[test]
fn fp16_checkpoint_runs_in_mixed_f16() {
    let device = device();
    let whisper = load(DType::F16, &device);
    assert!(
        decode_one(&whisper, true, &device)
            .iter()
            .all(|v| v.is_finite())
    );
}
