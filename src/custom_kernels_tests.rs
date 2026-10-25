//! Every fused kernel ([`crate::custom_kernels`]) against the plain burn computation it replaces,
//! in f32, on the backend this crate was built for. Run it on each backend you ship: a kernel that
//! is correct on one GPU API can be wrong on another.
//!
//! ```text
//! cargo test --release --no-default-features --features vulkan --lib custom_kernels
//! cargo test --release --no-default-features --features cuda --lib custom_kernels
//! ```
//!
//! CUDA (and HIP) allocate a tensor's rows at a pitch: each row is padded to the next power of
//! two of its bytes, clamped to 16–512. The kernels write their outputs densely, so the widths
//! below include ones whose rows are not already a multiple of that pitch, where a pitched
//! output would scramble every row after the first.

use super::*;
use burn::tensor::{Device, DeviceKind, FloatDType, TensorData};

fn device() -> Device {
    #[cfg(feature = "vulkan")]
    {
        Device::vulkan(DeviceKind::DefaultDevice)
    }
    #[cfg(all(not(feature = "vulkan"), feature = "metal"))]
    {
        Device::metal(DeviceKind::DefaultDevice)
    }
    #[cfg(all(not(feature = "vulkan"), not(feature = "metal"), feature = "wgpu"))]
    {
        Device::wgpu(DeviceKind::DefaultDevice)
    }
    #[cfg(all(
        not(feature = "vulkan"),
        not(feature = "metal"),
        not(feature = "wgpu"),
        feature = "cuda"
    ))]
    {
        Device::cuda(0)
    }
}

/// Deterministic values in `[-scale, scale]`.
fn values<const D: usize>(
    shape: [usize; D],
    seed: u64,
    scale: f32,
    device: &Device,
) -> BurnTensor<D> {
    let mut state = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let count = shape.iter().product();
    let data: Vec<f32> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect();
    BurnTensor::from_data(TensorData::new(data, shape), device)
}

fn host<const D: usize>(t: BurnTensor<D>) -> Vec<f32> {
    t.cast(FloatDType::F32)
        .into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .unwrap()
}

/// Fails if any value is not finite or differs from the reference by more than
/// `tolerance` times the reference's largest magnitude (at least 1).
fn assert_close(what: &str, got: Vec<f32>, want: Vec<f32>, tolerance: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let scale = want.iter().fold(1.0_f32, |m, v| m.max(v.abs()));
    let (worst, at) = got
        .iter()
        .zip(&want)
        .map(|(g, w)| {
            if g.is_finite() {
                (g - w).abs()
            } else {
                f32::INFINITY
            }
        })
        .enumerate()
        .fold(
            (0.0_f32, 0),
            |(m, i), (j, d)| if d > m { (d, j) } else { (m, i) },
        );
    assert!(
        worst <= tolerance * scale,
        "{what}: largest difference {worst} at {at} (got {}, want {}), allowed {}",
        got[at],
        want[at],
        tolerance * scale
    );
}

/// Whisper tiny's width and large-v3's: an f16 row of 384 is 768 bytes, pitched to 1024 on CUDA;
/// one of 1280 is 2560 bytes, already a multiple of 512.
#[test]
fn layer_norm_f16_matches_reference() {
    let device = device();
    for d in [384, 1280] {
        let x = values([2, 7, d], 1, 4.0, &device);
        let gamma = values([d], 2, 1.0, &device);
        let beta = values([d], 3, 0.5, &device);

        let mean = x.clone().mean_dim(2);
        let centred = x.clone() - mean;
        let var = centred.clone().powi_scalar(2).mean_dim(2);
        let want = centred / (var + 1e-5).sqrt() * gamma.clone().unsqueeze::<3>()
            + beta.clone().unsqueeze::<3>();

        let got = layer_norm_f16(x.cast(FloatDType::F16), gamma, beta);
        assert_close(
            &format!("layer_norm_f16 d={d}"),
            host(got),
            host(want),
            1e-2,
        );
    }
}

/// Decoder self-attention over a short prefix and the encoder's 1500 frames: both pitched on CUDA.
#[test]
fn softmax_f16_matches_reference() {
    let device = device();
    for n in [7, 1500] {
        let x = values([1, 20, 1, n], 4, 8.0, &device);
        let want = burn::tensor::activation::softmax(x.clone(), 3);
        let got = softmax_f16(x.cast(FloatDType::F16));
        assert_close(&format!("softmax_f16 n={n}"), host(got), host(want), 1e-2);
    }
}

/// `linear_f16` takes its weight as `[d_out, d_in]`. Rows of 384 outputs are pitched on CUDA.
#[test]
fn linear_f16_matches_reference() {
    let device = device();
    let (d_in, d_out) = (512, 384);
    let x = values([2, 7, d_in], 10, 2.0, &device);
    let weight = values([d_out, d_in], 11, 0.1, &device);
    let bias = values([d_out], 12, 0.5, &device);

    let want = x
        .clone()
        .matmul(weight.clone().transpose().unsqueeze::<3>())
        + bias.clone().unsqueeze::<3>();
    let got = linear_f16(x.cast(FloatDType::F16), weight, Some(bias));
    assert_close("linear_f16", host(got), host(want), 1e-2);
}

fn attention_reference(q: BurnTensor<4>, k: BurnTensor<4>, v: BurnTensor<4>) -> BurnTensor<4> {
    let d_k = q.dims()[3];
    let scores = q.matmul(k.transpose()) * (d_k as f32).sqrt().recip();
    burn::tensor::activation::softmax(scores, 3).matmul(v)
}

#[test]
fn fused_single_query_attn_matches_reference() {
    let device = device();
    // Self-attention (a few cached tokens, one stripe) and cross-attention (1500 encoder
    // frames, several stripes), as the decoder uses them.
    for n_kv in [7, 1500] {
        let q = values([1, 20, 1, 64], 5, 2.0, &device);
        let k = values([1, 20, n_kv, 64], 6, 2.0, &device);
        let v = values([1, 20, n_kv, 64], 7, 2.0, &device);
        let want = host(attention_reference(q.clone(), k.clone(), v.clone()));

        let got = fused_single_query_attn(q.clone(), k.clone(), v.clone());
        assert_close(
            &format!("fused attention f32 n_kv={n_kv}"),
            host(got),
            want.clone(),
            1e-3,
        );

        let got = fused_single_query_attn(
            q.cast(FloatDType::F16),
            k.cast(FloatDType::F16),
            v.cast(FloatDType::F16),
        );
        assert_close(
            &format!("fused attention f16 n_kv={n_kv}"),
            host(got),
            want,
            2e-2,
        );
    }
}
