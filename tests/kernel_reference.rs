//! Every fused kernel the model runs, checked against a plain host-side reference in f64.
//!
//! The kernels read f16 (or f32), accumulate in f32 and write back in the input dtype, so the
//! only error allowed is the final rounding of the output. Inputs are rounded to f16 on the host
//! before upload, so the reference starts from exactly the values the kernel reads; anything
//! beyond output rounding is a real difference in the kernel, not input quantization.
//!
//! Shapes are the ones Whisper large-v3 and Silero actually run: `d_model` 1280, 20 heads of 64,
//! 1500 encoder frames, and a 128-wide LSTM.
//!
//! ```text
//! cargo test --release --test kernel_reference -- --nocapture
//! ```

use burn::tensor::{Device, DeviceKind, FloatDType, Tensor, TensorData};
use fast_whisper_burn::custom_kernels::{
    fused_single_query_attn, layer_norm_f16, lstm_sequence_fused, softmax_f16,
};

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

/// Deterministic values in `[offset - scale, offset + scale)`, already rounded to f16.
fn values(n: usize, seed: u64, offset: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
            let unit = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            half::f16::from_f32(offset + unit * scale).to_f32()
        })
        .collect()
}

fn upload<const D: usize>(
    data: &[f32],
    shape: [usize; D],
    dtype: FloatDType,
    device: &Device,
) -> Tensor<D> {
    Tensor::<D>::from_data(TensorData::new(data.to_vec(), shape), device).cast(dtype)
}

fn download<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().convert::<f32>().try_into_vec::<f32>().unwrap()
}

/// Rounding the f32 result to the output dtype is the only error a kernel is allowed: half an
/// f16 ulp is ~4.9e-4 relative, so 1e-3 leaves room for f32 accumulation and nothing else.
fn assert_close(label: &str, got: &[f32], want: &[f64], dtype: FloatDType) {
    let rel = match dtype {
        FloatDType::F16 => 1e-3,
        _ => 1e-5,
    };
    assert_eq!(got.len(), want.len(), "{label}: length");
    let mut worst = 0.0f64;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (g as f64 - w).abs();
        assert!(
            err <= rel * (1.0 + w.abs()),
            "{label}: index {i} is {g}, reference says {w}"
        );
        worst = worst.max(err);
    }
    println!("{label}: max abs error {worst:.2e} over {} values", got.len());
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

fn layer_norm_reference(x: &[f32], gamma: &[f32], beta: &[f32], d: usize) -> Vec<f64> {
    x.chunks(d)
        .flat_map(|row| {
            let mean = row.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
            let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            row.iter()
                .zip(gamma)
                .zip(beta)
                .map(move |((&v, &g), &b)| (v as f64 - mean) * inv_std * g as f64 + b as f64)
        })
        .collect()
}

fn check_layer_norm(rows: usize, d: usize) {
    let device = device();
    // Off-centre, so a missing mean subtraction can't hide.
    let x = values(rows * d, 1, 2.0, 3.0);
    // In the mixed-precision model only activations are f16; LayerNorm parameters stay f32.
    let gamma = values(d, 2, 1.0, 0.5);
    let beta = values(d, 3, 0.0, 0.5);

    let out = layer_norm_f16(
        upload(&x, [rows, d], FloatDType::F16, &device),
        upload(&gamma, [d], FloatDType::F32, &device),
        upload(&beta, [d], FloatDType::F32, &device),
    );

    assert_eq!(out.dims(), [rows, d]);
    assert_close(
        &format!("layer_norm {rows}x{d}"),
        &download(out),
        &layer_norm_reference(&x, &gamma, &beta, d),
        FloatDType::F16,
    );
}

#[test]
fn layer_norm_large_v3_width() {
    check_layer_norm(7, 1280);
}

#[test]
fn layer_norm_base_width() {
    check_layer_norm(3, 512);
}

/// Narrower than the 128-thread block: most lanes contribute nothing to the reduction.
#[test]
fn layer_norm_narrower_than_block() {
    check_layer_norm(5, 100);
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

fn softmax_reference(x: &[f32], n: usize) -> Vec<f64> {
    x.chunks(n)
        .flat_map(|row| {
            let max = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
            let exps: Vec<f64> = row.iter().map(|&v| (v as f64 - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.into_iter().map(move |e| e / sum)
        })
        .collect()
}

fn check_softmax(shape: [usize; 4]) {
    let device = device();
    let n = shape[3];
    let mut x = values(shape.iter().product(), 7, 0.0, 8.0);
    // Masked positions, filled with the f16 minimum the way `mask_fill` does in the f16 path.
    for v in x.iter_mut().step_by(13) {
        *v = half::f16::MIN.to_f32();
    }

    let out = softmax_f16(upload(&x, shape, FloatDType::F16, &device));

    assert_eq!(out.dims(), shape);
    assert_close(
        &format!("softmax {shape:?}"),
        &download(out),
        &softmax_reference(&x, n),
        FloatDType::F16,
    );
}

/// Encoder self-attention rows: 1500 frames, far wider than the block.
#[test]
fn softmax_encoder_rows() {
    check_softmax([1, 4, 3, 1500]);
}

/// Decoder rows over a short prefix.
#[test]
fn softmax_short_rows() {
    check_softmax([2, 3, 5, 7]);
}

// ---------------------------------------------------------------------------
// Single-query attention
// ---------------------------------------------------------------------------

/// `heads` here is batch * n_heads, flattened: each one is independent.
fn attention_reference(q: &[f32], k: &[f32], v: &[f32], n_kv: usize, d_k: usize) -> Vec<f64> {
    let scale = 1.0 / (d_k as f64).sqrt();
    let heads = q.len() / d_k;
    let mut out = Vec::with_capacity(heads * d_k);
    for h in 0..heads {
        let q = &q[h * d_k..(h + 1) * d_k];
        let scores: Vec<f64> = (0..n_kv)
            .map(|j| {
                let k = &k[(h * n_kv + j) * d_k..][..d_k];
                q.iter().zip(k).map(|(&a, &b)| a as f64 * b as f64).sum::<f64>() * scale
            })
            .collect();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f64 = weights.iter().sum();
        for lane in 0..d_k {
            let acc: f64 = (0..n_kv)
                .map(|j| weights[j] * v[(h * n_kv + j) * d_k + lane] as f64)
                .sum();
            out.push(acc / sum);
        }
    }
    out
}

fn check_attention(batch: usize, n_heads: usize, n_kv: usize, dtype: FloatDType) {
    const D_K: usize = 64;
    let device = device();
    // A larger query spread makes the attention peaked, so outputs aren't all near the mean of V
    // (where a kernel returning zeros would slip under the tolerance).
    let q = values(batch * n_heads * D_K, 4, 0.0, 4.0);
    let k = values(batch * n_heads * n_kv * D_K, 5, 0.0, 1.0);
    let v = values(batch * n_heads * n_kv * D_K, 6, 0.0, 1.0);

    let out = fused_single_query_attn(
        upload(&q, [batch, n_heads, 1, D_K], dtype, &device),
        upload(&k, [batch, n_heads, n_kv, D_K], dtype, &device),
        upload(&v, [batch, n_heads, n_kv, D_K], dtype, &device),
    );

    assert_eq!(out.dims(), [batch, n_heads, 1, D_K]);
    assert_close(
        &format!("attention n_kv={n_kv} {dtype:?}"),
        &download(out),
        &attention_reference(&q, &k, &v, n_kv, D_K),
        dtype,
    );
}

/// The first decoding step: a single key.
#[test]
fn attention_single_key() {
    check_attention(2, 20, 1, FloatDType::F16);
}

/// Decoder self-attention over a short prefix: one stripe.
#[test]
fn attention_self_short() {
    check_attention(2, 20, 7, FloatDType::F16);
}

/// Just past the threshold where the kernel splits keys into parallel stripes.
#[test]
fn attention_first_striped_length() {
    check_attention(2, 20, 257, FloatDType::F16);
}

/// Cross-attention over all 1500 encoder frames, striped.
#[test]
fn attention_cross_f16() {
    check_attention(2, 20, 1500, FloatDType::F16);
}

/// The f32 instantiation used when running without `--f16`.
#[test]
fn attention_cross_f32() {
    check_attention(2, 20, 1500, FloatDType::F32);
}

// ---------------------------------------------------------------------------
// LSTM sequence
// ---------------------------------------------------------------------------

/// A PyTorch LSTM step, gate order (i, f, g, o), weight `[d_hidden, 4 * d_hidden]`.
fn lstm_reference(
    hidden: &[f32],
    cell: &[f32],
    gates: &[f32],
    weight: &[f32],
    bias: &[f32],
) -> (Vec<f64>, Vec<f64>) {
    let dh = hidden.len();
    let d_out = 4 * dh;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    let mut h: Vec<f64> = hidden.iter().map(|&v| v as f64).collect();
    let mut c: Vec<f64> = cell.iter().map(|&v| v as f64).collect();
    let mut all = Vec::with_capacity(gates.len() / 4);

    for step in gates.chunks(d_out) {
        let pre: Vec<f64> = (0..d_out)
            .map(|m| {
                let dot: f64 = (0..dh).map(|j| h[j] * weight[j * d_out + m] as f64).sum();
                dot + bias[m] as f64 + step[m] as f64
            })
            .collect();
        for k in 0..dh {
            let i = sigmoid(pre[k]);
            let f = sigmoid(pre[dh + k]);
            let g = pre[2 * dh + k].tanh();
            let o = sigmoid(pre[3 * dh + k]);
            c[k] = f * c[k] + i * g;
            h[k] = o * c[k].tanh();
        }
        all.extend_from_slice(&h);
    }
    (all, c)
}

#[test]
fn lstm_sequence_matches_reference() {
    const D_HIDDEN: usize = 128;
    const STEPS: usize = 37;
    let d_out = 4 * D_HIDDEN;
    let device = device();
    let hidden = values(D_HIDDEN, 11, 0.0, 0.5);
    let cell = values(D_HIDDEN, 22, 0.0, 0.5);
    let gates = values(STEPS * d_out, 33, 0.0, 0.4);
    // Scaled down because 128 accumulated products otherwise saturate every gate.
    let weight = values(D_HIDDEN * d_out, 44, 0.0, 0.05);
    let bias = values(d_out, 55, 0.0, 0.2);

    let (all_hidden, last_cell) = lstm_sequence_fused(
        upload(&hidden, [1, D_HIDDEN], FloatDType::F32, &device),
        upload(&cell, [1, D_HIDDEN], FloatDType::F32, &device),
        upload(&gates, [STEPS, d_out], FloatDType::F32, &device),
        upload(&weight, [D_HIDDEN, d_out], FloatDType::F32, &device),
        upload(&bias, [d_out], FloatDType::F32, &device),
    );

    let (want_hidden, want_cell) = lstm_reference(&hidden, &cell, &gates, &weight, &bias);
    assert_close("lstm hidden", &download(all_hidden), &want_hidden, FloatDType::F32);
    assert_close("lstm cell", &download(last_cell), &want_cell, FloatDType::F32);
}
