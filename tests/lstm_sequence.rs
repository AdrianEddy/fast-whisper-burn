//! `lstm_sequence_fused` must agree with the per-step `lstm_cell_fused` it replaced.
//!
//! The sequence kernel exists to turn `steps` dispatches into one, which is a performance change
//! and must be a no-op numerically. Both kernels do the same f32 arithmetic in the same order, so
//! the tolerance here is tight on purpose: anything larger would be hiding a real difference in
//! how the recurrence is carried (`h` through shared memory, `c` through a register) rather than
//! absorbing float noise.
//!
//! ```text
//! cargo test --release --features vulkan --test lstm_sequence
//! ```

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use fast_whisper_burn::custom_kernels::{lstm_cell_fused, lstm_sequence_fused};

/// The kernel's shared-memory slice is `BLOCK_SIZE` long and the launch uses one thread per
/// hidden unit, so this is the width both kernels are built for — and the one Silero uses.
const D_HIDDEN: usize = 128;

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

/// Deterministic values in a range the gates actually resolve in, with a few deliberate
/// outliers so the `tanh` clamp branches are exercised on both paths rather than only in theory.
fn pseudo_random(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u64).wrapping_mul(6364136223846793005).wrapping_add(seed);
            let unit = ((x >> 33) as f32 / (1u64 << 31) as f32) - 1.0;
            if i % 601 == 0 {
                // Past the kernel's |x| > 10 clamp.
                unit.signum() * 40.0
            } else {
                unit * scale
            }
        })
        .collect()
}

fn tensor2(data: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

struct Inputs {
    hidden: Tensor<2>,
    cell: Tensor<2>,
    gates: Tensor<2>,
    weight: Tensor<2>,
    bias: Tensor<1>,
}

fn inputs(steps: usize, device: &Device) -> Inputs {
    let d_out = 4 * D_HIDDEN;
    Inputs {
        hidden: tensor2(pseudo_random(D_HIDDEN, 11, 0.5), [1, D_HIDDEN], device),
        cell: tensor2(pseudo_random(D_HIDDEN, 22, 0.5), [1, D_HIDDEN], device),
        gates: tensor2(pseudo_random(steps * d_out, 33, 0.4), [steps, d_out], device),
        // `nn::Linear` stores its weight as [d_in, d_out], which is what both kernels index.
        // Scaled down because 128 accumulated products otherwise saturate every gate.
        weight: tensor2(
            pseudo_random(D_HIDDEN * d_out, 44, 0.05),
            [D_HIDDEN, d_out],
            device,
        ),
        bias: Tensor::<1>::from_data(
            TensorData::new(pseudo_random(d_out, 55, 0.2), [d_out]),
            device,
        ),
    }
}

/// The loop the sequence kernel replaced, kept here as the oracle.
fn reference(inputs: &Inputs, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let mut hidden = inputs.hidden.clone();
    let mut cell = inputs.cell.clone();
    let mut all = Vec::with_capacity(steps * D_HIDDEN);

    for step in 0..steps {
        let gates = inputs.gates.clone().slice([step..step + 1]);
        let (new_hidden, new_cell) = lstm_cell_fused(
            hidden,
            cell,
            gates,
            inputs.weight.clone(),
            inputs.bias.clone(),
        );
        hidden = new_hidden;
        cell = new_cell;
        all.extend_from_slice(&to_vec(hidden.clone()));
    }

    (all, to_vec(cell))
}

fn to_vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data().convert::<f32>().into_vec::<f32>().unwrap()
}

fn assert_matches(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5,
            "{label}: index {i} is {g}, per-step kernel says {w}"
        );
    }
}

fn check(steps: usize) {
    let device = device();
    let inputs = inputs(steps, &device);
    let (want_hidden, want_cell) = reference(&inputs, steps);

    let (all_hidden, last_cell) = lstm_sequence_fused(
        inputs.hidden.clone(),
        inputs.cell.clone(),
        inputs.gates.clone(),
        inputs.weight.clone(),
        inputs.bias.clone(),
    );

    assert_eq!(all_hidden.dims(), [steps, D_HIDDEN]);
    assert_eq!(last_cell.dims(), [1, D_HIDDEN]);
    assert_matches("hidden", &to_vec(all_hidden), &want_hidden);
    assert_matches("cell", &to_vec(last_cell), &want_cell);
}

/// A sequence long enough that the state has to survive many in-kernel iterations, and not a
/// round number — an off-by-one in the loop bound or in the packed output offset shows up here.
#[test]
fn sequence_matches_per_step_loop() {
    check(37);
}

/// One step: the loop body runs exactly once and the final cell is written to row 1.
#[test]
fn single_step_matches_per_step_loop() {
    check(1);
}

/// The block size Silero's VAD actually runs — `BURN_VAD_UPLOAD_WINDOWS` windows per upload
/// block, so this is the sequence length the kernel loops over in production.
#[test]
fn full_block_matches_per_step_loop() {
    check(1024);
}
