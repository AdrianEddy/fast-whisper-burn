//! Fused GPU kernels for mixed f16/f32 precision.
//!
//! Contains CubeCL kernels for:
//! - LayerNorm: reads f16, computes in f32, writes f16
//! - Softmax: reads f16, computes in f32, writes f16
//! - Linear (matvec): reads f16 input + f32 weights, computes in f32, writes f16
//! - Single-query attention: fused Q@K^T·scale→softmax→@V for seq_len=1 decoding

use burn::nn;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor as BurnTensor, TensorPrimitive};
use burn_backend::DType;
use burn_backend::TensorMetadata;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device_dtype;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime, FloatElement, IntElement};
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim};

const BLOCK_SIZE: u32 = 128;
const ATTN_D_K_MAX: u32 = 128; // Max d_k for shared memory in fused attention kernel
const ATTN_N_STRIPES_MAX: u32 = 4; // Max number of parallel stripes for attention
const ATTN_SMEM_PARTIALS: u32 = ATTN_N_STRIPES_MAX * ATTN_D_K_MAX; // partial_out shared mem

// ===========================================================================
// CubeCL Kernels
// ===========================================================================

/// Fused LayerNorm: f16 input -> f32 compute -> f16 output.
/// One cube per row (leading dims), BLOCK_SIZE threads for parallel reduction.
#[cube(launch)]
fn layer_norm_f16_kernel<FIn: Float, FComp: Float>(
    input: &Tensor<FIn>,
    gamma: &Tensor<FComp>,
    beta: &Tensor<FComp>,
    output: &mut Tensor<FIn>,
    d_model: u32,
) {
    let epsilon = FComp::new(1e-5);
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;
    let block_size = CUBE_DIM_X;
    let row_offset = (row * d_model) as usize;
    let tid_idx = tid as usize;

    let mut shared = SharedMemory::<FComp>::new(BLOCK_SIZE as usize);

    // --- Pass 1: compute mean via parallel reduction ---
    let mut local_sum = FComp::new(0.0);
    let mut i = tid;
    while i < d_model {
        local_sum += FComp::cast_from(input[row_offset + i as usize]);
        i += block_size;
    }
    shared[tid_idx] = local_sum;
    sync_cube();

    let mut stride = block_size / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let rhs = shared[(tid + stride) as usize];
            shared[tid_idx] += rhs;
        }
        sync_cube();
        stride = stride / 2u32;
    }

    let d_model_f = FComp::cast_from(d_model);
    let mean = shared[0] / d_model_f;
    sync_cube();

    // --- Pass 2: compute variance via parallel reduction ---
    let mut local_var = FComp::new(0.0);
    i = tid;
    while i < d_model {
        let val = FComp::cast_from(input[row_offset + i as usize]) - mean;
        local_var += val * val;
        i += block_size;
    }
    shared[tid_idx] = local_var;
    sync_cube();

    stride = block_size / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let rhs = shared[(tid + stride) as usize];
            shared[tid_idx] += rhs;
        }
        sync_cube();
        stride = stride / 2u32;
    }

    let variance = shared[0] / d_model_f;
    let inv_std = FComp::new(1.0) / FComp::sqrt(variance + epsilon);
    sync_cube();

    // --- Pass 3: normalize, scale, bias, and cast back ---
    i = tid;
    while i < d_model {
        let idx = row_offset + i as usize;
        let val = FComp::cast_from(input[idx]);
        let normalized = (val - mean) * inv_std;
        let result = normalized * gamma[i as usize] + beta[i as usize];
        output[idx] = FIn::cast_from(result);
        i += block_size;
    }
}

/// Fused Softmax: f16 input -> f32 compute -> f16 output.
/// Softmax along last dimension. One cube per row, BLOCK_SIZE threads.
#[cube(launch)]
fn softmax_f16_kernel<FIn: Float, FComp: Float>(
    input: &Tensor<FIn>,
    output: &mut Tensor<FIn>,
    row_size: u32,
) {
    let row = CUBE_POS_X;
    let tid = UNIT_POS_X;
    let block_size = CUBE_DIM_X;
    let row_offset = (row * row_size) as usize;
    let tid_idx = tid as usize;

    let mut shared = SharedMemory::<FComp>::new(BLOCK_SIZE as usize);

    // --- Pass 1: find max via parallel reduction ---
    let mut local_max = FComp::new(-65504.0);
    let mut i = tid;
    while i < row_size {
        let val = FComp::cast_from(input[row_offset + i as usize]);
        local_max = FComp::max(local_max, val);
        i += block_size;
    }
    shared[tid_idx] = local_max;
    sync_cube();

    let mut stride = block_size / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let rhs = shared[(tid + stride) as usize];
            shared[tid_idx] = FComp::max(shared[tid_idx], rhs);
        }
        sync_cube();
        stride = stride / 2u32;
    }

    let max_val = shared[0];
    sync_cube();

    // --- Pass 2: compute exp(x - max) and sum ---
    let mut local_sum = FComp::new(0.0);
    i = tid;
    while i < row_size {
        let val = FComp::cast_from(input[row_offset + i as usize]);
        local_sum += FComp::exp(val - max_val);
        i += block_size;
    }
    shared[tid_idx] = local_sum;
    sync_cube();

    stride = block_size / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let rhs = shared[(tid + stride) as usize];
            shared[tid_idx] += rhs;
        }
        sync_cube();
        stride = stride / 2u32;
    }

    let sum = shared[0];
    sync_cube();

    // --- Pass 3: normalize ---
    i = tid;
    while i < row_size {
        let idx = row_offset + i as usize;
        let val = FComp::cast_from(input[idx]);
        let result = FComp::exp(val - max_val) / sum;
        output[idx] = FIn::cast_from(result);
        i += block_size;
    }
}

/// Fused Linear (matrix-vector multiply): f16 input + f32 weight/bias -> f16 output.
/// One cube per (row, output_col) pair, BLOCK_SIZE threads for dot product reduction.
/// input: [total_rows, d_in], weight: [d_out, d_in], bias: [d_out], output: [total_rows, d_out]
#[cube(launch)]
fn linear_f16_kernel<FIn: Float, FComp: Float>(
    input: &Tensor<FIn>,
    weight: &Tensor<FComp>,
    bias: &Tensor<FComp>,
    output: &mut Tensor<FIn>,
    d_in: u32,
    d_out: u32,
) {
    let cube_id = CUBE_POS_X;
    let tid = UNIT_POS_X;
    let block_size = CUBE_DIM_X;
    let tid_idx = tid as usize;

    let row = cube_id / d_out;
    let col = cube_id % d_out;

    let input_offset = (row * d_in) as usize;
    let weight_offset = (col * d_in) as usize;

    let mut shared = SharedMemory::<FComp>::new(BLOCK_SIZE as usize);

    // Parallel dot product: input[row, :] . weight[col, :]
    let mut local_sum = FComp::new(0.0);
    let mut i = tid;
    while i < d_in {
        let in_val = FComp::cast_from(input[input_offset + i as usize]);
        let w_val = weight[weight_offset + i as usize];
        local_sum += in_val * w_val;
        i += block_size;
    }
    shared[tid_idx] = local_sum;
    sync_cube();

    // Reduction
    let mut stride = block_size / 2u32;
    while stride > 0u32 {
        if tid < stride {
            let rhs = shared[(tid + stride) as usize];
            shared[tid_idx] += rhs;
        }
        sync_cube();
        stride = stride / 2u32;
    }

    if tid == 0u32 {
        let result = shared[0] + bias[col as usize];
        output[(row * d_out + col) as usize] = FIn::cast_from(result);
    }
}

/// Fused LSTM cell: matmul(hidden, weight) + bias + input_gates -> gate activations -> cell/hidden update.
/// Single kernel replaces: matmul + add + split + 3×sigmoid + 2×tanh + 3×mul + add per LSTM step.
/// 128 threads, one per hidden element. Each thread computes 4 dot products (one per gate).
/// Output: packed [new_hidden | new_cell] of shape [2 * d_hidden].
/// Weight is Col-layout [d_hidden, 4*d_hidden] row-major after into_contiguous.
#[cube(launch)]
fn lstm_cell_kernel<F: Float>(
    hidden: &Tensor<F>,      // [d_hidden] flat
    cell: &Tensor<F>,        // [d_hidden] flat
    input_gates: &Tensor<F>, // [4*d_hidden] flat
    weight: &Tensor<F>,      // [d_hidden, 4*d_hidden] row-major (Col layout after contiguous)
    bias: &Tensor<F>,        // [4*d_hidden] flat
    output: &mut Tensor<F>,  // [2*d_hidden] flat: [new_hidden | new_cell]
    d_hidden: u32,
) {
    let k = UNIT_POS_X;
    if k < d_hidden {
        let k_idx = k as usize;

        // Load hidden into shared memory for all threads to read
        let mut shared_h = SharedMemory::<F>::new(BLOCK_SIZE as usize);
        shared_h[k_idx] = hidden[k_idx];
        sync_cube();

        // Compute 4 gate dot products: gates[m] = sum_j(hidden[j] * weight[j, m]) + bias[m]
        // Weight is [d_hidden, d_out] row-major, so weight[j, m] = weight_flat[j * d_out + m]
        // d_out = 4 * d_hidden
        let dh = d_hidden;
        let d_out = dh + dh + dh + dh;
        let idx0 = k; // gate i, output position k
        let idx1 = dh + k; // gate f, output position dh+k
        let idx2 = dh + dh + k; // gate g, output position 2*dh+k
        let idx3 = dh + dh + dh + k; // gate o, output position 3*dh+k
        let mut g0 = bias[idx0 as usize] + input_gates[idx0 as usize];
        let mut g1 = bias[idx1 as usize] + input_gates[idx1 as usize];
        let mut g2 = bias[idx2 as usize] + input_gates[idx2 as usize];
        let mut g3 = bias[idx3 as usize] + input_gates[idx3 as usize];

        let mut j = 0u32;
        while j < dh {
            let h_j = shared_h[j as usize];
            let wrow = j * d_out; // weight[j, :] starts at j * d_out
            g0 += h_j * weight[(wrow + idx0) as usize];
            g1 += h_j * weight[(wrow + idx1) as usize];
            g2 += h_j * weight[(wrow + idx2) as usize];
            g3 += h_j * weight[(wrow + idx3) as usize];
            j += 1u32;
        }

        // Gate activations: sigmoid(i), sigmoid(f), tanh(g), sigmoid(o)
        let one = F::new(1.0);
        let zero = F::new(0.0);
        let two = F::new(2.0);
        let ten = F::new(10.0);
        let sig_i = one / (one + F::exp(zero - g0));
        let sig_f = one / (one + F::exp(zero - g1));
        let sig_o = one / (one + F::exp(zero - g3));
        // Numerically stable tanh: clamp to avoid exp overflow
        let tanh_g = if g2 > ten {
            one
        } else if g2 < zero - ten {
            zero - one
        } else {
            let exp2g = F::exp(two * g2);
            (exp2g - one) / (exp2g + one)
        };

        // Cell update: new_cell = f * cell + i * tanh(g)
        let new_c = sig_f * cell[k_idx] + sig_i * tanh_g;

        // Hidden update: new_hidden = o * tanh(new_cell)
        let tanh_c = if new_c > ten {
            one
        } else if new_c < zero - ten {
            zero - one
        } else {
            let exp2c = F::exp(two * new_c);
            (exp2c - one) / (exp2c + one)
        };
        let new_h = sig_o * tanh_c;

        output[k_idx] = new_h;
        let cell_offset = dh + k;
        output[cell_offset as usize] = new_c;
    }
}

/// Fused single-query attention: Q@K^T · scale → softmax → @V in one kernel.
/// One cube per (batch, head) pair. d_k threads per cube, one per output dimension.
/// Fused single-query attention with striped n_kv parallelism.
///
/// Uses N_STRIPES groups of d_k threads (total = N_STRIPES * d_k threads per cube).
/// Each stripe handles every N_STRIPES-th KV position using the redundant-dot-product
/// online softmax approach. After the loop, stripes merge partial results via shared memory.
///
/// For n_stripes=1 this is identical to the original single-stripe kernel.
/// For n_stripes=4 with n_kv=1500, each stripe processes ~375 KV positions
/// instead of 1500, giving ~4× speedup on cross-attention.
///
/// Input tensors must be contiguous in [batch*heads, ...] layout.
#[cube(launch)]
fn fused_single_query_attn_kernel<FIn: Float, FComp: Float>(
    q: &Tensor<FIn>,          // [B*H * d_k] flat (seq_len=1 squeezed)
    k: &Tensor<FIn>,          // [B*H * n_kv * d_k] flat
    v: &Tensor<FIn>,          // [B*H * n_kv * d_k] flat
    output: &mut Tensor<FIn>, // [B*H * d_k] flat
    n_kv: u32,
    d_k: u32,
    n_stripes: u32,
) {
    let cube_id = CUBE_POS_X;
    let tid = UNIT_POS_X;

    let stripe_id = tid / d_k;
    let lane = tid % d_k;

    // Shared memory
    let mut q_shared = SharedMemory::<FComp>::new(ATTN_D_K_MAX as usize);
    // partial_out[stripe * d_k + lane]: per-stripe rescaled weighted V accumulator
    let mut partial_out = SharedMemory::<FComp>::new(ATTN_SMEM_PARTIALS as usize);
    // Per-stripe max and sum (one per stripe, written by lane 0)
    let mut stripe_max_s = SharedMemory::<FComp>::new(ATTN_N_STRIPES_MAX as usize);
    let mut stripe_sum_s = SharedMemory::<FComp>::new(ATTN_N_STRIPES_MAX as usize);

    // Load Q into shared memory
    if tid < d_k {
        q_shared[tid as usize] = FComp::cast_from(q[(cube_id * d_k + tid) as usize]);
    }
    sync_cube();

    if stripe_id < n_stripes {
        let scale = FComp::new(1.0) / FComp::sqrt(FComp::cast_from(d_k));
        let kv_base = (cube_id * n_kv * d_k) as usize;

        // Online softmax over this stripe's KV positions
        let mut running_max = FComp::new(-65504.0);
        let mut running_sum = FComp::new(0.0);
        let mut running_out = FComp::new(0.0);

        let mut t = stripe_id;
        while t < n_kv {
            let row = kv_base + (t * d_k) as usize;

            // Redundant dot product Q · K[t]
            let mut score = FComp::new(0.0);
            let mut d = 0u32;
            while d < d_k {
                score += q_shared[d as usize] * FComp::cast_from(k[row + d as usize]);
                d += 1u32;
            }
            score = score * scale;

            if score > running_max {
                let correction = FComp::exp(running_max - score);
                running_sum = running_sum * correction;
                running_out = running_out * correction;
                running_max = score;
            }
            let w = FComp::exp(score - running_max);
            running_sum += w;
            running_out += w * FComp::cast_from(v[row + lane as usize]);

            t += n_stripes;
        }

        // Write this stripe's max and sum for the merge phase
        if lane == 0u32 {
            stripe_max_s[stripe_id as usize] = running_max;
            stripe_sum_s[stripe_id as usize] = running_sum;
        }
        sync_cube();

        // --- Merge phase ---
        // Find global max across all stripes
        let mut global_max = FComp::new(-65504.0);
        let mut s = 0u32;
        while s < n_stripes {
            if stripe_max_s[s as usize] > global_max {
                global_max = stripe_max_s[s as usize];
            }
            s += 1u32;
        }

        // Rescale this stripe's partial output to global max and store
        let correction = FComp::exp(running_max - global_max);
        partial_out[(stripe_id * d_k + lane) as usize] = running_out * correction;

        // Rescale this stripe's sum
        if lane == 0u32 {
            stripe_sum_s[stripe_id as usize] = running_sum * correction;
        }
        sync_cube();

        // Stripe 0 aggregates final output
        if stripe_id == 0u32 {
            let mut global_sum = FComp::new(0.0);
            let mut s2 = 0u32;
            while s2 < n_stripes {
                global_sum += stripe_sum_s[s2 as usize];
                s2 += 1u32;
            }

            let mut final_val = FComp::new(0.0);
            let mut s3 = 0u32;
            while s3 < n_stripes {
                final_val += partial_out[(s3 * d_k + lane) as usize];
                s3 += 1u32;
            }

            let out_idx = (cube_id * d_k + lane) as usize;
            if global_sum > FComp::new(0.0) {
                output[out_idx] = FIn::cast_from(final_val / global_sum);
            } else {
                output[out_idx] = FIn::new(0.0);
            }
        }
    }
}

// ===========================================================================
// Low-level launch functions
// ===========================================================================

fn launch_layer_norm_f16<R: CubeRuntime>(
    input: CubeTensor<R>,
    gamma: CubeTensor<R>,
    beta: CubeTensor<R>,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let gamma = into_contiguous(gamma);
    let beta = into_contiguous(beta);

    let d_model = *input.shape().last().unwrap();
    let total_elements: usize = input.shape().iter().product();
    let total_rows = total_elements / d_model;

    let client = input.client.clone();

    let output = empty_device_dtype(
        client.clone(),
        input.device.clone(),
        input.shape(),
        input.dtype,
    );

    let cube_dim = CubeDim {
        x: BLOCK_SIZE,
        y: 1,
        z: 1,
    };
    let cube_count = CubeCount::Static(total_rows as u32, 1, 1);

    layer_norm_f16_kernel::launch::<half::f16, f32, R>(
        &client,
        cube_count,
        cube_dim,
        input.into_tensor_arg(),
        gamma.into_tensor_arg(),
        beta.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        d_model as u32,
    );

    output
}

fn launch_softmax_f16<R: CubeRuntime>(input: CubeTensor<R>) -> CubeTensor<R> {
    let input = into_contiguous(input);

    let row_size = *input.shape().last().unwrap();
    let total_elements: usize = input.shape().iter().product();
    let total_rows = total_elements / row_size;

    let client = input.client.clone();

    let output = empty_device_dtype(
        client.clone(),
        input.device.clone(),
        input.shape(),
        input.dtype,
    );

    let cube_dim = CubeDim {
        x: BLOCK_SIZE,
        y: 1,
        z: 1,
    };
    let cube_count = CubeCount::Static(total_rows as u32, 1, 1);

    softmax_f16_kernel::launch::<half::f16, f32, R>(
        &client,
        cube_count,
        cube_dim,
        input.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        row_size as u32,
    );

    output
}

fn launch_linear_f16<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);

    let d_in = *input.shape().last().unwrap();
    let d_out = weight.shape()[0];
    let total_elements: usize = input.shape().iter().product();
    let total_rows = total_elements / d_in;

    // Output shape: replace last dim with d_out
    let mut out_shape = input.shape();
    *out_shape.last_mut().unwrap() = d_out;

    let client = input.client.clone();

    let output = empty_device_dtype(client.clone(), input.device.clone(), out_shape, input.dtype);

    let cube_dim = CubeDim {
        x: BLOCK_SIZE,
        y: 1,
        z: 1,
    };
    let cube_count = CubeCount::Static((total_rows * d_out) as u32, 1, 1);

    linear_f16_kernel::launch::<half::f16, f32, R>(
        &client,
        cube_count,
        cube_dim,
        input.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        d_in as u32,
        d_out as u32,
    );

    output
}

fn launch_lstm_cell_fused<R: CubeRuntime>(
    hidden: CubeTensor<R>,
    cell: CubeTensor<R>,
    input_gates: CubeTensor<R>,
    weight: CubeTensor<R>,
    bias: CubeTensor<R>,
) -> CubeTensor<R> {
    let hidden = into_contiguous(hidden);
    let cell = into_contiguous(cell);
    let input_gates = into_contiguous(input_gates);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);

    let d_hidden = hidden.shape().iter().product::<usize>();
    let client = hidden.client.clone();

    let out_shape = burn_backend::Shape::from(vec![2 * d_hidden]);
    let output = empty_device_dtype(
        client.clone(),
        hidden.device.clone(),
        out_shape,
        hidden.dtype,
    );

    let cube_dim = CubeDim {
        x: d_hidden as u32,
        y: 1,
        z: 1,
    };
    let cube_count = CubeCount::Static(1, 1, 1);

    lstm_cell_kernel::launch::<f32, R>(
        &client,
        cube_count,
        cube_dim,
        hidden.into_tensor_arg(),
        cell.into_tensor_arg(),
        input_gates.into_tensor_arg(),
        weight.into_tensor_arg(),
        bias.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        d_hidden as u32,
    );

    output
}

fn launch_fused_single_query_attn<R: CubeRuntime>(
    q: CubeTensor<R>, // [batch, n_heads, 1, d_k]
    k: CubeTensor<R>, // [batch, n_heads, n_kv, d_k]
    v: CubeTensor<R>, // [batch, n_heads, n_kv, d_k]
) -> CubeTensor<R> {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let v = into_contiguous(v);

    let batch = q.shape()[0];
    let n_heads = q.shape()[1];
    let d_k = q.shape()[3];
    let n_kv = k.shape()[2];
    let n_cubes = batch * n_heads;

    // Choose stripe count based on n_kv:
    // Small n_kv (self-attention, ~1-50): 1 stripe (no overhead)
    // Large n_kv (cross-attention, ~1500): 4 stripes (4× less work per stripe)
    let n_stripes: usize = if n_kv > 256 {
        ATTN_N_STRIPES_MAX as usize
    } else {
        1
    };
    let threads_per_cube = n_stripes * d_k;

    let client = q.client.clone();
    let output = empty_device_dtype(client.clone(), q.device.clone(), q.shape().clone(), q.dtype);

    let cube_dim = CubeDim {
        x: threads_per_cube as u32,
        y: 1,
        z: 1,
    };
    let cube_count = CubeCount::Static(n_cubes as u32, 1, 1);

    match q.dtype {
        DType::F16 => {
            fused_single_query_attn_kernel::launch::<half::f16, f32, R>(
                &client,
                cube_count,
                cube_dim,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                n_kv as u32,
                d_k as u32,
                n_stripes as u32,
            );
        }
        _ => {
            fused_single_query_attn_kernel::launch::<f32, f32, R>(
                &client,
                cube_count,
                cube_dim,
                q.into_tensor_arg(),
                k.into_tensor_arg(),
                v.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                n_kv as u32,
                d_k as u32,
                n_stripes as u32,
            );
        }
    }

    output
}

// ===========================================================================
// Backend trait
// ===========================================================================

/// Backend extension for fused f16 mixed-precision kernels.
pub trait CustomKernelsBackend: Backend {
    fn layer_norm_f16(
        input: <Self as Backend>::FloatTensorPrimitive,
        gamma: <Self as Backend>::FloatTensorPrimitive,
        beta: <Self as Backend>::FloatTensorPrimitive,
    ) -> <Self as Backend>::FloatTensorPrimitive;

    fn softmax_f16(
        input: <Self as Backend>::FloatTensorPrimitive,
    ) -> <Self as Backend>::FloatTensorPrimitive;

    fn linear_f16(
        input: <Self as Backend>::FloatTensorPrimitive,
        weight: <Self as Backend>::FloatTensorPrimitive,
        bias: <Self as Backend>::FloatTensorPrimitive,
    ) -> <Self as Backend>::FloatTensorPrimitive;

    /// Fused LSTM cell: combines matmul + gate activations + cell/hidden update.
    /// Returns packed [new_hidden | new_cell] tensor of shape [2 * d_hidden].
    fn lstm_cell_fused(
        hidden: <Self as Backend>::FloatTensorPrimitive,
        cell: <Self as Backend>::FloatTensorPrimitive,
        input_gates: <Self as Backend>::FloatTensorPrimitive,
        weight: <Self as Backend>::FloatTensorPrimitive,
        bias: <Self as Backend>::FloatTensorPrimitive,
    ) -> <Self as Backend>::FloatTensorPrimitive;

    /// Fused single-query attention: Q@K^T·scale → softmax → @V in one kernel.
    /// All inputs are 4D [batch, n_heads, seq/n_kv, d_k]. Q has seq=1.
    fn fused_single_query_attn(
        q: <Self as Backend>::FloatTensorPrimitive,
        k: <Self as Backend>::FloatTensorPrimitive,
        v: <Self as Backend>::FloatTensorPrimitive,
    ) -> <Self as Backend>::FloatTensorPrimitive;
}

// Impl for CubeBackend (non-fusion)
impl<R, F, I, BT> CustomKernelsBackend for CubeBackend<R, F, I, BT>
where
    R: CubeRuntime,
    F: FloatElement,
    I: IntElement,
    BT: BoolElement,
{
    fn layer_norm_f16(
        input: CubeTensor<R>,
        gamma: CubeTensor<R>,
        beta: CubeTensor<R>,
    ) -> CubeTensor<R> {
        launch_layer_norm_f16(input, gamma, beta)
    }

    fn softmax_f16(input: CubeTensor<R>) -> CubeTensor<R> {
        launch_softmax_f16(input)
    }

    fn linear_f16(
        input: CubeTensor<R>,
        weight: CubeTensor<R>,
        bias: CubeTensor<R>,
    ) -> CubeTensor<R> {
        launch_linear_f16(input, weight, bias)
    }

    fn lstm_cell_fused(
        hidden: CubeTensor<R>,
        cell: CubeTensor<R>,
        input_gates: CubeTensor<R>,
        weight: CubeTensor<R>,
        bias: CubeTensor<R>,
    ) -> CubeTensor<R> {
        launch_lstm_cell_fused(hidden, cell, input_gates, weight, bias)
    }

    fn fused_single_query_attn(
        q: CubeTensor<R>,
        k: CubeTensor<R>,
        v: CubeTensor<R>,
    ) -> CubeTensor<R> {
        launch_fused_single_query_attn(q, k, v)
    }
}

// ===========================================================================
// Fusion wrapper
// ===========================================================================

mod fusion_impl {
    use super::*;
    use burn_fusion::stream::{Operation, OperationStreams};
    use burn_fusion::{Fusion, FusionBackend, FusionRuntime};
    use burn_ir::{CustomOpIr, HandleContainer, OperationIr, OperationOutput, TensorIr};
    use std::marker::PhantomData;

    impl<B> CustomKernelsBackend for Fusion<B>
    where
        B: FusionBackend + CustomKernelsBackend,
    {
        fn layer_norm_f16(
            input: <Self as Backend>::FloatTensorPrimitive,
            gamma: <Self as Backend>::FloatTensorPrimitive,
            beta: <Self as Backend>::FloatTensorPrimitive,
        ) -> <Self as Backend>::FloatTensorPrimitive {
            let client = input.client.clone();
            let out_shape = input.shape.clone();
            let out_dtype = input.dtype;

            #[derive(Clone, Debug)]
            struct LnF16Op<B1> {
                desc: CustomOpIr,
                _b: PhantomData<B1>,
            }

            impl<B1: FusionBackend + CustomKernelsBackend> Operation<B1::FusionRuntime> for LnF16Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([input, gamma, beta], [output]) = self.desc.as_fixed::<3, 1>();
                    let input = handles.get_float_tensor::<B1>(input);
                    let gamma = handles.get_float_tensor::<B1>(gamma);
                    let beta = handles.get_float_tensor::<B1>(beta);
                    let result = B1::layer_norm_f16(input, gamma, beta);
                    handles.register_float_tensor::<B1>(&output.id, result);
                }
            }

            let streams = OperationStreams::with_inputs([&input, &gamma, &beta]);
            let out_ir = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);

            let desc = CustomOpIr::new(
                "layer_norm_f16",
                &[input.into_ir(), gamma.into_ir(), beta.into_ir()],
                &[out_ir],
            );

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    LnF16Op::<B> {
                        desc,
                        _b: PhantomData,
                    },
                )
                .output()
        }

        fn softmax_f16(
            input: <Self as Backend>::FloatTensorPrimitive,
        ) -> <Self as Backend>::FloatTensorPrimitive {
            let client = input.client.clone();
            let out_shape = input.shape.clone();
            let out_dtype = input.dtype;

            #[derive(Clone, Debug)]
            struct SoftmaxF16Op<B1> {
                desc: CustomOpIr,
                _b: PhantomData<B1>,
            }

            impl<B1: FusionBackend + CustomKernelsBackend> Operation<B1::FusionRuntime> for SoftmaxF16Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([input], [output]) = self.desc.as_fixed::<1, 1>();
                    let input = handles.get_float_tensor::<B1>(input);
                    let result = B1::softmax_f16(input);
                    handles.register_float_tensor::<B1>(&output.id, result);
                }
            }

            let streams = OperationStreams::with_inputs([&input]);
            let out_ir = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);

            let desc = CustomOpIr::new("softmax_f16", &[input.into_ir()], &[out_ir]);

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    SoftmaxF16Op::<B> {
                        desc,
                        _b: PhantomData,
                    },
                )
                .output()
        }

        fn linear_f16(
            input: <Self as Backend>::FloatTensorPrimitive,
            weight: <Self as Backend>::FloatTensorPrimitive,
            bias: <Self as Backend>::FloatTensorPrimitive,
        ) -> <Self as Backend>::FloatTensorPrimitive {
            let client = input.client.clone();
            let out_dtype = input.dtype;

            // Output shape: same as input but last dim replaced by weight's first dim (d_out)
            let mut out_shape = input.shape.clone();
            let d_out = weight.shape[0];
            *out_shape.last_mut().unwrap() = d_out;

            #[derive(Clone, Debug)]
            struct LinearF16Op<B1> {
                desc: CustomOpIr,
                _b: PhantomData<B1>,
            }

            impl<B1: FusionBackend + CustomKernelsBackend> Operation<B1::FusionRuntime> for LinearF16Op<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([input, weight, bias], [output]) = self.desc.as_fixed::<3, 1>();
                    let input = handles.get_float_tensor::<B1>(input);
                    let weight = handles.get_float_tensor::<B1>(weight);
                    let bias = handles.get_float_tensor::<B1>(bias);
                    let result = B1::linear_f16(input, weight, bias);
                    handles.register_float_tensor::<B1>(&output.id, result);
                }
            }

            let streams = OperationStreams::with_inputs([&input, &weight, &bias]);
            let out_ir = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);

            let desc = CustomOpIr::new(
                "linear_f16",
                &[input.into_ir(), weight.into_ir(), bias.into_ir()],
                &[out_ir],
            );

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    LinearF16Op::<B> {
                        desc,
                        _b: PhantomData,
                    },
                )
                .output()
        }

        fn lstm_cell_fused(
            hidden: <Self as Backend>::FloatTensorPrimitive,
            cell: <Self as Backend>::FloatTensorPrimitive,
            input_gates: <Self as Backend>::FloatTensorPrimitive,
            weight: <Self as Backend>::FloatTensorPrimitive,
            bias: <Self as Backend>::FloatTensorPrimitive,
        ) -> <Self as Backend>::FloatTensorPrimitive {
            let client = hidden.client.clone();
            let out_dtype = hidden.dtype;
            let d_hidden: usize = hidden.shape.iter().product();
            let out_shape = vec![2 * d_hidden].into();

            #[derive(Clone, Debug)]
            struct LstmCellOp<B1> {
                desc: CustomOpIr,
                _b: PhantomData<B1>,
            }

            impl<B1: FusionBackend + CustomKernelsBackend> Operation<B1::FusionRuntime> for LstmCellOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([hidden, cell, input_gates, weight, bias], [output]) =
                        self.desc.as_fixed::<5, 1>();
                    let hidden = handles.get_float_tensor::<B1>(hidden);
                    let cell = handles.get_float_tensor::<B1>(cell);
                    let input_gates = handles.get_float_tensor::<B1>(input_gates);
                    let weight = handles.get_float_tensor::<B1>(weight);
                    let bias = handles.get_float_tensor::<B1>(bias);
                    let result = B1::lstm_cell_fused(hidden, cell, input_gates, weight, bias);
                    handles.register_float_tensor::<B1>(&output.id, result);
                }
            }

            let streams =
                OperationStreams::with_inputs([&hidden, &cell, &input_gates, &weight, &bias]);
            let out_ir = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);

            let desc = CustomOpIr::new(
                "lstm_cell_fused",
                &[
                    hidden.into_ir(),
                    cell.into_ir(),
                    input_gates.into_ir(),
                    weight.into_ir(),
                    bias.into_ir(),
                ],
                &[out_ir],
            );

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    LstmCellOp::<B> {
                        desc,
                        _b: PhantomData,
                    },
                )
                .output()
        }

        fn fused_single_query_attn(
            q: <Self as Backend>::FloatTensorPrimitive,
            k: <Self as Backend>::FloatTensorPrimitive,
            v: <Self as Backend>::FloatTensorPrimitive,
        ) -> <Self as Backend>::FloatTensorPrimitive {
            let client = q.client.clone();
            let out_shape = q.shape.clone();
            let out_dtype = q.dtype;

            #[derive(Clone, Debug)]
            struct FusedAttnOp<B1> {
                desc: CustomOpIr,
                _b: PhantomData<B1>,
            }

            impl<B1: FusionBackend + CustomKernelsBackend> Operation<B1::FusionRuntime> for FusedAttnOp<B1> {
                fn execute(
                    &self,
                    handles: &mut HandleContainer<
                        <B1::FusionRuntime as FusionRuntime>::FusionHandle,
                    >,
                ) {
                    let ([q, k, v], [output]) = self.desc.as_fixed::<3, 1>();
                    let q = handles.get_float_tensor::<B1>(q);
                    let k = handles.get_float_tensor::<B1>(k);
                    let v = handles.get_float_tensor::<B1>(v);
                    let result = B1::fused_single_query_attn(q, k, v);
                    handles.register_float_tensor::<B1>(&output.id, result);
                }
            }

            let streams = OperationStreams::with_inputs([&q, &k, &v]);
            let out_ir = TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype);

            let desc = CustomOpIr::new(
                "fused_single_query_attn",
                &[q.into_ir(), k.into_ir(), v.into_ir()],
                &[out_ir],
            );

            client
                .register(
                    streams,
                    OperationIr::Custom(desc.clone()),
                    FusedAttnOp::<B> {
                        desc,
                        _b: PhantomData,
                    },
                )
                .output()
        }
    }
}

// ===========================================================================
// Public high-level helpers
// ===========================================================================

/// Fused LayerNorm for f16 tensors: reads f16, computes in f32, writes f16.
pub fn layer_norm_f16<B: CustomKernelsBackend, const D: usize>(
    input: BurnTensor<B, D>,
    gamma: BurnTensor<B, 1>,
    beta: BurnTensor<B, 1>,
) -> BurnTensor<B, D> {
    let output = B::layer_norm_f16(
        input.into_primitive().tensor(),
        gamma.into_primitive().tensor(),
        beta.into_primitive().tensor(),
    );
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Convenience: run LayerNorm, choosing the fused f16 kernel when `use_f16` is
/// true and falling back to the standard burn `nn::LayerNorm` otherwise.
pub fn layer_norm_mixed<B: CustomKernelsBackend, const D: usize>(
    ln: &nn::LayerNorm<B>,
    x: BurnTensor<B, D>,
    use_f16: bool,
) -> BurnTensor<B, D> {
    if use_f16 {
        layer_norm_f16::<B, D>(
            x,
            ln.gamma.val(),
            ln.beta
                .as_ref()
                .expect("LayerNorm must have bias for fused f16 kernel")
                .val(),
        )
    } else {
        ln.forward(x)
    }
}

/// Fused Softmax for f16 tensors: reads f16, computes in f32, writes f16.
/// Always operates on the last dimension.
pub fn softmax_f16<B: CustomKernelsBackend, const D: usize>(
    input: BurnTensor<B, D>,
) -> BurnTensor<B, D> {
    let output = B::softmax_f16(input.into_primitive().tensor());
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Convenience: run softmax, choosing the fused f16 kernel when `use_f16` is
/// true and falling back to `burn::tensor::activation::softmax` otherwise.
pub fn softmax_mixed<B: CustomKernelsBackend, const D: usize>(
    x: BurnTensor<B, D>,
    dim: usize,
    use_f16: bool,
) -> BurnTensor<B, D> {
    if use_f16 {
        assert_eq!(
            dim,
            D - 1,
            "Fused f16 softmax only supports the last dimension"
        );
        softmax_f16::<B, D>(x)
    } else {
        burn::tensor::activation::softmax(x, dim)
    }
}

/// Fused Linear for f16 I/O: reads f16 input + f32 weight/bias, computes in f32, writes f16.
/// Equivalent to `cast(F32) -> Linear -> cast(F16)` but in a single kernel dispatch.
pub fn linear_f16<B: CustomKernelsBackend, const D: usize>(
    input: BurnTensor<B, D>,
    weight: BurnTensor<B, 2>,
    bias: Option<BurnTensor<B, 1>>,
) -> BurnTensor<B, D> {
    let device = input.device();
    let d_out = weight.dims()[0];
    let bias = bias.unwrap_or_else(|| BurnTensor::zeros([d_out], &device));
    let output = B::linear_f16(
        input.into_primitive().tensor(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}

/// Convenience: run Linear, choosing the fused f16 kernel when `use_f16` is
/// true and falling back to `nn::Linear::forward` otherwise.
pub fn linear_mixed<B: CustomKernelsBackend, const D: usize>(
    linear: &nn::Linear<B>,
    x: BurnTensor<B, D>,
    use_f16: bool,
) -> BurnTensor<B, D> {
    if use_f16 {
        let weight = linear.weight.val();
        let bias: Option<BurnTensor<B, 1>> = linear.bias.as_ref().map(|b| b.val());
        linear_f16::<B, D>(x, weight, bias)
    } else {
        linear.forward(x)
    }
}

/// Fused LSTM cell step: matmul + gate activations + cell/hidden update in a single kernel.
/// Returns (new_hidden [1, d_hidden], new_cell [1, d_hidden]).
pub fn lstm_cell_fused<B: CustomKernelsBackend>(
    hidden: BurnTensor<B, 2>,      // [1, d_hidden]
    cell: BurnTensor<B, 2>,        // [1, d_hidden]
    input_gates: BurnTensor<B, 2>, // [1, 4*d_hidden]
    weight: BurnTensor<B, 2>,      // [4*d_hidden, d_hidden]
    bias: BurnTensor<B, 1>,        // [4*d_hidden]
) -> (BurnTensor<B, 2>, BurnTensor<B, 2>) {
    let d_hidden = hidden.dims()[1];
    let output = B::lstm_cell_fused(
        hidden.flatten::<1>(0, 1).into_primitive().tensor(),
        cell.flatten::<1>(0, 1).into_primitive().tensor(),
        input_gates.flatten::<1>(0, 1).into_primitive().tensor(),
        weight.into_primitive().tensor(),
        bias.into_primitive().tensor(),
    );
    let combined: BurnTensor<B, 1> = BurnTensor::from_primitive(TensorPrimitive::Float(output));
    let new_hidden = combined.clone().slice([0..d_hidden]).reshape([1, d_hidden]);
    let new_cell = combined
        .slice([d_hidden..2 * d_hidden])
        .reshape([1, d_hidden]);
    (new_hidden, new_cell)
}

/// Fused single-query attention for f16 tensors.
/// Computes Q@K^T·scale → softmax → @V in a single kernel (no intermediate tensors).
/// Q: [batch, n_heads, 1, d_k], K/V: [batch, n_heads, n_kv, d_k].
/// Returns context: [batch, n_heads, 1, d_k].
pub fn fused_single_query_attn<B: CustomKernelsBackend>(
    q: BurnTensor<B, 4>,
    k: BurnTensor<B, 4>,
    v: BurnTensor<B, 4>,
) -> BurnTensor<B, 4> {
    let output = B::fused_single_query_attn(
        q.into_primitive().tensor(),
        k.into_primitive().tensor(),
        v.into_primitive().tensor(),
    );
    BurnTensor::from_primitive(TensorPrimitive::Float(output))
}
