mod silero_vad_op18_ifless;

mod util;
pub use util::*;

use burn::{prelude::*, tensor::ops::PadMode};
use burn_store::{ModuleSnapshot, ModuleStore};
use silero_vad_op18_ifless::Model as SileroModel;

pub struct PredictState<B: Backend> {
    pub context_size: usize,
    pub context: Tensor<B, 2>,
    pub state: Tensor<B, 3>,
}

/// The chunk size for processing audio_16k samples
pub const CHUNK_SIZE: usize = 512;

impl<B: Backend> PredictState<B> {
    /// Create a new PredictState with given batch size and context size
    /// # Arguments
    /// * `context_size` - The size of the context window, which last chunk retains
    /// * `batch_size` - The number of samples processed in parallel, default is 1
    pub fn new(device: &Device<B>, batch_size: usize, context_size: usize) -> Self {
        Self {
            context_size,
            context: Tensor::zeros([batch_size, context_size], device),
            state: Self::init_state(device, batch_size),
        }
    }

    pub fn default(device: &Device<B>) -> Self {
        Self::new(device, 1, 64)
    }

    pub fn input_size(&self) -> usize {
        512
    }

    pub fn init_state(device: &Device<B>, batch_size: usize) -> Tensor<B, 3> {
        Tensor::zeros([2, batch_size, 128], device)
    }
}

pub struct SileroVAD6Model<B: Backend> {
    pub model: SileroModel<B>,
    pub use_f16: bool,
}

#[derive(Debug)]
pub enum SileroVAD6Error {
    InvalidInputSize { expected: usize, found: usize },
}

impl<B: crate::custom_kernels::CustomKernelsBackend> SileroVAD6Model<B> {
    pub const SILERO_VAD6_WEIGHTS: &[u8] = include_bytes!("silero_vad_op18_ifless.bpk");

    pub fn new(
        device: &Device<B>,
        use_f16: bool,
    ) -> Result<Self, <burn_store::BurnpackStore as ModuleStore>::Error> {
        let mut model = SileroModel::<B>::new(device);

        let bytes = burn::tensor::Bytes::from_bytes_vec(Self::SILERO_VAD6_WEIGHTS.to_vec());
        let mut store = burn_store::BurnpackStore::from_bytes(Some(bytes));
        if use_f16 {
            store = store.with_from_adapter(burn_store::HalfPrecisionAdapter::new());
        }

        model.load_from(&mut store)?;

        Ok(Self { model, use_f16 })
    }

    /// Forward pass for 16kHz audio input
    pub fn predict(
        &self,
        predict_state: PredictState<B>,
        mut input: Tensor<B, 2>,
    ) -> Result<(PredictState<B>, Tensor<B, 2>), SileroVAD6Error> {
        let input_size = predict_state.input_size();
        if input.shape()[1] > input_size {
            return Err(SileroVAD6Error::InvalidInputSize {
                expected: input_size,
                found: input.shape()[1],
            });
        } else if input.shape()[1] < input_size {
            // Pad input to the expected size
            let pad_size = input_size - input.shape()[1];
            input = input.pad((0, pad_size, 0, 0), PadMode::Constant(0.0));
        }

        let PredictState {
            context_size,
            context,
            state,
        } = predict_state;

        let input_data = burn::Tensor::cat(vec![context, input], 1);
        let context = input_data
            .clone()
            .slice(s![.., -(context_size as i32)..])
            .clone();

        let (out, new_state) = self.model.forward(input_data, 16000, state);
        Ok((
            PredictState {
                context_size,
                context,
                state: new_state,
            },
            out,
        ))
    }

    /// Forward pass for a sequence of consecutive 16kHz audio chunks.
    ///
    /// The input shape is `[steps, chunk_size]` and the recurrent state is advanced one step at a time. Outputs are concatenated and returned as `[steps, 1]`.
    pub fn predict_sequence(
        &self,
        predict_state: PredictState<B>,
        mut input: Tensor<B, 2>,
    ) -> Result<(PredictState<B>, Tensor<B, 2>), SileroVAD6Error> {
        let input_size = predict_state.input_size();
        if input.shape()[1] > input_size {
            return Err(SileroVAD6Error::InvalidInputSize {
                expected: input_size,
                found: input.shape()[1],
            });
        } else if input.shape()[1] < input_size {
            let pad_size = input_size - input.shape()[1];
            input = input.pad((0, pad_size, 0, 0), PadMode::Constant(0.0));
        }

        let steps = input.shape()[0];
        let PredictState {
            context_size,
            context,
            state,
        } = predict_state;

        // Build all context-prepended inputs using batched tensor ops
        // instead of a per-step loop (eliminates N slice+cat kernel launches).
        // Step 0 uses the initial context; step k uses input[k-1, -context_size:].
        let all_contexts = if steps > 1 {
            let from_input = input
                .clone()
                .slice(s![0..steps as i64 - 1, -(context_size as i32)..]);
            Tensor::cat(vec![context, from_input], 0)
        } else {
            context
        };
        let input_data = Tensor::cat(vec![all_contexts, input.clone()], 1);

        let new_context = input.slice(s![steps as i64 - 1..steps as i64, -(context_size as i32)..]);

        let (out, state) = self
            .model
            .forward_sequence_16khz(input_data, state, self.use_f16);

        Ok((
            PredictState {
                context_size,
                context: new_context,
                state,
            },
            out,
        ))
    }
}
