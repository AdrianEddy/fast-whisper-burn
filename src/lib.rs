pub mod audio;
pub mod beam;
pub mod custom_kernels;
pub mod helper;
pub mod model;
pub mod pacing;
pub mod token;
pub mod transcribe;
pub mod vad;

use burn_store::{ModuleAdapter, ModuleContext, StoreTensor, bridge};

/// Mixed precision adapter: casts most weights to the given dtype (F16 for faster compute),
/// but loads the precision-critical ones as F32 whatever dtype the source stores them in:
/// the model reads them as F32 in both precisions.
#[derive(Debug, Clone)]
pub struct MixedPrecisionAdapter(pub burn::tensor::DType);

impl MixedPrecisionAdapter {
    fn is_precision_critical(name: &str) -> bool {
        // LayerNorm, the stem's convolutions, cross-attention and both embeddings
        name.contains("ln")
            || name.contains("conv1")
            || name.contains("conv2")
            || name.contains("cross_attn")
            || name.contains("positional_embedding")
            || name.contains("token_embedding")
    }
}

impl ModuleAdapter for MixedPrecisionAdapter {
    fn adapt(&self, tensor: StoreTensor, _ctx: ModuleContext<'_>) -> StoreTensor {
        use burn::tensor::DType;
        let dtype = self.0;

        if tensor.dtype() != DType::F32 && tensor.dtype() != DType::F16 {
            return tensor;
        }
        // Precision-critical weights are always promoted to f32, even when the checkpoint
        // stores them as f16 (as OpenAI's official `.pt` files do); passing them through
        // unchanged leaves f16 weights next to f32 activations.
        let dtype = if Self::is_precision_critical(tensor.name()) {
            DType::F32
        } else {
            dtype
        };
        if tensor.dtype() == dtype {
            return tensor;
        }

        let (name, shape) = (tensor.name().to_string(), tensor.shape().into());
        bridge::map_data(tensor, name, dtype, shape, move |data| {
            data.convert_dtype(dtype)
        })
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
