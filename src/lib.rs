pub mod audio;
pub mod beam;
pub mod custom_kernels;
pub mod helper;
pub mod model;
pub mod token;
pub mod transcribe;
pub mod vad;

use burn_store::burn_pack::Tensor as PackTensor;
use burn_store::{ModuleAdapter, ModuleContext, bridge};

/// Mixed precision adapter: casts most F32 weights to F16 for faster compute,
/// but keeps LayerNorm and embedding weights in F32 for numerical stability.
#[derive(Debug, Clone)]
pub struct MixedPrecisionAdapter(pub burn::tensor::DType);

impl MixedPrecisionAdapter {
    fn is_precision_critical(name: &str) -> bool {
        // Keep LayerNorm, positional_embedding, and token_embedding in f32
        name.contains("ln")
            || name.contains("conv1")
            || name.contains("conv2")
            || name.contains("cross_attn")
            || name.contains("positional_embedding")
            || name.contains("token_embedding")
    }
}

impl ModuleAdapter for MixedPrecisionAdapter {
    fn adapt(&self, tensor: PackTensor, _ctx: ModuleContext<'_>) -> PackTensor {
        use burn::tensor::DType;
        let dtype = self.0;

        if tensor.dtype != DType::F32 && tensor.dtype != DType::F16 {
            return tensor;
        }
        if tensor.dtype == dtype {
            return tensor;
        }

        if Self::is_precision_critical(&tensor.name) {
            return tensor;
        }

        let (name, shape) = (tensor.name.clone(), tensor.shape.clone());
        bridge::map_data(tensor, name, dtype, shape, move |data| {
            data.convert_dtype(dtype)
        })
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
