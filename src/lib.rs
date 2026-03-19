pub mod audio;
pub mod beam;
pub mod custom_kernels;
pub mod helper;
pub mod model;
pub mod token;
pub mod transcribe;
pub mod vad;

use burn_store::{ModuleAdapter, TensorSnapshot};

/// Mixed precision adapter: casts most F32 weights to F16 for faster compute,
/// but keeps LayerNorm and embedding weights in F32 for numerical stability.
#[derive(Debug, Clone)]
pub struct MixedPrecisionAdapter(pub burn::tensor::DType);

impl MixedPrecisionAdapter {
    fn is_precision_critical(path_stack: &[String]) -> bool {
        // Keep LayerNorm, positional_embedding, and token_embedding in f32
        path_stack.iter().any(|segment| {
            segment.contains("ln")
                || segment.contains("conv1")
                || segment.contains("conv2")
                || segment.contains("cross_attn")
                || segment.contains("positional_embedding")
                || segment.contains("token_embedding")
        })
    }
}

impl ModuleAdapter for MixedPrecisionAdapter {
    fn adapt(&self, snapshot: &TensorSnapshot) -> TensorSnapshot {
        use burn::tensor::DType;
        use std::rc::Rc;
        let dtype = self.0;

        if snapshot.dtype != DType::F32 && snapshot.dtype != DType::F16 {
            return snapshot.clone();
        }

        let path = snapshot.path_stack.as_deref().unwrap_or(&[]);
        if Self::is_precision_critical(path) {
            return snapshot.clone();
        }

        let original_data_fn = snapshot.clone_data_fn();
        let cast_data_fn = Rc::new(move || {
            let data = original_data_fn()?;
            Ok(data.convert_dtype(dtype))
        });

        TensorSnapshot::from_closure(
            cast_data_fn,
            dtype,
            snapshot.shape.clone(),
            snapshot.path_stack.clone().unwrap_or_default(),
            snapshot.container_stack.clone().unwrap_or_default(),
            snapshot.tensor_id.unwrap_or_default(),
        )
    }

    fn clone_box(&self) -> Box<dyn ModuleAdapter> {
        Box::new(self.clone())
    }
}
