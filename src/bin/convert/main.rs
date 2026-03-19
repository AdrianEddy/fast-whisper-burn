use burn::{backend::wgpu::Wgpu, config::Config};
use burn_store::pytorch::PytorchReader;
use burn_store::{BurnpackStore, ModuleSnapshot, PytorchStore};
use fast_whisper_burn::MixedPrecisionAdapter;
use fast_whisper_burn::custom_kernels::CustomKernelsBackend;
use fast_whisper_burn::model::*;
use std::error::Error;

fn save_whisper<B: CustomKernelsBackend>(
    whisper: Whisper<B>,
    name: &str,
) -> Result<(), burn_store::BurnpackError> {
    let mut store = BurnpackStore::from_file(&format!("{name}.bpk")).overwrite(true);
    whisper.save_into(&mut store)?;

    let mut storef16 = BurnpackStore::from_file(&format!("{name}-f16.bpk"))
        .overwrite(true)
        .with_to_adapter(MixedPrecisionAdapter(burn::tensor::DType::F16));
    whisper.save_into(&mut storef16)
}

use std::env;
use std::path::Path;

fn main() {
    let pt_path = match env::args().nth(1) {
        Some(name) => name,
        None => {
            eprintln!("Usage: convert <model.pt>");
            eprintln!("  e.g. convert base.en.pt");
            return;
        }
    };

    // Derive output name from .pt filename (e.g. "base.en.pt" → "base.en")
    let output_name = Path::new(&pt_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(&pt_path)
        .to_string();

    println!("Loading model from {pt_path}...");
    let (whisper, whisper_config): (Whisper<Wgpu>, WhisperConfig) = match load_whisper(&pt_path) {
        Ok(model) => model,
        Err(e) => {
            eprintln!("Error loading model from {pt_path}: {e}");
            return;
        }
    };

    println!("Saving model...");
    if let Err(e) = save_whisper(whisper, &output_name) {
        eprintln!("Error saving model {output_name}: {e}");
        return;
    }

    println!("Saving config...");
    if let Err(e) = whisper_config.save(format!("{output_name}.cfg")) {
        eprintln!("Error saving config for {output_name}: {e}");
        return;
    }

    println!("Finished.");
}

/// Load a Whisper model directly from a PyTorch .pt checkpoint file.
///
/// The .pt file is expected to have the standard OpenAI Whisper format with
/// a `model_state_dict` top-level key containing the model weights.
///
/// Config values are inferred from tensor shapes in the checkpoint.
pub fn load_whisper<B: CustomKernelsBackend>(
    pt_path: &str,
) -> Result<(Whisper<B>, WhisperConfig), Box<dyn Error>> {
    // 1. Inspect the .pt file to infer model config from tensor shapes
    let reader = PytorchReader::with_top_level_key(pt_path, "model_state_dict")?;

    let conv1_shape = reader
        .get("encoder.conv1.weight")
        .ok_or("missing encoder.conv1.weight")?
        .shape
        .clone();
    let enc_pos_shape = reader
        .get("encoder.positional_embedding")
        .ok_or("missing encoder.positional_embedding")?
        .shape
        .clone();
    let tok_emb_shape = reader
        .get("decoder.token_embedding.weight")
        .ok_or("missing decoder.token_embedding.weight")?
        .shape
        .clone();
    let dec_pos_shape = reader
        .get("decoder.positional_embedding")
        .ok_or("missing decoder.positional_embedding")?
        .shape
        .clone();

    let n_mels = conv1_shape[1];
    let n_audio_state = conv1_shape[0];
    let n_audio_ctx = enc_pos_shape[0];
    let n_audio_layer = reader
        .keys()
        .iter()
        .filter(|k| k.starts_with("encoder.blocks.") && k.ends_with(".attn_ln.weight"))
        .count();
    let n_audio_head = n_audio_state / 64; // Whisper always uses d_head=64

    let n_vocab = tok_emb_shape[0];
    let n_text_ctx = dec_pos_shape[0];
    let n_text_state = dec_pos_shape[1];
    let n_text_layer = reader
        .keys()
        .iter()
        .filter(|k| k.starts_with("decoder.blocks.") && k.ends_with(".attn_ln.weight"))
        .count();
    let n_text_head = n_text_state / 64;

    drop(reader);

    let config = WhisperConfig::new(
        AudioEncoderConfig::new(
            n_mels,
            n_audio_ctx,
            n_audio_state,
            n_audio_head,
            n_audio_layer,
        ),
        TextDecoderConfig::new(n_vocab, n_text_ctx, n_text_state, n_text_head, n_text_layer),
    );

    println!("Inferred config: {config:?}");

    // 2. Init model with random weights, then load from .pt
    let device = Default::default();
    let mut whisper: Whisper<B> = config.init(&device);

    let mut store = PytorchStore::from_file(pt_path)
        .with_top_level_key("model_state_dict")
        // PyTorch "out" → Burn "output" in MultiHeadAttention
        .with_key_remapping(r"\.out\.", ".output.")
        // PyTorch nn.Sequential indices → Burn MLP named fields
        .with_key_remapping(r"\.mlp\.0\.", ".mlp.lin1.")
        .with_key_remapping(r"\.mlp\.2\.", ".mlp.lin2.")
        // PyTorch Embedding .weight → Burn bare Param (no .weight suffix)
        .with_key_remapping(
            r"^decoder\.token_embedding\.weight$",
            "decoder.token_embedding",
        )
        // Whisper uses bias=False for key projections, but Burn MHA always includes key bias
        .allow_partial(true);

    let result = whisper.load_from(&mut store)?;
    println!("Loaded {} tensors from {pt_path}", result.applied.len());
    if !result.missing.is_empty() {
        println!("Missing: {}", result);
    }

    Ok((whisper, config))
}
