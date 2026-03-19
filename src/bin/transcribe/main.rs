use fast_whisper_burn::MixedPrecisionAdapter;
use fast_whisper_burn::custom_kernels::CustomKernelsBackend;
use fast_whisper_burn::model::*;
use fast_whisper_burn::token::{Gpt2Tokenizer, Language};
use fast_whisper_burn::transcribe::{
    SamplingStrategy, TokenTimestamp, WhisperParams, transcribe_regions_batched,
};
use fast_whisper_burn::vad::*;

use strum::IntoEnumIterator;

use burn::config::Config;
use burn_store::ModuleSnapshot;
use hound::{self, SampleFormat};
use std::{env, fs, io::Write, path::Path, process, time::Instant};

type WgpuF32 = burn::backend::Wgpu<f32>;

const TARGET_SAMPLE_RATE: u32 = 16_000;

#[derive(Debug, Clone)]
struct TranscriptSegment {
    start_ms: i64,
    end_ms: i64,
    text: String,
    token_timestamps: Vec<TokenTimestamp>,
}

fn load_audio_waveform(filename: &str) -> hound::Result<(Vec<f32>, usize)> {
    let reader = hound::WavReader::open(filename)?;
    let spec = reader.spec();

    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate as usize;
    let bits_per_sample = spec.bits_per_sample;
    let sample_format = spec.sample_format;

    assert_eq!(
        sample_rate, TARGET_SAMPLE_RATE as usize,
        "The audio sample rate must be 16k."
    );
    assert_eq!(channels, 1, "The audio must be single-channel.");

    let max_int_val = 2_u32.pow(bits_per_sample as u32 - 1) - 1;

    let floats = match sample_format {
        SampleFormat::Float => reader.into_samples::<f32>().collect::<hound::Result<_>>()?,
        SampleFormat::Int => reader
            .into_samples::<i32>()
            .map(|s| s.map(|s| s as f32 / max_int_val as f32))
            .collect::<hound::Result<_>>()?,
    };

    Ok((floats, sample_rate))
}

fn clean_segment_text(text: &str) -> String {
    text.replace("[BLANK_AUDIO]", "")
        .replace("(speaking in foreign language)", "")
        .replace("[NON-ENGLISH SPEECH]", "")
        .replace("(mumbles)", "")
        .replace("(mumbling)", "")
        .trim()
        .to_string()
}

fn format_srt_ts(ms: i64) -> String {
    let total_ms = ms.max(0);
    let hours = total_ms / 3_600_000;
    let minutes = (total_ms % 3_600_000) / 60_000;
    let seconds = (total_ms % 60_000) / 1_000;
    let millis = total_ms % 1_000;
    format!("{hours:02}:{minutes:02}:{seconds:02},{millis:03}")
}

fn segments_to_srt(segments: &[TranscriptSegment]) -> String {
    let mut srt = String::new();
    for (index, segment) in segments.iter().enumerate() {
        srt.push_str(&(index + 1).to_string());
        srt.push('\n');
        srt.push_str(&format!(
            "{} --> {}",
            format_srt_ts(segment.start_ms),
            format_srt_ts(segment.end_ms)
        ));
        srt.push('\n');
        srt.push_str(segment.text.trim());
        srt.push_str("\n\n");
    }
    srt
}

fn segments_to_text(segments: &[TranscriptSegment]) -> String {
    let mut text = String::new();
    for segment in segments {
        text.push_str(segment.text.trim());
        text.push('\n');
    }
    text
}

fn main() {
    let total_started = Instant::now();
    let tensor_device = <WgpuF32 as burn::tensor::backend::Backend>::Device::default();

    let args: Vec<String> = env::args().collect();
    let use_f16 = args.iter().any(|a| a == "--f16");
    let use_greedy = args.iter().any(|a| a == "--greedy");
    let use_token_ts = args.iter().any(|a| a == "--token-timestamps");
    let pos_args: Vec<&str> = args
        .iter()
        .map(|s| s.as_str())
        .filter(|a| !a.starts_with("--"))
        .collect();

    if pos_args.len() < 5 {
        eprintln!(
            "Usage: {} <model name> <audio file> <lang> <transcription file> [--f16] [--greedy] [--beam] [--token-timestamps]",
            pos_args[0]
        );
        process::exit(1);
    }

    let wav_file = pos_args[2];
    let text_file = pos_args[4];

    let lang_str = pos_args[3];
    let lang = if lang_str == "auto" || Language::iter().any(|lang| lang.as_str() == lang_str) {
        lang_str
    } else {
        eprintln!("Invalid language abbreviation: {lang_str}");
        process::exit(1);
    };

    let model_name = pos_args[1];

    println!("Loading waveform...");
    let waveform_started = Instant::now();
    let (waveform, sample_rate) = match load_audio_waveform(wav_file) {
        Ok((w, sr)) => (w, sr),
        Err(e) => {
            eprintln!("Failed to load audio file: {e}");
            process::exit(1);
        }
    };
    println!("Waveform loaded in {:.2?}", waveform_started.elapsed());

    if use_f16 {
        println!("Using f16 mixed-precision compute.");
    } else {
        println!("Using f32 precision.");
    }

    let use_beam_search = !use_greedy;

    run_transcription::<WgpuF32>(
        &tensor_device,
        model_name,
        &waveform,
        sample_rate,
        lang,
        text_file,
        use_f16,
        use_beam_search,
        use_token_ts,
        total_started,
    );
}

fn run_transcription<B: CustomKernelsBackend>(
    tensor_device: &B::Device,
    model_name: &str,
    waveform: &[f32],
    sample_rate: usize,
    lang_str: &str,
    text_file: &str,
    use_f16: bool,
    use_beam_search: bool,
    use_token_timestamps: bool,
    total_started: Instant,
) {
    let model_started = Instant::now();
    let (bpe, _whisper_config, whisper) = load_model::<B>(model_name, tensor_device, use_f16);
    println!("Model loaded in {:.2?}", model_started.elapsed());

    let vad_started = Instant::now();
    let vad = match SileroVAD6Model::<B>::new(tensor_device, false) {
        Ok(vad) => vad,
        Err(e) => {
            eprintln!("Failed to initialize Silero VAD: {e}");
            process::exit(1);
        }
    };

    let speech_regions = match detect_speech_regions(
        &vad,
        tensor_device,
        waveform,
        Some(|offset, len| {
            print!(
                "\rVAD progress: {:.2}%",
                (offset as f64 / len as f64) * 100.0
            );
            std::io::stdout().flush().unwrap();
            true // Continue processing
        }),
    ) {
        Ok(regions) => regions,
        Err(e) => {
            eprintln!("Failed to detect speech regions: {e}");
            process::exit(1);
        }
    };
    println!(
        "\rVAD finished in {:.2?}. {} speech region(s) detected.",
        vad_started.elapsed(),
        speech_regions.len()
    );

    let mut params = WhisperParams::default();
    params.print_special = false;
    params.language = lang_str.to_string();
    if use_beam_search {
        params.strategy = SamplingStrategy::BeamSearch {
            beam_size: 3,
            patience: -1.0,
        };
    } else {
        params.strategy = SamplingStrategy::Greedy { best_of: -1 };
    }
    params.use_f16_compute = use_f16;
    params.token_timestamps = use_token_timestamps;

    let mut segments = Vec::new();
    let transcription_started = Instant::now();

    // Collect waveform slices for batched transcription
    let region_waveforms: Vec<&[f32]> = speech_regions
        .iter()
        .map(|region| &waveform[region.start_sample..region.end_sample])
        .collect();

    let batch_results = match transcribe_regions_batched(
        &whisper,
        &bpe,
        &region_waveforms,
        sample_rate,
        &params,
        None, // use default max batch size
        Some(|completed, total| {
            print!(
                "\rTranscribing: {:.2}%",
                (completed as f64 / total as f64) * 100.0
            );
            std::io::stdout().flush().unwrap();
            true // Continue processing
        }),
    ) {
        Ok(results) => results,
        Err(e) => {
            eprintln!("Error during transcription: {e}");
            process::exit(1);
        }
    };

    for (_index, (region, result)) in speech_regions.iter().zip(batch_results.iter()).enumerate() {
        /*println!(
            "Region {}/{} [{} ms -> {} ms] ({} segments)",
            _index + 1,
            speech_regions.len(),
            region.start_ms,
            ((region.end_sample as i64) * 1000) / TARGET_SAMPLE_RATE as i64,
            result.segments.len(),
        );*/

        for segment in &result.segments {
            let text = clean_segment_text(&segment.text);
            if text.is_empty() {
                continue;
            }
            let token_timestamps: Vec<TokenTimestamp> = segment
                .token_timestamps
                .iter()
                .map(|tt| TokenTimestamp {
                    token_id: tt.token_id,
                    text: tt.text.clone(),
                    t0: region.start_ms / 10 + tt.t0,
                    t1: region.start_ms / 10 + tt.t1,
                    pt: tt.pt,
                })
                .collect();
            segments.push(TranscriptSegment {
                start_ms: region.start_ms + segment.t0 * 10,
                end_ms: region.start_ms + segment.t1 * 10,
                text,
                token_timestamps,
            });
        }
    }
    println!(
        "\rTranscription finished in {:.2?}",
        transcription_started.elapsed()
    );

    let text = segments_to_text(&segments);
    let srt = segments_to_srt(&segments);

    // Write plain text transcription
    fs::write(text_file, &text).unwrap_or_else(|e| {
        eprintln!("Error writing transcription file: {e}");
        process::exit(1);
    });

    // Write SRT file alongside
    let srt_file = Path::new(text_file).with_extension("srt");
    fs::write(&srt_file, &srt).unwrap_or_else(|e| {
        eprintln!("Error writing SRT file: {e}");
        process::exit(1);
    });

    // Write token-level timestamps if enabled
    if use_token_timestamps {
        let tsv_file = Path::new(text_file).with_extension("tokens.tsv");
        let mut tsv = String::from("start_ms\tend_ms\ttoken\tconfidence\n");
        for seg in &segments {
            for tt in &seg.token_timestamps {
                let t = tt.text.replace('\t', " ").replace('\n', " ");
                tsv.push_str(&format!(
                    "{}\t{}\t{}\t{:.4}\n",
                    tt.t0 * 10,
                    tt.t1 * 10,
                    t,
                    tt.pt
                ));
            }
        }
        fs::write(&tsv_file, &tsv).unwrap_or_else(|e| {
            eprintln!("Error writing token timestamps file: {e}");
            process::exit(1);
        });

        // Print a sample of token timestamps to console
        println!("\n--- Token-level timestamps (first 30 tokens) ---");
        println!(
            "{:<12} {:<12} {:<8} {}",
            "start_ms", "end_ms", "conf", "token"
        );
        let mut count = 0;
        for seg in &segments {
            for tt in &seg.token_timestamps {
                if count >= 30 {
                    break;
                }
                println!(
                    "{:<12} {:<12} {:<8.4} {:?}",
                    tt.t0 * 10,
                    tt.t1 * 10,
                    tt.pt,
                    tt.text
                );
                count += 1;
            }
            if count >= 30 {
                break;
            }
        }
        let total_tokens: usize = segments.iter().map(|s| s.token_timestamps.len()).sum();
        if total_tokens > 30 {
            println!("... ({} more tokens)", total_tokens - 30);
        }
        println!("Token TSV: {}", tsv_file.display());
    }

    println!("\nTranscription finished. {} segments.", segments.len());
    println!("Text: {}", text_file);
    println!("SRT:  {}", srt_file.display());
    println!("Total elapsed: {:.2?}", total_started.elapsed());
}

fn load_model<B: CustomKernelsBackend>(
    model_name: &str,
    tensor_device_ref: &B::Device,
    use_f16: bool,
) -> (Gpt2Tokenizer, WhisperConfig, Whisper<B>) {
    let bpe = match Gpt2Tokenizer::new(&format!("models/{model_name}-tokenizer.json")) {
        Ok(bpe) => bpe,
        Err(e) => {
            eprintln!("Failed to load tokenizer from models/{model_name}-tokenizer.json: {e}");
            process::exit(1);
        }
    };

    let whisper_config = match WhisperConfig::load(format!("models/{model_name}.cfg")) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Failed to load whisper config from models/{model_name}.cfg: {e}");
            process::exit(1);
        }
    };

    let bpk_path = if use_f16 {
        format!("models/{model_name}-f16.bpk")
    } else {
        format!("models/{model_name}.bpk")
    };
    println!("Loading model from {bpk_path}...");
    let whisper: Whisper<B> = {
        let mut store = burn_store::BurnpackStore::from_file(&bpk_path);
        let target_dtype = if use_f16 {
            // Mixed precision: cast most weights to f16, keep LayerNorm/embeddings in f32
            burn::tensor::DType::F16
        } else {
            // This allows us to load the f16 model but still run it in f32
            burn::tensor::DType::F32
        };
        store = store.with_from_adapter(MixedPrecisionAdapter(target_dtype));
        let mut whisper_model = whisper_config.init(tensor_device_ref);
        let load_result = whisper_model.load_from(&mut store);
        match load_result {
            Ok(_) => whisper_model,
            Err(e) => {
                eprintln!("Failed to load whisper model file from {bpk_path}: {e}");
                process::exit(1);
            }
        }
    };

    (bpe, whisper_config, whisper)
}
