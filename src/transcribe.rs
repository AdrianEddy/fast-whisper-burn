use crate::audio::prep_audio;
use crate::beam::{
    decode_segment_beam, decoder_count_for_iteration, effective_beam_size, effective_best_of,
};
use crate::custom_kernels::CustomKernelsBackend;
use crate::model::*;
use crate::token::{self, *};
use burn::tensor::TensorData;
use burn::{
    backend::ndarray::NdArray,
    module::Module,
    tensor::{
        ElementConversion, Int, Tensor,
        activation::{log_softmax, softmax},
        backend::Backend,
    },
};
use mt19937::MT19937;
use std::{
    collections::HashMap,
    f32,
    io::{Error as IoError, ErrorKind},
};

/// Compute mel spectrogram on CPU (NdArray backend) to avoid GPU kernel launch
/// overhead for the many small STFT operations, then upload to the target device.
fn compute_mel_cpu<B: CustomKernelsBackend>(
    waveform: &[f32],
    sample_rate: usize,
    n_mels: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let cpu_device = <NdArray as Backend>::Device::default();
    let wav: Tensor<NdArray, 1> = Tensor::from_floats(waveform, &cpu_device);
    let mel: Tensor<NdArray, 3> = prep_audio(wav.unsqueeze(), sample_rate as f64, n_mels);
    Tensor::<B, 3>::from_data(mel.into_data(), device)
}

// Non-speech tokens to suppress when suppress_nst is enabled
// ref: https://github.com/openai/whisper/blob/7858aa9c08d98f75575035ecd6481f462d66ca27/whisper/tokenizer.py#L224-L253
const NON_SPEECH_TOKENS: &[&str] = &[
    "\"",
    "#",
    "(",
    ")",
    "*",
    "+",
    "/",
    ":",
    ";",
    "<",
    "=",
    ">",
    "@",
    "[",
    "\\",
    "]",
    "^",
    "_",
    "`",
    "{",
    "|",
    "}",
    "~",
    "\u{300c}",
    "\u{300d}",
    "\u{300e}",
    "\u{300f}",
    "<<",
    ">>",
    "<<<",
    ">>>",
    "--",
    "---",
    "-(",
    "-[",
    "('",
    "(\"",
    "((",
    "))",
    "(((",
    ")))",
    "[[",
    "]]",
    "{{",
    "}}",
    "\u{266a}\u{266a}",
    "\u{266a}\u{266a}\u{266a}",
    "\u{266a9}",
    "\u{266a}",
    "\u{266b}",
    "\u{266c}",
    "\u{266d}",
    "\u{266e}",
    "\u{266f}",
];

// ═══════════════════════════════════════════
// Config types matching whisper_full_params
// ═══════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum SamplingStrategy {
    Greedy { best_of: i32 },
    BeamSearch { beam_size: i32, patience: f32 },
}

#[derive(Clone, Debug)]
pub struct WhisperParams {
    pub strategy: SamplingStrategy,
    pub n_max_text_ctx: usize,
    pub offset_ms: usize,
    pub duration_ms: usize,
    pub translate: bool,
    pub no_context: bool,
    pub carry_initial_prompt: bool,
    pub no_timestamps: bool,
    pub single_segment: bool,
    pub print_special: bool,
    pub token_timestamps: bool,
    pub thold_pt: f32,
    pub thold_ptsum: f32,
    pub max_len: usize,
    pub split_on_word: bool,
    pub max_tokens: usize,
    pub debug_mode: bool,
    pub audio_ctx: usize,
    pub tdrz_enable: bool,
    pub initial_prompt: Option<String>,
    pub prompt_tokens: Option<Vec<usize>>,
    pub language: String,
    pub detect_language: bool,
    pub suppress_blank: bool,
    pub suppress_nst: bool,
    pub temperature: f32,
    pub max_initial_ts: f32,
    pub length_penalty: f32,
    pub temperature_inc: f32,
    pub entropy_thold: f32,
    pub logprob_thold: f32,
    pub no_speech_thold: f32,
    pub use_f16_compute: bool,
}

impl Default for WhisperParams {
    fn default() -> Self {
        Self {
            strategy: SamplingStrategy::Greedy { best_of: -1 },
            n_max_text_ctx: 16384,
            offset_ms: 0,
            duration_ms: 0,
            translate: false,
            no_context: true,
            carry_initial_prompt: false,
            no_timestamps: false,
            single_segment: false,
            print_special: false,
            token_timestamps: false,
            thold_pt: 0.01,
            thold_ptsum: 0.01,
            max_len: 0,
            split_on_word: false,
            max_tokens: 0,
            debug_mode: false,
            audio_ctx: 0,
            tdrz_enable: false,
            initial_prompt: None,
            prompt_tokens: None,
            language: "en".to_string(),
            detect_language: false,
            suppress_blank: true,
            suppress_nst: false,
            temperature: 0.0,
            max_initial_ts: 1.0,
            length_penalty: -1.0,
            temperature_inc: 0.2,
            entropy_thold: 2.4,
            logprob_thold: -1.0,
            no_speech_thold: 0.6,
            use_f16_compute: false,
        }
    }
}

// ═══════════════════
// Result types
// ═══════════════════

#[derive(Debug, Clone)]
pub struct TranscriptionSegment {
    /// Start time in 10ms frames
    pub t0: i64,
    /// End time in 10ms frames
    pub t1: i64,
    pub text: String,
    pub no_speech_prob: f32,
    /// Per-token timestamps (only populated when token_timestamps=true)
    pub token_timestamps: Vec<TokenTimestamp>,
    /// [TDRZ] Indicates the decoder predicted a speaker turn after this segment.
    pub speaker_turn_next: bool,
}

#[derive(Debug, Clone)]
pub struct TokenTimestamp {
    pub token_id: usize,
    pub text: String,
    pub t0: i64,
    pub t1: i64,
    /// Peak cross-attention weight used for alignment.
    pub pt: f32,
}

impl TranscriptionSegment {
    pub fn start_ms(&self) -> i64 {
        self.t0 * 10
    }
    pub fn end_ms(&self) -> i64 {
        self.t1 * 10
    }
}

/// Result of language detection
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    pub language: Language,
    pub probability: f32,
}

#[derive(Debug)]
pub struct TranscriptionResult {
    pub segments: Vec<TranscriptionSegment>,
    pub text: String,
}

// ═══════════════════
// Constants
// ═══════════════════

pub(crate) const CHUNK_SIZE: usize = 30; // seconds
const HISTORY_CONDITIONING_TEMP_CUTOFF: f32 = 0.5;

fn parse_requested_language(language: &str) -> token::Result<Option<Language>> {
    if language.is_empty() || language.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }

    Language::from_code(language).map(Some).ok_or_else(|| {
        IoError::new(
            ErrorKind::InvalidInput,
            format!("invalid language code: {language}"),
        )
        .into()
    })
}

/// Detect the language of the audio using the first 30s chunk.
/// Returns detected language and its probability.
pub fn detect_language<B: CustomKernelsBackend>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    use_f16_compute: bool,
) -> token::Result<LanguageDetectionResult> {
    assert!(
        bpe.is_multilingual(),
        "Language detection requires a multilingual model"
    );

    let device = whisper.devices()[0].clone();
    let n_mels = whisper.encoder_mel_size();
    let n_audio_ctx = whisper.encoder_ctx_size();
    let mel_frame_count = n_audio_ctx * 2;

    let full_mel = compute_mel_cpu::<B>(waveform, sample_rate, n_mels, &device);
    let [nb, nm, total] = full_mel.dims();

    let mel_chunk = if total >= mel_frame_count {
        full_mel.slice([0..nb, 0..nm, 0..mel_frame_count])
    } else {
        let pad = Tensor::zeros([nb, nm, mel_frame_count - total], &device);
        Tensor::cat(vec![full_mel, pad], 2)
    };

    let encoder_output = if use_f16_compute {
        whisper.forward_encoder_f16(mel_chunk)
    } else {
        whisper.forward_encoder(mel_chunk)
    };

    // Feed SOT token to decoder to get language logits
    let sot_token = bpe.special_token(SpecialToken::StartofTranscript).unwrap();
    let token_tensor = Tensor::from_ints(TensorData::new(vec![sot_token as u32], [1, 1]), &device);
    let logits_tensor = if use_f16_compute {
        whisper
            .forward_decoder_cached_with_cross_attention_f16(
                token_tensor,
                whisper.create_decoder_cache_f16(encoder_output),
            )
            .logits
    } else {
        whisper
            .forward_decoder_cached_with_cross_attention(
                token_tensor,
                whisper.create_decoder_cache(encoder_output),
            )
            .logits
    };
    let probs = softmax(logits_tensor.flatten::<1>(0, 2), 0);

    let mut language_ids = Vec::new();
    let mut languages = Vec::new();
    for lang in <Language as strum::IntoEnumIterator>::iter() {
        if let Some(lang_id) = bpe.special_token(SpecialToken::Language(lang)) {
            language_ids.push(lang_id as i32);
            languages.push(lang);
        }
    }

    let (best_lang, best_prob) = if language_ids.is_empty() {
        (Language::English, 0.0f32)
    } else {
        let language_indices = Tensor::<B, 1, Int>::from_ints(
            TensorData::new(language_ids, [languages.len()]),
            &device,
        );
        let language_probs = probs.select(0, language_indices);
        let (best_prob_tensor, best_index_tensor) = language_probs.topk_with_indices(1, 0);
        let best_prob: f32 = best_prob_tensor.into_scalar().elem();
        let best_index: i32 = best_index_tensor.into_scalar().elem();
        (languages[best_index as usize], best_prob)
    };

    Ok(LanguageDetectionResult {
        language: best_lang,
        probability: best_prob,
    })
}

pub fn transcribe<B: CustomKernelsBackend, F: FnMut(usize, usize) -> bool>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    params: &WhisperParams,
    progress_callback: Option<F>,
) -> token::Result<TranscriptionResult> {
    transcribe_inner(
        whisper,
        bpe,
        waveform,
        sample_rate,
        params,
        None,
        progress_callback,
    )
}

/// Like `transcribe`, but accepts a pre-computed encoder output to skip encoding.
/// Used by `transcribe_regions_batched` to amortise encoder cost across regions.
fn transcribe_inner<B: CustomKernelsBackend, F: FnMut(usize, usize) -> bool>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    waveform: &[f32],
    sample_rate: usize,
    params: &WhisperParams,
    pre_encoded: Option<Tensor<B, 3>>,
    mut progress_callback: Option<F>,
) -> token::Result<TranscriptionResult> {
    let device = whisper.devices()[0].clone();
    let n_mels = whisper.encoder_mel_size();
    let n_audio_ctx = whisper.encoder_ctx_size();
    let n_text_ctx = whisper.decoder_ctx_size();
    let mut effective_params = params.clone();

    // First-release distilled models require the no_timestamps token.
    if whisper.decoder_layer_count() == 2
        && bpe.vocab_size() != 51866
        && !effective_params.no_timestamps
    {
        log::warn!("using first release distilled model behavior: forcing no_timestamps");
        effective_params.no_timestamps = true;
    }
    let params = &effective_params;

    // audio_ctx override (per whisper.cpp)
    let effective_audio_ctx = if params.audio_ctx > 0 {
        assert!(
            params.audio_ctx <= n_audio_ctx,
            "audio_ctx ({}) exceeds model maximum ({})",
            params.audio_ctx,
            n_audio_ctx
        );
        params.audio_ctx
    } else {
        n_audio_ctx
    };
    let effective_mel_frame_count = effective_audio_ctx * 2;

    let full_mel = compute_mel_cpu::<B>(waveform, sample_rate, n_mels, &device);
    let [_, _, total_mel_frames] = full_mel.dims();

    // Seek range (in 10ms mel frames)
    let seek_start = params.offset_ms / 10;
    let seek_end = if params.duration_ms == 0 {
        total_mel_frames
    } else {
        seek_start + params.duration_ms / 10
    };
    let delta_min = 10; // 100 ms minimum

    if seek_end < seek_start + delta_min {
        if params.debug_mode {
            log::warn!(
                "input is too short: {} ms < 100 ms",
                (seek_end.saturating_sub(seek_start)) * 10
            );
        }
        return Ok(TranscriptionResult {
            segments: Vec::new(),
            text: String::new(),
        });
    }

    // Language detection
    let lang = if bpe.is_multilingual() {
        let requested_lang = parse_requested_language(&params.language)?;
        if params.detect_language || requested_lang.is_none() {
            let det = detect_language(whisper, bpe, waveform, sample_rate, params.use_f16_compute)?;
            eprintln!(
                "Auto-detected language: {} (p = {:.4})",
                det.language.as_str(),
                det.probability
            );
            det.language
        } else {
            requested_lang.unwrap()
        }
    } else {
        Language::English
    };
    effective_params.language = lang.as_str().to_string();
    let params = &effective_params;

    // Model type
    let is_multilingual = bpe.is_multilingual();

    // Build SOT sequence
    let mut sot_sequence = vec![bpe.special_token(SpecialToken::StartofTranscript).unwrap()];
    if is_multilingual {
        sot_sequence.push(bpe.special_token(SpecialToken::Language(lang)).unwrap());
        if params.translate {
            sot_sequence.push(bpe.special_token(SpecialToken::Translate).unwrap());
        } else {
            sot_sequence.push(bpe.special_token(SpecialToken::Transcribe).unwrap());
        }
    }
    if params.no_timestamps {
        sot_sequence.push(bpe.special_token(SpecialToken::NoTimeStamps).unwrap());
    }

    // Important token IDs
    let token_eot = bpe.special_token(SpecialToken::EndofText).unwrap();
    let token_beg = bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap();
    let token_not = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();
    let token_nosp = bpe
        .special_token(SpecialToken::NoSpeech)
        .unwrap_or(token_eot);
    let token_solm = bpe.special_token(SpecialToken::StartofLM);
    let vocab_size = bpe.vocab_size();

    // Space token for suppress_blank
    let space_tokens = bpe.encode(" ");
    let space_token_id = space_tokens.first().copied();

    // Build suppression mask: suppress all special tokens except EOT and timestamps
    let suppress_mask: Vec<bool> = (0..vocab_size)
        .map(|id| {
            if id == token_eot {
                return false;
            }
            if id >= token_beg {
                return false; // timestamp tokens are not suppressed by default
            }
            if params.tdrz_enable && token_solm == Some(id) {
                return false;
            }
            bpe.is_special(id)
        })
        .collect();

    // Build non-speech token suppression set (for suppress_nst)
    let nst_suppress_ids: Vec<usize> = if params.suppress_nst {
        let mut ids = Vec::new();
        for &token_str in NON_SPEECH_TOKENS {
            // Suppress both "token" and " token" variants
            for variant in &[token_str.to_string(), format!(" {}", token_str)] {
                let encoded = bpe.encode(variant);
                if encoded.len() == 1 {
                    ids.push(encoded[0]);
                }
            }
        }
        // Suppress " -" and " '" at word boundaries
        for extra in &[" -", " '"] {
            let encoded = bpe.encode(extra);
            if encoded.len() == 1 {
                ids.push(encoded[0]);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    } else {
        Vec::new()
    };

    // Temperature schedule
    let mut temperatures = vec![params.temperature];
    if params.temperature_inc > 0.0 {
        let mut t = params.temperature + params.temperature_inc;
        while t < 1.0 + 1e-6 {
            temperatures.push(t);
            t += params.temperature_inc;
        }
    }

    // Max decode length per segment (whisper.cpp: n_text_ctx/2 - 4)
    let n_max = n_text_ctx / 2 - 4;

    // Prompt context budget (per whisper.cpp: min(n_max_text_ctx, n_text_ctx/2))
    let max_prompt_ctx = if params.n_max_text_ctx > 0 {
        params.n_max_text_ctx.min(n_text_ctx / 2)
    } else {
        0
    };

    let max_decoder_count = match &params.strategy {
        SamplingStrategy::Greedy { .. } => effective_best_of(&params.strategy),
        SamplingStrategy::BeamSearch { .. } => {
            effective_best_of(&params.strategy).max(effective_beam_size(&params.strategy))
        }
    };
    let mut decoder_rngs: Vec<MT19937> = (0..max_decoder_count.max(1))
        .map(|index| MT19937::new_with_slice_seed(&[index as u32]))
        .collect();

    let mut segments: Vec<TranscriptionSegment> = Vec::new();
    let mut seek = seek_start;
    let mut prompt_past0: Vec<usize> = Vec::new();
    let mut prompt_past1: Vec<usize> = Vec::new();

    if params.no_context {
        prompt_past0.clear();
        prompt_past1.clear();
    }

    let explicit_prompt_tokens = params
        .prompt_tokens
        .as_ref()
        .filter(|tokens| !tokens.is_empty())
        .cloned()
        .or_else(|| {
            params
                .initial_prompt
                .as_ref()
                .filter(|prompt| !prompt.is_empty())
                .map(|prompt| bpe.encode(prompt))
        });

    if let Some(prompt_tokens) = explicit_prompt_tokens {
        if params.carry_initial_prompt {
            if prompt_past0.is_empty() {
                let max_tokens = max_prompt_ctx.saturating_sub(1).max(1);
                let n_take = prompt_tokens.len().min(max_tokens);
                prompt_past0.extend_from_slice(&prompt_tokens[prompt_tokens.len() - n_take..]);
            }
        } else {
            prompt_past1.extend(prompt_tokens);
        }
    }

    // Main seek loop
    while seek + delta_min < seek_end {
        if seek > seek_start && seek + 500 >= seek_end {
            prompt_past0.clear();
            prompt_past1.clear();
        }

        // Extract mel chunk [1, n_mels, effective_mel_frame_count] at seek position.
        let mel_chunk = {
            let [nb, nm, _] = full_mel.dims();
            let end = (seek + effective_mel_frame_count).min(total_mel_frames);
            let avail = end.saturating_sub(seek);
            if avail == 0 {
                Tensor::zeros([nb, nm, effective_mel_frame_count], &device)
            } else {
                let chunk = full_mel.clone().slice([0..nb, 0..nm, seek..end]);
                if avail < effective_mel_frame_count {
                    let pad = Tensor::zeros([nb, nm, effective_mel_frame_count - avail], &device);
                    Tensor::cat(vec![chunk, pad], 2)
                } else {
                    chunk
                }
            }
        };

        // Use pre-computed encoder output if available (first seek iteration only),
        // otherwise encode the mel chunk.
        let encoder_output = if let Some(ref enc) = pre_encoded {
            if seek == seek_start {
                enc.clone()
            } else if params.use_f16_compute {
                whisper.forward_encoder_f16(mel_chunk)
            } else {
                whisper.forward_encoder(mel_chunk)
            }
        } else if params.use_f16_compute {
            whisper.forward_encoder_f16(mel_chunk)
        } else {
            whisper.forward_encoder(mel_chunk)
        };

        // Temperature fallback loop
        let mut best_result: Option<SegmentDecodeResult> = None;
        let mut best_prompt: Vec<usize> = Vec::new();

        for (it, &t_cur) in temperatures.iter().enumerate() {
            let mut prompt: Vec<usize> = Vec::new();
            if params.n_max_text_ctx > 0
                && t_cur < HISTORY_CONDITIONING_TEMP_CUTOFF
                && !params.no_context
            {
                let can_take0 = params.carry_initial_prompt && !prompt_past0.is_empty();
                let can_take1 = !prompt_past1.is_empty();

                if max_prompt_ctx > 0 && (can_take0 || can_take1) {
                    if let Some(prev_tok) = bpe.special_token(SpecialToken::StartofPrev) {
                        prompt.push(prev_tok);

                        let mut n_take0 = 0usize;
                        if can_take0 {
                            n_take0 = prompt_past0.len();
                            prompt.extend_from_slice(&prompt_past0[prompt_past0.len() - n_take0..]);
                        }

                        let n_take1 =
                            (max_prompt_ctx.saturating_sub(n_take0 + 1)).min(prompt_past1.len());
                        prompt.extend_from_slice(&prompt_past1[prompt_past1.len() - n_take1..]);
                    }
                }
            }
            prompt.extend_from_slice(&sot_sequence);

            let decoder_count = decoder_count_for_iteration(params, t_cur);

            let result = decode_segment(
                whisper,
                bpe,
                &encoder_output,
                &prompt,
                t_cur,
                n_max,
                vocab_size,
                token_eot,
                token_beg,
                token_not,
                token_nosp,
                space_token_id,
                &suppress_mask,
                &nst_suppress_ids,
                params,
                seek,
                seek_end,
                delta_min,
                effective_audio_ctx,
                decoder_count,
                &mut decoder_rngs,
                &device,
            );

            // Quality check
            // Fallback condition: failed, OR (avg_logprobs < threshold AND no_speech_prob < no_speech_thold)
            let mut success = true;
            if it < temperatures.len() - 1 {
                if result.failed {
                    success = false;
                } else if result.avg_logprobs < params.logprob_thold as f64
                    && result.no_speech_prob < params.no_speech_thold
                {
                    success = false;
                } else if result.result_len > 32 && result.entropy < params.entropy_thold as f64 {
                    success = false;
                }
            }

            if success {
                best_prompt = prompt;
                best_result = Some(result);
                break;
            }
        }

        let result = best_result.unwrap();

        // no_speech_thold: skip segments with high no-speech probability and low confidence
        let is_no_speech = result.no_speech_prob > params.no_speech_thold
            && result.avg_logprobs < params.logprob_thold as f64;
        let tokens_cur: Vec<(usize, f64, usize)> =
            result.tokens[..result.result_len.min(result.tokens.len())].to_vec();
        let token_alignments_cur: Vec<Vec<f32>> = result.token_alignments
            [..result.result_len.min(result.token_alignments.len())]
            .to_vec();
        let mut seek_delta = result.seek_delta;
        let no_speech_prob = result.no_speech_prob;

        // Build segments from decoded tokens (skip if no-speech detected)
        if !tokens_cur.is_empty() && !is_no_speech {
            // Use tid (best timestamp token) for t0, per whisper.cpp:
            //   auto t0 = seek + 2*(tokens_cur.front().tid - whisper_token_beg(ctx))
            let mut t0 = seek as i64 + 2 * (tokens_cur[0].2 as i64 - token_beg as i64);

            let mut text = String::new();
            let mut seg_token_indices: Vec<usize> = Vec::new();
            let mut speaker_turn_next = false;
            let mut i = 0;

            while i < tokens_cur.len() {
                let (token_id, _, tid) = tokens_cur[i];

                if params.print_special || token_id < token_eot {
                    text += &bpe.decode(&[token_id], true)?;
                }
                seg_token_indices.push(i);

                if params.tdrz_enable && token_solm == Some(token_id) {
                    speaker_turn_next = true;
                }

                // Timestamp token > token_beg → segment boundary
                if token_id > token_beg && !params.single_segment {
                    // Use tid for time computation (per whisper.cpp)
                    let t1 = seek as i64 + 2 * (tid as i64 - token_beg as i64);

                    if !text.is_empty() {
                        // Adjust t0 using text tokens' cross-attention tids.
                        // The model's timestamp tokens (and max_initial_ts
                        // constraint) can place t0 earlier than speech actually
                        // starts. Text tokens' tids reflect where the decoder's
                        // cross-attention focuses in the audio, giving a better
                        // estimate of speech onset. Skip first 2 text tokens
                        // (noisy tids right after timestamp transitions).
                        let adjusted_t0 = t0;

                        // Compute token timestamps if requested
                        let tok_ts = if params.token_timestamps {
                            compute_token_timestamps_for_segment(
                                bpe,
                                &tokens_cur,
                                &token_alignments_cur,
                                &seg_token_indices,
                                seek as i64,
                                adjusted_t0,
                                t1,
                                token_beg,
                                token_eot,
                            )
                        } else {
                            Vec::new()
                        };

                        let seg = TranscriptionSegment {
                            t0: adjusted_t0,
                            t1,
                            text: text.clone(),
                            no_speech_prob,
                            token_timestamps: tok_ts,
                            speaker_turn_next,
                        };

                        // max_len: split segment if it exceeds character limit
                        if params.max_len > 0 && params.token_timestamps {
                            let mut split =
                                split_segment_by_length(&seg, params.max_len, params.split_on_word);
                            segments.append(&mut split);
                        } else {
                            segments.push(seg);
                        }
                    }

                    text.clear();
                    seg_token_indices.clear();
                    // Skip consecutive timestamp tokens
                    while i < tokens_cur.len() && tokens_cur[i].0 > token_beg {
                        i += 1;
                    }
                    i = i.saturating_sub(1);
                    t0 = t1;
                    speaker_turn_next = false;
                }

                i += 1;
            }

            // Remaining text → final segment of this chunk
            if !text.is_empty() {
                let t1 = seek as i64 + seek_delta as i64;

                let adjusted_t0 = t0;

                let tok_ts = if params.token_timestamps {
                    compute_token_timestamps_for_segment(
                        bpe,
                        &tokens_cur,
                        &token_alignments_cur,
                        &seg_token_indices,
                        seek as i64,
                        adjusted_t0,
                        t1,
                        token_beg,
                        token_eot,
                    )
                } else {
                    Vec::new()
                };

                let seg = TranscriptionSegment {
                    t0: adjusted_t0,
                    t1,
                    text,
                    no_speech_prob,
                    token_timestamps: tok_ts,
                    speaker_turn_next,
                };

                if params.max_len > 0 && params.token_timestamps {
                    let mut split =
                        split_segment_by_length(&seg, params.max_len, params.split_on_word);
                    segments.append(&mut split);
                } else {
                    segments.push(seg);
                }
            }
        }

        if !params.no_context {
            prompt_past1.clear();

            if !params.carry_initial_prompt {
                if let Some(prev_tok) = bpe.special_token(SpecialToken::StartofPrev) {
                    if !best_prompt.is_empty()
                        && best_prompt[0] == prev_tok
                        && best_prompt.len() >= sot_sequence.len() + 1
                    {
                        prompt_past1.extend_from_slice(
                            &best_prompt[1..best_prompt.len() - sot_sequence.len()],
                        );
                    }
                }
            }

            if !is_no_speech {
                for &(token_id, _, _) in tokens_cur.iter().take(result.result_len) {
                    prompt_past1.push(token_id);
                }
            }
        }

        // Single timestamp ending → skip entire 30s chunk
        if tokens_cur.len() > 1 {
            let last = tokens_cur[tokens_cur.len() - 1].0;
            let penult = tokens_cur[tokens_cur.len() - 2].0;
            if penult < token_beg && last > token_beg {
                seek_delta = (seek_end - seek).min(CHUNK_SIZE * 100);
            }
        }

        seek += seek_delta;

        if let Some(callback) = progress_callback.as_mut() {
            if !callback(seek, seek_end) {
                break;
            }
        }
    }

    let full_text = segments.iter().map(|s| s.text.as_str()).collect::<String>();

    Ok(TranscriptionResult {
        segments,
        text: full_text,
    })
}

// ═══════════════════════════════════════════════════
// Batched region transcription (encoder + decoder)
// ═══════════════════════════════════════════════════

/// Maximum number of regions to process in a single GPU batch.
const DEFAULT_MAX_BATCH_SIZE: usize = 10;

/// Transcribe multiple waveform regions in batched GPU passes.
///
/// Regions are processed in chunks of `max_batch_size`. Within each chunk,
/// mel spectrogram computation, encoder forward, and greedy decode all run
/// with batch=N to amortise kernel-launch overhead.
///
/// Falls back to sequential `transcribe()` for beam-search or sampling
/// (temperature > 0) strategies.
pub fn transcribe_regions_batched<B: CustomKernelsBackend, F: FnMut(usize, usize) -> bool>(
    whisper: &Whisper<B>,
    bpe: &Gpt2Tokenizer,
    regions: &[&[f32]],
    sample_rate: usize,
    params: &WhisperParams,
    max_batch_size: Option<usize>,
    mut progress_callback: Option<F>,
) -> token::Result<Vec<TranscriptionResult>> {
    let max_batch = max_batch_size.unwrap_or(DEFAULT_MAX_BATCH_SIZE).max(1);
    let total_regions = regions.len();
    let mut completed_regions = 0usize;

    // Both greedy and beam search get fully batched decode.
    // Only fall back to sequential for sampling (temperature > 0).
    if params.temperature > 0.0 {
        let mut results = Vec::with_capacity(total_regions);
        for (i, waveform) in regions.iter().enumerate() {
            results.push(transcribe(
                whisper,
                bpe,
                waveform,
                sample_rate,
                params,
                None::<fn(usize, usize) -> bool>,
            )?);
            if let Some(callback) = progress_callback.as_mut() {
                if !callback(i + 1, total_regions) {
                    results.resize_with(total_regions, || TranscriptionResult {
                        segments: Vec::new(),
                        text: String::new(),
                    });
                    return Ok(results);
                }
            }
        }
        return Ok(results);
    }

    let use_beam = matches!(params.strategy, SamplingStrategy::BeamSearch { .. });
    let beam_size = if use_beam {
        crate::beam::effective_beam_size(&params.strategy)
    } else {
        1
    };

    let device = whisper.devices()[0].clone();
    let n_mels = whisper.encoder_mel_size();
    let n_audio_ctx = whisper.encoder_ctx_size();
    let n_text_ctx = whisper.decoder_ctx_size();
    let effective_audio_ctx = if params.audio_ctx > 0 {
        params.audio_ctx.min(n_audio_ctx)
    } else {
        n_audio_ctx
    };
    let mel_frame_count = effective_audio_ctx * 2;

    // Token setup (same as transcribe())
    let mut effective_params = params.clone();
    if whisper.decoder_layer_count() == 2
        && bpe.vocab_size() != 51866
        && !effective_params.no_timestamps
    {
        effective_params.no_timestamps = true;
    }
    let params = &effective_params;

    // Language
    let is_multilingual = bpe.is_multilingual();
    let lang = if is_multilingual {
        parse_requested_language(&params.language)?.unwrap_or(Language::English)
    } else {
        Language::English
    };

    // SOT sequence
    let mut sot_sequence = vec![bpe.special_token(SpecialToken::StartofTranscript).unwrap()];
    if is_multilingual {
        sot_sequence.push(bpe.special_token(SpecialToken::Language(lang)).unwrap());
        if params.translate {
            sot_sequence.push(bpe.special_token(SpecialToken::Translate).unwrap());
        } else {
            sot_sequence.push(bpe.special_token(SpecialToken::Transcribe).unwrap());
        }
    }
    if params.no_timestamps {
        sot_sequence.push(bpe.special_token(SpecialToken::NoTimeStamps).unwrap());
    }

    let token_eot = bpe.special_token(SpecialToken::EndofText).unwrap();
    let token_beg = bpe.special_token(SpecialToken::Timestamp(0.0)).unwrap();
    let token_not = bpe.special_token(SpecialToken::NoTimeStamps).unwrap();
    let token_nosp = bpe
        .special_token(SpecialToken::NoSpeech)
        .unwrap_or(token_eot);
    let vocab_size = bpe.vocab_size();
    let space_tokens = bpe.encode(" ");
    let space_token_id = space_tokens.first().copied();

    let suppress_mask: Vec<bool> = (0..vocab_size)
        .map(|id| {
            if id == token_eot {
                return false;
            }
            if id >= token_beg {
                return false;
            }
            bpe.is_special(id)
        })
        .collect();

    let nst_suppress_ids: Vec<usize> = if params.suppress_nst {
        let mut ids = Vec::new();
        for &token_str in NON_SPEECH_TOKENS {
            for variant in &[token_str.to_string(), format!(" {}", token_str)] {
                let encoded = bpe.encode(variant);
                if encoded.len() == 1 {
                    ids.push(encoded[0]);
                }
            }
        }
        for extra in &[" -", " '"] {
            let encoded = bpe.encode(extra);
            if encoded.len() == 1 {
                ids.push(encoded[0]);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    } else {
        Vec::new()
    };

    let n_max = n_text_ctx / 2 - 4;
    let delta_min = 10usize;

    // Pre-build GPU suppression tensors (shared across all batches)
    let static_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            let is_suppressed = suppress_mask.get(id).copied().unwrap_or(false)
                || nst_suppress_ids.contains(&id)
                || id == token_not;
            if is_suppressed {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let blank_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            if id == token_eot || space_token_id == Some(id) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();

    // Pre-compute fused weights once (shared across all batches)
    let fused_weights = if params.use_f16_compute {
        Some(whisper.build_fused_decoder_weights())
    } else {
        None
    };

    // Process regions in chunks.
    // Pre-classify: regions >30s need multiple seek iterations → sequential fallback.
    let max_samples_for_batch = CHUNK_SIZE * sample_rate; // 30s of audio
    let mut all_results: Vec<TranscriptionResult> = (0..regions.len())
        .map(|_| TranscriptionResult {
            segments: Vec::new(),
            text: String::new(),
        })
        .collect();
    let mut batchable_indices: Vec<usize> = Vec::new();
    for (i, waveform) in regions.iter().enumerate() {
        if waveform.len() > max_samples_for_batch {
            // Too long for single-pass batching → sequential with full seek loop
            all_results[i] = transcribe(
                whisper,
                bpe,
                waveform,
                sample_rate,
                params,
                None::<fn(usize, usize) -> bool>,
            )?;
            completed_regions += 1;
            if let Some(callback) = progress_callback.as_mut() {
                if !callback(completed_regions, total_regions) {
                    return Ok(all_results);
                }
            }
        } else {
            batchable_indices.push(i);
        }
    }

    for chunk_start in (0..batchable_indices.len()).step_by(max_batch) {
        let chunk_end = (chunk_start + max_batch).min(batchable_indices.len());
        let batch_indices = &batchable_indices[chunk_start..chunk_end];
        let batch_regions: Vec<&[f32]> = batch_indices.iter().map(|&i| regions[i]).collect();
        let batch_size = batch_regions.len();

        // 1. Compute mel spectrograms on CPU, then stack and upload
        let mut mel_tensors: Vec<Tensor<B, 3>> = Vec::with_capacity(batch_size);
        let mut seek_end_per_region: Vec<usize> = Vec::with_capacity(batch_size);
        for waveform in batch_regions.iter() {
            let mel = compute_mel_cpu::<B>(waveform, sample_rate, n_mels, &device);
            let [nb, nm, total] = mel.dims();
            seek_end_per_region.push(total);
            let padded = if total >= mel_frame_count {
                mel.slice([0..nb, 0..nm, 0..mel_frame_count])
            } else {
                let pad = Tensor::zeros([nb, nm, mel_frame_count - total], &device);
                Tensor::cat(vec![mel, pad], 2)
            };
            mel_tensors.push(padded);
        }
        // Stack [1, n_mels, mel_frames] × N → [N, n_mels, mel_frames]
        let batched_mel = Tensor::cat(mel_tensors, 0);

        // 2. Batched encoder forward
        let batched_encoder_output = if params.use_f16_compute {
            whisper.forward_encoder_f16(batched_mel)
        } else {
            whisper.forward_encoder(batched_mel)
        };

        // Beam search: use multi-region batched beam decode
        if use_beam {
            let beam_results = decode_regions_beam_batched(
                whisper,
                batched_encoder_output,
                batch_size,
                &sot_sequence,
                n_max,
                token_eot,
                token_beg,
                token_not,
                token_nosp,
                space_token_id,
                &suppress_mask,
                &nst_suppress_ids,
                params,
                &seek_end_per_region,
                effective_audio_ctx,
                beam_size,
                &fused_weights,
                &device,
            );

            for (r, decode_result) in beam_results.into_iter().enumerate() {
                // Quality check: fall back to sequential transcribe() with
                // temperature fallback for degenerate outputs.
                let needs_fallback = decode_result.failed
                    || (decode_result.avg_logprobs < params.logprob_thold as f64
                        && decode_result.no_speech_prob < params.no_speech_thold)
                    || (decode_result.result_len > 32
                        && decode_result.entropy < params.entropy_thold as f64);

                let out_idx = batch_indices[r];
                if needs_fallback && params.temperature_inc > 0.0 {
                    all_results[out_idx] = transcribe(
                        whisper,
                        bpe,
                        batch_regions[r],
                        sample_rate,
                        params,
                        None::<fn(usize, usize) -> bool>,
                    )?;
                    continue;
                }

                let seek_end = seek_end_per_region[r];
                let no_speech_prob = decode_result.no_speech_prob;
                let is_no_speech = no_speech_prob > params.no_speech_thold
                    && decode_result.avg_logprobs < params.logprob_thold as f64;

                let tokens_cur = decode_result.tokens;
                let alignments_cur = decode_result.token_alignments;
                let mut segments = Vec::new();
                if !tokens_cur.is_empty() && !is_no_speech {
                    build_segments_from_tokens(
                        bpe,
                        &tokens_cur,
                        &alignments_cur,
                        token_eot,
                        token_beg,
                        0, // seek
                        seek_end,
                        delta_min,
                        decode_result.seek_delta,
                        no_speech_prob,
                        params,
                        &mut segments,
                    );
                }
                let text = segments.iter().map(|s| s.text.as_str()).collect::<String>();
                all_results[out_idx] = TranscriptionResult { segments, text };
            }
            completed_regions += batch_indices.len();
            if let Some(callback) = progress_callback.as_mut() {
                if !callback(completed_regions, total_regions) {
                    return Ok(all_results);
                }
            }
            continue;
        }

        let encoder_output = batched_encoder_output;

        // 3. Create batched decoder cache from batched encoder output
        let cache = if params.use_f16_compute {
            whisper.create_decoder_cache_f16(encoder_output)
        } else {
            whisper.create_decoder_cache(encoder_output)
        };

        // 4. Batched prompt processing
        let prompt: Vec<usize> = sot_sequence.clone();
        let prompt_len = prompt.len();
        let prompt_data: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
        // Repeat prompt for batch: [N, prompt_len]
        let single_prompt =
            Tensor::<B, 2, Int>::from_ints(TensorData::new(prompt_data, [1, prompt_len]), &device);
        let batched_prompt = single_prompt.repeat_dim(0, batch_size);

        let prompt_output = if let Some(ref fused) = fused_weights {
            whisper.forward_decoder_cached_with_cross_attention_fused(
                batched_prompt,
                cache,
                fused,
                true,
            )
        } else if params.use_f16_compute {
            whisper.forward_decoder_cached_with_cross_attention_f16(batched_prompt, cache)
        } else {
            whisper.forward_decoder_cached_with_cross_attention(batched_prompt, cache)
        };

        let mut cache = prompt_output.cache;
        // Extract last-position logits for each region: [N, vocab_size]
        let last_pos = prompt_len - 1;
        let batched_logits: Tensor<B, 2> = prompt_output
            .logits
            .slice([0..batch_size, last_pos..last_pos + 1])
            .reshape([batch_size, vocab_size]);

        // Pull all logits to CPU at once
        let all_logits_data = batched_logits.into_data().convert::<f32>();
        let all_logits_flat = all_logits_data.to_vec::<f32>().unwrap();

        // Extract per-region cross-attention from prompt output
        let mut pending_attention: Vec<Vec<f32>> = if params.token_timestamps {
            average_cross_attention_batched(
                &prompt_output.cross_attention_weights,
                prompt_len - 1,
                batch_size,
            )
        } else {
            vec![Vec::new(); batch_size]
        };

        // 5. Per-region decode state
        let mut no_speech_probs: Vec<f32> = Vec::with_capacity(batch_size);
        let mut states: Vec<RegionStateInner> = (0..batch_size)
            .map(|r| {
                let logits_slice = &all_logits_flat[r * vocab_size..(r + 1) * vocab_size];

                // no_speech_prob from first step
                let no_speech_prob = if token_nosp < vocab_size {
                    let max_logit = logits_slice
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    let sum_exp: f32 = logits_slice.iter().map(|&x| (x - max_logit).exp()).sum();
                    ((logits_slice[token_nosp] - max_logit).exp()) / sum_exp
                } else {
                    0.0
                };
                no_speech_probs.push(no_speech_prob);

                RegionStateInner {
                    tokens_out: Vec::new(),
                    token_alignments: Vec::new(),
                    seek_delta: CHUNK_SIZE * 100,
                    result_len: 0,
                    has_ts: false,
                    failed: false,
                    finished: false,
                    next_token: token_eot,
                }
            })
            .collect();

        // Compute first token for each region from prompt logits
        for r in 0..batch_size {
            let logits_slice = &all_logits_flat[r * vocab_size..(r + 1) * vocab_size];
            let token = greedy_decode_step(
                logits_slice,
                vocab_size,
                token_eot,
                token_beg,
                &static_suppress_data,
                &blank_suppress_data,
                params,
                &states[r].tokens_out,
                states[r].has_ts,
                states[r].seek_delta,
                0, // seek
                seek_end_per_region[r],
                delta_min,
                effective_audio_ctx,
            );
            states[r].next_token = token.0;
            states[r]
                .token_alignments
                .push(std::mem::take(&mut pending_attention[r]));
            // Process token
            process_decoded_token(
                &mut states[r],
                token,
                token_eot,
                token_beg,
                params,
                0, // seek
                seek_end_per_region[r],
                delta_min,
                n_max,
                0, // step index
            );
        }

        // 6. Autoregressive decode loop
        for step in 1..n_max {
            let n_active = states.iter().filter(|s| !s.finished).count();
            if n_active == 0 {
                break;
            }

            // Build next token tensor [N, 1]
            let next_tokens: Vec<u32> = states
                .iter()
                .map(|s| {
                    if s.finished {
                        token_eot as u32
                    } else {
                        s.next_token as u32
                    }
                })
                .collect();
            let token_tensor = Tensor::<B, 2, Int>::from_ints(
                TensorData::new(next_tokens, [batch_size, 1]),
                &device,
            );

            // Batched forward pass
            let step_output = if let Some(ref fused) = fused_weights {
                whisper.forward_decoder_cached_with_cross_attention_fused(
                    token_tensor,
                    cache,
                    fused,
                    true,
                )
            } else if params.use_f16_compute {
                whisper.forward_decoder_cached_with_cross_attention_f16(token_tensor, cache)
            } else {
                whisper.forward_decoder_cached_with_cross_attention(token_tensor, cache)
            };

            cache = step_output.cache;

            // Extract cross-attention for token timestamps
            pending_attention = if params.token_timestamps {
                average_cross_attention_batched(
                    &step_output.cross_attention_weights,
                    0, // single-token step → position 0
                    batch_size,
                )
            } else {
                vec![Vec::new(); batch_size]
            };

            // Pull logits to CPU: [N, 1, vocab_size] → [N * vocab_size]
            let logits_2d: Tensor<B, 2> = step_output.logits.reshape([batch_size, vocab_size]);
            let logits_data = logits_2d.into_data().convert::<f32>();
            let logits_flat = logits_data.to_vec::<f32>().unwrap();

            // Per-region decode on CPU
            for r in 0..batch_size {
                if states[r].finished {
                    continue;
                }

                let logits_slice = &logits_flat[r * vocab_size..(r + 1) * vocab_size];
                let token = greedy_decode_step(
                    logits_slice,
                    vocab_size,
                    token_eot,
                    token_beg,
                    &static_suppress_data,
                    &blank_suppress_data,
                    params,
                    &states[r].tokens_out,
                    states[r].has_ts,
                    states[r].seek_delta,
                    0,
                    seek_end_per_region[r],
                    delta_min,
                    effective_audio_ctx,
                );
                states[r].next_token = token.0;
                states[r]
                    .token_alignments
                    .push(std::mem::take(&mut pending_attention[r]));
                process_decoded_token(
                    &mut states[r],
                    token,
                    token_eot,
                    token_beg,
                    params,
                    0,
                    seek_end_per_region[r],
                    delta_min,
                    n_max,
                    step,
                );
            }
        }

        // 7. Build TranscriptionResult per region, with quality fallback
        for (r, state) in states.into_iter().enumerate() {
            let no_speech_prob = no_speech_probs[r];
            let avg_lp = compute_avg_logprob(&state.tokens_out, state.result_len);
            let entropy =
                compute_entropy(&state.tokens_out[..state.result_len.min(state.tokens_out.len())]);

            // Quality check: fall back to sequential transcribe() with
            // temperature fallback for degenerate outputs.
            let needs_fallback = state.failed
                || (avg_lp < params.logprob_thold as f64
                    && no_speech_prob < params.no_speech_thold)
                || (state.result_len > 32 && entropy < params.entropy_thold as f64);

            let out_idx = batch_indices[r];
            if needs_fallback && params.temperature_inc > 0.0 {
                all_results[out_idx] = transcribe(
                    whisper,
                    bpe,
                    batch_regions[r],
                    sample_rate,
                    params,
                    None::<fn(usize, usize) -> bool>,
                )?;
                continue;
            }

            let seek = 0usize;
            let seek_end = seek_end_per_region[r];
            let is_no_speech =
                no_speech_prob > params.no_speech_thold && avg_lp < params.logprob_thold as f64;

            let result_end = state.result_len.min(state.tokens_out.len());
            let tokens_cur: Vec<(usize, f64, usize)> = state.tokens_out[..result_end].to_vec();
            let alignments_cur: Vec<Vec<f32>> =
                state.token_alignments[..result_end.min(state.token_alignments.len())].to_vec();
            let seek_delta = state.seek_delta;

            let mut segments = Vec::new();
            if !tokens_cur.is_empty() && !is_no_speech {
                build_segments_from_tokens(
                    bpe,
                    &tokens_cur,
                    &alignments_cur,
                    token_eot,
                    token_beg,
                    seek,
                    seek_end,
                    delta_min,
                    seek_delta,
                    no_speech_prob,
                    params,
                    &mut segments,
                );
            }

            let text = segments.iter().map(|s| s.text.as_str()).collect::<String>();
            all_results[out_idx] = TranscriptionResult { segments, text };
        }

        completed_regions += batch_indices.len();
        if let Some(callback) = progress_callback.as_mut() {
            if !callback(completed_regions, total_regions) {
                return Ok(all_results);
            }
        }
    }

    Ok(all_results)
}

/// Apply greedy logit masking and return (token_id, log_prob, tid).
/// Mirrors the CPU-side logic from decode_segment_candidate but operates on a
/// pre-pulled f32 logits slice (no GPU ops).
fn greedy_decode_step(
    logits_raw: &[f32],
    vocab_size: usize,
    token_eot: usize,
    token_beg: usize,
    static_suppress: &[f32],
    blank_suppress: &[f32],
    params: &WhisperParams,
    tokens_out: &[(usize, f64, usize)],
    has_ts: bool,
    seek_delta: usize,
    _seek: usize,
    _seek_end: usize,
    _delta_min: usize,
    n_audio_ctx: usize,
) -> (usize, f64, usize) {
    let mut lp: Vec<f32> = logits_raw.to_vec();
    let is_initial = tokens_out.is_empty();

    // Static suppression (special tokens + nst + token_not)
    for i in 0..vocab_size.min(lp.len()) {
        lp[i] += static_suppress[i];
    }

    // Suppress blank on initial step
    if params.suppress_blank && is_initial {
        for i in 0..vocab_size.min(lp.len()) {
            lp[i] += blank_suppress[i];
        }
    }

    // No-timestamps: suppress all timestamp tokens
    if params.no_timestamps {
        for i in token_beg..vocab_size.min(lp.len()) {
            lp[i] = f32::NEG_INFINITY;
        }
    }

    // Timestamp constraints
    if !params.no_timestamps {
        let last_was_ts = !tokens_out.is_empty() && tokens_out.last().unwrap().0 >= token_beg;
        let penult_was_ts = tokens_out.len() < 2 || tokens_out[tokens_out.len() - 2].0 >= token_beg;

        if last_was_ts {
            if penult_was_ts {
                // Two consecutive timestamps → suppress all timestamps
                for i in token_beg..vocab_size.min(lp.len()) {
                    lp[i] = f32::NEG_INFINITY;
                }
            } else {
                // Force timestamp or EOT
                for i in 0..token_eot.min(lp.len()) {
                    lp[i] = f32::NEG_INFINITY;
                }
            }
        }

        if is_initial && params.max_initial_ts > 0.0 {
            let precision = CHUNK_SIZE as f32 / n_audio_ctx as f32;
            let tid0 = (params.max_initial_ts / precision).round() as usize;
            let cutoff = (token_beg + tid0 + 1).min(vocab_size);
            for i in cutoff..vocab_size.min(lp.len()) {
                lp[i] = f32::NEG_INFINITY;
            }
        }

        if has_ts {
            let tid0 = seek_delta / 2;
            let end = (token_beg + tid0).min(vocab_size);
            for i in token_beg..end.min(lp.len()) {
                lp[i] = f32::NEG_INFINITY;
            }
        }
    }

    // Log softmax
    let max_lp = lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = lp.iter().map(|&x| (x - max_lp).exp()).sum();
    let log_sum = max_lp + sum_exp.ln();
    for v in lp.iter_mut() {
        *v -= log_sum;
    }

    // Timestamp vs text logprob comparison
    if !params.no_timestamps {
        let ts_slice = &lp[token_beg..vocab_size];
        let ts_max = ts_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ts_logprob = ts_max
            + ts_slice
                .iter()
                .map(|&x| (x - ts_max).exp())
                .sum::<f32>()
                .ln();
        let max_text_logprob = lp[..token_beg]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        if ts_logprob > max_text_logprob {
            for v in lp[..token_beg].iter_mut() {
                *v = f32::NEG_INFINITY;
            }
        }
    }

    // tid: argmax over timestamp range
    let tid = if token_beg < vocab_size {
        let (offset, _) = lp[token_beg..vocab_size]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        token_beg + offset
    } else {
        token_beg
    };

    // Greedy argmax
    let sampled_id = lp
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let final_tid = if sampled_id >= token_beg {
        sampled_id
    } else {
        tid
    };
    let log_prob = lp[sampled_id] as f64;

    (sampled_id, log_prob, final_tid)
}

/// Process a decoded token: update region state, check stopping conditions.
fn process_decoded_token(
    rs: &mut RegionStateInner,
    token: (usize, f64, usize),
    token_eot: usize,
    token_beg: usize,
    params: &WhisperParams,
    seek: usize,
    seek_end: usize,
    delta_min: usize,
    n_max: usize,
    step: usize,
) {
    let (sampled_id, log_prob, final_tid) = token;

    rs.tokens_out.push((sampled_id, log_prob, final_tid));

    if sampled_id > token_beg {
        let seek_delta_new = 2 * (sampled_id - token_beg);
        if rs.has_ts && rs.seek_delta > seek_delta_new && rs.result_len < step {
            rs.failed = true;
            rs.finished = true;
            return;
        }
        rs.seek_delta = seek_delta_new;
        rs.result_len = step + 1;
        rs.has_ts = true;
    }

    if sampled_id == token_eot
        || (params.max_tokens > 0 && step >= params.max_tokens)
        || (rs.has_ts && seek + rs.seek_delta + delta_min >= seek_end)
    {
        if rs.result_len == 0 && !params.no_timestamps {
            if seek + rs.seek_delta + delta_min >= seek_end {
                rs.result_len = step + 1;
            } else {
                rs.failed = true;
            }
        }
        if params.single_segment || params.no_timestamps {
            rs.result_len = step + 1;
            rs.seek_delta = CHUNK_SIZE * 100;
        }
        rs.finished = true;
        return;
    }

    if step == n_max - 1 && (rs.result_len == 0 || rs.seek_delta < CHUNK_SIZE * 100 / 2) {
        rs.failed = true;
        rs.finished = true;
    }
}

/// Per-region decode state for batched transcription.
struct RegionStateInner {
    tokens_out: Vec<(usize, f64, usize)>,
    token_alignments: Vec<Vec<f32>>,
    seek_delta: usize,
    result_len: usize,
    has_ts: bool,
    failed: bool,
    finished: bool,
    next_token: usize,
}

fn compute_avg_logprob(tokens: &[(usize, f64, usize)], result_len: usize) -> f64 {
    let n = result_len.max(1);
    let sum: f64 = tokens.iter().take(result_len).map(|t| t.1).sum();
    sum / n as f64
}

/// Build TranscriptionSegment list from decoded tokens (shared between sequential and batched).
fn build_segments_from_tokens(
    bpe: &Gpt2Tokenizer,
    tokens_cur: &[(usize, f64, usize)],
    token_alignments: &[Vec<f32>],
    token_eot: usize,
    token_beg: usize,
    seek: usize,
    _seek_end: usize,
    _delta_min: usize,
    _seek_delta: usize,
    no_speech_prob: f32,
    params: &WhisperParams,
    segments: &mut Vec<TranscriptionSegment>,
) {
    let token_solm = bpe.special_token(SpecialToken::StartofLM);

    let mut t0 = seek as i64 + 2 * (tokens_cur[0].2 as i64 - token_beg as i64);
    let mut text = String::new();
    let mut speaker_turn_next = false;
    let mut seg_token_indices: Vec<usize> = Vec::new();
    let mut i = 0;

    while i < tokens_cur.len() {
        let (token_id, _, tid) = tokens_cur[i];

        if params.print_special || token_id < token_eot {
            if let Ok(decoded) = bpe.decode(&[token_id], true) {
                text += &decoded;
            }
            seg_token_indices.push(i);
        }

        if params.tdrz_enable && token_solm == Some(token_id) {
            speaker_turn_next = true;
        }

        if token_id > token_beg && !params.single_segment {
            let t1 = seek as i64 + 2 * (tid as i64 - token_beg as i64);
            if !text.is_empty() {
                let tok_ts = if params.token_timestamps {
                    compute_token_timestamps_for_segment(
                        bpe,
                        tokens_cur,
                        token_alignments,
                        &seg_token_indices,
                        seek as i64,
                        t0,
                        t1,
                        token_beg,
                        token_eot,
                    )
                } else {
                    Vec::new()
                };

                let seg = TranscriptionSegment {
                    t0,
                    t1,
                    text: text.clone(),
                    no_speech_prob,
                    token_timestamps: tok_ts,
                    speaker_turn_next,
                };

                if params.max_len > 0 && params.token_timestamps {
                    let mut split =
                        split_segment_by_length(&seg, params.max_len, params.split_on_word);
                    segments.append(&mut split);
                } else {
                    segments.push(seg);
                }
            }
            text.clear();
            seg_token_indices.clear();
            speaker_turn_next = false;
            t0 = t1;
        }

        i += 1;
    }

    if !text.is_empty() {
        let t1 = if params.no_timestamps || params.single_segment {
            seek as i64 + (CHUNK_SIZE as i64 * 100)
        } else {
            t0
        };

        let tok_ts = if params.token_timestamps {
            compute_token_timestamps_for_segment(
                bpe,
                tokens_cur,
                token_alignments,
                &seg_token_indices,
                seek as i64,
                t0,
                t1,
                token_beg,
                token_eot,
            )
        } else {
            Vec::new()
        };

        let seg = TranscriptionSegment {
            t0,
            t1,
            text,
            no_speech_prob,
            token_timestamps: tok_ts,
            speaker_turn_next,
        };

        if params.max_len > 0 && params.token_timestamps {
            let mut split = split_segment_by_length(&seg, params.max_len, params.split_on_word);
            segments.append(&mut split);
        } else {
            segments.push(seg);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// Multi-region batched beam search
// ═══════════════════════════════════════════════════════════

/// Per-beam state within a region's beam search.
#[derive(Clone)]
struct BatchedBeamDecoder {
    tokens_out: Vec<(usize, f64, usize)>,
    token_alignments: Vec<Vec<f32>>,
    full_tokens: Vec<usize>,
    sum_logprobs_all: f64,
    seek_delta: usize,
    has_ts: bool,
    failed: bool,
    completed: bool,
    result_len: usize,
}

/// Decode multiple regions' beam searches simultaneously.
///
/// The combined GPU batch dimension is `n_regions × beam_size`, allowing
/// a single forward pass to serve all beams across all regions.
fn decode_regions_beam_batched<B: CustomKernelsBackend>(
    whisper: &Whisper<B>,
    encoder_output: Tensor<B, 3>,
    n_regions: usize,
    prompt: &[usize],
    n_max: usize,
    token_eot: usize,
    token_beg: usize,
    token_not: usize,
    token_nosp: usize,
    space_token_id: Option<usize>,
    suppress_mask: &[bool],
    nst_suppress_ids: &[usize],
    params: &WhisperParams,
    seek_ends: &[usize],
    n_audio_ctx: usize,
    beam_size: usize,
    fused_weights: &Option<FusedDecoderWeights<B>>,
    device: &B::Device,
) -> Vec<SegmentDecodeResult> {
    let total_beams = n_regions * beam_size;
    let delta_min = 10usize;
    let vocab_size = suppress_mask.len();

    // Expand encoder output: each region's encoder output repeated beam_size times
    // encoder_output is [n_regions, enc_seq, enc_dim]
    // We need [n_regions * beam_size, enc_seq, enc_dim]
    let [_, enc_seq, enc_dim] = encoder_output.dims();
    let expanded_encoder = {
        // Repeat each region's output beam_size times via repeat_interleave-like operation
        let mut slices: Vec<Tensor<B, 3>> = Vec::with_capacity(total_beams);
        for r in 0..n_regions {
            let region_enc = encoder_output
                .clone()
                .slice([r..r + 1, 0..enc_seq, 0..enc_dim]);
            for _ in 0..beam_size {
                slices.push(region_enc.clone());
            }
        }
        Tensor::cat(slices, 0)
    };

    // Create cache and process prompt for all beams
    let cache = if params.use_f16_compute {
        whisper.create_decoder_cache_f16(expanded_encoder)
    } else {
        whisper.create_decoder_cache(expanded_encoder)
    };

    let prompt_data: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let single_prompt =
        Tensor::<B, 2, Int>::from_ints(TensorData::new(prompt_data, [1, prompt.len()]), device);
    let batched_prompt = single_prompt.repeat_dim(0, total_beams);

    let prompt_output = if let Some(fused) = fused_weights.as_ref() {
        whisper.forward_decoder_cached_with_cross_attention_fused(
            batched_prompt,
            cache,
            fused,
            true,
        )
    } else if params.use_f16_compute {
        whisper.forward_decoder_cached_with_cross_attention_f16(batched_prompt, cache)
    } else {
        whisper.forward_decoder_cached_with_cross_attention(batched_prompt, cache)
    };

    let prompt_last_pos = prompt.len() - 1;
    let mut batched_cache = prompt_output.cache;

    // Extract per-beam logits: [total_beams, vocab_size]
    let all_logits: Tensor<B, 2> = prompt_output
        .logits
        .slice([0..total_beams, prompt_last_pos..prompt_last_pos + 1])
        .reshape([total_beams, vocab_size]);
    let all_logits_data = all_logits.into_data().convert::<f32>();
    let all_logits_flat = all_logits_data.to_vec::<f32>().unwrap();

    // Compute no_speech_prob per region (from first beam of each region)
    let mut no_speech_probs: Vec<f32> = Vec::with_capacity(n_regions);
    for r in 0..n_regions {
        let beam0_offset = r * beam_size * vocab_size;
        let logits_slice = &all_logits_flat[beam0_offset..beam0_offset + vocab_size];
        let max_logit = logits_slice
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = logits_slice.iter().map(|&x| (x - max_logit).exp()).sum();
        let nsp = if token_nosp < vocab_size {
            ((logits_slice[token_nosp] - max_logit).exp()) / sum_exp
        } else {
            0.0
        };
        no_speech_probs.push(nsp);
    }

    // Extract per-beam cross-attention from prompt output
    let mut beam_pending_attention: Vec<Vec<f32>> = if params.token_timestamps {
        average_cross_attention_batched(
            &prompt_output.cross_attention_weights,
            prompt.len() - 1,
            total_beams,
        )
    } else {
        vec![Vec::new(); total_beams]
    };

    // Initialize per-region beam states
    let initial_tokens = prompt.to_vec();
    let mut region_beams: Vec<Vec<BatchedBeamDecoder>> = (0..n_regions)
        .map(|_| {
            (0..beam_size)
                .map(|_| BatchedBeamDecoder {
                    tokens_out: Vec::new(),
                    token_alignments: Vec::new(),
                    full_tokens: initial_tokens.clone(),
                    sum_logprobs_all: 0.0,
                    seek_delta: CHUNK_SIZE * 100,
                    has_ts: false,
                    failed: false,
                    completed: false,
                    result_len: 0,
                })
                .collect()
        })
        .collect();

    // Store pending logits per beam (CPU-side)
    let mut pending_logits: Vec<Vec<f32>> = (0..total_beams)
        .map(|i| {
            let offset = i * vocab_size;
            all_logits_flat[offset..offset + vocab_size].to_vec()
        })
        .collect();

    // Pre-build suppression data
    let static_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            let is_suppressed = suppress_mask.get(id).copied().unwrap_or(false)
                || nst_suppress_ids.contains(&id)
                || id == token_not;
            if is_suppressed {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let blank_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            if id == token_eot || space_token_id == Some(id) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();

    for step in 0..n_max {
        // Per-region beam management on CPU
        let mut global_reorder: Vec<i32> = (0..total_beams as i32).collect();
        let mut global_next_tokens: Vec<u32> = vec![token_eot as u32; total_beams];
        let mut any_active = false;

        for r in 0..n_regions {
            let beams = &mut region_beams[r];
            let seek_end = seek_ends[r];
            let base = r * beam_size;

            // Check if all beams done
            if beams.iter().all(|b| b.completed || b.failed) {
                for b in 0..beam_size {
                    global_next_tokens[base + b] = token_eot as u32;
                }
                continue;
            }
            any_active = true;

            // Generate candidates from each active beam
            #[derive(Clone)]
            #[allow(dead_code)]
            struct Candidate {
                beam_idx: usize,
                token_id: usize,
                log_prob: f64,
                tid: usize,
                sum_logprobs_all: f64,
                full_tokens: Vec<usize>,
                tokens_out: Vec<(usize, f64, usize)>,
                token_alignments: Vec<Vec<f32>>,
                seek_delta: usize,
                has_ts: bool,
            }

            let mut candidates: Vec<Candidate> = Vec::new();

            for b in 0..beam_size {
                if beams[b].completed || beams[b].failed {
                    continue;
                }

                let global_idx = base + b;
                let logits_raw = &pending_logits[global_idx];

                // Apply masking (CPU version of beam.rs logic)
                let mut lp = logits_raw.clone();
                let is_initial = beams[b].tokens_out.is_empty();

                for i in 0..vocab_size {
                    lp[i] += static_suppress_data[i];
                }

                if params.suppress_blank && is_initial {
                    for i in 0..vocab_size {
                        lp[i] += blank_suppress_data[i];
                    }
                }

                if params.no_timestamps {
                    for i in token_beg..vocab_size {
                        lp[i] = f32::NEG_INFINITY;
                    }
                }

                if !params.no_timestamps {
                    let decoded = &beams[b].tokens_out;
                    let last_was_ts = !decoded.is_empty() && decoded.last().unwrap().0 >= token_beg;
                    let penult_was_ts =
                        decoded.len() < 2 || decoded[decoded.len() - 2].0 >= token_beg;

                    if last_was_ts {
                        if penult_was_ts {
                            for i in token_beg..vocab_size {
                                lp[i] = f32::NEG_INFINITY;
                            }
                        } else {
                            for i in 0..token_eot {
                                lp[i] = f32::NEG_INFINITY;
                            }
                        }
                    }

                    if is_initial && params.max_initial_ts > 0.0 {
                        let precision = CHUNK_SIZE as f32 / n_audio_ctx as f32;
                        let tid0 = (params.max_initial_ts / precision).round() as usize;
                        let cutoff = (token_beg + tid0 + 1).min(vocab_size);
                        for i in cutoff..vocab_size {
                            lp[i] = f32::NEG_INFINITY;
                        }
                    }

                    if beams[b].has_ts {
                        let tid0 = beams[b].seek_delta / 2;
                        let end = (token_beg + tid0).min(vocab_size);
                        for i in token_beg..end {
                            lp[i] = f32::NEG_INFINITY;
                        }
                    }
                }

                // Log softmax
                let max_lp = lp.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = lp.iter().map(|&x| (x - max_lp).exp()).sum();
                let log_sum = max_lp + sum_exp.ln();
                for v in lp.iter_mut() {
                    *v -= log_sum;
                }

                // Timestamp vs text logprob comparison
                if !params.no_timestamps {
                    let ts_slice = &lp[token_beg..vocab_size];
                    let ts_max = ts_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let ts_lp = ts_max
                        + ts_slice
                            .iter()
                            .map(|&x| (x - ts_max).exp())
                            .sum::<f32>()
                            .ln();
                    let max_text_lp = lp[..token_beg]
                        .iter()
                        .cloned()
                        .fold(f32::NEG_INFINITY, f32::max);
                    if ts_lp > max_text_lp {
                        for v in lp[..token_beg].iter_mut() {
                            *v = f32::NEG_INFINITY;
                        }
                    }
                }

                let tid = if token_beg < vocab_size {
                    let (offset, _) = lp[token_beg..vocab_size]
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .unwrap();
                    token_beg + offset
                } else {
                    token_beg
                };

                // Top-k candidates
                let top_k = beam_size.min(vocab_size).max(1);
                let mut indexed: Vec<(usize, f32)> = lp.iter().cloned().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed.truncate(top_k);

                for (sampled_id, log_prob_f32) in indexed {
                    let log_prob = log_prob_f32 as f64;
                    let final_tid = if sampled_id >= token_beg {
                        sampled_id
                    } else {
                        tid
                    };

                    let mut new_tokens_out = beams[b].tokens_out.clone();
                    new_tokens_out.push((sampled_id, log_prob, final_tid));
                    let mut new_full_tokens = beams[b].full_tokens.clone();
                    new_full_tokens.push(sampled_id);

                    // Carry alignment history from source beam + current step's attention
                    let mut new_alignments = beams[b].token_alignments.clone();
                    new_alignments.push(beam_pending_attention[global_idx].clone());

                    candidates.push(Candidate {
                        beam_idx: b,
                        token_id: sampled_id,
                        log_prob,
                        tid: final_tid,
                        sum_logprobs_all: beams[b].sum_logprobs_all + log_prob,
                        full_tokens: new_full_tokens,
                        tokens_out: new_tokens_out,
                        token_alignments: new_alignments,
                        seek_delta: beams[b].seek_delta,
                        has_ts: beams[b].has_ts,
                    });
                }
            }

            if candidates.is_empty() {
                continue;
            }

            // Sort by total logprob
            candidates.sort_by(|a, b| b.sum_logprobs_all.partial_cmp(&a.sum_logprobs_all).unwrap());

            // Assign best candidates to beams
            let mut used = 0usize;
            for b in 0..beam_size {
                if beams[b].completed || beams[b].failed {
                    continue;
                }
                if used >= candidates.len() {
                    used = 0;
                }

                let sel = &candidates[used];
                global_reorder[base + b] = (base + sel.beam_idx) as i32;

                beams[b].tokens_out = sel.tokens_out.clone();
                beams[b].token_alignments = sel.token_alignments.clone();
                beams[b].full_tokens = sel.full_tokens.clone();
                beams[b].sum_logprobs_all = sel.sum_logprobs_all;
                beams[b].seek_delta = sel.seek_delta;
                beams[b].has_ts = sel.has_ts;
                beams[b].failed = false;
                beams[b].completed = false;

                used += 1;
                // Skip duplicates
                while used < candidates.len()
                    && step > 0
                    && tokens_equal(&candidates[used].tokens_out, &sel.tokens_out)
                {
                    used += 1;
                }

                // Process the selected token
                let token_id = sel.token_id;
                if token_id > token_beg {
                    let seek_delta_new = 2 * (token_id - token_beg);
                    if beams[b].has_ts
                        && beams[b].seek_delta > seek_delta_new
                        && beams[b].result_len < step
                    {
                        beams[b].failed = true;
                        global_next_tokens[base + b] = token_eot as u32;
                        continue;
                    }
                    beams[b].seek_delta = seek_delta_new;
                    beams[b].result_len = step + 1;
                    beams[b].has_ts = true;
                }

                if token_id == token_eot
                    || (params.max_tokens > 0 && step >= params.max_tokens)
                    || (beams[b].has_ts && beams[b].seek_delta + delta_min >= seek_end)
                {
                    if beams[b].result_len == 0 && !params.no_timestamps {
                        if beams[b].seek_delta + delta_min >= seek_end {
                            beams[b].result_len = step + 1;
                        } else {
                            beams[b].failed = true;
                            global_next_tokens[base + b] = token_eot as u32;
                            continue;
                        }
                    }
                    if params.single_segment || params.no_timestamps {
                        beams[b].result_len = step + 1;
                        beams[b].seek_delta = CHUNK_SIZE * 100;
                    }
                    beams[b].completed = true;
                    global_next_tokens[base + b] = token_eot as u32;
                    continue;
                }

                if step == n_max - 1
                    && (beams[b].result_len == 0 || beams[b].seek_delta < CHUNK_SIZE * 100 / 2)
                {
                    beams[b].failed = true;
                    global_next_tokens[base + b] = token_eot as u32;
                    continue;
                }

                global_next_tokens[base + b] = token_id as u32;
            }
        }

        if !any_active {
            break;
        }

        // Reorder cache and run forward pass
        let reorder_tensor =
            Tensor::<B, 1, Int>::from_ints(TensorData::new(global_reorder, [total_beams]), device);
        batched_cache = batched_cache.reorder_beams(reorder_tensor);

        let token_tensor = Tensor::<B, 2, Int>::from_ints(
            TensorData::new(global_next_tokens, [total_beams, 1]),
            device,
        );

        let step_output = if let Some(fused) = fused_weights.as_ref() {
            whisper.forward_decoder_cached_with_cross_attention_fused(
                token_tensor,
                batched_cache,
                fused,
                true,
            )
        } else if params.use_f16_compute {
            whisper.forward_decoder_cached_with_cross_attention_f16(token_tensor, batched_cache)
        } else {
            whisper.forward_decoder_cached_with_cross_attention(token_tensor, batched_cache)
        };

        batched_cache = step_output.cache;

        // Extract per-beam cross-attention for token timestamps
        beam_pending_attention = if params.token_timestamps {
            average_cross_attention_batched(
                &step_output.cross_attention_weights,
                0, // single-token step → position 0
                total_beams,
            )
        } else {
            vec![Vec::new(); total_beams]
        };

        // Extract per-beam logits
        let logits_2d: Tensor<B, 2> = step_output.logits.reshape([total_beams, vocab_size]);
        let logits_data = logits_2d.into_data().convert::<f32>();
        let logits_flat = logits_data.to_vec::<f32>().unwrap();

        for i in 0..total_beams {
            pending_logits[i] = logits_flat[i * vocab_size..(i + 1) * vocab_size].to_vec();
        }
    }

    // Build results per region
    (0..n_regions)
        .map(|r| {
            let beams = &region_beams[r];
            let best = beams
                .iter()
                .filter(|b| !b.failed && b.result_len > 0)
                .max_by(|a, b| {
                    let score_a = sequence_score(
                        a.tokens_out.iter().take(a.result_len).map(|t| t.1).sum(),
                        a.result_len,
                        params.length_penalty,
                    );
                    let score_b = sequence_score(
                        b.tokens_out.iter().take(b.result_len).map(|t| t.1).sum(),
                        b.result_len,
                        params.length_penalty,
                    );
                    score_a.partial_cmp(&score_b).unwrap()
                });

            if let Some(best) = best {
                let tokens: Vec<(usize, f64, usize)> = best.tokens_out[..best.result_len].to_vec();
                let alignments: Vec<Vec<f32>> = best.token_alignments
                    [..best.result_len.min(best.token_alignments.len())]
                    .to_vec();
                let sum_logprobs: f64 = tokens.iter().map(|t| t.1).sum();
                let avg_logprobs = sum_logprobs / best.result_len.max(1) as f64;
                let entropy = compute_entropy(&tokens);

                SegmentDecodeResult {
                    tokens,
                    token_alignments: alignments,
                    seek_delta: best.seek_delta,
                    result_len: best.result_len,
                    sum_logprobs,
                    avg_logprobs,
                    entropy,
                    no_speech_prob: no_speech_probs[r],
                    failed: false,
                }
            } else {
                SegmentDecodeResult {
                    tokens: Vec::new(),
                    token_alignments: Vec::new(),
                    seek_delta: CHUNK_SIZE * 100,
                    result_len: 0,
                    sum_logprobs: f64::NEG_INFINITY,
                    avg_logprobs: f64::NEG_INFINITY,
                    entropy: 0.0,
                    no_speech_prob: no_speech_probs[r],
                    failed: true,
                }
            }
        })
        .collect()
}

fn tokens_equal(a: &[(usize, f64, usize)], b: &[(usize, f64, usize)]) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.0 == y.0)
}

// ═══════════════════
// Decode one segment
// ═══════════════════

#[derive(Clone)]
pub(crate) struct SegmentDecodeResult {
    /// Each token: (id, logprob, tid) where tid = best timestamp token at that step
    pub(crate) tokens: Vec<(usize, f64, usize)>,
    pub(crate) token_alignments: Vec<Vec<f32>>,
    pub(crate) seek_delta: usize,
    pub(crate) result_len: usize,
    pub(crate) sum_logprobs: f64,
    pub(crate) avg_logprobs: f64,
    pub(crate) entropy: f64,
    pub(crate) no_speech_prob: f32,
    pub(crate) failed: bool,
}

fn sequence_score_from_result(result: &SegmentDecodeResult, length_penalty: f32) -> f64 {
    sequence_score(result.sum_logprobs, result.result_len, length_penalty)
}

fn decode_segment<B: CustomKernelsBackend>(
    whisper: &Whisper<B>,
    _bpe: &Gpt2Tokenizer,
    encoder_output: &Tensor<B, 3>,
    prompt: &[usize],
    temperature: f32,
    n_max: usize,
    _vocab_size: usize,
    token_eot: usize,
    token_beg: usize,
    token_not: usize,
    token_nosp: usize,
    space_token_id: Option<usize>,
    suppress_mask: &[bool],
    nst_suppress_ids: &[usize],
    params: &WhisperParams,
    seek: usize,
    seek_end: usize,
    delta_min: usize,
    n_audio_ctx: usize,
    beam_size: usize,
    beam_rngs: &mut [MT19937],
    device: &B::Device,
) -> SegmentDecodeResult {
    if matches!(params.strategy, SamplingStrategy::BeamSearch { .. })
        && temperature <= 0.0
        && beam_size > 1
    {
        return decode_segment_beam(
            whisper,
            encoder_output,
            prompt,
            temperature,
            n_max,
            token_eot,
            token_beg,
            token_not,
            token_nosp,
            space_token_id,
            suppress_mask,
            nst_suppress_ids,
            params,
            seek,
            seek_end,
            delta_min,
            n_audio_ctx,
            beam_size,
            device,
        );
    }

    let use_sampling = temperature > 0.0;
    let mut best_result: Option<SegmentDecodeResult> = None;
    let mut fallback_result: Option<SegmentDecodeResult> = None;

    for decoder_index in 0..beam_size.max(1) {
        let result = decode_segment_candidate(
            whisper,
            encoder_output,
            prompt,
            temperature,
            n_max,
            token_eot,
            token_beg,
            token_not,
            token_nosp,
            space_token_id,
            suppress_mask,
            nst_suppress_ids,
            params,
            seek,
            seek_end,
            delta_min,
            n_audio_ctx,
            use_sampling,
            if use_sampling {
                Some(&mut beam_rngs[decoder_index])
            } else {
                None
            },
            device,
        );

        if fallback_result.is_none() {
            fallback_result = Some(result.clone());
        }

        if result.failed {
            if let Some(current) = &fallback_result {
                if sequence_score_from_result(&result, params.length_penalty)
                    > sequence_score_from_result(current, params.length_penalty)
                {
                    fallback_result = Some(result);
                }
            }
            continue;
        }

        let should_replace = match &best_result {
            Some(current) => {
                sequence_score_from_result(&result, params.length_penalty)
                    > sequence_score_from_result(current, params.length_penalty)
            }
            None => true,
        };

        if should_replace {
            best_result = Some(result);
        }
    }

    best_result.or(fallback_result).unwrap()
}

fn decode_segment_candidate<B: CustomKernelsBackend>(
    whisper: &Whisper<B>,
    encoder_output: &Tensor<B, 3>,
    prompt: &[usize],
    temperature: f32,
    n_max: usize,
    token_eot: usize,
    token_beg: usize,
    token_not: usize,
    token_nosp: usize,
    space_token_id: Option<usize>,
    suppress_mask: &[bool],
    nst_suppress_ids: &[usize],
    params: &WhisperParams,
    seek: usize,
    seek_end: usize,
    delta_min: usize,
    n_audio_ctx: usize,
    use_sampling: bool,
    rng: Option<&mut MT19937>,
    device: &B::Device,
) -> SegmentDecodeResult {
    let mut tokens_out: Vec<(usize, f64, usize)> = Vec::new();
    let mut token_alignments: Vec<Vec<f32>> = Vec::new();
    let mut seek_delta = CHUNK_SIZE * 100;
    let mut result_len = 0usize;
    let mut has_ts = false;
    let mut failed = false;
    let mut no_speech_prob = 0.0f32;

    let mut full_tokens: Vec<usize> = prompt.to_vec();
    let mut rng = rng;

    let fused_weights = if params.use_f16_compute {
        Some(whisper.build_fused_decoder_weights())
    } else {
        None
    };

    // Initialize KV cache: precomputes cross-attention K/V from encoder output
    // and processes the prompt in one shot.
    let prompt_token_data: Vec<u32> = full_tokens.iter().map(|&t| t as u32).collect();
    let prompt_tensor = Tensor::from_ints(
        TensorData::new(prompt_token_data, [1, full_tokens.len()]),
        device,
    );
    let prompt_output = if params.use_f16_compute {
        whisper.forward_decoder_cached_with_cross_attention_f16(
            prompt_tensor,
            whisper.create_decoder_cache_f16(encoder_output.clone()),
        )
    } else {
        whisper.forward_decoder_cached_with_cross_attention(
            prompt_tensor,
            whisper.create_decoder_cache(encoder_output.clone()),
        )
    };
    let prompt_last_pos = full_tokens.len() - 1;
    let mut cache = prompt_output.cache;
    let mut pending_logits: Tensor<B, 1> = prompt_output
        .logits
        .slice([0..1, prompt_last_pos..prompt_last_pos + 1])
        .flatten::<1>(0, 2);
    let mut pending_attention: Vec<f32> = if params.token_timestamps {
        average_cross_attention_for_token(&prompt_output.cross_attention_weights, prompt_last_pos)
    } else {
        Vec::new()
    };

    // Pre-build GPU tensors for logit masking (avoids per-step CPU loops)
    let vocab_size = suppress_mask.len();
    let static_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            let is_suppressed = suppress_mask.get(id).copied().unwrap_or(false)
                || nst_suppress_ids.contains(&id)
                || id == token_not;
            if is_suppressed {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let gpu_static_suppress: Tensor<B, 1> =
        Tensor::from_floats(static_suppress_data.as_slice(), device);
    let blank_suppress_data: Vec<f32> = (0..vocab_size)
        .map(|id| {
            if id == token_eot || space_token_id == Some(id) {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        })
        .collect();
    let gpu_blank_suppress: Tensor<B, 1> =
        Tensor::from_floats(blank_suppress_data.as_slice(), device);
    let gpu_indices: Tensor<B, 1, Int> = Tensor::arange(0..vocab_size as i64, device);

    for i in 0..n_max {
        // Use cached logits from previous step (or prompt output)
        let logits = pending_logits.clone();
        let attention = pending_attention.clone();

        if i == 0 {
            let probs = softmax(logits.clone(), 0);
            if token_nosp < vocab_size {
                no_speech_prob = probs
                    .slice([token_nosp..token_nosp + 1])
                    .into_scalar()
                    .elem();
            }
        }

        let is_initial = tokens_out.is_empty();

        // Temperature scaling on GPU
        let logits = if temperature > 0.0 {
            logits / (temperature as f64)
        } else {
            logits
        };

        // Apply pre-built static suppression mask on GPU
        // (covers suppress_mask, nst_suppress_ids, token_not)
        let logits = logits + gpu_static_suppress.clone();

        // suppress_blank on initial step
        let logits = if params.suppress_blank && is_initial {
            logits + gpu_blank_suppress.clone()
        } else {
            logits
        };

        // No-timestamps: suppress all timestamp tokens on GPU
        let logits = if params.no_timestamps {
            let mask = gpu_indices.clone().greater_equal_elem(token_beg as i64);
            logits.mask_fill(mask, f32::NEG_INFINITY)
        } else {
            logits
        };

        // Timestamp constraints on GPU
        let logits = if !params.no_timestamps {
            let last_was_ts = !tokens_out.is_empty() && tokens_out.last().unwrap().0 >= token_beg;
            let penult_was_ts =
                tokens_out.len() < 2 || tokens_out[tokens_out.len() - 2].0 >= token_beg;

            let logits = if last_was_ts {
                if penult_was_ts {
                    let mask = gpu_indices.clone().greater_equal_elem(token_beg as i64);
                    logits.mask_fill(mask, f32::NEG_INFINITY)
                } else {
                    let mask = gpu_indices.clone().lower_elem(token_eot as i64);
                    logits.mask_fill(mask, f32::NEG_INFINITY)
                }
            } else {
                logits
            };

            let logits = if is_initial && params.max_initial_ts > 0.0 {
                let precision = CHUNK_SIZE as f32 / n_audio_ctx as f32;
                let tid0 = (params.max_initial_ts / precision).round() as usize;
                let cutoff = (token_beg + tid0 + 1).min(vocab_size);
                if cutoff < vocab_size {
                    let mask = gpu_indices.clone().greater_equal_elem(cutoff as i64);
                    logits.mask_fill(mask, f32::NEG_INFINITY)
                } else {
                    logits
                }
            } else {
                logits
            };

            let logits = if has_ts {
                let tid0 = seek_delta / 2;
                let end = (token_beg + tid0).min(vocab_size);
                if end > token_beg {
                    let mask = (gpu_indices
                        .clone()
                        .greater_equal_elem(token_beg as i64)
                        .float()
                        * gpu_indices.clone().lower_elem(end as i64).float())
                    .greater_equal_elem(1.0);
                    logits.mask_fill(mask, f32::NEG_INFINITY)
                } else {
                    logits
                }
            } else {
                logits
            };

            logits
        } else {
            logits
        };

        // Log softmax on GPU, then pull to CPU once (single GPU sync instead of ~5 per step)
        let logprobs_gpu = log_softmax(logits, 0);
        let mut lp: Vec<f32> = logprobs_gpu
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .unwrap();

        // Timestamp vs text logprob comparison on CPU
        if !params.no_timestamps {
            let ts_slice = &lp[token_beg..vocab_size];
            let ts_max = ts_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ts_logprob = ts_max
                + ts_slice
                    .iter()
                    .map(|&x| (x - ts_max).exp())
                    .sum::<f32>()
                    .ln();
            let max_text_logprob = lp[..token_beg]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);

            if ts_logprob > max_text_logprob {
                for v in lp[..token_beg].iter_mut() {
                    *v = f32::NEG_INFINITY;
                }
            }
        }

        // tid: argmax over timestamp range on CPU
        let tid = if token_beg < vocab_size {
            let (offset, _) = lp[token_beg..vocab_size]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            token_beg + offset
        } else {
            token_beg
        };

        // Token selection on CPU
        let sampled_id = if use_sampling {
            let probs: Vec<f32> = lp.iter().map(|&x| x.exp()).collect();
            sample_from_probs(&probs, rng.as_deref_mut().expect("sampling requires RNG"))
        } else {
            lp.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };

        let final_tid = if sampled_id >= token_beg {
            sampled_id
        } else {
            tid
        };

        let log_prob: f64 = lp[sampled_id] as f64;
        tokens_out.push((sampled_id, log_prob, final_tid));
        token_alignments.push(attention);
        full_tokens.push(sampled_id);

        if sampled_id > token_beg {
            let seek_delta_new = 2 * (sampled_id - token_beg);

            if has_ts && seek_delta > seek_delta_new && result_len < i {
                failed = true;
                break;
            }

            seek_delta = seek_delta_new;
            result_len = i + 1;
            has_ts = true;
        }

        if sampled_id == token_eot
            || (params.max_tokens > 0 && i >= params.max_tokens)
            || (has_ts && seek + seek_delta + delta_min >= seek_end)
        {
            if result_len == 0 && !params.no_timestamps {
                if seek + seek_delta + delta_min >= seek_end {
                    result_len = i + 1;
                } else {
                    failed = true;
                    break;
                }
            }

            if params.single_segment || params.no_timestamps {
                result_len = i + 1;
                seek_delta = CHUNK_SIZE * 100;
            }

            break;
        }

        if i == n_max - 1 && (result_len == 0 || seek_delta < CHUNK_SIZE * 100 / 2) {
            failed = true;
            break;
        }

        // Advance KV cache: feed only the new token through the cached decoder
        let next_token_tensor =
            Tensor::from_ints(TensorData::new(vec![sampled_id as u32], [1, 1]), device);
        let step_output = if let Some(ref fused) = fused_weights {
            whisper.forward_decoder_cached_with_cross_attention_fused(
                next_token_tensor,
                cache,
                fused,
                true,
            )
        } else {
            whisper.forward_decoder_cached_with_cross_attention(next_token_tensor, cache)
        };
        cache = step_output.cache;
        pending_logits = step_output.logits.slice([0..1, 0..1]).flatten::<1>(0, 2);
        pending_attention = if params.token_timestamps {
            average_cross_attention_for_token(&step_output.cross_attention_weights, 0)
        } else {
            Vec::new()
        };
    }

    let mut sum_logprobs_subset = 0.0;
    for &(_, logprob, _) in tokens_out.iter().take(result_len) {
        sum_logprobs_subset += logprob;
    }
    let n_tokens = result_len.max(1);
    let avg_logprobs = sum_logprobs_subset / n_tokens as f64;
    let entropy = compute_entropy(&tokens_out[..result_len]);

    SegmentDecodeResult {
        tokens: tokens_out,
        token_alignments,
        seek_delta,
        result_len,
        sum_logprobs: sum_logprobs_subset,
        avg_logprobs,
        entropy,
        no_speech_prob,
        failed,
    }
}

/// Compute sequence score with optional length penalty (per whisper.cpp)
pub(crate) fn sequence_score(sum_logprobs: f64, result_len: usize, length_penalty: f32) -> f64 {
    if result_len == 0 {
        return f64::NEG_INFINITY;
    }
    let penalty = if length_penalty > 0.0 {
        ((5.0 + result_len as f64) / 6.0).powf(length_penalty as f64)
    } else {
        result_len as f64
    };
    sum_logprobs / penalty
}

fn sample_from_probs(probs: &[f32], rng: &mut MT19937) -> usize {
    let total: f64 = probs.iter().map(|&prob| prob.max(0.0) as f64).sum();

    if total <= 0.0 {
        return probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);
    }

    let mut target = mt19937::gen_res53(rng) * total;
    let mut last_non_zero = 0usize;

    for (index, &prob) in probs.iter().enumerate() {
        let weight = prob.max(0.0) as f64;
        if weight <= 0.0 {
            continue;
        }

        last_non_zero = index;
        if target < weight {
            return index;
        }
        target -= weight;
    }

    last_non_zero
}

/// Compute token-level timestamps for a segment from decoded token data
fn compute_token_timestamps_for_segment(
    bpe: &Gpt2Tokenizer,
    tokens_cur: &[(usize, f64, usize)],
    token_alignments: &[Vec<f32>],
    seg_indices: &[usize],
    seek: i64,
    t0: i64,
    t1: i64,
    token_beg: usize,
    token_eot: usize,
) -> Vec<TokenTimestamp> {
    let text_indices: Vec<usize> = seg_indices
        .iter()
        .copied()
        .filter(|&idx| {
            let (id, _, _) = tokens_cur[idx];
            id < token_eot
                && id < token_beg
                && idx < token_alignments.len()
                && !token_alignments[idx].is_empty()
        })
        .collect();

    if text_indices.is_empty() {
        return compute_uniform_token_timestamps_for_segment(
            bpe,
            tokens_cur,
            seg_indices,
            t0,
            t1,
            token_beg,
            token_eot,
        );
    }

    let mut centers = Vec::with_capacity(text_indices.len());
    let mut peaks = Vec::with_capacity(text_indices.len());

    for &idx in &text_indices {
        let alignment = &token_alignments[idx];
        let total: f32 = alignment.iter().sum();

        if total <= 0.0 {
            return compute_uniform_token_timestamps_for_segment(
                bpe,
                tokens_cur,
                seg_indices,
                t0,
                t1,
                token_beg,
                token_eot,
            );
        }

        // Use argmax (peak attention position) instead of weighted mean.
        // Weighted mean is dominated by background noise when attention is
        // only moderately focused (e.g. peak 7% vs uniform 0.07%), causing
        // the center of mass to fall near the middle of the context rather
        // than at the actual speech position.
        let (peak_pos, peak) = alignment
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &v)| (i, v))
            .unwrap();

        centers.push(seek as f32 + peak_pos as f32 * 2.0);
        peaks.push(peak);
    }

    for index in 1..centers.len() {
        if centers[index] < centers[index - 1] {
            centers[index] = centers[index - 1];
        }
    }

    let mut boundaries = Vec::with_capacity(centers.len() + 1);
    boundaries.push(t0);
    for index in 0..centers.len().saturating_sub(1) {
        let midpoint = ((centers[index] + centers[index + 1]) * 0.5).round() as i64;
        boundaries.push(midpoint.clamp(t0, t1));
    }
    boundaries.push(t1);

    for index in 1..boundaries.len() {
        if boundaries[index] < boundaries[index - 1] {
            boundaries[index] = boundaries[index - 1];
        }
    }

    text_indices
        .iter()
        .enumerate()
        .map(|(index, &idx)| {
            let (token_id, _, _) = tokens_cur[idx];
            TokenTimestamp {
                token_id,
                text: bpe.decode(&[token_id], true).unwrap_or_default(),
                t0: boundaries[index],
                t1: boundaries[index + 1],
                pt: peaks[index],
            }
        })
        .collect()
}

fn compute_uniform_token_timestamps_for_segment(
    bpe: &Gpt2Tokenizer,
    tokens_cur: &[(usize, f64, usize)],
    seg_indices: &[usize],
    t0: i64,
    t1: i64,
    token_beg: usize,
    token_eot: usize,
) -> Vec<TokenTimestamp> {
    let text_indices: Vec<usize> = seg_indices
        .iter()
        .copied()
        .filter(|&idx| {
            let (id, _, _) = tokens_cur[idx];
            id < token_eot && id < token_beg
        })
        .collect();

    if text_indices.is_empty() {
        return Vec::new();
    }

    let n = text_indices.len() as i64;
    let duration = t1 - t0;
    let step = if n > 0 { duration / n } else { duration };

    text_indices
        .iter()
        .enumerate()
        .map(|(index, &idx)| {
            let (token_id, _, _) = tokens_cur[idx];
            let tok_t0 = t0 + index as i64 * step;
            let tok_t1 = if index as i64 == n - 1 {
                t1
            } else {
                tok_t0 + step
            };
            TokenTimestamp {
                token_id,
                text: bpe.decode(&[token_id], true).unwrap_or_default(),
                t0: tok_t0,
                t1: tok_t1,
                pt: 0.0,
            }
        })
        .collect()
}

/// Extract cross-attention alignment for a single token.
///
/// Uses only the **last decoder layer** and takes the **element-wise max across
/// heads** instead of averaging all layers/heads.  Most attention heads don't
/// track audio-text alignment; averaging them drowns the signal from the few
/// "alignment heads" that do.  The last layer has the strongest alignment signal,
/// and max-over-heads preserves it even when most heads attend elsewhere.
pub(crate) fn average_cross_attention_for_token<B: CustomKernelsBackend>(
    cross_attention_weights: &[Tensor<B, 4>],
    token_index: usize,
) -> Vec<f32> {
    if cross_attention_weights.is_empty() {
        return Vec::new();
    }

    // Use only the last layer (best alignment signal)
    let layer_weights = cross_attention_weights.last().unwrap();
    let [_batch, n_head, n_qctx, n_ctx] = layer_weights.dims();
    if token_index >= n_qctx {
        return vec![0.0f32; n_ctx];
    }

    // [1, n_head, 1, n_ctx] → [n_head, n_ctx], then max across heads → [n_ctx]
    let heads = layer_weights
        .clone()
        .slice([0..1, 0..n_head, token_index..token_index + 1, 0..n_ctx])
        .reshape([n_head, n_ctx]);

    heads
        .max_dim(0)
        .reshape([n_ctx])
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .unwrap()
}

/// Like `average_cross_attention_for_token` but for a batched forward pass.
/// Returns one `Vec<f32>` per batch element, each of length `n_audio_ctx`.
/// Batched version of `average_cross_attention_for_token`.
/// Uses last layer + max-over-heads (see docs on the single-element version).
/// Performs a single GPU→CPU transfer for the whole batch.
fn average_cross_attention_batched<B: CustomKernelsBackend>(
    cross_attention_weights: &[Tensor<B, 4>],
    token_index: usize,
    batch_size: usize,
) -> Vec<Vec<f32>> {
    if cross_attention_weights.is_empty() || batch_size == 0 {
        return vec![Vec::new(); batch_size];
    }

    // Use only the last layer (best alignment signal)
    let layer_weights = cross_attention_weights.last().unwrap();
    let [_b, n_head, n_qctx, n_ctx] = layer_weights.dims();
    if token_index >= n_qctx {
        return vec![vec![0.0f32; n_ctx]; batch_size];
    }

    // [batch, n_head, 1, n_ctx] → [batch, n_head, n_ctx], max over heads → [batch, n_ctx]
    let heads = layer_weights
        .clone()
        .slice([
            0..batch_size,
            0..n_head,
            token_index..token_index + 1,
            0..n_ctx,
        ])
        .reshape([batch_size, n_head, n_ctx]);

    let maxed: Tensor<B, 2> = heads.max_dim(1).reshape([batch_size, n_ctx]);
    let flat = maxed.into_data().convert::<f32>().to_vec::<f32>().unwrap();

    (0..batch_size)
        .map(|b| flat[b * n_ctx..(b + 1) * n_ctx].to_vec())
        .collect()
}

/// Split a segment into multiple sub-segments if it exceeds max_len characters
fn split_segment_by_length(
    seg: &TranscriptionSegment,
    max_len: usize,
    split_on_word: bool,
) -> Vec<TranscriptionSegment> {
    if seg.token_timestamps.is_empty() || seg.text.chars().count() <= max_len {
        return vec![seg.clone()];
    }

    let mut result = Vec::new();
    let mut acc_chars = 0usize;
    let mut current_tokens: Vec<&TokenTimestamp> = Vec::new();
    let mut seg_t0 = seg.t0;

    for tt in &seg.token_timestamps {
        let token_chars = tt.text.chars().count();

        if acc_chars + token_chars > max_len && !current_tokens.is_empty() {
            // Check split_on_word: only split at word boundaries (tokens starting with space)
            if split_on_word && !tt.text.starts_with(' ') {
                current_tokens.push(tt);
                acc_chars += token_chars;
                continue;
            }

            // Create segment from accumulated tokens
            let text: String = current_tokens.iter().map(|t| t.text.as_str()).collect();
            let t1 = tt.t0;
            result.push(TranscriptionSegment {
                t0: seg_t0,
                t1,
                text,
                no_speech_prob: seg.no_speech_prob,
                token_timestamps: current_tokens.iter().map(|t| (*t).clone()).collect(),
                speaker_turn_next: false,
            });

            seg_t0 = t1;
            current_tokens.clear();
            acc_chars = 0;
        }

        current_tokens.push(tt);
        acc_chars += token_chars;
    }

    // Remaining tokens
    if !current_tokens.is_empty() {
        let text: String = current_tokens.iter().map(|t| t.text.as_str()).collect();
        result.push(TranscriptionSegment {
            t0: seg_t0,
            t1: seg.t1,
            text,
            no_speech_prob: seg.no_speech_prob,
            token_timestamps: current_tokens.iter().map(|t| (*t).clone()).collect(),
            speaker_turn_next: seg.speaker_turn_next,
        });
    }

    if result.is_empty() {
        vec![seg.clone()]
    } else {
        result
    }
}

/// Compute entropy of token frequency distribution (last 32 tokens)
pub(crate) fn compute_entropy(tokens: &[(usize, f64, usize)]) -> f64 {
    let start = if tokens.len() > 32 {
        tokens.len() - 32
    } else {
        0
    };
    let window = &tokens[start..];
    if window.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &(id, _, _) in window {
        *counts.entry(id).or_insert(0) += 1;
    }

    let n = window.len() as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / n;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

/// Format 10ms frame count as HH:MM:SS,mmm for SRT
pub fn format_timestamp(frames: i64) -> String {
    let ms = frames * 10;
    let s = ms / 1000;
    let m = s / 60;
    let h = m / 60;
    format!("{:02}:{:02}:{:02},{:03}", h, m % 60, s % 60, ms % 1000)
}
