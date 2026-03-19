use crate::custom_kernels::CustomKernelsBackend;
use crate::model::*;
use crate::transcribe::{
    CHUNK_SIZE, SamplingStrategy, SegmentDecodeResult, WhisperParams,
    average_cross_attention_for_token, compute_entropy, sequence_score,
};
use burn::tensor::TensorData;
use burn::tensor::{
    ElementConversion, Int, Tensor,
    activation::{log_softmax, softmax},
};
use std::f32;

const DEFAULT_BEST_OF: usize = 5;
const DEFAULT_BEAM_SIZE: usize = 5;

#[derive(Clone)]
struct BeamSequence {
    tokens: Vec<(usize, f64, usize)>,
    result_len: usize,
    sum_logprobs_all: f64,
    sum_logprobs: f64,
    avg_logprobs: f64,
    entropy: f64,
    score: f64,
}

#[derive(Clone)]
struct BeamDecoder<B: CustomKernelsBackend> {
    pending_logits: Tensor<B, 1>,
    pending_attention: Vec<f32>,
    full_tokens: Vec<usize>,
    sequence: BeamSequence,
    alignments: Vec<Vec<f32>>,
    seek_delta: usize,
    has_ts: bool,
    failed: bool,
    completed: bool,
}

#[derive(Clone)]
struct BeamCandidate {
    decoder_idx: usize,
    full_tokens: Vec<usize>,
    sequence: BeamSequence,
    alignments: Vec<Vec<f32>>,
    seek_delta: usize,
    has_ts: bool,
}

pub(crate) fn effective_best_of(strategy: &SamplingStrategy) -> usize {
    let best_of = match strategy {
        SamplingStrategy::Greedy { best_of } => *best_of,
        SamplingStrategy::BeamSearch { .. } => -1,
    };

    if best_of > 0 {
        best_of as usize
    } else {
        DEFAULT_BEST_OF
    }
}

pub(crate) fn effective_beam_size(strategy: &SamplingStrategy) -> usize {
    let beam_size = match strategy {
        SamplingStrategy::BeamSearch { beam_size, .. } => *beam_size,
        SamplingStrategy::Greedy { .. } => -1,
    };

    if beam_size > 0 {
        beam_size as usize
    } else {
        DEFAULT_BEAM_SIZE
    }
}

pub(crate) fn decoder_count_for_iteration(params: &WhisperParams, temperature: f32) -> usize {
    let count = match &params.strategy {
        SamplingStrategy::Greedy { .. } => {
            if temperature > 0.0 {
                effective_best_of(&params.strategy)
            } else {
                1
            }
        }
        SamplingStrategy::BeamSearch { .. } => {
            if temperature > 0.0 {
                effective_best_of(&params.strategy)
            } else {
                effective_beam_size(&params.strategy)
            }
        }
    };

    count.max(1)
}

pub(crate) fn decode_segment_beam<B: CustomKernelsBackend>(
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
    beam_size: usize,
    device: &B::Device,
) -> SegmentDecodeResult {
    let mut no_speech_prob = 0.0f32;
    let fused_weights = if params.use_f16_compute {
        Some(whisper.build_fused_decoder_weights())
    } else {
        None
    };
    let initial_tokens = prompt.to_vec();

    let prompt_token_data: Vec<u32> = initial_tokens.iter().map(|&t| t as u32).collect();
    let prompt_tensor = Tensor::from_ints(
        TensorData::new(prompt_token_data, [1, initial_tokens.len()]),
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
    let prompt_last_pos = initial_tokens.len() - 1;
    let prompt_attention = if params.token_timestamps {
        average_cross_attention_for_token(&prompt_output.cross_attention_weights, prompt_last_pos)
    } else {
        Vec::new()
    };
    let prompt_logits: Tensor<B, 1> = prompt_output
        .logits
        .slice([0..1, prompt_last_pos..prompt_last_pos + 1])
        .flatten::<1>(0, 2);

    let probs = softmax(prompt_logits.clone(), 0);
    let prob_size = probs.dims()[0];
    if token_nosp < prob_size {
        no_speech_prob = probs
            .slice([token_nosp..token_nosp + 1])
            .into_scalar()
            .elem();
    }

    // Persistent batched cache: expand prompt cache to beam_size along batch dim.
    // This single cache persists across all beam steps (no per-step stack/unstack).
    let mut batched_cache = DecoderCache::stack(
        (0..beam_size)
            .map(|_| prompt_output.cache.clone())
            .collect(),
    );

    let mut decoders: Vec<BeamDecoder<B>> = (0..beam_size)
        .map(|_| BeamDecoder {
            pending_logits: prompt_logits.clone(),
            pending_attention: prompt_attention.clone(),
            full_tokens: initial_tokens.clone(),
            sequence: BeamSequence {
                tokens: Vec::new(),
                result_len: 0,
                sum_logprobs_all: 0.0,
                sum_logprobs: f64::NEG_INFINITY,
                avg_logprobs: f64::NEG_INFINITY,
                entropy: 0.0,
                score: f64::NEG_INFINITY,
            },
            alignments: Vec::new(),
            seek_delta: CHUNK_SIZE * 100,
            has_ts: false,
            failed: false,
            completed: false,
        })
        .collect();

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

    for step in 0..n_max {
        let mut candidates_per_decoder: Vec<Vec<BeamCandidate>> = vec![Vec::new(); beam_size];

        for (decoder_idx, decoder) in decoders.iter().enumerate() {
            if decoder.completed || decoder.failed {
                continue;
            }

            let attention = decoder.pending_attention.clone();
            let logits = decoder.pending_logits.clone();
            let is_initial = decoder.sequence.tokens.is_empty();

            let logits = if temperature > 0.0 {
                logits / (temperature as f64)
            } else {
                logits
            };

            let logits = logits + gpu_static_suppress.clone();

            let logits = if params.suppress_blank && is_initial {
                logits + gpu_blank_suppress.clone()
            } else {
                logits
            };

            let logits = if params.no_timestamps {
                let mask = gpu_indices.clone().greater_equal_elem(token_beg as i64);
                logits.mask_fill(mask, f32::NEG_INFINITY)
            } else {
                logits
            };

            let logits = if !params.no_timestamps {
                let decoded = &decoder.sequence.tokens;
                let last_was_ts = !decoded.is_empty() && decoded.last().unwrap().0 >= token_beg;
                let penult_was_ts = decoded.len() < 2 || decoded[decoded.len() - 2].0 >= token_beg;

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

                let logits = if decoder.has_ts {
                    let tid0 = decoder.seek_delta / 2;
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

            let logprobs_gpu = log_softmax(logits, 0);
            let mut lp: Vec<f32> = logprobs_gpu
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()
                .unwrap();

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
                    for v in &mut lp[..token_beg] {
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

                let mut candidate = BeamCandidate {
                    decoder_idx,
                    full_tokens: decoder.full_tokens.clone(),
                    sequence: decoder.sequence.clone(),
                    alignments: decoder.alignments.clone(),
                    seek_delta: decoder.seek_delta,
                    has_ts: decoder.has_ts,
                };

                candidate
                    .sequence
                    .tokens
                    .push((sampled_id, log_prob, final_tid));
                candidate.sequence.sum_logprobs_all += log_prob;
                candidate.alignments.push(attention.clone());
                candidate.full_tokens.push(sampled_id);

                candidates_per_decoder[decoder_idx].push(candidate);
            }
        }

        let mut beam_candidates: Vec<BeamCandidate> = Vec::new();
        for decoder_candidates in candidates_per_decoder {
            beam_candidates.extend(decoder_candidates);
        }

        if beam_candidates.is_empty() {
            break;
        }

        beam_candidates.sort_by(|a, b| {
            match b
                .sequence
                .sum_logprobs_all
                .partial_cmp(&a.sequence.sum_logprobs_all)
                .unwrap()
            {
                std::cmp::Ordering::Equal => a.decoder_idx.cmp(&b.decoder_idx),
                ordering => ordering,
            }
        });

        let mut next_decoders = decoders.clone();
        let mut reorder_indices: Vec<i32> = (0..beam_size as i32).collect();
        let mut cur_candidate = 0usize;

        for decoder_idx in 0..next_decoders.len() {
            if next_decoders[decoder_idx].completed || next_decoders[decoder_idx].failed {
                continue;
            }

            if cur_candidate >= beam_candidates.len() {
                cur_candidate = 0;
            }

            let selected = beam_candidates[cur_candidate].clone();
            reorder_indices[decoder_idx] = selected.decoder_idx as i32;
            cur_candidate += 1;

            while cur_candidate < beam_candidates.len()
                && step > 0
                && decoded_tokens_equal(
                    &beam_candidates[cur_candidate].sequence.tokens,
                    &selected.sequence.tokens,
                )
            {
                cur_candidate += 1;
            }

            next_decoders[decoder_idx].pending_logits = Tensor::zeros([1], device);
            next_decoders[decoder_idx].pending_attention.clear();
            next_decoders[decoder_idx].full_tokens = selected.full_tokens;
            next_decoders[decoder_idx].sequence = selected.sequence;
            next_decoders[decoder_idx].alignments = selected.alignments;
            next_decoders[decoder_idx].seek_delta = selected.seek_delta;
            next_decoders[decoder_idx].has_ts = selected.has_ts;
            next_decoders[decoder_idx].failed = false;
            next_decoders[decoder_idx].completed = false;
        }

        // Reorder the persistent batched cache based on beam selection
        let indices_tensor: Tensor<B, 1, Int> =
            Tensor::from_ints(TensorData::new(reorder_indices, [beam_size]), device);

        decoders = next_decoders;

        for decoder in &mut decoders {
            if decoder.completed || decoder.failed {
                continue;
            }

            let token_id = if let Some(&(token_id, _, _)) = decoder.sequence.tokens.last() {
                token_id
            } else {
                decoder.failed = true;
                continue;
            };

            if token_id > token_beg {
                let seek_delta_new = 2 * (token_id - token_beg);
                if decoder.has_ts
                    && decoder.seek_delta > seek_delta_new
                    && decoder.sequence.result_len < step
                {
                    decoder.failed = true;
                    continue;
                }

                decoder.seek_delta = seek_delta_new;
                decoder.sequence.result_len = step + 1;
                decoder.has_ts = true;
            }

            if token_id == token_eot
                || (params.max_tokens > 0 && step >= params.max_tokens)
                || (decoder.has_ts && seek + decoder.seek_delta + delta_min >= seek_end)
            {
                if decoder.sequence.result_len == 0 && !params.no_timestamps {
                    if seek + decoder.seek_delta + delta_min >= seek_end {
                        decoder.sequence.result_len = step + 1;
                    } else {
                        decoder.failed = true;
                        continue;
                    }
                }

                if params.single_segment || params.no_timestamps {
                    decoder.sequence.result_len = step + 1;
                    decoder.seek_delta = CHUNK_SIZE * 100;
                }

                decoder.completed = true;
                continue;
            }

            if step == n_max - 1
                && (decoder.sequence.result_len == 0 || decoder.seek_delta < CHUNK_SIZE * 100 / 2)
            {
                decoder.failed = true;
            }
        }

        if decoders
            .iter()
            .all(|decoder| decoder.completed || decoder.failed)
        {
            break;
        }

        // Reorder batched cache then run forward pass for all beams
        batched_cache = batched_cache.reorder_beams(indices_tensor);

        // Build token tensor for all beams (dummy token 0 for completed/failed)
        let token_ids: Vec<u32> = decoders
            .iter()
            .map(|d| {
                if d.completed || d.failed {
                    0u32
                } else {
                    d.sequence
                        .tokens
                        .last()
                        .map(|(t, _, _)| *t as u32)
                        .unwrap_or(0)
                }
            })
            .collect();
        let batched_tokens: Tensor<B, 2, Int> =
            Tensor::from_ints(TensorData::new(token_ids, [beam_size, 1]), device);

        let batched_output = if let Some(ref fused) = fused_weights {
            whisper.forward_decoder_cached_with_cross_attention_fused(
                batched_tokens,
                batched_cache,
                fused,
                true,
            )
        } else {
            whisper.forward_decoder_cached_with_cross_attention(batched_tokens, batched_cache)
        };

        batched_cache = batched_output.cache;

        // Extract per-beam logits and attention (skip completed/failed beams)
        for decoder_idx in 0..beam_size {
            if decoders[decoder_idx].completed || decoders[decoder_idx].failed {
                continue;
            }

            let attention = if params.token_timestamps {
                let beam_attn: Vec<Tensor<B, 4>> = batched_output
                    .cross_attention_weights
                    .iter()
                    .map(|w| w.clone().slice([decoder_idx..decoder_idx + 1]))
                    .collect();
                average_cross_attention_for_token(&beam_attn, 0)
            } else {
                Vec::new()
            };

            decoders[decoder_idx].pending_logits = batched_output
                .logits
                .clone()
                .slice([decoder_idx..decoder_idx + 1, 0..1])
                .flatten::<1>(0, 2);
            decoders[decoder_idx].pending_attention = attention;
        }
    }

    let best = decoders
        .into_iter()
        .filter_map(|mut decoder| {
            if decoder.failed || decoder.sequence.result_len == 0 {
                return None;
            }

            decoder
                .sequence
                .tokens
                .truncate(decoder.sequence.result_len);
            decoder.alignments.truncate(decoder.sequence.result_len);

            let sum_logprobs = decoder
                .sequence
                .tokens
                .iter()
                .map(|&(_, logprob, _)| logprob)
                .sum::<f64>();
            let avg_logprobs = sum_logprobs / decoder.sequence.result_len as f64;
            let entropy = compute_entropy(&decoder.sequence.tokens);

            if decoder.sequence.result_len > 32 && entropy < params.entropy_thold as f64 {
                return None;
            }

            decoder.sequence.sum_logprobs = sum_logprobs;
            decoder.sequence.avg_logprobs = avg_logprobs;
            decoder.sequence.entropy = entropy;
            decoder.sequence.score = sequence_score(
                sum_logprobs,
                decoder.sequence.result_len,
                params.length_penalty,
            );

            Some(decoder)
        })
        .max_by(|a, b| a.sequence.score.partial_cmp(&b.sequence.score).unwrap());

    if let Some(best) = best {
        SegmentDecodeResult {
            tokens: best.sequence.tokens,
            token_alignments: best.alignments,
            seek_delta: best.seek_delta,
            result_len: best.sequence.result_len,
            sum_logprobs: best.sequence.sum_logprobs,
            avg_logprobs: best.sequence.avg_logprobs,
            entropy: best.sequence.entropy,
            no_speech_prob,
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
            no_speech_prob,
            failed: true,
        }
    }
}

fn decoded_tokens_equal(a: &[(usize, f64, usize)], b: &[(usize, f64, usize)]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for index in (0..a.len()).rev() {
        if a[index].0 != b[index].0 {
            return false;
        }
    }

    true
}
