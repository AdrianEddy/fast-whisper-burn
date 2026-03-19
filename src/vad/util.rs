const VAD_MERGE_GAP_MS: i64 = 1000;
const VAD_TAIL_PADDING_MS: i64 = 500;
const BURN_VAD_UPLOAD_WINDOWS: usize = 256;
const TARGET_SAMPLE_RATE: u32 = 16_000;

use super::PredictState;
use super::SileroVAD6Model;
use burn::prelude::Tensor;

#[derive(Clone, Copy)]
pub struct SpeechRegion {
    pub start_ms: i64,
    pub start_sample: usize,
    pub end_sample: usize,
}

#[derive(Clone, Copy)]
pub struct VadSegment {
    pub start: u64,
    pub end: u64,
}

pub fn whisper_vad_segments_from_probs(probs: &[f32], n_window: usize) -> Vec<VadSegment> {
    const THRESHOLD: f32 = 0.5;
    const MIN_SPEECH_MS: usize = 250;
    const MIN_SILENCE_MS: usize = 100;
    const MAX_SPEECH_DURATION_S: f32 = f32::MAX;
    const SPEECH_PAD_MS: usize = 30;
    const MAX_MERGE_GAP_MS: usize = 200;
    const MIN_SILENCE_AT_MAX_SPEECH_MS: usize = 98;

    #[derive(Clone, Copy)]
    struct SpeechSegment {
        start: usize,
        end: usize,
    }

    let sample_rate = TARGET_SAMPLE_RATE as usize;
    let min_silence_samples = sample_rate * MIN_SILENCE_MS / 1000;
    let audio_length_samples = probs.len() * n_window;
    let min_speech_samples = sample_rate * MIN_SPEECH_MS / 1000;
    let speech_pad_samples = sample_rate * SPEECH_PAD_MS / 1000;
    let min_silence_samples_at_max_speech = sample_rate * MIN_SILENCE_AT_MAX_SPEECH_MS / 1000;
    let max_merge_gap_samples = sample_rate * MAX_MERGE_GAP_MS / 1000;

    let max_speech_samples = if MAX_SPEECH_DURATION_S > 100000.0 {
        usize::MAX / 2
    } else {
        let temp = (sample_rate as i64 * MAX_SPEECH_DURATION_S as i64)
            - n_window as i64
            - 2 * speech_pad_samples as i64;
        if temp <= 0 {
            usize::MAX / 2
        } else {
            temp.min((usize::MAX / 2) as i64) as usize
        }
    };

    let neg_threshold = (THRESHOLD - 0.15).max(0.01);

    let mut speeches = Vec::<SpeechSegment>::with_capacity(256);
    let mut is_speech_segment = false;
    let mut temp_end = 0usize;
    let mut prev_end = 0usize;
    let mut next_start = 0usize;
    let mut curr_speech_start = 0usize;
    let mut has_curr_speech = false;

    for (index, &curr_prob) in probs.iter().enumerate() {
        let curr_sample = n_window * index;

        if curr_prob >= THRESHOLD && temp_end != 0 {
            temp_end = 0;
            if next_start < prev_end {
                next_start = curr_sample;
            }
        }

        if curr_prob >= THRESHOLD && !is_speech_segment {
            is_speech_segment = true;
            curr_speech_start = curr_sample;
            has_curr_speech = true;
            continue;
        }

        if is_speech_segment && (curr_sample - curr_speech_start) > max_speech_samples {
            if prev_end != 0 {
                speeches.push(SpeechSegment {
                    start: curr_speech_start,
                    end: prev_end,
                });
                has_curr_speech = true;

                if next_start < prev_end {
                    is_speech_segment = false;
                    has_curr_speech = false;
                } else {
                    curr_speech_start = next_start;
                }

                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            } else {
                speeches.push(SpeechSegment {
                    start: curr_speech_start,
                    end: curr_sample,
                });
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                is_speech_segment = false;
                has_curr_speech = false;
                continue;
            }
        }

        if curr_prob < neg_threshold && is_speech_segment {
            if temp_end == 0 {
                temp_end = curr_sample;
            }

            if (curr_sample - temp_end) > min_silence_samples_at_max_speech {
                prev_end = temp_end;
            }

            if (curr_sample - temp_end) < min_silence_samples {
                continue;
            }

            if (temp_end - curr_speech_start) > min_speech_samples {
                speeches.push(SpeechSegment {
                    start: curr_speech_start,
                    end: temp_end,
                });
            }

            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            is_speech_segment = false;
            has_curr_speech = false;
        }
    }

    if has_curr_speech && (audio_length_samples - curr_speech_start) > min_speech_samples {
        speeches.push(SpeechSegment {
            start: curr_speech_start,
            end: audio_length_samples,
        });
    }

    let mut index = 0usize;
    while index + 1 < speeches.len() {
        if speeches[index + 1].start - speeches[index].end < max_merge_gap_samples {
            speeches[index].end = speeches[index + 1].end;
            speeches.remove(index + 1);
        } else {
            index += 1;
        }
    }

    speeches.retain(|segment| segment.end.saturating_sub(segment.start) >= min_speech_samples);

    for index in 0..speeches.len() {
        if index == 0 {
            speeches[index].start = speeches[index].start.saturating_sub(speech_pad_samples);
        }

        if index + 1 < speeches.len() {
            let silence_duration = speeches[index + 1]
                .start
                .saturating_sub(speeches[index].end);
            if silence_duration < 2 * speech_pad_samples {
                speeches[index].end += silence_duration / 2;
                speeches[index + 1].start = speeches[index + 1]
                    .start
                    .saturating_sub(silence_duration / 2);
            } else {
                speeches[index].end =
                    (speeches[index].end + speech_pad_samples).min(audio_length_samples);
                speeches[index + 1].start =
                    speeches[index + 1].start.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[index].end =
                (speeches[index].end + speech_pad_samples).min(audio_length_samples);
        }
    }

    speeches
        .into_iter()
        .filter(|segment| segment.end > segment.start)
        .map(|segment| VadSegment {
            start: samples_to_centiseconds(segment.start),
            end: samples_to_centiseconds(segment.end),
        })
        .collect()
}

pub fn merge_vad_segments(
    audio_len: usize,
    segments: impl IntoIterator<Item = VadSegment>,
) -> Vec<SpeechRegion> {
    let mut ranges = Vec::<(i64, i64)>::new();
    for segment in segments {
        let start_ms = ((segment.start as f64 / 100.0) * 1000.0).round() as i64;
        let end_ms = ((segment.end as f64 / 100.0) * 1000.0).round() as i64;
        if end_ms <= start_ms {
            continue;
        }
        if let Some((_, last_end_ms)) = ranges.last_mut() {
            if start_ms - *last_end_ms <= VAD_MERGE_GAP_MS {
                *last_end_ms = (*last_end_ms).max(end_ms);
                continue;
            }
        }
        ranges.push((start_ms, end_ms));
    }

    let audio_len_ms = (audio_len as i64 * 1000) / TARGET_SAMPLE_RATE as i64;
    let last_index = ranges.len().saturating_sub(1);

    ranges
        .into_iter()
        .enumerate()
        .filter_map(|(index, (start_ms, end_ms))| {
            let end_ms = if index == last_index {
                (end_ms + VAD_TAIL_PADDING_MS).min(audio_len_ms)
            } else {
                (end_ms + VAD_TAIL_PADDING_MS).min(audio_len_ms)
            };
            let start_sample = ((start_ms.max(0) as usize) * TARGET_SAMPLE_RATE as usize) / 1000;
            let end_sample =
                (((end_ms.max(0) as usize) * TARGET_SAMPLE_RATE as usize) / 1000).min(audio_len);
            if end_sample <= start_sample {
                return None;
            }
            Some(SpeechRegion {
                start_ms,
                start_sample,
                end_sample,
            })
        })
        .collect()
}

pub fn detect_speech_regions<
    B: crate::custom_kernels::CustomKernelsBackend,
    F: FnMut(usize, usize) -> bool,
>(
    vad: &SileroVAD6Model<B>,
    device: &B::Device,
    waveform: &[f32],
    mut progress_callback: Option<F>,
) -> Result<Vec<SpeechRegion>, String> {
    let mut predict_state = PredictState::default(device);
    let chunk_size = predict_state.input_size();
    let mut speech_probs =
        Vec::<f32>::with_capacity((waveform.len() + chunk_size - 1) / chunk_size);

    let mut offset = 0usize;
    while offset < waveform.len() {
        let block_end = (offset + chunk_size * BURN_VAD_UPLOAD_WINDOWS).min(waveform.len());
        let block_len = block_end - offset;
        let padded_block_len = if block_len % chunk_size == 0 {
            block_len
        } else {
            block_len + (chunk_size - (block_len % chunk_size))
        };
        let num_windows = padded_block_len / chunk_size;

        let mut block = vec![0.0f32; padded_block_len];
        block[..block_len].copy_from_slice(&waveform[offset..block_end]);

        let block_tensor = Tensor::<B, 1>::from_floats(block.as_slice(), device)
            .reshape([num_windows, chunk_size]);
        let (new_state, output) = vad
            .predict_sequence(predict_state, block_tensor)
            .map_err(|_| "Burn Silero VAD failed".to_string())?;
        predict_state = new_state;

        let output_data = output.into_data().convert::<f32>();
        speech_probs.extend_from_slice(
            output_data
                .as_slice::<f32>()
                .map_err(|_| "Unexpected Silero VAD output type".to_string())?,
        );

        offset = block_end;
        if let Some(callback) = progress_callback.as_mut() {
            if !callback(offset, waveform.len()) {
                break;
            }
        }
    }

    let vad_segments = whisper_vad_segments_from_probs(&speech_probs, chunk_size);
    Ok(merge_vad_segments(waveform.len(), vad_segments))
}

fn samples_to_centiseconds(samples: usize) -> u64 {
    ((samples as u64 * 100) + (TARGET_SAMPLE_RATE as u64 / 2)) / TARGET_SAMPLE_RATE as u64
}
