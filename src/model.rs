use burn::{
    config::Config,
    module::{Module, Param},
    nn::{
        self, PaddingConfig1d,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        conv::{Conv1d, Conv1dConfig},
    },
    tensor::{Bool, Distribution, FloatDType, Int, Tensor, backend::Backend, module::embedding},
};

use crate::custom_kernels::{
    CustomKernelsBackend, fused_single_query_attn, layer_norm_mixed, softmax_mixed,
};

#[derive(Config, Debug)]
pub struct WhisperConfig {
    audio_encoder_config: AudioEncoderConfig,
    text_decoder_config: TextDecoderConfig,
}

impl WhisperConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> Whisper<B> {
        let n_audio_state = self.audio_encoder_config.n_audio_state;
        let n_text_state = self.text_decoder_config.n_text_state;

        assert!(
            n_audio_state == n_text_state,
            "Audio encoder state size {n_audio_state} must be equal to text decoder state size {n_text_state}."
        );

        let encoder = self.audio_encoder_config.init(tensor_device_ref);
        let decoder = self.text_decoder_config.init(tensor_device_ref);

        Whisper { encoder, decoder }
    }
}

#[derive(Module, Debug)]
pub struct Whisper<B: Backend> {
    encoder: AudioEncoder<B>,
    decoder: TextDecoder<B>,
}

impl<B: CustomKernelsBackend> Whisper<B> {
    pub fn forward(&self, mel: Tensor<B, 3>, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let encoder_output = self.encoder.forward(mel, false);
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn forward_encoder(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel, false)
    }

    pub fn forward_encoder_f16(&self, mel: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encoder.forward(mel, true)
    }

    pub fn forward_decoder(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.decoder.forward(tokens, encoder_output)
    }

    pub fn forward_decoder_with_cross_attention(
        &self,
        tokens: Tensor<B, 2, Int>,
        encoder_output: Tensor<B, 3>,
    ) -> DecoderForwardOutput<B> {
        self.decoder
            .forward_with_cross_attention(tokens, encoder_output)
    }

    pub fn create_decoder_cache(&self, encoder_output: Tensor<B, 3>) -> DecoderCache<B> {
        self.decoder.create_cache(encoder_output, false)
    }

    pub fn create_decoder_cache_f16(&self, encoder_output: Tensor<B, 3>) -> DecoderCache<B> {
        self.decoder.create_cache(encoder_output, true)
    }

    pub fn forward_decoder_cached_with_cross_attention(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: DecoderCache<B>,
    ) -> CachedDecoderForwardOutput<B> {
        self.decoder
            .forward_cached_with_cross_attention(tokens, cache, false, None)
    }

    pub fn forward_decoder_cached_with_cross_attention_f16(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: DecoderCache<B>,
    ) -> CachedDecoderForwardOutput<B> {
        self.decoder
            .forward_cached_with_cross_attention(tokens, cache, true, None)
    }

    pub fn encoder_ctx_size(&self) -> usize {
        self.encoder.ctx_size()
    }
    pub fn encoder_mel_size(&self) -> usize {
        self.encoder.n_mels
    }

    pub fn decoder_ctx_size(&self) -> usize {
        self.decoder.ctx_size()
    }

    pub fn decoder_layer_count(&self) -> usize {
        self.decoder.layer_count()
    }

    /// Pre-compute fused QKV weights for all decoder layers (call once before inference).
    pub fn build_fused_decoder_weights(&self) -> FusedDecoderWeights<B> {
        let self_attn_qkv = self
            .decoder
            .blocks
            .iter()
            .map(|block| {
                let w = Tensor::cat(
                    vec![
                        block.attn.query.weight.val(),
                        block.attn.key.weight.val(),
                        block.attn.value.weight.val(),
                    ],
                    1,
                );
                let b = Tensor::cat(
                    vec![
                        block.attn.query.bias.as_ref().unwrap().val(),
                        block.attn.key.bias.as_ref().unwrap().val(),
                        block.attn.value.bias.as_ref().unwrap().val(),
                    ],
                    0,
                );
                (w, b)
            })
            .collect();

        // Pre-compute logit projection matrix: transpose + cast to f16 once
        let logit_embed = self
            .decoder
            .token_embedding
            .val()
            .transpose()
            .unsqueeze::<3>()
            .cast(FloatDType::F16);

        FusedDecoderWeights {
            self_attn_qkv,
            logit_embed,
        }
    }

    pub fn forward_decoder_cached_with_cross_attention_fused(
        &self,
        tokens: Tensor<B, 2, Int>,
        cache: DecoderCache<B>,
        fused: &FusedDecoderWeights<B>,
        use_f16: bool,
    ) -> CachedDecoderForwardOutput<B> {
        self.decoder
            .forward_cached_with_cross_attention(tokens, cache, use_f16, Some(fused))
    }
}

/// Pre-computed fused weights for all decoder layers.
/// Computed once before inference to avoid per-step overhead.
pub struct FusedDecoderWeights<B: Backend> {
    /// For each layer: self-attn QKV weight [d_model, 3*d_model] and bias [3*d_model]
    self_attn_qkv: Vec<(Tensor<B, 2>, Tensor<B, 1>)>,
    /// Pre-computed logit projection matrix: token_embedding transposed + cast to f16
    /// Shape: [1, d_model, vocab_size]
    logit_embed: Tensor<B, 3>,
}

pub struct DecoderForwardOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub cross_attention_weights: Vec<Tensor<B, 4>>,
}

/// Self-attention KV cache stored in head-split 4D format [batch, heads, seq, d_k].
#[derive(Clone, Debug)]
pub struct DecoderSelfAttentionCache<B: Backend> {
    key: Option<Tensor<B, 4>>,
    value: Option<Tensor<B, 4>>,
}

/// Cross-attention KV cache stored in head-split 4D format [batch, heads, seq, d_k].
#[derive(Clone, Debug)]
pub struct DecoderCrossAttentionCache<B: Backend> {
    key: Tensor<B, 4>,
    value: Tensor<B, 4>,
}

#[derive(Clone, Debug)]
pub struct DecoderLayerCache<B: Backend> {
    self_attention: DecoderSelfAttentionCache<B>,
    cross_attention: DecoderCrossAttentionCache<B>,
}

#[derive(Clone, Debug)]
pub struct DecoderCache<B: Backend> {
    layers: Vec<DecoderLayerCache<B>>,
    n_past: usize,
}

impl<B: Backend> DecoderCache<B> {
    /// Stack multiple batch=1 caches into one cache with batch=N (along dim 0).
    /// All caches must have the same `n_past` and layer count.
    pub fn stack(caches: Vec<Self>) -> Self {
        assert!(!caches.is_empty());
        let n_past = caches[0].n_past;
        let n_layers = caches[0].layers.len();

        let layers = (0..n_layers)
            .map(|i| {
                let self_keys: Vec<Tensor<B, 4>> = caches
                    .iter()
                    .map(|c| c.layers[i].self_attention.key.clone().unwrap())
                    .collect();
                let self_values: Vec<Tensor<B, 4>> = caches
                    .iter()
                    .map(|c| c.layers[i].self_attention.value.clone().unwrap())
                    .collect();
                let cross_keys: Vec<Tensor<B, 4>> = caches
                    .iter()
                    .map(|c| c.layers[i].cross_attention.key.clone())
                    .collect();
                let cross_values: Vec<Tensor<B, 4>> = caches
                    .iter()
                    .map(|c| c.layers[i].cross_attention.value.clone())
                    .collect();

                DecoderLayerCache {
                    self_attention: DecoderSelfAttentionCache {
                        key: Some(Tensor::cat(self_keys, 0)),
                        value: Some(Tensor::cat(self_values, 0)),
                    },
                    cross_attention: DecoderCrossAttentionCache {
                        key: Tensor::cat(cross_keys, 0),
                        value: Tensor::cat(cross_values, 0),
                    },
                }
            })
            .collect();

        DecoderCache { layers, n_past }
    }

    /// Unstack a batched cache (batch=N) into N individual batch=1 caches.
    pub fn unstack(self, n: usize) -> Vec<Self> {
        let n_past = self.n_past;
        let n_layers = self.layers.len();

        // Pre-split all layers' tensors
        let mut layer_splits: Vec<(
            Vec<Tensor<B, 4>>,
            Vec<Tensor<B, 4>>,
            Vec<Tensor<B, 4>>,
            Vec<Tensor<B, 4>>,
        )> = Vec::with_capacity(n_layers);

        for layer in self.layers {
            let sk = layer.self_attention.key.unwrap();
            let sv = layer.self_attention.value.unwrap();
            let ck = layer.cross_attention.key;
            let cv = layer.cross_attention.value;

            let sk_chunks: Vec<_> = (0..n).map(|i| sk.clone().slice([i..i + 1])).collect();
            let sv_chunks: Vec<_> = (0..n).map(|i| sv.clone().slice([i..i + 1])).collect();
            let ck_chunks: Vec<_> = (0..n).map(|i| ck.clone().slice([i..i + 1])).collect();
            let cv_chunks: Vec<_> = (0..n).map(|i| cv.clone().slice([i..i + 1])).collect();

            layer_splits.push((sk_chunks, sv_chunks, ck_chunks, cv_chunks));
        }

        (0..n)
            .map(|i| {
                let layers = layer_splits
                    .iter()
                    .map(|(sk, sv, ck, cv)| DecoderLayerCache {
                        self_attention: DecoderSelfAttentionCache {
                            key: Some(sk[i].clone()),
                            value: Some(sv[i].clone()),
                        },
                        cross_attention: DecoderCrossAttentionCache {
                            key: ck[i].clone(),
                            value: cv[i].clone(),
                        },
                    })
                    .collect();

                DecoderCache { layers, n_past }
            })
            .collect()
    }

    /// Reorder beams in a batched cache using index_select along the batch dimension.
    /// `indices[i]` = source beam index for new beam position `i`.
    pub fn reorder_beams(self, indices: Tensor<B, 1, Int>) -> Self {
        DecoderCache {
            layers: self
                .layers
                .into_iter()
                .map(|layer| DecoderLayerCache {
                    self_attention: DecoderSelfAttentionCache {
                        key: layer
                            .self_attention
                            .key
                            .map(|k| k.select(0, indices.clone())),
                        value: layer
                            .self_attention
                            .value
                            .map(|v| v.select(0, indices.clone())),
                    },
                    cross_attention: DecoderCrossAttentionCache {
                        key: layer.cross_attention.key.select(0, indices.clone()),
                        value: layer.cross_attention.value.select(0, indices.clone()),
                    },
                })
                .collect(),
            n_past: self.n_past,
        }
    }
}

pub struct CachedDecoderForwardOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub cross_attention_weights: Vec<Tensor<B, 4>>,
    pub cache: DecoderCache<B>,
}

struct DecoderBlockCachedOutput<B: Backend> {
    output: Tensor<B, 3>,
    cross_attention_weights: Tensor<B, 4>,
    cache: DecoderLayerCache<B>,
}

#[derive(Config, Debug)]
pub struct TextDecoderConfig {
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize,
    n_text_head: usize,
    n_text_layer: usize,
}

impl TextDecoderConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> TextDecoder<B> {
        let token_embedding = Param::from_tensor(Tensor::random(
            [self.n_vocab, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_text_ctx, self.n_text_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let blocks: Vec<_> = (0..self.n_text_layer)
            .map(|_| {
                ResidualDecoderAttentionBlockConfig::new(self.n_text_state, self.n_text_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln = nn::LayerNormConfig::new(self.n_text_state).init(tensor_device_ref);

        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        TextDecoder {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            n_vocab,
            n_text_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct TextDecoder<B: Backend> {
    token_embedding: Param<Tensor<B, 2>>,
    positional_embedding: Param<Tensor<B, 2>>,
    blocks: Vec<ResidualDecoderAttentionBlock<B>>,
    ln: nn::LayerNorm<B>,
    n_vocab: usize,
    n_text_ctx: usize,
}

impl<B: CustomKernelsBackend> TextDecoder<B> {
    fn forward(&self, x: Tensor<B, 2, Int>, xa: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let device = x.device();
        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mask = causal_mask::<B>(seq_len, 0, &device);

        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(x, xa.clone(), mask.clone());
        }

        let x = self.ln.forward(x);
        x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>())
    }

    fn forward_with_cross_attention(
        &self,
        x: Tensor<B, 2, Int>,
        xa: Tensor<B, 3>,
    ) -> DecoderForwardOutput<B> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            seq_len <= self.n_text_ctx,
            "Token sequence length {} must not exceed {}.",
            seq_len,
            self.n_text_ctx
        );

        let device = x.device();
        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([0..seq_len])
                .unsqueeze::<3>();

        let mask = causal_mask::<B>(seq_len, 0, &device);

        let mut x = x;
        let mut cross_attention_weights = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter() {
            let (block_output, weights) =
                block.forward_with_cross_attention(x, xa.clone(), mask.clone());
            x = block_output;
            cross_attention_weights.push(weights);
        }

        let x = self.ln.forward(x);
        let logits = x.matmul(self.token_embedding.val().transpose().unsqueeze::<3>());

        DecoderForwardOutput {
            logits,
            cross_attention_weights,
        }
    }

    fn create_cache(&self, xa: Tensor<B, 3>, use_f16: bool) -> DecoderCache<B> {
        let layers = self
            .blocks
            .iter()
            .map(|block| DecoderLayerCache {
                self_attention: DecoderSelfAttentionCache {
                    key: None,
                    value: None,
                },
                cross_attention: block.build_cross_cache(xa.clone(), use_f16),
            })
            .collect();

        DecoderCache { layers, n_past: 0 }
    }

    fn forward_cached_with_cross_attention(
        &self,
        x: Tensor<B, 2, Int>,
        cache: DecoderCache<B>,
        use_f16: bool,
        fused_qkv: Option<&FusedDecoderWeights<B>>,
    ) -> CachedDecoderForwardOutput<B> {
        let [_n_batch, seq_len] = x.dims();

        assert!(
            cache.n_past + seq_len <= self.n_text_ctx,
            "Token sequence length {} with {} cached tokens must not exceed {}.",
            seq_len,
            cache.n_past,
            self.n_text_ctx
        );
        assert!(
            cache.layers.len() == self.blocks.len(),
            "Decoder cache layer count {} must match decoder layer count {}.",
            cache.layers.len(),
            self.blocks.len()
        );

        let device = x.device();
        let position_start = cache.n_past;
        let x = embedding(self.token_embedding.val(), x)
            + self
                .positional_embedding
                .val()
                .slice([position_start..position_start + seq_len])
                .unsqueeze::<3>();

        // For seq_len=1 (autoregressive step), the causal mask is trivially all-false
        // (the single query can attend to all past tokens), so skip mask generation entirely.
        let mask = if seq_len > 1 {
            Some(causal_mask::<B>(seq_len, cache.n_past, &device))
        } else {
            None
        };

        // Cast to f16 for the heavy decoder block compute
        let mut x = if use_f16 { x.cast(FloatDType::F16) } else { x };
        let mut next_layers = Vec::with_capacity(self.blocks.len());
        let mut cross_attention_weights = Vec::with_capacity(self.blocks.len());

        for (i, (block, layer_cache)) in
            self.blocks.iter().zip(cache.layers.into_iter()).enumerate()
        {
            let fused_self = fused_qkv.map(|f| &f.self_attn_qkv[i]);
            let block_output = block.forward_cached_with_cross_attention(
                x,
                mask.clone(),
                layer_cache,
                use_f16,
                fused_self,
            );
            x = block_output.output;
            next_layers.push(block_output.cache);
            cross_attention_weights.push(block_output.cross_attention_weights);
        }

        // Final LayerNorm + logit projection.
        let x = layer_norm_mixed(&self.ln, x, use_f16);
        let logits = if let Some(fused) = fused_qkv {
            // Use pre-computed f16 embedding matrix (avoids per-step transpose + cast)
            x.matmul(fused.logit_embed.clone()).cast(FloatDType::F32)
        } else if use_f16 {
            let embed = self.token_embedding.val().transpose().unsqueeze::<3>();
            x.matmul(embed.cast(FloatDType::F16)).cast(FloatDType::F32)
        } else {
            let embed = self.token_embedding.val().transpose().unsqueeze::<3>();
            x.matmul(embed)
        };

        CachedDecoderForwardOutput {
            logits,
            cross_attention_weights,
            cache: DecoderCache {
                layers: next_layers,
                n_past: position_start + seq_len,
            },
        }
    }

    fn ctx_size(&self) -> usize {
        self.n_text_ctx
    }

    fn layer_count(&self) -> usize {
        self.blocks.len()
    }
}

#[derive(Config, Debug)]
pub struct AudioEncoderConfig {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize,
    n_audio_head: usize,
    n_audio_layer: usize,
}

impl AudioEncoderConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> AudioEncoder<B> {
        let conv1 = Conv1dConfig::new(self.n_mels, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .init(tensor_device_ref);
        let gelu1 = nn::Gelu::new();
        let conv2 = Conv1dConfig::new(self.n_audio_state, self.n_audio_state, 3)
            .with_padding(PaddingConfig1d::Explicit(1, 1))
            .with_stride(2)
            .init(tensor_device_ref);
        let gelu2 = nn::Gelu::new();
        let blocks: Vec<_> = (0..self.n_audio_layer)
            .map(|_| {
                ResidualEncoderAttentionBlockConfig::new(self.n_audio_state, self.n_audio_head)
                    .init(tensor_device_ref)
            })
            .collect();
        let ln_post = nn::LayerNormConfig::new(self.n_audio_state).init(tensor_device_ref);
        let positional_embedding = Param::from_tensor(Tensor::random(
            [self.n_audio_ctx, self.n_audio_state],
            Distribution::Normal(0.0, 1.0),
            tensor_device_ref,
        ));
        let n_mels = self.n_mels;
        let n_audio_ctx = self.n_audio_ctx;

        AudioEncoder {
            conv1,
            gelu1,
            conv2,
            gelu2,
            blocks,
            ln_post,
            positional_embedding,
            n_mels,
            n_audio_ctx,
        }
    }
}

#[derive(Module, Debug)]
pub struct AudioEncoder<B: Backend> {
    conv1: Conv1d<B>,
    gelu1: nn::Gelu,
    conv2: Conv1d<B>,
    gelu2: nn::Gelu,
    blocks: Vec<ResidualEncoderAttentionBlock<B>>,
    ln_post: nn::LayerNorm<B>,
    positional_embedding: Param<Tensor<B, 2>>,
    n_mels: usize,
    n_audio_ctx: usize,
}

impl<B: CustomKernelsBackend> AudioEncoder<B> {
    fn forward(&self, x: Tensor<B, 3>, use_f16: bool) -> Tensor<B, 3> {
        let [_, n_mels, _n_ctx] = x.dims();

        assert!(
            n_mels == self.n_mels,
            "Audio mel spectrum size must be {}.",
            self.n_mels
        );

        // Conv weights are kept in f32 for accuracy; run convs in f32
        let x = self.gelu1.forward(self.conv1.forward(x));
        let x = self.gelu2.forward(self.conv2.forward(x));

        // Cast to f16 after convs for the attention blocks
        let x = if use_f16 { x.cast(FloatDType::F16) } else { x };

        let x = x.swap_dims(1, 2);
        let k = x.dims()[1];

        assert!(
            k <= self.n_audio_ctx,
            "Encoded audio context {} exceeds maximum {}. Input mel frames should be at most {}.",
            k,
            self.n_audio_ctx,
            self.n_audio_ctx * 2
        );

        #[allow(clippy::single_range_in_vec_init)]
        let pos_emb = self
            .positional_embedding
            .val()
            .slice([0..k])
            .unsqueeze::<3>();
        // positional_embedding is kept in f32 for stability, cast to match x
        let pos_emb = if use_f16 {
            pos_emb.cast(FloatDType::F16)
        } else {
            pos_emb
        };
        let x = x + pos_emb;

        let mut x = x;
        for block in self.blocks.iter() {
            x = block.forward(x, use_f16);
        }

        // Cast back to f32 for final LayerNorm
        let x = if use_f16 { x.cast(FloatDType::F32) } else { x };
        self.ln_post.forward(x)
    }

    fn ctx_size(&self) -> usize {
        self.n_audio_ctx
    }
}

#[derive(Config, Debug)]
pub struct ResidualEncoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualEncoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
    ) -> ResidualEncoderAttentionBlock<B> {
        let attn = MultiHeadAttentionConfig::new(self.n_state, self.n_head)
            .with_dropout(0.0)
            .init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);
        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualEncoderAttentionBlock {
            attn,
            attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualEncoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: nn::LayerNorm<B>,
}

/// Number of query positions to process at once in chunked encoder attention.
/// Reduces peak VRAM from O(batch * heads * seq² ) to O(batch * heads * chunk * seq).
/// 256 is a good balance: for seq=1500, peak attention scores drop from
/// 1500×1500 = 2.25M to 256×1500 = 384K entries per head (~6× reduction).
const ENCODER_ATTN_CHUNK: usize = 256;

impl<B: CustomKernelsBackend> ResidualEncoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, use_f16: bool) -> Tensor<B, 3> {
        let ln_out = layer_norm_mixed(&self.attn_ln, x.clone(), use_f16);
        let attn_out = self.chunked_self_attention(ln_out, use_f16);

        let x = x + attn_out;

        let mlp_input = layer_norm_mixed(&self.mlp_ln, x.clone(), use_f16);
        x + self.mlp.forward(mlp_input)
    }

    /// Chunked self-attention: processes queries in blocks of ENCODER_ATTN_CHUNK
    /// to avoid allocating the full [batch, heads, seq, seq] score matrix.
    fn chunked_self_attention(&self, x: Tensor<B, 3>, use_f16: bool) -> Tensor<B, 3> {
        let mha = &self.attn;
        let [batch, seq_len, d_model] = x.dims();
        let n_heads = mha.n_heads;
        let d_k = mha.d_k;

        // Project Q, K, V: [batch, seq, d_model] -> [batch, heads, seq, d_k]
        let q = mha
            .query
            .forward(x.clone())
            .reshape([batch, seq_len, n_heads, d_k])
            .swap_dims(1, 2);
        let k = mha
            .key
            .forward(x.clone())
            .reshape([batch, seq_len, n_heads, d_k])
            .swap_dims(1, 2);
        let v = mha
            .value
            .forward(x)
            .reshape([batch, seq_len, n_heads, d_k])
            .swap_dims(1, 2);

        let scale: f64 = 1.0 / (d_k as f64).sqrt();

        // If sequence is short enough, fall back to standard full attention
        if seq_len <= ENCODER_ATTN_CHUNK {
            let scores = q.matmul(k.transpose()) * scale;
            let weights = softmax_mixed(scores, 3, use_f16);
            let context = weights
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([batch, seq_len, d_model]);
            return mha.output.forward(context);
        }

        // Chunked: process query blocks against full K, V
        let n_chunks = (seq_len + ENCODER_ATTN_CHUNK - 1) / ENCODER_ATTN_CHUNK;
        let mut chunk_outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(n_chunks);

        for chunk_idx in 0..n_chunks {
            let q_start = chunk_idx * ENCODER_ATTN_CHUNK;
            let q_end = (q_start + ENCODER_ATTN_CHUNK).min(seq_len);

            // q_chunk: [batch, heads, chunk_len, d_k]
            let q_chunk = q
                .clone()
                .slice([0..batch, 0..n_heads, q_start..q_end, 0..d_k]);

            // scores: [batch, heads, chunk_len, seq_len] — much smaller than [seq, seq]
            let scores = q_chunk.matmul(k.clone().transpose()) * scale;
            let weights = softmax_mixed(scores, 3, use_f16);
            // out_chunk: [batch, heads, chunk_len, d_k]
            let out_chunk = weights.matmul(v.clone());
            chunk_outputs.push(out_chunk);
        }

        // Concatenate chunks along the sequence dimension (dim 2)
        let context = Tensor::cat(chunk_outputs, 2)
            .swap_dims(1, 2)
            .reshape([batch, seq_len, d_model]);
        mha.output.forward(context)
    }
}

#[derive(Config, Debug)]
pub struct ResidualDecoderAttentionBlockConfig {
    n_state: usize,
    n_head: usize,
}

impl ResidualDecoderAttentionBlockConfig {
    pub fn init<B: Backend>(
        &self,
        tensor_device_ref: &B::Device,
    ) -> ResidualDecoderAttentionBlock<B> {
        let attn = MultiHeadAttentionConfig::new(self.n_state, self.n_head)
            .with_dropout(0.0)
            .init(tensor_device_ref);
        let attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let cross_attn = MultiHeadAttentionConfig::new(self.n_state, self.n_head)
            .with_dropout(0.0)
            .init(tensor_device_ref);
        let cross_attn_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        let mlp = MLPConfig::new(self.n_state).init(tensor_device_ref);
        let mlp_ln = nn::LayerNormConfig::new(self.n_state).init(tensor_device_ref);

        ResidualDecoderAttentionBlock {
            attn,
            attn_ln,
            cross_attn,
            cross_attn_ln,
            mlp,
            mlp_ln,
        }
    }
}

#[derive(Module, Debug)]
pub struct ResidualDecoderAttentionBlock<B: Backend> {
    pub attn: MultiHeadAttention<B>,
    attn_ln: nn::LayerNorm<B>,
    pub cross_attn: MultiHeadAttention<B>,
    cross_attn_ln: nn::LayerNorm<B>,
    mlp: MLP<B>,
    mlp_ln: nn::LayerNorm<B>,
}

impl<B: CustomKernelsBackend> ResidualDecoderAttentionBlock<B> {
    fn forward(&self, x: Tensor<B, 3>, xa: Tensor<B, 3>, mask: Tensor<B, 3, Bool>) -> Tensor<B, 3> {
        let self_attn_out = self
            .attn
            .forward(MhaInput::self_attn(self.attn_ln.forward(x.clone())).mask_attn(mask))
            .context;
        let x = x + self_attn_out;

        let cross_attn_out = self.cross_attn.forward(MhaInput::new(
            self.cross_attn_ln.forward(x.clone()),
            xa.clone(),
            xa,
        ));
        let x = x + cross_attn_out.context;

        x.clone() + self.mlp.forward(self.mlp_ln.forward(x))
    }

    fn forward_with_cross_attention(
        &self,
        x: Tensor<B, 3>,
        xa: Tensor<B, 3>,
        mask: Tensor<B, 3, Bool>,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let self_attn_out = self
            .attn
            .forward(MhaInput::self_attn(self.attn_ln.forward(x.clone())).mask_attn(mask))
            .context;
        let x = x + self_attn_out;

        let cross_attn_out = self.cross_attn.forward(MhaInput::new(
            self.cross_attn_ln.forward(x.clone()),
            xa.clone(),
            xa,
        ));
        let weights = cross_attn_out.weights.clone();
        let x = x + cross_attn_out.context;

        let output = x.clone() + self.mlp.forward(self.mlp_ln.forward(x));
        (output, weights)
    }

    /// Build pre-projected cross-attention KV cache in 4D head-split format.
    fn build_cross_cache(&self, xa: Tensor<B, 3>, use_f16: bool) -> DecoderCrossAttentionCache<B> {
        // Cross-attn weights are f32 for accuracy; project in f32, then cast cache to f16
        let [batch, seq, _] = xa.dims();
        let n_heads = self.cross_attn.n_heads;
        let d_k = self.cross_attn.d_k;
        let key = self
            .cross_attn
            .key
            .forward(xa.clone())
            .reshape([batch, seq, n_heads, d_k])
            .swap_dims(1, 2);
        let value = self
            .cross_attn
            .value
            .forward(xa)
            .reshape([batch, seq, n_heads, d_k])
            .swap_dims(1, 2);
        DecoderCrossAttentionCache {
            key: if use_f16 {
                key.cast(FloatDType::F16)
            } else {
                key
            },
            value: if use_f16 {
                value.cast(FloatDType::F16)
            } else {
                value
            },
        }
    }

    fn forward_cached_with_cross_attention(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, Bool>>,
        cache: DecoderLayerCache<B>,
        use_f16: bool,
        fused_qkv: Option<&(Tensor<B, 2>, Tensor<B, 1>)>,
    ) -> DecoderBlockCachedOutput<B> {
        let ln_out = layer_norm_mixed(&self.attn_ln, x.clone(), use_f16);
        let (self_attn_output, self_attention) =
            self.cached_self_attention(ln_out, mask, cache.self_attention, use_f16, fused_qkv);
        let x = x + self_attn_output;

        let ln_out = layer_norm_mixed(&self.cross_attn_ln, x.clone(), use_f16);
        let (cross_attn_output, cross_weights, cross_attention) =
            self.cached_cross_attention(ln_out, cache.cross_attention, use_f16);
        let x = x + cross_attn_output;

        let ln_out = layer_norm_mixed(&self.mlp_ln, x.clone(), use_f16);
        let output = x + self.mlp.forward(ln_out);

        DecoderBlockCachedOutput {
            output,
            cross_attention_weights: cross_weights,
            cache: DecoderLayerCache {
                self_attention,
                cross_attention,
            },
        }
    }

    /// Self-attention with KV cache in 4D head-split format.
    fn cached_self_attention(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 3, Bool>>,
        cache: DecoderSelfAttentionCache<B>,
        use_f16: bool,
        fused_qkv: Option<&(Tensor<B, 2>, Tensor<B, 1>)>,
    ) -> (Tensor<B, 3>, DecoderSelfAttentionCache<B>) {
        let [batch_size, seq_len, _] = x.dims();
        let n_heads = self.attn.n_heads;
        let d_k = self.attn.d_k;
        let d_model = n_heads * d_k;

        let (q, k_new, v_new) = if let Some((qkv_w, qkv_b)) = fused_qkv {
            // Fused QKV: one matmul with pre-computed [d_model, 3*d_model] weight
            let x_flat = x.reshape([batch_size * seq_len, d_model]);
            let qkv = (x_flat.matmul(qkv_w.clone()) + qkv_b.clone().unsqueeze()).reshape([
                batch_size,
                seq_len,
                3 * d_model,
            ]);

            let q = qkv
                .clone()
                .slice([0..batch_size, 0..seq_len, 0..d_model])
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            let k_new = qkv
                .clone()
                .slice([0..batch_size, 0..seq_len, d_model..2 * d_model])
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            let v_new = qkv
                .slice([0..batch_size, 0..seq_len, 2 * d_model..3 * d_model])
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            (q, k_new, v_new)
        } else {
            // Standard: 3 separate projections
            let q = self
                .attn
                .query
                .forward(x.clone())
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            let k_new = self
                .attn
                .key
                .forward(x.clone())
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            let v_new = self
                .attn
                .value
                .forward(x)
                .reshape([batch_size, seq_len, n_heads, d_k])
                .swap_dims(1, 2);
            (q, k_new, v_new)
        };

        let k = if let Some(cached_k) = cache.key {
            Tensor::cat(vec![cached_k, k_new], 2)
        } else {
            k_new
        };
        let v = if let Some(cached_v) = cache.value {
            Tensor::cat(vec![cached_v, v_new], 2)
        } else {
            v_new
        };

        // For single-query decoding, use fused attention kernel
        // (Q@K^T·scale → softmax → @V in one pass, no intermediate tensors)
        let context = if seq_len == 1 && mask.is_none() {
            fused_single_query_attn::<B>(q, k.clone(), v.clone())
                .swap_dims(1, 2)
                .reshape([batch_size, 1, n_heads * d_k])
        } else {
            let scale = (d_k as f32).sqrt().recip();
            let scores = q.matmul(k.clone().transpose()) * scale;

            let scores = if let Some(mask) = mask {
                let [mask_batch, mask_seq_q, mask_seq_k] = mask.dims();
                scores.mask_fill(
                    mask.reshape([mask_batch, 1, mask_seq_q, mask_seq_k]),
                    self.attn.min_float,
                )
            } else {
                scores
            };

            let weights = softmax_mixed(scores, 3, use_f16);
            weights
                .matmul(v.clone())
                .swap_dims(1, 2)
                .reshape([batch_size, seq_len, n_heads * d_k])
        };
        let output = self.attn.output.forward(context);

        (
            output,
            DecoderSelfAttentionCache {
                key: Some(k),
                value: Some(v),
            },
        )
    }

    /// Cross-attention using pre-computed KV cache (4D head-split).
    fn cached_cross_attention(
        &self,
        x: Tensor<B, 3>,
        cache: DecoderCrossAttentionCache<B>,
        use_f16: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>, DecoderCrossAttentionCache<B>) {
        let [batch_size, seq_len, _] = x.dims();
        let n_heads = self.cross_attn.n_heads;
        let d_k = self.cross_attn.d_k;
        let d_model = n_heads * d_k;

        // Cross-attn weights are f32; cast input for projection, then back to f16
        let q_input = if use_f16 { x.cast(FloatDType::F32) } else { x };
        let q = self
            .cross_attn
            .query
            .forward(q_input)
            .reshape([batch_size, seq_len, n_heads, d_k])
            .swap_dims(1, 2);
        let q = if use_f16 { q.cast(FloatDType::F16) } else { q };

        let scale = (d_k as f32).sqrt().recip();
        let scores = q.matmul(cache.key.clone().transpose()) * scale;
        let weights = softmax_mixed(scores, 3, use_f16);
        let context = weights
            .clone()
            .matmul(cache.value.clone())
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, d_model]);

        // Cross-attn output weights are f32; cast for projection, then back
        let context_proj = if use_f16 {
            context.cast(FloatDType::F32)
        } else {
            context
        };
        let output = self.cross_attn.output.forward(context_proj);
        let output = if use_f16 {
            output.cast(FloatDType::F16)
        } else {
            output
        };

        (output, weights, cache)
    }
}

#[derive(Config, Debug)]
pub struct MLPConfig {
    n_state: usize,
}

impl MLPConfig {
    pub fn init<B: Backend>(&self, tensor_device_ref: &B::Device) -> MLP<B> {
        let lin1 = nn::LinearConfig::new(self.n_state, 4 * self.n_state).init(tensor_device_ref);
        let gelu = nn::Gelu::new();
        let lin2 = nn::LinearConfig::new(4 * self.n_state, self.n_state).init(tensor_device_ref);

        MLP { lin1, gelu, lin2 }
    }
}

#[derive(Module, Debug)]
pub struct MLP<B: Backend> {
    lin1: nn::Linear<B>,
    gelu: nn::Gelu,
    lin2: nn::Linear<B>,
}

impl<B: Backend> MLP<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.lin1.forward(x);
        let x = self.gelu.forward(x);

        self.lin2.forward(x)
    }
}

/// Generate a Bool causal mask [1, seq_len, n_past + seq_len].
/// `true` = masked (future positions blocked), `false` = attend.
fn causal_mask<B: Backend>(
    seq_len: usize,
    n_past: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let total = n_past + seq_len;
    Tensor::<B, 2, Bool>::tril_mask([seq_len, total], n_past as i64, device).unsqueeze::<3>()
}
