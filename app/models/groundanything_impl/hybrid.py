"""Qwen3.5 Direct-Conversion model for GAM.

Qwen3.5 interleaves causal GatedDeltaNet and full-attention layers.  A
Fast-dVLM block mask cannot be applied to the recurrent linear layers.  This
adapter therefore keeps noisy and clean states isolated through those layers
and applies the exact Direct-Conversion N2N/N2C/C2C mask in every full-
attention layer.  Vision is encoded once and appears only in the clean stream.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
from typing import Any

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

MASK_TOKEN = "|<MASK>|"
STABLE_BLOCK_MASK_ENV = "GAM_DLM_STABLE_BLOCK_MASK"
FLEX_CUDAGRAPHS_ENV = "GAM_DLM_FLEX_CUDAGRAPHS"
ATTENTION_PROFILE_ENV = "GAM_DLM_ATTENTION_PROFILE"


@contextmanager
def _attention_profile_region(name: str):
    if os.environ.get(ATTENTION_PROFILE_ENV) == "1":
        with torch.profiler.record_function(name):
            yield
    else:
        yield


def _flex_attention_kernel(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask: Any,
) -> torch.Tensor:
    return flex_attention(
        query, key, value, block_mask=block_mask, enable_gqa=True
    )


# Calling flex_attention from eager mode deliberately uses its dense reference
# backend.  Compile this boundary so Inductor lowers the HOP to the block-sparse
# Triton kernel.  Dynamic shapes avoid recompiling for every variable-length
# multimodal batch.
_flex_cudagraphs = os.environ.get(FLEX_CUDAGRAPHS_ENV, "1") == "1"
_compiled_flex_attention = torch.compile(
    _flex_attention_kernel,
    dynamic=True,
    fullgraph=True,
    options={
        # Packed multimodal lengths vary on every optimizer step.  Retaining
        # one CUDA Graph pool per concrete shape causes live HBM to grow until
        # OOM even though the dynamic Inductor/Triton kernel itself is stable.
        # Keep the historical default for other routes; PPU formal training
        # opts out explicitly through the environment contract.
        "triton.cudagraphs": _flex_cudagraphs,
        "triton.cudagraph_trees": _flex_cudagraphs,
    },
)


class GAMQwen35DLM(nn.Module):
    """Direct Conversion wrapper around a Qwen3.5 VLM.

    Training uses complementary noisy views.  Evaluation uses the same hybrid
    noisy/clean attention contract through :meth:`draft_block`; causal
    verification is deliberately delegated to the unchanged base VLM.
    """

    def __init__(
        self,
        base_model: nn.Module,
        mask_token_id: int,
        im_end_token_id: int,
        block_size: int = 32,
        minimum_noise_level: float = 0.0,
        allow_nondefault_block_size: bool = False,
    ) -> None:
        super().__init__()
        if block_size not in (8, 32):
            raise ValueError(
                "GAM Direct Conversion only validates fixed block_size 8 or 32"
            )
        if block_size != 32 and not allow_nondefault_block_size:
            raise ValueError(
                "non-default DLM block size requires the isolated fixed_block_size_experiment route"
            )
        if not 0.0 <= minimum_noise_level < 1.0:
            raise ValueError("minimum_noise_level must be in [0, 1)")
        self.base_model = base_model
        self.mask_token_id = int(mask_token_id)
        self.im_end_token_id = int(im_end_token_id)
        self.block_size = block_size
        self.minimum_noise_level = minimum_noise_level
        # Backward-compatible default: historical DLM runs optimize the
        # equally weighted MDM and causal objectives. New experiments must
        # opt out explicitly through ``casuallossenable``.
        self.casuallossenable = True
        self.mdm_loss_weight = 0.5
        self.causal_loss_weight = 0.5
        self.config = base_model.config
        self.gradient_checkpointing = False
        self.language_checkpoint_stride = 0
        # Padding-free DLM uses a separate, opt-in checkpoint cadence.  A
        # stride of one preserves the established route (every language
        # layer); larger values checkpoint every Nth layer uniformly for all
        # samples.  The legacy non-packed stride above remains untouched.
        self.packing_language_checkpoint_stride = 1
        self.padding_free_packing = False
        self.packing_linear_kernel = "chunk"
        self.packing_activation_cpu_offload = False
        self.packing_offload_threshold_bytes = 8 << 20
        self.packing_activation_offload_min_tokens = 0
        self.packing_activation_offload_layer_stride = 1
        self.packing_activation_offload_pin_memory = True
        self.packing_lm_head_loss_backend = "fused"
        self.packing_lm_head_loss_chunk_tokens = 128
        self._last_packing_activation_cpu_offload = False
        self._last_packing_offloaded_tensor_count = 0
        self._last_packing_offloaded_bytes = 0
        self._gradient_checkpointing_kwargs: dict[str, Any] = {
            "use_reentrant": False
        }

    @property
    def multimodal_model(self) -> nn.Module:
        return self.base_model.model

    @property
    def language_model(self) -> nn.Module:
        return self.multimodal_model.language_model

    @property
    def lm_head(self) -> nn.Module:
        return self.base_model.lm_head

    def _checkpoint_language_layer(self, layer_index: int) -> bool:
        if not self.training:
            return False
        if self.gradient_checkpointing:
            return True
        stride = self.language_checkpoint_stride
        return stride > 0 and (layer_index + 1) % stride == 0

    def _checkpoint_full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        clean_causal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from torch.utils.checkpoint import checkpoint

        def custom_forward(
            noisy_states: torch.Tensor,
            clean_states: torch.Tensor,
            noisy_cos: torch.Tensor,
            noisy_sin: torch.Tensor,
            clean_cos: torch.Tensor,
            clean_sin: torch.Tensor,
            noisy_states_valid: torch.Tensor,
            clean_states_valid: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._full_layer(
                layer,
                noisy_states,
                clean_states,
                (noisy_cos, noisy_sin),
                (clean_cos, clean_sin),
                block_mask,
                noisy_states_valid,
                clean_states_valid,
                clean_causal_mask,
            )

        return checkpoint(
            custom_forward,
            noisy,
            clean,
            noisy_position_embeddings[0],
            noisy_position_embeddings[1],
            clean_position_embeddings[0],
            clean_position_embeddings[1],
            noisy_valid,
            clean_valid,
            **self._gradient_checkpointing_kwargs,
        )

    @staticmethod
    def _block_mask(
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        noisy_turn: torch.Tensor,
        clean_turn: torch.Tensor,
        num_heads: int,
    ) -> Any:
        from torch.nn.attention.flex_attention import create_block_mask

        noisy_length = noisy_valid.shape[1]
        total_length = noisy_length + clean_valid.shape[1]
        clean_batch_size = clean_valid.shape[0]
        noisy_batch_size = noisy_valid.shape[0]
        if noisy_batch_size not in (clean_batch_size, 2 * clean_batch_size):
            raise ValueError(
                "noisy batch must contain one inference view or two complementary training views per clean sample"
            )

        def mask_mod(
            b: torch.Tensor, h: torch.Tensor, q: torch.Tensor, kv: torch.Tensor
        ) -> torch.Tensor:
            del h
            kv_clean = kv >= noisy_length
            kv_pos = torch.where(kv_clean, kv - noisy_length, kv)
            clean_b = b % clean_batch_size
            q_noisy_valid = noisy_valid[
                b, torch.clamp(q, max=noisy_length - 1)
            ]
            kv_noisy_valid = noisy_valid[
                b, torch.clamp(kv_pos, max=noisy_length - 1)
            ]
            kv_clean_valid = clean_valid[
                clean_b, torch.clamp(kv_pos, max=clean_valid.shape[1] - 1)
            ]
            q_valid = q_noisy_valid
            kv_valid = torch.where(kv_clean, kv_clean_valid, kv_noisy_valid)
            q_turn = noisy_turn[b, torch.clamp(q, max=noisy_turn.shape[1] - 1)]
            kv_turn = torch.where(
                kv_clean,
                clean_turn[
                    clean_b, torch.clamp(kv_pos, max=clean_turn.shape[1] - 1)
                ],
                noisy_turn[
                    b, torch.clamp(kv_pos, max=noisy_turn.shape[1] - 1)
                ],
            )
            n2n = ~kv_clean & q_turn.eq(kv_turn)
            n2c = kv_clean & q_turn.gt(kv_turn)
            allowed = q_valid & kv_valid & (n2n | n2c)
            # Flex attention requires at least one key per query. Padding rows
            # attend only to their noisy-stream diagonal and are zeroed after
            # each layer.
            return allowed | (~q_valid & q.eq(kv))

        return create_block_mask(
            mask_mod,
            B=noisy_batch_size,
            H=num_heads,
            Q_LEN=noisy_length,
            KV_LEN=total_length,
            device=str(noisy_valid.device),
            _compile=os.environ.get(STABLE_BLOCK_MASK_ENV, "0") != "1",
        )

    def _full_layer(
        self,
        layer: nn.Module,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
        noisy_valid: torch.Tensor,
        clean_valid: torch.Tensor,
        clean_causal_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_residual, clean_residual = noisy, clean
        noisy_norm = layer.input_layernorm(noisy)
        clean_norm = layer.input_layernorm(clean)
        noisy = self._flex_qwen_attention(
            layer.self_attn,
            noisy_norm,
            clean_norm,
            noisy_position_embeddings,
            clean_position_embeddings,
            block_mask,
        )
        clean, _ = layer.self_attn(
            hidden_states=clean_norm,
            position_embeddings=clean_position_embeddings,
            attention_mask=clean_causal_mask,
            past_key_values=None,
        )
        noisy = noisy_residual + noisy
        clean = clean_residual + clean
        noisy_residual, clean_residual = noisy, clean
        noisy = noisy_residual + layer.mlp(
            layer.post_attention_layernorm(noisy)
        )
        clean = clean_residual + layer.mlp(
            layer.post_attention_layernorm(clean)
        )
        noisy = noisy * noisy_valid.unsqueeze(-1).to(noisy.dtype)
        clean = clean * clean_valid.unsqueeze(-1).to(clean.dtype)
        return noisy, clean

    def _build_inference_streams(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        clean_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        block_start: int,
    ) -> dict[str, torch.Tensor]:
        """Build one Direct-Conversion draft view for batch-size-one inference.

        ``block_start`` is the known first token of the speculative block.  All
        tokens from there onward form the current noisy turn.  Earlier text is
        retained because Qwen3.5 linear-attention layers carry recurrent state;
        vision placeholders remain clean-stream-only, exactly as in training.
        """

        if input_ids.shape[0] != 1:
            raise ValueError(
                "DLM block drafting currently requires batch size 1"
            )
        valid = attention_mask[0].bool()
        if not bool(valid.all()):
            raise ValueError("DLM block drafting expects an unpadded sequence")
        sequence_length = input_ids.shape[1]
        if not 0 <= block_start < sequence_length:
            raise ValueError(
                f"invalid block_start={block_start} for length={sequence_length}"
            )

        ids = input_ids[0]
        cfg = self.config
        vision_ids = {
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        }
        text = torch.ones_like(ids, dtype=torch.bool)
        for token_id in vision_ids:
            text &= ids.ne(token_id)

        # The current block is one new turn.  N2N is bidirectional within this
        # turn; strict N2C (q_turn > kv_turn) exposes only completed context.
        clean_turn = torch.zeros_like(ids)
        clean_turn[block_start:] = 1
        noisy_turn = clean_turn[text]
        noisy_ids = ids[text]
        noisy = self.language_model.embed_tokens(noisy_ids).unsqueeze(0)
        noisy_positions = position_ids[:, 0, text].unsqueeze(1)
        return {
            "noisy": noisy,
            "noisy_valid": torch.ones_like(
                noisy_ids, dtype=torch.bool
            ).unsqueeze(0),
            "noisy_turn": noisy_turn.unsqueeze(0),
            "noisy_positions": noisy_positions,
            "clean": clean_embeds,
            "clean_valid": attention_mask.bool(),
            "clean_turn": clean_turn.unsqueeze(0),
            "clean_positions": position_ids,
        }

    @torch.inference_mode()
    def draft_block_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_start: int,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        precomputed_clean_embeds: torch.Tensor | None = None,
        precomputed_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits for every position after ``block_start`` in one NFE.

        The returned tensor contains predictions only for positions strictly
        after ``block_start``.  Token shift matches the training loss: hidden
        state at position ``i-1`` predicts token ``i``.
        """

        was_training = self.training
        self.eval()
        if (precomputed_clean_embeds is None) != (
            precomputed_position_ids is None
        ):
            raise ValueError(
                "clean embeddings and position ids must be precomputed together"
            )
        if precomputed_clean_embeds is None:
            clean_embeds, position_ids = self._embed_clean(
                input_ids,
                attention_mask,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                mm_token_type_ids,
            )
        else:
            clean_embeds = precomputed_clean_embeds
            position_ids = precomputed_position_ids
            if clean_embeds.shape[:2] != input_ids.shape:
                raise ValueError(
                    "precomputed clean embeddings do not match input ids"
                )
            if position_ids.shape[1:] != input_ids.shape:
                raise ValueError(
                    "precomputed position ids do not match input ids"
                )
        streams = self._build_inference_streams(
            input_ids,
            attention_mask,
            clean_embeds,
            position_ids,
            block_start,
        )
        dump_streams_path = os.environ.get("GAM_DLM_DUMP_DLM_STREAMS_PATH")
        if dump_streams_path and not os.path.exists(dump_streams_path):
            # Tiny semantic-audit artifact: embeddings/positions before layer
            # 0.  It is opt-in and CPU-only; ordinary inference/training is
            # untouched.  The caller can compare the current block segment
            # with SGLang without serializing the full model state.
            torch.save(
                {
                    "input_ids": input_ids.detach().cpu(),
                    "block_start": int(block_start),
                    "noisy": streams["noisy"].detach().float().cpu(),
                    "clean": streams["clean"].detach().float().cpu(),
                    "noisy_positions": streams["noisy_positions"]
                    .detach()
                    .cpu(),
                    "clean_positions": streams["clean_positions"]
                    .detach()
                    .cpu(),
                    "noisy_turn": streams["noisy_turn"].detach().cpu(),
                    "clean_turn": streams["clean_turn"].detach().cpu(),
                },
                dump_streams_path,
            )
        noisy_hidden, _ = self._hybrid_language_forward(streams)

        cfg = self.config
        text = torch.ones_like(input_ids[0], dtype=torch.bool)
        for token_id in (
            int(cfg.image_token_id),
            int(cfg.video_token_id),
            int(cfg.vision_start_token_id),
        ):
            text &= input_ids[0].ne(token_id)
        text_indices = torch.nonzero(text, as_tuple=False).flatten()
        target_positions = torch.arange(
            block_start + 1,
            input_ids.shape[1],
            device=input_ids.device,
        )
        # map original positions to compact noisy-stream positions
        compact = torch.searchsorted(text_indices, target_positions)
        if not torch.equal(text_indices[compact], target_positions):
            raise RuntimeError(
                "speculative target unexpectedly contains a vision placeholder"
            )
        prediction_hidden = noisy_hidden[0, compact - 1]
        logits = self.lm_head(prediction_hidden)
        if was_training:
            self.train()
        return logits

    @staticmethod
    def _extend_optional_sequence(
        value: torch.Tensor | None, count: int, fill: int = 0
    ) -> torch.Tensor | None:
        if value is None:
            return None
        extension = torch.full(
            (value.shape[0], count),
            fill,
            dtype=value.dtype,
            device=value.device,
        )
        return torch.cat([value, extension], dim=1)

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        history: torch.Tensor,
        penalty: float,
    ) -> torch.Tensor:
        """Apply the Transformers greedy repetition-penalty contract."""

        if penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")
        if penalty == 1.0:
            return logits
        penalized = logits.clone()
        seen_scores = torch.gather(penalized, 1, history)
        seen_scores = torch.where(
            seen_scores < 0,
            seen_scores * penalty,
            seen_scores / penalty,
        )
        penalized.scatter_(1, history, seen_scores)
        return penalized

    @staticmethod
    def _generation_position_ids(
        cache_positions: torch.Tensor,
        axes: int,
        batch_size: int,
        rope_deltas: torch.Tensor | None,
    ) -> torch.Tensor:
        """Build Qwen3.5 mRoPE positions for generated text tokens."""

        positions = cache_positions.view(1, 1, -1).expand(axes, batch_size, -1)
        if rope_deltas is None:
            return positions
        deltas = rope_deltas.to(
            device=cache_positions.device, dtype=cache_positions.dtype
        )
        deltas = deltas.reshape(batch_size, -1)[:, :1]
        return positions + deltas.view(1, batch_size, 1)

    @torch.inference_mode()
    def draft_block(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        block_start: int,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        precomputed_clean_embeds: torch.Tensor | None = None,
        precomputed_position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict every position after ``block_start`` in one DLM NFE."""

        logits = self.draft_block_logits(
            input_ids,
            attention_mask,
            block_start,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            precomputed_clean_embeds=precomputed_clean_embeds,
            precomputed_position_ids=precomputed_position_ids,
        )
        return logits.argmax(dim=-1)

    @torch.inference_mode()
    def speculative_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        inference_block_size: int = 16,
        stop_token_ids: tuple[int, ...] = (),
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        cuda_cache_cleanup_interval_tokens: int = 0,
        cuda_cache_cleanup_fraction: float = 0.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        """Greedy self-speculative decoding with an exact Qwen3.5 verifier.

        Qwen3.5 interleaves full attention and recurrent GatedDeltaNet layers.
        A multi-token causal chunk is mathematically equivalent to incremental
        decoding, but its convolution/recurrent kernels need not produce the
        same floating-point cache state as one-token cached generation.  Even
        when every immediate argmax agrees, promoting that chunk state can
        therefore change a later near-tied token.  The verifier deliberately
        advances the real cache one token at a time and never promotes a
        speculative chunk cache.  Rejected draft tokens cannot contaminate
        either GatedDeltaNet state or full-attention KV.
        """

        if input_ids.shape[0] != 1:
            raise ValueError(
                "DLM speculative generation currently requires batch size 1"
            )
        if inference_block_size < 2 or inference_block_size > self.block_size:
            raise ValueError(
                f"inference_block_size must be in [2, {self.block_size}], got {inference_block_size}"
            )
        if max_new_tokens < 1:
            return input_ids[:, :0]
        if cuda_cache_cleanup_interval_tokens < 0:
            raise ValueError(
                "cuda_cache_cleanup_interval_tokens must be non-negative"
            )
        if not 0.0 <= cuda_cache_cleanup_fraction <= 1.0:
            raise ValueError("cuda_cache_cleanup_fraction must be in [0, 1]")
        if repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be strictly positive")
        if no_repeat_ngram_size == 1 or no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size must be 0 or at least 2")

        prompt_length = input_ids.shape[1]
        generated = input_ids
        generated_attention = attention_mask
        generated_mm_types = mm_token_type_ids
        stop_ids = set(int(value) for value in stop_token_ids)
        accepted_draft_tokens = 0
        drafted_tokens = 0
        draft_nfe = 0
        causal_nfe = 0
        promoted_cache_blocks = 0
        rejection_cache_commits = 0
        sequential_fallback_blocks = 0
        sequential_verifier_blocks = 0
        cuda_cache_cleanups = 0
        last_cache_check_tokens = 0
        peak_cuda_reserved_bytes = 0
        repeated_ngram_suppressions = 0

        def suppress_repeated_ngram(
            logits: torch.Tensor, history: torch.Tensor
        ) -> None:
            nonlocal repeated_ngram_suppressions
            if no_repeat_ngram_size < 2:
                return
            # Match the standard causal no-repeat contract: the prompt is part
            # of the token history, even though only generated tokens are
            # returned.  Slicing it away permits the first repeated n-gram to
            # straddle the prompt/response boundary and made the verifier
            # safety weaker than ordinary causal decoding.
            tokens = history[0].tolist()
            if len(tokens) < no_repeat_ngram_size - 1:
                return
            suffix = tuple(tokens[-(no_repeat_ngram_size - 1) :])
            banned = {
                tokens[index + no_repeat_ngram_size - 1]
                for index in range(len(tokens) - no_repeat_ngram_size + 1)
                if tuple(tokens[index : index + no_repeat_ngram_size - 1])
                == suffix
            }
            for token in banned:
                if 0 <= token < logits.shape[-1] and torch.isfinite(
                    logits[0, token]
                ):
                    logits[0, token] = -torch.inf
                    repeated_ngram_suppressions += 1

        def update_progress(active: bool = True) -> None:
            self._inflight_generation_stats = {
                "active": active,
                "output_tokens": int(generated.shape[1] - prompt_length),
                "nfe": draft_nfe + causal_nfe,
                "draft_nfe": draft_nfe,
                "causal_nfe": causal_nfe,
                "promoted_cache_blocks": promoted_cache_blocks,
                "rejection_cache_commits": rejection_cache_commits,
                "sequential_fallback_blocks": sequential_fallback_blocks,
                "sequential_verifier_blocks": sequential_verifier_blocks,
                "cuda_cache_cleanups": cuda_cache_cleanups,
                "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
            }

        def maybe_release_fragmented_cuda_cache() -> None:
            """Bound allocator fragmentation from monotonically growing prefixes.

            Each draft iteration creates a new full-prefix tensor shape.
            CUDA's caching allocator otherwise retains the old
            blocks even though live allocations stay small.  Releasing only
            when reserved memory is both above a device fraction and more than
            twice live memory is numerically inert and avoids work on normal
            short structured responses.
            """

            nonlocal cuda_cache_cleanups, last_cache_check_tokens, peak_cuda_reserved_bytes
            if (
                cuda_cache_cleanup_interval_tokens == 0
                or not generated.is_cuda
            ):
                return
            output_tokens = int(generated.shape[1] - prompt_length)
            if (
                output_tokens - last_cache_check_tokens
                < cuda_cache_cleanup_interval_tokens
            ):
                return
            last_cache_check_tokens = output_tokens
            reserved = int(torch.cuda.memory_reserved(generated.device))
            allocated = int(torch.cuda.memory_allocated(generated.device))
            peak_cuda_reserved_bytes = max(peak_cuda_reserved_bytes, reserved)
            total = int(
                torch.cuda.get_device_properties(generated.device).total_memory
            )
            if (
                reserved < int(total * cuda_cache_cleanup_fraction)
                or reserved <= 2 * allocated
            ):
                return
            torch.cuda.empty_cache()
            cuda_cache_cleanups += 1

        # Encode vision and exact multimodal positions once.  Subsequent draft
        # iterations append only ordinary text/mask embeddings and positions.
        prompt_clean_embeds, prompt_position_ids = self._embed_clean(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            mm_token_type_ids,
        )
        rope_deltas = getattr(self.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        position_axes = int(prompt_position_ids.shape[0])
        batch_size = int(input_ids.shape[0])

        prompt_cache_positions = torch.arange(
            prompt_length, device=input_ids.device
        )
        causal_outputs = self.language_model(
            input_ids=None,
            inputs_embeds=prompt_clean_embeds,
            position_ids=self._native_position_ids(prompt_position_ids),
            attention_mask=attention_mask,
            past_key_values=None,
            use_cache=True,
            cache_position=prompt_cache_positions,
            return_dict=True,
        )
        causal_cache = causal_outputs.past_key_values
        if causal_cache is None:
            raise RuntimeError("Qwen3.5 causal prefill did not return a cache")
        first_logits = self.lm_head(causal_outputs.last_hidden_state[:, -1])
        first_logits = self._apply_repetition_penalty(
            first_logits, generated, repetition_penalty
        )
        suppress_repeated_ngram(first_logits, generated)
        next_token = first_logits.argmax(dim=-1, keepdim=True)
        causal_nfe += 1
        generated = torch.cat([generated, next_token], dim=1)
        generated_attention = self._extend_optional_sequence(
            generated_attention, 1, fill=1
        )
        generated_mm_types = self._extend_optional_sequence(
            generated_mm_types, 1, fill=0
        )
        update_progress()

        def cached_causal_logits(
            tokens_to_commit: torch.Tensor, cache: Any
        ) -> torch.Tensor:
            nonlocal causal_nfe
            cache_start = int(cache.get_seq_length())
            cache_position = torch.arange(
                cache_start,
                cache_start + tokens_to_commit.shape[1],
                dtype=torch.long,
                device=tokens_to_commit.device,
            )
            position_ids = self._generation_position_ids(
                cache_position,
                position_axes,
                batch_size,
                rope_deltas,
            )
            outputs = self.language_model(
                input_ids=tokens_to_commit,
                attention_mask=None,
                position_ids=self._native_position_ids(position_ids),
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
                return_dict=True,
            )
            causal_nfe += 1
            return self.lm_head(outputs.last_hidden_state)

        def exact_verify_block(
            drafted: torch.Tensor,
            remaining: int,
        ) -> tuple[torch.Tensor, int, bool]:
            """Verify and commit against the native one-token cache contract."""

            parts: list[torch.Tensor] = []
            accepted_count = 0
            stopped = False
            commit_token = generated[:, -1:]
            for index in range(drafted.shape[1]):
                history = torch.cat([generated, drafted[:, :index]], dim=1)
                logits = cached_causal_logits(commit_token, causal_cache)[
                    :, -1
                ]
                logits = self._apply_repetition_penalty(
                    logits, history, repetition_penalty
                )
                suppress_repeated_ngram(logits, history)
                verifier_token = logits.argmax(dim=-1, keepdim=True)
                draft_token = drafted[:, index : index + 1]
                if not torch.equal(verifier_token, draft_token):
                    parts.append(verifier_token)
                    stopped = int(verifier_token[0, 0]) in stop_ids
                    break
                parts.append(draft_token)
                accepted_count += 1
                if int(draft_token[0, 0]) in stop_ids:
                    stopped = True
                    break
                commit_token = draft_token
            else:
                if drafted.shape[1] < remaining:
                    history = torch.cat([generated, drafted], dim=1)
                    logits = cached_causal_logits(commit_token, causal_cache)[
                        :, -1
                    ]
                    logits = self._apply_repetition_penalty(
                        logits, history, repetition_penalty
                    )
                    suppress_repeated_ngram(logits, history)
                    verifier_token = logits.argmax(dim=-1, keepdim=True)
                    parts.append(verifier_token)
                    stopped = int(verifier_token[0, 0]) in stop_ids
            return torch.cat(parts, dim=1), accepted_count, stopped

        while generated.shape[1] - prompt_length < max_new_tokens:
            if int(generated[0, -1]) in stop_ids:
                break
            remaining = max_new_tokens - (generated.shape[1] - prompt_length)
            draft_width = min(inference_block_size - 1, remaining)
            if draft_width <= 0:
                break
            block_start = generated.shape[1] - 1
            masks = torch.full(
                (1, draft_width),
                self.mask_token_id,
                dtype=generated.dtype,
                device=generated.device,
            )
            draft_input = torch.cat([generated, masks], dim=1)
            draft_attention = self._extend_optional_sequence(
                generated_attention, draft_width, fill=1
            )
            draft_mm_types = self._extend_optional_sequence(
                generated_mm_types, draft_width, fill=0
            )
            draft_suffix_embeds = self.language_model.embed_tokens(
                draft_input[:, prompt_length:]
            )
            draft_clean_embeds = torch.cat(
                [prompt_clean_embeds, draft_suffix_embeds], dim=1
            )
            draft_cache_positions = torch.arange(
                prompt_length,
                draft_input.shape[1],
                dtype=torch.long,
                device=draft_input.device,
            )
            draft_suffix_positions = self._generation_position_ids(
                draft_cache_positions,
                position_axes,
                batch_size,
                rope_deltas,
            )
            draft_position_ids = torch.cat(
                [prompt_position_ids, draft_suffix_positions], dim=2
            )
            drafted = self.draft_block(
                draft_input,
                draft_attention,
                block_start,
                mm_token_type_ids=draft_mm_types,
                precomputed_clean_embeds=draft_clean_embeds,
                precomputed_position_ids=draft_position_ids,
            ).unsqueeze(0)
            draft_nfe += 1
            drafted_tokens += int(drafted.numel())

            # Never promote or commit a multi-token verifier cache.  The
            # accepted prefix is replayed through the exact one-token path so
            # its recurrent/conv state stays identical to cached causal decode.
            append, accepted, terminated = exact_verify_block(
                drafted, remaining
            )
            sequential_verifier_blocks += 1
            accepted_draft_tokens += accepted

            generated = torch.cat([generated, append], dim=1)
            generated_attention = self._extend_optional_sequence(
                generated_attention, append.shape[1], fill=1
            )
            generated_mm_types = self._extend_optional_sequence(
                generated_mm_types, append.shape[1], fill=0
            )

            # Drop references to the old, differently sized prefix tensors
            # before asking the caching allocator to release inactive blocks.
            del (
                masks,
                draft_input,
                draft_attention,
                draft_mm_types,
                draft_suffix_embeds,
                draft_clean_embeds,
                draft_cache_positions,
                draft_suffix_positions,
                draft_position_ids,
                drafted,
                append,
            )
            maybe_release_fragmented_cuda_cache()
            update_progress()
            if terminated:
                break

        response = generated[:, prompt_length : prompt_length + max_new_tokens]
        nfe = draft_nfe + causal_nfe
        self._last_generation_stats = {
            "output_tokens": int(response.shape[1]),
            "nfe": nfe,
            "draft_nfe": draft_nfe,
            "causal_nfe": causal_nfe,
            "promoted_cache_blocks": promoted_cache_blocks,
            "rejection_cache_commits": rejection_cache_commits,
            "sequential_fallback_blocks": sequential_fallback_blocks,
            "sequential_verifier_blocks": sequential_verifier_blocks,
            "drafted_tokens": drafted_tokens,
            "accepted_draft_tokens": accepted_draft_tokens,
            "no_repeat_ngram_size": no_repeat_ngram_size,
            "repeated_ngram_suppressions": repeated_ngram_suppressions,
            "tokens_per_nfe": float(response.shape[1]) / max(nfe, 1),
            "draft_acceptance": float(accepted_draft_tokens)
            / max(drafted_tokens, 1),
            "causal_cache_tokens": int(causal_cache.get_seq_length()),
            "cuda_cache_cleanups": cuda_cache_cleanups,
            "peak_cuda_reserved_gib": peak_cuda_reserved_bytes / (1 << 30),
        }
        update_progress(active=False)
        return response
