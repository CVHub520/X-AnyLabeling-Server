"""GroundAnything/Kimi-K3-ViT + plain-Qwen3 Direct-Conversion DLM route.

This module deliberately subclasses only the model-independent Direct-
Conversion machinery.  Qwen3.5 keeps its original hybrid GatedDeltaNet route;
plain Qwen3 uses its own 1-D RoPE and GQA attention implementation here.
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn as nn

from .hybrid import (
    GAMQwen35DLM,
    _attention_profile_region,
    _compiled_flex_attention,
)

PACKED_CLEAN_ATTENTION_BACKEND_ENV = "GAM_DLM_PACKED_CLEAN_ATTENTION_BACKEND"


class GAMQwen3DLM(GAMQwen35DLM):
    """Direct Conversion wrapper for GroundAnything-VLM's plain Qwen3 backbone."""

    def _native_position_ids(self, positions):
        return self._text_positions(positions)

    def _embed_clean(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
        patch_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mm_token_type_ids
        model = self.multimodal_model
        embeds = model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_features = model.get_image_features(
                pixel_values,
                image_grid_thw,
                patch_positions=patch_positions,
            )
            image_embeds = torch.cat(image_features, dim=0).to(
                embeds.device, embeds.dtype
            )
            image_mask, _ = model.get_placeholder_mask(
                input_ids,
                embeds,
                image_features=image_embeds,
            )
            embeds = embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_features = model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                patch_positions=patch_positions,
            )
            video_embeds = torch.cat(video_features, dim=0).to(
                embeds.device, embeds.dtype
            )
            _, video_mask = model.get_placeholder_mask(
                input_ids,
                embeds,
                video_features=video_embeds,
            )
            embeds = embeds.masked_scatter(video_mask, video_embeds)

        # GroundAnything's language tower is plain Qwen3, not Qwen-VL mRoPE.  Keep a
        # singleton axis for the inherited stream builder and remove it again
        # before Qwen3RotaryEmbedding.
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 1)
        return embeds, positions.unsqueeze(0)

    @staticmethod
    def _flex_qwen_attention(
        attention: nn.Module,
        noisy_states: torch.Tensor,
        clean_states: torch.Tensor,
        noisy_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        clean_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        block_mask: Any,
    ) -> torch.Tensor:
        from transformers.models.qwen3.modeling_qwen3 import (
            apply_rotary_pos_emb,
        )

        noisy_shape = noisy_states.shape[:-1]
        noisy_q_shape = (*noisy_shape, -1, attention.head_dim)
        noisy_query = attention.q_norm(
            attention.q_proj(noisy_states).view(noisy_q_shape)
        ).transpose(1, 2)
        noisy_key = attention.k_norm(
            attention.k_proj(noisy_states).view(noisy_q_shape)
        ).transpose(1, 2)
        noisy_value = (
            attention.v_proj(noisy_states).view(noisy_q_shape).transpose(1, 2)
        )
        q_pre = noisy_query
        k_pre = noisy_key
        noisy_query, noisy_key = apply_rotary_pos_emb(
            noisy_query,
            noisy_key,
            *noisy_position_embeddings,
        )
        dump_path = os.environ.get("GAM_DLM_DUMP_DLM_QKV_PATH")
        if dump_path and not os.path.exists(dump_path):
            # Layer-0 pre-attention audit only.  The artifact is intentionally
            # opt-in and bounded to this one NFE; it does not participate in
            # the forward graph or alter training/inference numerics.
            torch.save(
                {
                    "noisy_states": noisy_states.detach().float().cpu(),
                    "q_pre": q_pre.detach().float().cpu(),
                    "k_pre": k_pre.detach().float().cpu(),
                    "q": noisy_query.detach().float().cpu(),
                    "k": noisy_key.detach().float().cpu(),
                    "v": noisy_value.detach().float().cpu(),
                    "positions": noisy_position_embeddings[0].detach().cpu(),
                },
                dump_path,
            )

        clean_shape = clean_states.shape[:-1]
        clean_kv_shape = (*clean_shape, -1, attention.head_dim)
        clean_key = attention.k_norm(
            attention.k_proj(clean_states).view(clean_kv_shape)
        ).transpose(1, 2)
        clean_value = (
            attention.v_proj(clean_states).view(clean_kv_shape).transpose(1, 2)
        )
        _, clean_key = apply_rotary_pos_emb(
            clean_key,
            clean_key,
            *clean_position_embeddings,
        )
        repeats = noisy_states.shape[0] // clean_states.shape[0]
        clean_key = clean_key.repeat((repeats, 1, 1, 1))
        clean_value = clean_value.repeat((repeats, 1, 1, 1))
        key = torch.cat([noisy_key, clean_key], dim=2)
        value = torch.cat([noisy_value, clean_value], dim=2)
        with _attention_profile_region("gam_dlm.noisy.flex_attention"):
            output = _compiled_flex_attention(
                noisy_query, key, value, block_mask
            )
        output = output.transpose(1, 2).reshape(*noisy_shape, -1).contiguous()
        return attention.o_proj(output)

    @staticmethod
    def _text_positions(positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 3 or positions.shape[0] != 1:
            raise ValueError(
                f"plain Qwen3 positions must be [1, batch, seq], got {tuple(positions.shape)}"
            )
        return positions[0]

    def _hybrid_language_forward(
        self,
        streams: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noisy, clean = streams["noisy"], streams["clean"]
        noisy_valid, clean_valid = (
            streams["noisy_valid"],
            streams["clean_valid"],
        )
        noisy_positions = self._text_positions(streams["noisy_positions"])
        clean_positions = self._text_positions(streams["clean_positions"])
        noisy_position_embeddings = self.language_model.rotary_emb(
            noisy, noisy_positions
        )
        clean_position_embeddings = self.language_model.rotary_emb(
            clean, clean_positions
        )

        from transformers.masking_utils import create_causal_mask

        clean_causal_mask = create_causal_mask(
            config=self.language_model.config,
            inputs_embeds=clean,
            attention_mask=clean_valid,
            cache_position=torch.arange(clean.shape[1], device=clean.device),
            past_key_values=None,
            position_ids=clean_positions,
        )
        block_mask = self._block_mask(
            noisy_valid,
            clean_valid,
            streams["noisy_turn"],
            streams["clean_turn"],
            int(self.config.text_config.num_attention_heads),
        )
        for layer_index, layer in enumerate(self.language_model.layers):
            if self._checkpoint_language_layer(layer_index):
                noisy, clean = self._checkpoint_full_layer(
                    layer,
                    noisy,
                    clean,
                    noisy_position_embeddings,
                    clean_position_embeddings,
                    block_mask,
                    noisy_valid,
                    clean_valid,
                    clean_causal_mask,
                )
            else:
                noisy, clean = self._full_layer(
                    layer,
                    noisy,
                    clean,
                    noisy_position_embeddings,
                    clean_position_embeddings,
                    block_mask,
                    noisy_valid,
                    clean_valid,
                    clean_causal_mask,
                )
        return self.language_model.norm(noisy), self.language_model.norm(clean)
