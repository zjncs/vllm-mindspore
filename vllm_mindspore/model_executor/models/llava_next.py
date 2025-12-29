# SPDX-License-Identifier: Apache-2.0
# Adapted from
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava_next/modeling_llava_next.py
#
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright 2025 The vLLM team.
# Copyright 2025 The HuggingFace Inc. team.
#
# This file provides a MindSpore-compatible draft implementation for
# LLaVA-NeXT (LLaVA-v1.6) integration in vLLM.

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypeAlias, TypeVar, TypedDict

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from transformers import BatchFeature, LlavaNextConfig, LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import (
    get_anyres_image_grid_shape,
    unpad_image,
)

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.parse import (
    ImageEmbeddingItems,
    ImageProcessorItems,
    ImageSize,
    MultiModalDataItems,
)
from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
)
from vllm.sequence import IntermediateTensors
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.distributed import get_pp_group

from vllm_mindspore.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm_mindspore.model_executor.layers.logits_processor import (
    LogitsProcessor,
)
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
)
from vllm_mindspore.model_executor.model_loader.weight_utils import (
    default_weight_loader,
)
from vllm_mindspore.model_executor.models.attention_mask import (
    MultiModalLowerTriangularMask,
)
from vllm_mindspore.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
)
from vllm_mindspore.model_executor.models.model_base import (
    AttentionWrapper,
    NativeModel,
)
from vllm_mindspore.model_executor.models.llama import LlamaModel
from vllm_mindspore.model_executor.models.qwen2 import Qwen2Model
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    maybe_prefix,
    merge_multimodal_embeddings,
)


def get_num_selected_vision_tokens(
    num_vision_tokens: int,
    strategy: Any,
) -> int:
    if callable(strategy):
        dummy_features = ops.zeros((1, num_vision_tokens, 1), ms.float32)
        selected = strategy(dummy_features)
        return int(selected.shape[1])

    if strategy == "class":
        return 1
    if strategy == "default":
        return num_vision_tokens - 1
    if strategy == "full":
        return num_vision_tokens
    return num_vision_tokens


class LlavaNextImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    pixel_values: Tensor | list[Tensor]
    image_sizes: Tensor | None


class LlavaNextImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: Tensor


LlavaNextImageInputs: TypeAlias = (
    LlavaNextImagePixelInputs | LlavaNextImageEmbeddingInputs
)


@dataclass
class LlavaNextLikeConfig:
    vision_config: dict[str, Any] = field(default_factory=dict)
    image_token_index: int = 32000
    vision_feature_select_strategy: str = "spatial_unpad"
    vision_feature_layer: int | list[int] = 22
    image_grid_pinpoints: list[list[int]] = field(default_factory=list)
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = True


class LlavaMultiModalProjector(nn.Cell):
    """MindSpore port of the LLaVA projector."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str,
        multimodal_projector_bias: bool,
        *,
        prefix: str = "",
    ):
        super().__init__()
        self.linear_1 = ColumnParallelLinear(
            vision_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            prefix=f"{prefix}.linear_1",
        )
        self.act = nn.get_activation(projector_hidden_act)
        self.linear_2 = RowParallelLinear(
            text_hidden_size,
            text_hidden_size,
            bias=multimodal_projector_bias,
            prefix=f"{prefix}.linear_2",
        )

    def construct(self, image_features: Tensor) -> Tensor:
        hidden_states, _ = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


def init_vision_tower_for_llava(
    hf_config: LlavaNextLikeConfig,
    *,
    prefix: str = "",
) -> nn.Cell:
    return CLIPVisionModel(hf_config.vision_config, prefix=prefix)


def _get_config_attr(config: object, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


class CLIPVisionEmbeddings(nn.Cell):

    def __init__(self, vision_config: object):
        super().__init__()
        self.hidden_size = _get_config_attr(vision_config, "hidden_size", 1024)
        self.image_size = _get_config_attr(vision_config, "image_size", 224)
        self.patch_size = _get_config_attr(vision_config, "patch_size", 14)
        self.num_channels = _get_config_attr(vision_config, "num_channels", 3)
        self.layer_norm_eps = _get_config_attr(vision_config, "layer_norm_eps",
                                               1e-5)

        self.patch_embedding = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            has_bias=False,
        )
        num_patches = (self.image_size // self.patch_size) ** 2
        self.class_embedding = Parameter(
            ops.zeros((self.hidden_size,), ms.float32),
            name="class_embedding",
        )
        self.position_embedding = Parameter(
            ops.zeros((1, num_patches + 1, self.hidden_size), ms.float32),
            name="position_embedding",
        )

    def interpolate_pos_encoding(self, embeddings: Tensor,
                                 height: int, width: int) -> Tensor:
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embedding.shape[1] - 1
        if num_patches == num_positions and height == width == self.image_size:
            return self.position_embedding

        class_pos_embed = self.position_embedding[:, :1, :]
        patch_pos_embed = self.position_embedding[:, 1:, :]
        grid_size = int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            (1, grid_size, grid_size, self.hidden_size))
        patch_pos_embed = ops.transpose(patch_pos_embed, (0, 3, 1, 2))

        new_height = height // self.patch_size
        new_width = width // self.patch_size
        patch_pos_embed = ops.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = ops.transpose(patch_pos_embed, (0, 2, 3, 1))
        patch_pos_embed = patch_pos_embed.reshape(
            (1, new_height * new_width, self.hidden_size))

        return ops.concat((class_pos_embed, patch_pos_embed), axis=1)

    def construct(self, pixel_values: Tensor) -> Tensor:
        batch_size, _, height, width = pixel_values.shape
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = ops.transpose(patch_embeds, (0, 2, 3, 1))
        patch_embeds = patch_embeds.reshape(
            (batch_size, -1, self.hidden_size))

        class_embeds = ops.expand_dims(self.class_embedding, 0)
        class_embeds = ops.tile(class_embeds, (batch_size, 1, 1))
        embeddings = ops.concat((class_embeds, patch_embeds), axis=1)

        position_embedding = self.interpolate_pos_encoding(
            embeddings, height=height, width=width)
        return embeddings + position_embedding


class CLIPVisionEncoderLayer(nn.Cell):

    def __init__(self, vision_config: object):
        super().__init__()
        self.hidden_size = _get_config_attr(vision_config, "hidden_size", 1024)
        self.num_attention_heads = _get_config_attr(vision_config,
                                                    "num_attention_heads",
                                                    16)
        self.intermediate_size = _get_config_attr(vision_config,
                                                  "intermediate_size", 4096)
        self.layer_norm_eps = _get_config_attr(vision_config,
                                               "layer_norm_eps", 1e-5)

        self.self_attn = nn.MultiheadAttention(
            self.hidden_size,
            self.num_attention_heads,
            batch_first=True,
        )
        self.layer_norm1 = nn.LayerNorm(
            (self.hidden_size,), epsilon=self.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(
            (self.hidden_size,), epsilon=self.layer_norm_eps)
        self.mlp_fc1 = nn.Dense(self.hidden_size, self.intermediate_size)
        self.mlp_act = nn.GELU()
        self.mlp_fc2 = nn.Dense(self.intermediate_size, self.hidden_size)

    def construct(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output = self.self_attn(hidden_states, hidden_states,
                                     hidden_states)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp_fc2(self.mlp_act(self.mlp_fc1(hidden_states)))
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPVisionEncoder(nn.Cell):

    def __init__(self, vision_config: object):
        super().__init__()
        num_layers = _get_config_attr(vision_config, "num_hidden_layers", 24)
        self.layers = nn.CellList(
            [CLIPVisionEncoderLayer(vision_config) for _ in range(num_layers)]
        )

    def construct(
        self,
        hidden_states: Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        all_hidden_states: list[Tensor] | None = [] if output_hidden_states else None
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class CLIPVisionTransformer(nn.Cell):

    def __init__(self, vision_config: object):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(vision_config)
        self.pre_layrnorm = nn.LayerNorm(
            (self.embeddings.hidden_size,),
            epsilon=self.embeddings.layer_norm_eps,
        )
        self.encoder = CLIPVisionEncoder(vision_config)
        self.post_layernorm = nn.LayerNorm(
            (self.embeddings.hidden_size,),
            epsilon=self.embeddings.layer_norm_eps,
        )

    def construct(
        self,
        pixel_values: Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> tuple[Tensor, list[Tensor] | None]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states, hidden_states_list = self.encoder(
            hidden_states, output_hidden_states=output_hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states, hidden_states_list


class CLIPVisionModel(nn.Cell):

    def __init__(self, vision_config: object, *, prefix: str = ""):
        super().__init__()
        self.vision_model = CLIPVisionTransformer(vision_config)

    def construct(
        self,
        pixel_values: Tensor,
        *,
        select_layers: Optional[list[int]] = None,
        feature_select_strategy: str | None = None,
    ) -> Tensor:
        output_hidden_states = select_layers is not None
        last_hidden, hidden_states = self.vision_model(
            pixel_values, output_hidden_states=output_hidden_states)
        if select_layers is None:
            return last_hidden
        if hidden_states is None:
            raise ValueError("Hidden states are not available for selection.")
        selected = [hidden_states[idx] for idx in select_layers]
        return ops.concat(selected, axis=-1)


class LlavaNextProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> LlavaNextLikeConfig:
        return self.ctx.get_hf_config(LlavaNextConfig)

    def get_hf_processor(self, **kwargs: object) -> LlavaNextProcessor:
        hf_processor = self.ctx.get_hf_processor(LlavaNextProcessor, **kwargs)
        if getattr(hf_processor, "patch_size", None) is None:
            vision_config = self.get_hf_config().vision_config
            patch_size = getattr(vision_config, "patch_size", 14)
            hf_processor.patch_size = patch_size
        return hf_processor

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_num_image_tokens(
        self,
        *,
        image_width: int,
        image_height: int,
    ) -> int:
        hf_config = self.get_hf_config()
        vision_config = hf_config.vision_config

        patch_size = getattr(vision_config, "patch_size", 1)
        base_image_size = getattr(vision_config, "image_size", patch_size)
        base_tokens = (image_height // patch_size) * (image_width // patch_size)

        base_feature_size = get_num_selected_vision_tokens(
            base_tokens, hf_config.vision_feature_select_strategy
        )

        num_patch_height, num_patch_width = get_anyres_image_grid_shape(
            image_size=(image_height, image_width),
            grid_pinpoints=hf_config.image_grid_pinpoints,
            patch_size=base_image_size,
        )

        (
            unpadded_feature_size,
            newline_feature_size,
        ) = self._get_num_unpadded_features(
            original_height=image_height,
            original_width=image_width,
            npatches=base_image_size // patch_size,
            num_patch_height=num_patch_height,
            num_patch_width=num_patch_width,
        )

        return unpadded_feature_size + newline_feature_size + base_feature_size

    def _get_num_unpadded_features(
        self,
        *,
        original_height: int,
        original_width: int,
        npatches: int,
        num_patch_height: int,
        num_patch_width: int,
    ) -> tuple[int, int]:
        current_height = npatches * num_patch_height
        current_width = npatches * num_patch_width

        aspect_ratio = original_width / original_height
        current_aspect_ratio = current_width / current_height

        if aspect_ratio > current_aspect_ratio:
            new_height = int(
                round(original_height * (current_width / original_width), 7)
            )
            padding = (current_height - new_height) // 2
            current_height = current_height - (2 * padding)
        else:
            new_width = int(
                round(original_width * (current_height / original_height), 7)
            )
            padding = (current_width - new_width) // 2
            current_width = current_width - (2 * padding)

        unpadded_features = current_height * current_width
        newline_features = current_height
        return (unpadded_features, newline_features)

    def get_image_size_with_most_features(self) -> ImageSize:
        hf_config = self.get_hf_config()

        largest_feature_size, largest_feature_pinpoint = 0, None
        for height, width in hf_config.image_grid_pinpoints:
            feat_size = self.get_num_image_tokens(
                image_width=width, image_height=height
            )
            if feat_size > largest_feature_size:
                largest_feature_size = feat_size
                largest_feature_pinpoint = ImageSize(width=width, height=height)

        if largest_feature_pinpoint is None:
            return ImageSize(width=336, height=336)

        return largest_feature_pinpoint


_I = TypeVar("_I", bound=LlavaNextProcessingInfo)


class BaseLlavaNextMultiModalProcessor(BaseMultiModalProcessor[_I]):
    @abstractmethod
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        raise NotImplementedError

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: Mapping[str, object],
    ):
        hf_config = self.info.get_hf_config()
        image_token_id = hf_config.image_token_index

        def get_replacement(item_idx: int):
            images = mm_items.get_items(
                "image", (ImageEmbeddingItems, ImageProcessorItems)
            )

            if isinstance(images, ImageEmbeddingItems):
                num_image_tokens = images.get_feature_size(item_idx)
            else:
                image_size = images.get_image_size(item_idx)
                num_image_tokens = self.info.get_num_image_tokens(
                    image_width=image_size.width,
                    image_height=image_size.height,
                )

            return [image_token_id] * num_image_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            ),
        ]


class LlavaNextMultiModalProcessor(
    BaseLlavaNextMultiModalProcessor[LlavaNextProcessingInfo]
):
    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            image_sizes=MultiModalFieldConfig.batched("image"),
            image_embeds=MultiModalFieldConfig.batched("image"),
        )


@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=LlavaNextProcessingInfo,
    dummy_inputs=None,
)
class LlavaNextForConditionalGeneration(NativeModel, SupportsMultiModal):
    supports_multimodal: Literal[True] = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<image>"
        raise ValueError("Only image modality is supported")

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        text_config = config.text_config

        vision_feature_layer = config.vision_feature_layer
        if isinstance(vision_feature_layer, int):
            vision_hidden_size = config.vision_config.hidden_size
            self.select_layers: Optional[list[int]] = None
        elif isinstance(vision_feature_layer, (list, tuple)):
            vision_hidden_size = config.vision_config.hidden_size * len(
                vision_feature_layer
            )
            self.select_layers = list(vision_feature_layer)
        else:
            raise TypeError(
                f"vision_layer_feature type: {type(vision_feature_layer)}"
                " is not supported"
            )

        self.llava_config = config
        self.text_config = text_config
        self.config = text_config
        self.multimodal_config = multimodal_config

        self.vision_tower = init_vision_tower_for_llava(
            config,
            prefix=maybe_prefix(prefix, "vision_tower"),
        )
        self.image_newline = Parameter(
            ops.zeros((text_config.hidden_size,), ms.float32),
            name="image_newline",
        )
        self.multi_modal_projector = LlavaMultiModalProjector(
            vision_hidden_size=vision_hidden_size,
            text_hidden_size=text_config.hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
            prefix=maybe_prefix(prefix, "multi_modal_projector"),
        )

        text_model_type = getattr(text_config, "model_type", None)
        original_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = text_config
        if text_model_type in {"llama", "mistral"}:
            self.language_model = LlamaModel(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model.model"),
            )
        elif text_model_type in {"qwen2", "qwen2_moe"}:
            self.language_model = Qwen2Model(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "language_model.model"),
            )
        else:
            vllm_config.model_config.hf_config = original_hf_config
            raise ValueError(
                f"Unsupported text model type: {text_model_type}"
            )
        vllm_config.model_config.hf_config = original_hf_config

        if get_pp_group().is_last_rank:
            if text_config.tie_word_embeddings:
                self.lm_head = self.language_model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    text_config.vocab_size,
                    text_config.hidden_size,
                    org_num_embeddings=text_config.vocab_size,
                    padding_size=(
                        DEFAULT_VOCAB_PADDING_SIZE
                        if not self.lora_config else
                        self.lora_config.lora_vocab_padding_size),
                    quant_config=self.quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
            logit_scale = getattr(text_config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(text_config.vocab_size,
                                                    text_config.vocab_size,
                                                    logit_scale)
        else:
            self.lm_head = PPMissingLayer()

        self.model = self.language_model
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self._common_multimodal_preprocess(vllm_config, prefix)

    def _common_multimodal_preprocess(self, vllm_config, prefix=""):
        self.set_modules({"model": self.model, "lm_head": self.lm_head})
        self.casual_mask = MultiModalLowerTriangularMask(
            dtype=self.model_config.dtype,
            max_model_len=self.model_config.max_model_len)
        self.kv_caches = [
            AttentionWrapper() for _ in range(self.text_config.num_hidden_layers)
        ]
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        for i in range(self.text_config.num_hidden_layers):
            compilation_config.static_forward_context[str(
                i)] = self.kv_caches[i]

    def _parse_and_validate_image_input(
        self, **kwargs: object
    ) -> LlavaNextImageInputs | None:
        pixel_values = kwargs.pop("pixel_values", None)
        image_sizes = kwargs.pop("image_sizes", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            return LlavaNextImagePixelInputs(
                type="pixel_values",
                pixel_values=pixel_values,
                image_sizes=image_sizes,
            )

        if image_embeds is not None:
            return LlavaNextImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _select_vision_features(self, image_features: Tensor) -> Tensor:
        strategy = self.llava_config.vision_feature_select_strategy
        if callable(strategy):
            return strategy(image_features)
        if strategy == "class":
            return image_features[:, :1, :]
        if strategy == "full":
            return image_features
        if strategy in {"default", "spatial", "spatial_unpad"}:
            return image_features[:, 1:, :]
        return image_features

    def _image_pixels_to_features(
        self,
        vision_tower: nn.Cell,
        pixel_values: Tensor,
    ) -> Tensor:
        image_features = vision_tower(
            pixel_values,
            select_layers=self.select_layers,
            feature_select_strategy=self.llava_config.vision_feature_select_strategy,
        )
        return self._select_vision_features(image_features)

    def _merge_image_patch_embeddings(
        self, image_size: Tensor, patch_embeddings: Tensor, *, strategy: str
    ) -> Tensor:
        if strategy == "flat":
            return patch_embeddings.reshape((-1, patch_embeddings.shape[-1]))

        if strategy.startswith("spatial"):
            patch_size = self.llava_config.vision_config.patch_size
            base_image_size = self.llava_config.vision_config.image_size
            npatches = base_image_size // patch_size

            base_patch_embeds = patch_embeddings[0]
            other_patch_embeds = patch_embeddings[1:]

            if other_patch_embeds.shape[0] == 0:
                if "unpad" in strategy:
                    newline = ops.expand_dims(
                        self.image_newline.astype(base_patch_embeds.dtype), 0
                    )
                    return ops.concat((base_patch_embeds, newline), axis=0)
                return base_patch_embeds

            num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                image_size=(int(image_size[0]), int(image_size[1])),
                grid_pinpoints=self.llava_config.image_grid_pinpoints,
                patch_size=base_image_size,
            )

            expected_slices = num_patch_height * num_patch_width
            if other_patch_embeds.shape[0] != expected_slices:
                raise ValueError(
                    "Unexpected number of anyres slices: "
                    f"{other_patch_embeds.shape[0]} vs {expected_slices}"
                )

            if other_patch_embeds.shape[1] != npatches * npatches:
                raise ValueError(
                    "Unexpected patch count per slice: "
                    f"{other_patch_embeds.shape[1]} vs {npatches * npatches}"
                )

            grid = other_patch_embeds.reshape(
                num_patch_height,
                num_patch_width,
                npatches,
                npatches,
                -1,
            )
            grid = ops.transpose(grid, (0, 2, 1, 3, 4))
            grid = grid.reshape(
                num_patch_height * npatches, num_patch_width * npatches, -1
            )

            if "unpad" in strategy:
                grid = unpad_image(
                    grid,
                    image_size=(int(image_size[0]), int(image_size[1])),
                )

                newline = ops.expand_dims(
                    self.image_newline.astype(grid.dtype), 0
                )
                newline = ops.expand_dims(newline, 0)
                newline = ops.tile(
                    newline, (grid.shape[0], 1, 1)
                )
                grid = ops.concat((grid, newline), axis=1)

            merged_patch_embeddings = ops.concat(
                (base_patch_embeds, grid.reshape((-1, grid.shape[-1]))),
                axis=0,
            )
            return merged_patch_embeddings

        raise ValueError(f"Unexpected patch merge strategy: {strategy}")

    def _process_image_pixels(
        self,
        inputs: LlavaNextImagePixelInputs,
    ) -> Tensor | tuple[Tensor, ...]:
        assert self.vision_tower is not None

        pixel_values = inputs["pixel_values"]

        if isinstance(pixel_values, Tensor):
            b, num_patches, c, h, w = pixel_values.shape
            stacked_pixel_values = pixel_values.reshape((b * num_patches, c, h, w))
            stacked_image_features = self._image_pixels_to_features(
                self.vision_tower, stacked_pixel_values
            )
            stacked_patch_embeddings = self.multi_modal_projector(
                stacked_image_features
            )

            return stacked_patch_embeddings.reshape(
                (b, num_patches, *stacked_patch_embeddings.shape[1:])
            )

        num_patches_per_batch = [v.shape[0] for v in pixel_values]
        stacked_pixel_values = ops.concat(pixel_values, axis=0)
        stacked_image_features = self._image_pixels_to_features(
            self.vision_tower, stacked_pixel_values
        )
        projected = self.multi_modal_projector(stacked_image_features)

        outputs: list[Tensor] = []
        start = 0
        for size in num_patches_per_batch:
            end = start + size
            outputs.append(projected[start:end])
            start = end
        return tuple(outputs)

    def _process_image_input(
        self,
        image_input: LlavaNextImageInputs,
    ) -> Tensor | list[Tensor]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        patch_embeddings = self._process_image_pixels(image_input)

        image_sizes = image_input.get("image_sizes")
        if image_sizes is None:
            batch_size = len(image_input["pixel_values"])
            vision_config = self.llava_config.vision_config
            default_height = default_width = vision_config.image_size
            image_sizes = ms.Tensor(
                [[default_height, default_width] for _ in range(batch_size)],
                dtype=ms.int32,
            )

        if isinstance(patch_embeddings, Tensor):
            patch_embeddings = ops.unstack(patch_embeddings)

        return [
            self._merge_image_patch_embeddings(
                image_sizes[i], patch_features_batch, strategy="spatial_unpad"
            )
            for i, patch_features_batch in enumerate(patch_embeddings)
        ]

    def get_multimodal_embeddings(
        self, **kwargs: object
    ) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None
        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> Tensor:
        if not hasattr(self.language_model, "embed_tokens"):
            raise RuntimeError("language_model.embed_tokens is required.")
        inputs_embeds = self.language_model.embed_tokens(input_ids)
        if multimodal_embeddings is None:
            return inputs_embeds

        return merge_multimodal_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            placeholder_token_id=self.llava_config.image_token_index,
        )

    def construct(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs: object,
    ) -> Tensor | IntermediateTensors:
        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.exec_model(input_ids, positions,
                                        intermediate_tensors, inputs_embeds)
        return hidden_states

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        params_dict = self.get_params_dict()
        loaded: set[str] = set()
        for name, weight in self.hf_to_vllm_mapper.apply(weights):
            if name in params_dict:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, weight)
                loaded.add(name)
        return loaded