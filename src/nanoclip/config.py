from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CLIPTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    eos_token_id: int = 49407

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CLIPTextConfig":
        return cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            max_position_embeddings=int(data["max_position_embeddings"]),
            hidden_act=str(data.get("hidden_act", "quick_gelu")),
            layer_norm_eps=float(data.get("layer_norm_eps", 1e-5)),
            eos_token_id=int(data.get("eos_token_id", 49407)),
        )


@dataclass(slots=True)
class CLIPVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    image_size: int = 224
    patch_size: int = 32
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    num_channels: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CLIPVisionConfig":
        return cls(
            hidden_size=int(data["hidden_size"]),
            intermediate_size=int(data["intermediate_size"]),
            num_hidden_layers=int(data["num_hidden_layers"]),
            num_attention_heads=int(data["num_attention_heads"]),
            image_size=int(data["image_size"]),
            patch_size=int(data["patch_size"]),
            hidden_act=str(data.get("hidden_act", "quick_gelu")),
            layer_norm_eps=float(data.get("layer_norm_eps", 1e-5)),
            num_channels=int(data.get("num_channels", 3)),
        )


@dataclass(slots=True)
class CLIPConfig:
    text_config: CLIPTextConfig
    vision_config: CLIPVisionConfig
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CLIPConfig":
        text_config_data = dict(data["text_config"])
        text_config_dict = data.get("text_config_dict")
        if isinstance(text_config_dict, dict):
            base_text = CLIPTextConfig()
            for key, value in text_config_dict.items():
                if value is None or not hasattr(base_text, key):
                    continue
                setattr(base_text, key, value)
            text_config_data.update(asdict(base_text))

        vision_config_data = dict(data["vision_config"])
        vision_config_dict = data.get("vision_config_dict")
        if isinstance(vision_config_dict, dict):
            base_vision = CLIPVisionConfig()
            for key, value in vision_config_dict.items():
                if value is None or not hasattr(base_vision, key):
                    continue
                setattr(base_vision, key, value)
            vision_config_data.update(asdict(base_vision))

        return cls(
            text_config=CLIPTextConfig.from_dict(text_config_data),
            vision_config=CLIPVisionConfig.from_dict(vision_config_data),
            projection_dim=int(data["projection_dim"]),
            logit_scale_init_value=float(data.get("logit_scale_init_value", 2.6592)),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "CLIPConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls.from_dict(data)
