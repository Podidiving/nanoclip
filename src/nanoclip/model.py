from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import Tensor, nn

from .config import CLIPConfig


def _get_activation(name: str):
    if name == "quick_gelu":
        return lambda x: x * torch.sigmoid(1.702 * x)
    if name == "gelu":
        return nn.functional.gelu
    raise ValueError(f"Unsupported activation: {name}")


class CLIPAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        bsz, seq_len, dim = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        key_states = self.k_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_heads, self.head_dim
        )

        query_states = query_states.transpose(1, 2) * self.scale
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_weights = query_states @ key_states.transpose(-2, -1)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_probs = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(attn_weights.dtype)
        attn_output = attn_probs @ value_states
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(attn_output)


class CLIPMLP(nn.Module):
    def __init__(
        self, hidden_size: int, intermediate_size: int, hidden_act: str
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation_fn = _get_activation(hidden_act)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        return self.fc2(hidden_states)


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        hidden_act: str,
        eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = CLIPAttention(hidden_size, num_heads)
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = CLIPMLP(hidden_size, intermediate_size, hidden_act)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class CLIPEncoder(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_attention_heads,
                    hidden_act=config.hidden_act,
                    eps=config.layer_norm_eps,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )

    def forward(
        self, hidden_states: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class CLIPTextEmbeddings(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand(1, -1),
        )

    def forward(self, input_ids: Tensor) -> Tensor:
        seq_len = input_ids.shape[1]
        position_ids = self.position_ids[:, :seq_len]
        return self.token_embedding(input_ids) + self.position_embedding(position_ids)


class CLIPTextTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.eos_token_id = int(config.eos_token_id)
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def _build_causal_attention_mask(
        self, seq_len: int, dtype: torch.dtype, device: torch.device
    ) -> Tensor:
        mask = torch.full(
            (seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device
        )
        mask = torch.triu(mask, diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.embeddings(input_ids)
        causal_mask = self._build_causal_attention_mask(
            seq_len=input_ids.shape[1],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        hidden_states = self.encoder(hidden_states, causal_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        input_ids_int = input_ids.to(dtype=torch.int, device=hidden_states.device)
        if self.eos_token_id == 2:
            pooled_indices = input_ids_int.argmax(dim=-1)
        else:
            pooled_indices = (input_ids_int == self.eos_token_id).int().argmax(dim=-1)
        pooled_output = hidden_states[
            torch.arange(hidden_states.shape[0], device=hidden_states.device),
            pooled_indices,
        ]
        return hidden_states, pooled_output


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.class_embedding = nn.Parameter(torch.randn(config.hidden_size))
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        num_patches = (config.image_size // config.patch_size) ** 2
        self.num_positions = num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)
        self.register_buffer(
            "position_ids", torch.arange(self.num_positions).expand(1, -1)
        )

    def forward(self, pixel_values: Tensor) -> Tensor:
        batch_size = pixel_values.shape[0]
        patch_embeds = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        return embeddings + self.position_embedding(self.position_ids)


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = CLIPEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)
        hidden_states = self.encoder(hidden_states)
        pooled_output = self.post_layernorm(hidden_states[:, 0, :])
        return hidden_states, pooled_output


class CLIPModel(nn.Module):
    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        self.config = config
        self.text_model = CLIPTextTransformer(config.text_config)
        self.vision_model = CLIPVisionTransformer(config.vision_config)
        self.text_projection = nn.Linear(
            config.text_config.hidden_size, config.projection_dim, bias=False
        )
        self.visual_projection = nn.Linear(
            config.vision_config.hidden_size, config.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(torch.tensor(config.logit_scale_init_value))

    def get_text_features(self, input_ids: Tensor) -> Tensor:
        _, pooled_output = self.text_model(input_ids=input_ids)
        text_features = self.text_projection(pooled_output)
        return nn.functional.normalize(text_features, dim=-1)

    def get_image_features(self, pixel_values: Tensor) -> Tensor:
        _, pooled_output = self.vision_model(pixel_values=pixel_values)
        image_features = self.visual_projection(pooled_output)
        return nn.functional.normalize(image_features, dim=-1)

    def forward(self, input_ids: Tensor, pixel_values: Tensor) -> dict[str, Tensor]:
        text_embeds = self.get_text_features(input_ids)
        image_embeds = self.get_image_features(pixel_values)
        logit_scale = self.logit_scale.exp()
        logits_per_text = text_embeds @ image_embeds.t() * logit_scale
        logits_per_image = logits_per_text.t()
        return {
            "text_embeds": text_embeds,
            "image_embeds": image_embeds,
            "logits_per_text": logits_per_text,
            "logits_per_image": logits_per_image,
        }

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        revision: str = "main",
        device: str | torch.device | None = None,
        strict: bool = True,
        cache_dir: str | Path | None = None,
        only_local_files: bool = False,
    ) -> "CLIPModel":
        local_dir = _ensure_model_files(
            model_id_or_path,
            revision=revision,
            cache_dir=cache_dir,
            only_local_files=only_local_files,
        )
        config = CLIPConfig.from_json_file(local_dir / "config.json")
        model = cls(config)
        state_dict = _load_safetensors_state_dict(local_dir)
        model.load_state_dict(state_dict, strict=strict)
        if device is not None:
            model = model.to(device)
        return model


def _format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)}{unit}"
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{num_bytes}B"


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_percent = -1
    last_bytes = -1
    last_msg_len = 0
    prefix = f"Downloading {path.name}"

    def reporthook(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal last_percent, last_bytes, last_msg_len
        downloaded = block_count * block_size

        if total_size > 0:
            percent = min(100, int(downloaded * 100 / total_size))
            if percent == last_percent:
                return
            last_percent = percent
            bar_width = 24
            filled = int(bar_width * percent / 100)
            bar = "#" * filled + "-" * (bar_width - filled)
            msg = (
                f"\r{prefix} [{bar}] {percent:3d}% "
                f"{_format_size(min(downloaded, total_size))}/{_format_size(total_size)}"
            )
        else:
            if downloaded == last_bytes:
                return
            last_bytes = downloaded
            msg = f"\r{prefix} {_format_size(downloaded)}"

        padding = " " * max(0, last_msg_len - len(msg))
        sys.stderr.write(msg + padding)
        sys.stderr.flush()
        last_msg_len = len(msg)

    urllib.request.urlretrieve(url, path, reporthook=reporthook)
    sys.stderr.write("\n")
    sys.stderr.flush()


def _ensure_model_files(
    model_id_or_path: str | Path,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    only_local_files: bool = False,
) -> Path:
    path = Path(model_id_or_path)
    if path.exists():
        return path

    if cache_dir is None:
        resolved_cache_dir = Path.home() / ".cache" / "nanoclip" / "hf"
    else:
        resolved_cache_dir = Path(cache_dir)

    repo_id = str(model_id_or_path)
    legacy_cache_dir = resolved_cache_dir / repo_id.replace("/", "__") / revision
    if legacy_cache_dir.exists():
        return legacy_cache_dir

    repo_cache_dir = resolved_cache_dir / f"models--{repo_id.replace('/', '--')}"
    refs_dir = repo_cache_dir / "refs"
    snapshots_dir = repo_cache_dir / "snapshots"
    ref_file = refs_dir / revision
    if ref_file.exists():
        snapshot_ref = ref_file.read_text(encoding="utf-8").strip()
        if snapshot_ref:
            resolved_snapshot = snapshots_dir / snapshot_ref
            if resolved_snapshot.exists():
                return resolved_snapshot

    model_cache_dir = snapshots_dir / revision
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/{model_id_or_path}/resolve/{revision}"

    config_path = model_cache_dir / "config.json"
    if not config_path.exists():
        if only_local_files:
            raise FileNotFoundError(f"Missing local file: {config_path}")
        _download(f"{base}/config.json", config_path)

    safetensors_path = model_cache_dir / "model.safetensors"
    index_path = model_cache_dir / "model.safetensors.index.json"
    if not safetensors_path.exists() and not index_path.exists():
        if only_local_files:
            raise FileNotFoundError(
                f"Missing local model weights in {model_cache_dir}; expected model.safetensors "
                "or model.safetensors.index.json (+ shards)."
            )
        try:
            _download(f"{base}/model.safetensors", safetensors_path)
        except Exception:
            _download(f"{base}/model.safetensors.index.json", index_path)
            with index_path.open("r", encoding="utf-8") as handle:
                index_data = json.load(handle)
            shard_filenames = sorted(set(index_data["weight_map"].values()))
            for name in shard_filenames:
                shard_path = model_cache_dir / name
                if not shard_path.exists():
                    _download(f"{base}/{name}", shard_path)

    refs_dir.mkdir(parents=True, exist_ok=True)
    if not ref_file.exists():
        ref_file.write_text(f"{revision}\n", encoding="utf-8")

    return model_cache_dir


def _load_safetensors_state_dict(model_dir: Path) -> dict[str, Tensor]:
    single_file = model_dir / "model.safetensors"
    if single_file.exists():
        shard_paths = [single_file]
    else:
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"No model.safetensors or index found in {model_dir}"
            )
        with index_path.open("r", encoding="utf-8") as handle:
            index_data = json.load(handle)
        shard_paths = [
            model_dir / shard
            for shard in sorted(set(index_data["weight_map"].values()))
        ]

    state_dict: dict[str, Tensor] = {}
    for shard_path in shard_paths:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state_dict[key] = handle.get_tensor(key)
    return state_dict
