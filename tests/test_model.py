from __future__ import annotations

import json
from dataclasses import asdict

import pytest
import torch

from nanoclip import CLIPConfig, CLIPModel, CLIPTextConfig, CLIPVisionConfig
import nanoclip.model as clip_model_module


def tiny_config() -> CLIPConfig:
    return CLIPConfig(
        text_config=CLIPTextConfig(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=8,
        ),
        vision_config=CLIPVisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            image_size=16,
            patch_size=8,
        ),
        projection_dim=16,
    )


def test_forward_shapes() -> None:
    model = CLIPModel(tiny_config())
    input_ids = torch.randint(0, 32, (2, 8))
    pixel_values = torch.randn(2, 3, 16, 16)
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)

    assert outputs["text_embeds"].shape == (2, 16)
    assert outputs["image_embeds"].shape == (2, 16)
    assert outputs["logits_per_text"].shape == (2, 2)
    assert outputs["logits_per_image"].shape == (2, 2)
    assert torch.allclose(
        outputs["logits_per_image"], outputs["logits_per_text"].t(), atol=1e-5
    )


def test_position_ids_are_persistent_for_hf_compatibility() -> None:
    model = CLIPModel(tiny_config())
    state_keys = set(model.state_dict().keys())
    assert "text_model.embeddings.position_ids" in state_keys
    assert "vision_model.embeddings.position_ids" in state_keys


def test_text_pooling_uses_eos_token_id() -> None:
    cfg = tiny_config()
    cfg.text_config.eos_token_id = 7
    model = CLIPModel(cfg)
    input_ids = torch.tensor(
        [
            [1, 3, 7, 2, 0, 0, 0, 0],
            [4, 5, 6, 1, 2, 3, 4, 5],
        ]
    )
    hidden, pooled = model.text_model(input_ids)

    expected_indices = torch.tensor([2, 0])
    expected = hidden[torch.arange(hidden.shape[0]), expected_indices]
    assert torch.allclose(pooled, expected)


def test_from_pretrained_local_safetensors(tmp_path, monkeypatch) -> None:
    cfg = tiny_config()
    model = CLIPModel(cfg)
    model_dir = tmp_path / "clip"
    model_dir.mkdir()

    with (model_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "text_config": asdict(cfg.text_config),
                "vision_config": asdict(cfg.vision_config),
                "projection_dim": cfg.projection_dim,
                "logit_scale_init_value": cfg.logit_scale_init_value,
            },
            handle,
        )

    def fake_ensure_model_files(
        model_id_or_path,
        revision="main",
        cache_dir=None,
        only_local_files=False,
    ):
        del model_id_or_path, revision, cache_dir, only_local_files
        return model_dir

    def fake_load_safetensors_state_dict(_model_dir):
        del _model_dir
        return model.state_dict()

    monkeypatch.setattr(
        clip_model_module, "_ensure_model_files", fake_ensure_model_files
    )
    monkeypatch.setattr(
        clip_model_module,
        "_load_safetensors_state_dict",
        fake_load_safetensors_state_dict,
    )

    loaded = CLIPModel.from_pretrained(model_dir)
    for name, value in model.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[name]), name


def test_from_pretrained_passes_cache_options(tmp_path, monkeypatch) -> None:
    cfg = tiny_config()
    model = CLIPModel(cfg)
    expected_model_dir = tmp_path / "found"
    expected_model_dir.mkdir()
    seen = {}

    def fake_ensure_model_files(
        model_id_or_path,
        revision="main",
        cache_dir=None,
        only_local_files=False,
    ):
        seen["model_id_or_path"] = model_id_or_path
        seen["revision"] = revision
        seen["cache_dir"] = cache_dir
        seen["only_local_files"] = only_local_files
        return expected_model_dir

    def fake_load_safetensors_state_dict(_model_dir):
        del _model_dir
        return model.state_dict()

    monkeypatch.setattr(
        clip_model_module, "_ensure_model_files", fake_ensure_model_files
    )
    monkeypatch.setattr(
        clip_model_module,
        "_load_safetensors_state_dict",
        fake_load_safetensors_state_dict,
    )
    monkeypatch.setattr(clip_model_module.CLIPConfig, "from_json_file", lambda _: cfg)

    custom_cache_dir = tmp_path / "my-cache"
    CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        revision="main",
        cache_dir=custom_cache_dir,
        only_local_files=True,
    )

    assert seen["model_id_or_path"] == "openai/clip-vit-large-patch14"
    assert seen["revision"] == "main"
    assert seen["cache_dir"] == custom_cache_dir
    assert seen["only_local_files"] is True


def test_ensure_model_files_only_local_files_blocks_download(
    tmp_path, monkeypatch
) -> None:
    def fail_download(url, path):
        del url, path
        raise AssertionError(
            "download should not be attempted when only_local_files=True"
        )

    monkeypatch.setattr(clip_model_module, "_download", fail_download)

    with pytest.raises(FileNotFoundError, match="Missing local file"):
        clip_model_module._ensure_model_files(
            model_id_or_path="openai/clip-vit-large-patch14",
            cache_dir=tmp_path,
            only_local_files=True,
        )


def test_ensure_model_files_uses_hf_cache_layout(tmp_path, monkeypatch) -> None:
    model_id = "openai/clip-vit-large-patch14"
    revision = "main"
    expected_dir = (
        tmp_path / "models--openai--clip-vit-large-patch14" / "snapshots" / "main"
    )
    expected_dir.mkdir(parents=True)

    config_path = expected_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    (expected_dir / "model.safetensors").write_bytes(b"dummy")

    def fail_download(url, path):
        del url, path
        raise AssertionError("download should not be attempted")

    monkeypatch.setattr(clip_model_module, "_download", fail_download)

    resolved = clip_model_module._ensure_model_files(
        model_id_or_path=model_id,
        revision=revision,
        cache_dir=tmp_path,
        only_local_files=False,
    )
    assert resolved == expected_dir


def test_clip_config_applies_text_config_dict_defaults() -> None:
    cfg = CLIPConfig.from_dict(
        {
            "projection_dim": 16,
            "text_config": {
                "vocab_size": 32,
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "max_position_embeddings": 8,
                "eos_token_id": 2,
            },
            "vision_config": {
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 16,
                "patch_size": 8,
            },
            "text_config_dict": {},
        }
    )
    assert cfg.text_config.eos_token_id == 49407
