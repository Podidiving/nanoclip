from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tokenizers import Tokenizer


class NanoCLIPProcessor:
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        shortest_edge: int,
        crop_size: tuple[int, int],
        pad_token_id: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.image_mean = image_mean
        self.image_std = image_std
        self.shortest_edge = shortest_edge
        self.crop_size = crop_size
        self.pad_token_id = pad_token_id

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | Path,
        revision: str = "main",
        cache_dir: str | Path | None = None,
        only_local_files: bool = False,
    ) -> "NanoCLIPProcessor":
        snapshot_dir = _ensure_processor_files(
            model_id_or_path=model_id_or_path,
            revision=revision,
            cache_dir=cache_dir,
            only_local_files=only_local_files,
        )

        tokenizer = Tokenizer.from_file(str(snapshot_dir / "tokenizer.json"))
        with (snapshot_dir / "preprocessor_config.json").open(
            "r", encoding="utf-8"
        ) as handle:
            image_cfg = json.load(handle)

        size_cfg = image_cfg["size"]
        if isinstance(size_cfg, dict):
            shortest_edge = int(size_cfg["shortest_edge"])
        else:
            shortest_edge = int(size_cfg)

        crop_cfg = image_cfg["crop_size"]
        if isinstance(crop_cfg, dict):
            crop_size = (int(crop_cfg["height"]), int(crop_cfg["width"]))
        else:
            crop_size = (int(crop_cfg), int(crop_cfg))
        image_mean = tuple(float(x) for x in image_cfg["image_mean"])
        image_std = tuple(float(x) for x in image_cfg["image_std"])

        pad_token_id = tokenizer.token_to_id("<|endoftext|>")
        if pad_token_id is None:
            raise ValueError("Tokenizer is missing <|endoftext|> token id")

        return cls(
            tokenizer=tokenizer,
            image_mean=image_mean,
            image_std=image_std,
            shortest_edge=shortest_edge,
            crop_size=crop_size,
            pad_token_id=int(pad_token_id),
        )

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Image.Image | list[Image.Image] | None = None,
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        if text is None and images is None:
            raise ValueError("At least one of text or images must be provided")

        out: dict[str, torch.Tensor] = {}
        if text is not None:
            out.update(self._encode_text(text, padding=padding))
        if images is not None:
            out["pixel_values"] = self._encode_images(images)
        return out

    def _encode_text(
        self, text: str | list[str], padding: bool
    ) -> dict[str, torch.Tensor]:
        texts = [text] if isinstance(text, str) else text
        encodings = self.tokenizer.encode_batch(texts)
        sequences = [enc.ids for enc in encodings]
        max_len = max(len(seq) for seq in sequences) if padding else None

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for seq in sequences:
            if max_len is None:
                input_ids.append(seq)
                attention_mask.append([1] * len(seq))
                continue
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_token_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def _encode_images(self, images: Image.Image | list[Image.Image]) -> torch.Tensor:
        image_list = [images] if isinstance(images, Image.Image) else images
        tensors = [self._preprocess_one_image(image) for image in image_list]
        return torch.stack(tensors, dim=0)

    def _preprocess_one_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        arr = np.asarray(image)
        height, width = int(arr.shape[0]), int(arr.shape[1])
        if min(width, height) == 0:
            raise ValueError("Invalid image size")

        if width <= height:
            new_width = self.shortest_edge
            new_height = int(self.shortest_edge * height / width)
        else:
            new_height = self.shortest_edge
            new_width = int(self.shortest_edge * width / height)

        image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
        arr = np.asarray(image)

        crop_h, crop_w = self.crop_size
        top = (new_height - crop_h) // 2
        left = (new_width - crop_w) // 2
        bottom = top + crop_h
        right = left + crop_w

        if top >= 0 and left >= 0 and bottom <= new_height and right <= new_width:
            arr = arr[top:bottom, left:right, :]
        else:
            pad_h = max(crop_h, new_height)
            pad_w = max(crop_w, new_width)
            padded = np.zeros((pad_h, pad_w, 3), dtype=arr.dtype)
            top_pad = int(np.ceil((pad_h - new_height) / 2.0))
            left_pad = int(np.ceil((pad_w - new_width) / 2.0))
            padded[
                top_pad : top_pad + new_height, left_pad : left_pad + new_width, :
            ] = arr
            top = top + top_pad
            left = left + left_pad
            bottom = top + crop_h
            right = left + crop_w
            arr = padded[
                max(0, top) : min(pad_h, bottom), max(0, left) : min(pad_w, right), :
            ]

        arr = arr.astype(np.float32) * (1.0 / 255.0)
        mean = np.asarray(self.image_mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(self.image_std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        pixel_values = torch.from_numpy(arr)
        return pixel_values


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def _ensure_processor_files(
    model_id_or_path: str | Path,
    revision: str = "main",
    cache_dir: str | Path | None = None,
    only_local_files: bool = False,
) -> Path:
    required_files = ["tokenizer.json", "preprocessor_config.json"]

    path = Path(model_id_or_path)
    if path.exists():
        for filename in required_files:
            target = path / filename
            if not target.exists():
                raise FileNotFoundError(f"Missing local file: {target}")
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
                _ensure_required_processor_files(
                    snapshot_dir=resolved_snapshot,
                    model_id_or_path=model_id_or_path,
                    revision=revision,
                    required_files=required_files,
                    only_local_files=only_local_files,
                )
                return resolved_snapshot

    snapshot_dir = snapshots_dir / revision
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    _ensure_required_processor_files(
        snapshot_dir=snapshot_dir,
        model_id_or_path=model_id_or_path,
        revision=revision,
        required_files=required_files,
        only_local_files=only_local_files,
    )

    refs_dir.mkdir(parents=True, exist_ok=True)
    if not ref_file.exists():
        ref_file.write_text(f"{revision}\n", encoding="utf-8")

    return snapshot_dir


def _ensure_required_processor_files(
    snapshot_dir: Path,
    model_id_or_path: str | Path,
    revision: str,
    required_files: list[str],
    only_local_files: bool,
) -> None:
    base = f"https://huggingface.co/{model_id_or_path}/resolve/{revision}"
    for filename in required_files:
        target = snapshot_dir / filename
        if target.exists():
            continue
        if only_local_files:
            raise FileNotFoundError(f"Missing local file: {target}")
        _download(f"{base}/{filename}", target)
