from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from nanoclip import NanoCLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nanoclip and transformers CLIP processors."
    )
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache" / "nanoclip" / "hf")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser()

    np.random.seed(args.seed)
    image = Image.fromarray(
        (np.random.rand(319, 517, 3) * 255).astype("uint8"), mode="RGB"
    )
    texts = [
        "a photo of a dog",
        "a photo of a cat",
        "an airplane on the runway",
        "a glass of orange juice",
    ]

    nano = NanoCLIPProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=cache_dir,
        only_local_files=False,
    )
    org, repo = args.model_id.split("/")
    reference = CLIPProcessor.from_pretrained(
        cache_dir / f"models--{org}--{repo}" / "snapshots" / args.revision,
        local_files_only=True,
    )

    nano_out = nano(text=texts, images=image, return_tensors="pt", padding=True)
    ref_out = reference(text=texts, images=image, return_tensors="pt", padding=True)

    torch.testing.assert_close(
        nano_out["input_ids"], ref_out["input_ids"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        nano_out["attention_mask"], ref_out["attention_mask"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        nano_out["pixel_values"],
        ref_out["pixel_values"],
        rtol=args.rtol,
        atol=args.atol,
    )

    print("OK: nanoclip and transformers processors produce matching outputs.")


if __name__ == "__main__":
    main()
