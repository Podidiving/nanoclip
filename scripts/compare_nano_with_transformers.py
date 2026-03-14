from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import CLIPModel as TransformersCLIPModel

from nanoclip import CLIPModel as NanoCLIPModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nanoclip and transformers CLIP outputs."
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--cache-dir", default=str(Path.home() / ".cache" / "nanoclip" / "hf")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)

    torch.manual_seed(args.seed)
    if args.model_path is not None:
        nano = NanoCLIPModel.from_pretrained(
            args.model_path,
            only_local_files=True,
            device=device,
        ).eval()
        reference = (
            TransformersCLIPModel.from_pretrained(
                args.model_path,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )
    else:
        org, repo = args.model_id.split("/")
        nano = NanoCLIPModel.from_pretrained(
            args.model_id,
            revision=args.revision,
            cache_dir=cache_dir,
            only_local_files=False,
            device=device,
        ).eval()
        reference = (
            TransformersCLIPModel.from_pretrained(
                cache_dir / f"models--{org}--{repo}" / "snapshots" / args.revision,
                local_files_only=True,
            )
            .to(device)
            .eval()
        )

    text_cfg = reference.config.text_config
    vision_cfg = reference.config.vision_config

    input_ids = torch.randint(
        low=0,
        high=text_cfg.vocab_size,
        size=(args.batch_size, text_cfg.max_position_embeddings),
        device=device,
    )
    pixel_values = torch.randn(
        args.batch_size,
        vision_cfg.num_channels,
        vision_cfg.image_size,
        vision_cfg.image_size,
        device=device,
    )

    with torch.no_grad():
        nano_outputs = nano(input_ids=input_ids, pixel_values=pixel_values)
        ref_outputs = reference(input_ids=input_ids, pixel_values=pixel_values)

    torch.testing.assert_close(
        nano_outputs["text_embeds"],
        ref_outputs.text_embeds,
        rtol=args.rtol,
        atol=args.atol,
    )
    torch.testing.assert_close(
        nano_outputs["image_embeds"],
        ref_outputs.image_embeds,
        rtol=args.rtol,
        atol=args.atol,
    )
    torch.testing.assert_close(
        nano_outputs["logits_per_text"],
        ref_outputs.logits_per_text,
        rtol=args.rtol,
        atol=args.atol,
    )
    torch.testing.assert_close(
        nano_outputs["logits_per_image"],
        ref_outputs.logits_per_image,
        rtol=args.rtol,
        atol=args.atol,
    )

    print("OK: nanoclip and transformers outputs are close.")


if __name__ == "__main__":
    main()
