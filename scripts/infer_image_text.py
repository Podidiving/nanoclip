from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from nanoclip import CLIPModel, NanoCLIPProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CLIP inference for one image against multiple texts and return the closest match."
    )
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument(
        "--texts",
        nargs="+",
        action="append",
        required=True,
        help="Candidate texts to compare against image. Can be provided once with many values or repeated.",
    )
    parser.add_argument("--model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", default=str(Path.home() / ".cache" / "nanoclip" / "hf"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only-local-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cache_dir = Path(args.cache_dir)
    texts = [item for group in args.texts for item in group]

    model = CLIPModel.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=cache_dir,
        only_local_files=args.only_local_files,
        device=device,
    ).eval()

    processor = NanoCLIPProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        cache_dir=cache_dir,
        only_local_files=args.only_local_files,
    )

    image = Image.open(args.image).convert("RGB")
    encoded = processor(text=texts, images=image, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(device)
    pixel_values = encoded["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, pixel_values=pixel_values)
        probs = outputs["logits_per_image"].softmax(dim=-1)[0]

    best_idx = int(torch.argmax(probs).item())
    print(f"Best match: {texts[best_idx]}")
    print("\nScores:")
    for text, score in sorted(zip(texts, probs.tolist()), key=lambda item: item[1], reverse=True):
        print(f"{score:.6f}\t{text}")


if __name__ == "__main__":
    main()
