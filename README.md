# nanoclip

Minimal pure-PyTorch CLIP implementation compatible with OpenAI CLIP weights on Hugging Face (safetensors).

## Install

```bash
uv sync
```

## Usage

```python
import torch
from PIL import Image
from nanoclip import CLIPModel, NanoCLIPProcessor

# Loads from Hugging Face and caches under ~/.cache/nanoclip/hf/
model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    only_local_files=False,
)
processor = NanoCLIPProcessor.from_pretrained(
    "openai/clip-vit-large-patch14",
    only_local_files=False,
)
model.eval()

image = Image.open("/path/to/image.jpg").convert("RGB")
encoded = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)

with torch.no_grad():
    outputs = model(input_ids=encoded["input_ids"], pixel_values=encoded["pixel_values"])
    # outputs["logits_per_image"], outputs["logits_per_text"]
```

Set `only_local_files=True` to disallow downloads and require files to already exist in `cache_dir`.

## Tests

```bash
uv run pytest
```

## Compare with Transformers

```bash
uv run --group transformers python scripts/compare_nano_with_transformers.py \
  --model-id openai/clip-vit-large-patch14 \
  --cache-dir ~/.cache/nanoclip/hf
```

The script loads both models from the same `--cache-dir` root and asserts embeddings/logits are numerically close.

## Image-Text Inference

```bash
uv run python scripts/infer_image_text.py \
  --image /path/to/image.jpg \
  --texts "a photo of a cat" "a photo of a dog" "a landscape" \
  --model-id openai/clip-vit-large-patch14 \
  --cache-dir ~/.cache/nanoclip/hf
```

This preprocesses the image and texts, runs inference with `nanoclip`, and prints the closest text plus full ranking.

## Compare Processor with Transformers

```bash
uv run --group transformers python scripts/compare_nano_processor_with_transformers.py \
  --model-id openai/clip-vit-large-patch14 \
  --cache-dir ~/.cache/nanoclip/hf
```

The script compares `input_ids`, `attention_mask`, and `pixel_values` between `NanoCLIPProcessor` and `transformers.CLIPProcessor`.
