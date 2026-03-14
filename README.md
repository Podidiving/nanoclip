# nanoclip

Minimal pure-PyTorch CLIP implementation compatible with OpenAI CLIP weights on Hugging Face (safetensors).

## Install

```bash
uv sync
```

## Usage

```python
import torch
from nanoclip import CLIPModel

# Loads from Hugging Face and caches under ~/.cache/nanoclip/hf/
model = CLIPModel.from_pretrained(
    "openai/clip-vit-large-patch14",
    only_local_files=False,
)
model.eval()

input_ids = torch.randint(0, 49408, (2, 77))
pixel_values = torch.randn(2, 3, 224, 224)

with torch.no_grad():
    outputs = model(input_ids=input_ids, pixel_values=pixel_values)
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
