"""Golden fixture for WebGPU vision parity — Qwen3-VL.

Runs the REAL HF processor + vision tower on a deterministic synthetic image
and dumps every stage the browser must reproduce:
  - the source image (base64 PNG, so browser decode is bit-exact)
  - pixel_values (the patch matrix after preprocessing)
  - image_embeds (post-merger, text-space)
  - deepstack feature lists (if the tower returns them)

Run from the repo root with the project venv:
    ./venv/Scripts/python.exe webgpu/scripts/gen_vision_fixture.py

Browser side: load local/qwen3-vl-4b-instruct, then run
    __VISION_PARITY__()
in the console — it fetches this fixture, replays preprocessing + the WGSL
tower, and reports per-stage max-abs-diff.
"""
import base64
import io
import json
import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'qwen3-vl-4b-instruct')
OUT = os.path.join(os.path.dirname(__file__), '..', 'test-fixtures', 'vision-qwen3vl-golden.json')

def main():
    import numpy as np
    import torch
    from PIL import Image

    # Deterministic 320x256 gradient + structure (multiples of 32 — no resize)
    W, H = 320, 256
    x = np.linspace(0, 255, W, dtype=np.float64)[None, :, None]
    y = np.linspace(0, 255, H, dtype=np.float64)[:, None, None]
    img = np.concatenate([
        np.broadcast_to(x, (H, W, 1)),                      # R: horizontal ramp
        np.broadcast_to(y, (H, W, 1)),                      # G: vertical ramp
        np.broadcast_to((x + y) / 2 % 256, (H, W, 1)),      # B: diagonal
    ], axis=2).astype(np.uint8)
    img[96:160, 128:192] = [255, 0, 0]                      # a red square for structure
    pil = Image.fromarray(img)
    png = io.BytesIO()
    pil.save(png, format='PNG')

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    inputs = processor(images=[pil], text='<|vision_start|><|image_pad|><|vision_end|>',
                       return_tensors='pt')
    pixel_values = inputs['pixel_values']
    grid_thw = inputs['image_grid_thw']
    print(f'pixel_values: {tuple(pixel_values.shape)}, grid_thw: {grid_thw.tolist()}')

    # Vision tower only — load in bf16 to keep RAM sane, run in f32
    from transformers import AutoModelForImageTextToText
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16, device_map='cpu')
    visual = model.model.visual.float()
    visual.eval()

    with torch.no_grad():
        # get_image_features applies the merger (and returns deepstack for
        # Qwen3-VL); the raw visual() forward stops at pre-merger hidden.
        inner = getattr(model, 'model', model)
        if hasattr(inner, 'get_image_features'):
            out = inner.get_image_features(pixel_values.float(), grid_thw)
        elif hasattr(model, 'get_image_features'):
            out = model.get_image_features(pixel_values.float(), grid_thw)
        else:
            out = visual(pixel_values.float(), grid_thw=grid_thw)

    # HF returns either a tensor, a tuple (embeds, deepstack), or an object —
    # introspect defensively and dump whatever is there.
    image_embeds = None
    deepstack = []
    # BaseModelOutputWithDeepstackFeatures (probed on this checkpoint):
    #   last_hidden_state: (numPatches, towerHidden)  — PRE-merger tower out
    #   pooler_output:     (numTokens, textHidden)    — post-merger embeds
    #                      (get_image_features wraps it in a per-image list)
    #   deepstack_features: list of (numTokens, textHidden)
    tower_hidden = None
    def unwrap(v):
        if isinstance(v, (tuple, list)) and len(v) == 1:
            return v[0]
        return v
    if hasattr(out, 'pooler_output'):
        image_embeds = unwrap(out.pooler_output)
        tower_hidden = getattr(out, 'last_hidden_state', None)
        ds = getattr(out, 'deepstack_features', None)
        if ds is not None:
            deepstack = [unwrap(t) for t in ds]
    elif torch.is_tensor(out):
        image_embeds = out
    if image_embeds is None:
        print(f'!! could not find embeddings in {type(out)}: {dir(out)}')
        sys.exit(1)
    print(f'image_embeds: {tuple(image_embeds.shape)}, deepstack: {[tuple(t.shape) for t in deepstack]}')

    fixture = {
        'png_base64': base64.b64encode(png.getvalue()).decode(),
        'grid_thw': grid_thw.tolist(),
        'pixel_values_shape': list(pixel_values.shape),
        'pixel_values': pixel_values.flatten().tolist(),
        'image_embeds_shape': list(image_embeds.shape),
        'image_embeds': image_embeds.flatten().tolist(),
        'deepstack_shapes': [list(t.shape) for t in deepstack],
        'deepstack': [t.flatten().tolist() for t in deepstack],
        # PRE-merger tower output — extra localization stage for the harness
        'tower_hidden_shape': list(tower_hidden.shape) if tower_hidden is not None else None,
        'tower_hidden': tower_hidden.flatten().tolist() if tower_hidden is not None else None,
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(fixture, f)
    print(f'Wrote {os.path.normpath(OUT)} ({os.path.getsize(OUT) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
