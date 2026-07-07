#!/usr/bin/env python
"""Generate FLUX.2-klein-4B reference fixtures for WebGPU parity testing (Phase 0a).

Run each stage in a SEPARATE process -- the f32 components do not fit in 32 GB
RAM together (TE ~16 GB f32, DiT ~15.5 GB f32):

  ./venv/Scripts/python.exe webgpu/scripts/gen_flux2_fixture.py --stage scheduler
  ./venv/Scripts/python.exe webgpu/scripts/gen_flux2_fixture.py --stage te
  ./venv/Scripts/python.exe webgpu/scripts/gen_flux2_fixture.py --stage dit
  ./venv/Scripts/python.exe webgpu/scripts/gen_flux2_fixture.py --stage vae
  ./venv/Scripts/python.exe webgpu/scripts/gen_flux2_fixture.py --stage edit

Outputs raw little-endian .bin dumps + manifest.json into
webgpu/scripts/flux2_fixture/. All model math runs on CPU in float32
(bf16 weights are exact in f32, so both sides do f32 math on identical weights).

Normative reference: venv diffusers 0.39
  pipelines/flux2/pipeline_flux2_klein.py
  models/transformers/transformer_flux2.py
  models/autoencoders/autoencoder_kl_flux2.py
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "flux.2-klein-4b"
OUT_DIR = Path(__file__).resolve().parent / "flux2_fixture"

PROMPTS = [
    "A photo of a cat sitting on a windowsill at sunset",
    "A watercolor painting of a lighthouse on a rocky coast under a stormy sky",
]

MAX_SEQ = 512
NUM_STEPS = 4
HIDDEN_LAYERS = (9, 18, 27)

torch.set_num_threads(os.cpu_count() or 8)
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# compute_empirical_mu -- copied verbatim from pipeline_flux2_klein.py:63-78
# ---------------------------------------------------------------------------
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


# ---------------------------------------------------------------------------
# Fixture writer: raw .bin per tensor + shared manifest.json
# ---------------------------------------------------------------------------
class FixtureWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.out_dir / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        else:
            self.manifest = {"tensors": {}, "meta": {}}

    def dump(self, name: str, t: torch.Tensor):
        t = t.detach().cpu().contiguous()
        if t.dtype in (torch.int64, torch.int32, torch.bool):
            arr = t.to(torch.int32).numpy().astype("<i4")
            dtype = "i32"
        elif t.dtype == torch.float64:
            arr = t.numpy().astype("<f8")
            dtype = "f64"
        else:
            arr = t.to(torch.float32).numpy().astype("<f4")
            dtype = "f32"
        fname = name.replace("/", "_") + ".bin"
        arr.tofile(self.out_dir / fname)
        self.manifest["tensors"][name] = {
            "file": fname,
            "dtype": dtype,
            "shape": list(t.shape),
        }
        print(f"  dump {name}: {dtype} {list(t.shape)}")

    def meta(self, key: str, value):
        self.manifest["meta"][key] = value

    def save(self):
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=1), encoding="utf-8"
        )
        print(f"manifest saved: {self.manifest_path}")

    def load(self, name: str) -> torch.Tensor:
        info = self.manifest["tensors"][name]
        dt = {"f32": "<f4", "f64": "<f8", "i32": "<i4"}[info["dtype"]]
        arr = np.fromfile(self.out_dir / info["file"], dtype=dt)
        expected = int(np.prod(info["shape"])) if info["shape"] else 1
        assert arr.size == expected, f"{name}: {arr.size} != {expected}"
        return torch.from_numpy(arr.reshape(info["shape"]).copy())


# ---------------------------------------------------------------------------
# Shared pipeline math (replicated verbatim from pipeline staticmethods)
# ---------------------------------------------------------------------------
def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    # (B, C, H, W) -> (B, H*W, C)
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def prepare_latent_ids(latents_4d: torch.Tensor) -> torch.Tensor:
    b, _, h, w = latents_4d.shape
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(h), torch.arange(w), torch.arange(1)
    )
    return ids.unsqueeze(0).expand(b, -1, -1)


def prepare_text_ids(seq_len: int) -> torch.Tensor:
    ids = torch.cartesian_prod(
        torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(seq_len)
    )
    return ids.unsqueeze(0)


def make_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_DIR, subfolder="scheduler"
    )


def set_timesteps(sched, image_seq_len: int, num_steps: int):
    sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
    sched.set_timesteps(sigmas=sigmas.tolist(), mu=mu)
    if hasattr(sched, "set_begin_index"):
        sched.set_begin_index(0)
    return mu


# ---------------------------------------------------------------------------
# Stage: scheduler (no weights needed)
# ---------------------------------------------------------------------------
def stage_scheduler(w: FixtureWriter):
    for px, seq_len in [(256, 256), (512, 1024), (1024, 4096)]:
        sched = make_scheduler()
        mu = set_timesteps(sched, seq_len, NUM_STEPS)
        w.dump(f"sched.{px}.sigmas", sched.sigmas)  # includes trailing 0
        w.dump(f"sched.{px}.timesteps", sched.timesteps)
        w.meta(f"sched.{px}.mu", mu)
        w.meta(f"sched.{px}.image_seq_len", seq_len)
    w.meta("sched.num_steps", NUM_STEPS)
    w.save()


# ---------------------------------------------------------------------------
# Stage: text encoder (Qwen3, f32, ~16 GB RAM)
# ---------------------------------------------------------------------------
def stage_te(w: FixtureWriter):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR / "tokenizer"))
    print("loading text encoder (f32)...")
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR / "text_encoder"), torch_dtype=torch.float32
    )
    model.eval()

    for pi, prompt in enumerate(PROMPTS):
        # pipeline_flux2_klein.py:_get_qwen3_prompt_embeds, verbatim settings
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        w.meta(f"te.p{pi}.prompt", prompt)
        w.meta(f"te.p{pi}.templated_text", text)
        w.meta(f"te.p{pi}.valid_len", int(attention_mask.sum().item()))
        w.dump(f"te.p{pi}.input_ids", input_ids[0])
        w.dump(f"te.p{pi}.attention_mask", attention_mask[0])

        print(f"TE forward prompt {pi} (valid_len={int(attention_mask.sum())})...")
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hs = out.hidden_states  # 37 entries; index 0 = embeddings
        for k in HIDDEN_LAYERS:
            w.dump(f"te.p{pi}.hidden{k}", hs[k][0])  # (512, 2560)

        stacked = torch.stack([hs[k] for k in HIDDEN_LAYERS], dim=1)  # (1,3,512,2560)
        b, nc, sl, hd = stacked.shape
        prompt_embeds = stacked.permute(0, 2, 1, 3).reshape(b, sl, nc * hd)
        w.dump(f"te.p{pi}.prompt_embeds", prompt_embeds[0])  # (512, 7680)
        del out, hs, stacked, prompt_embeds
        gc.collect()

    w.save()


# ---------------------------------------------------------------------------
# Stage: DiT (Flux2Transformer2DModel, f32, ~15.5 GB RAM, CPU-slow)
# ---------------------------------------------------------------------------
def stage_dit(w: FixtureWriter, sizes):
    from diffusers import Flux2Transformer2DModel
    from diffusers.utils.torch_utils import randn_tensor

    assert "te.p0.prompt_embeds" in w.manifest["tensors"], (
        "run --stage te first (needs te.p0.prompt_embeds)"
    )
    prompt_embeds = w.load("te.p0.prompt_embeds").unsqueeze(0)  # (1, 512, 7680)

    print("loading DiT (f32)...")
    tr = Flux2Transformer2DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32
    )
    tr.eval()

    # --- capture hooks (active only for the first forward of the 256px run) ---
    capture = {"on": False}
    pos_embed_calls = []

    def cap_out(name):
        def hook(mod, args, kwargs, output):
            if not capture["on"]:
                return
            if isinstance(output, tuple):
                for i, o in enumerate(output):
                    w.dump(f"dit.cap.{name}.out{i}", o)
            else:
                w.dump(f"dit.cap.{name}.out", output)

        return hook

    def cap_block(name):
        def pre(mod, args, kwargs):
            if not capture["on"]:
                return
            hsx = kwargs.get("hidden_states", args[0] if args else None)
            w.dump(f"dit.cap.{name}.in_hidden", hsx)
            enc = kwargs.get("encoder_hidden_states", None)
            if enc is not None:
                w.dump(f"dit.cap.{name}.in_encoder", enc)

        return pre

    def pos_embed_hook(mod, args, kwargs, output):
        if not capture["on"]:
            return
        i = len(pos_embed_calls)
        pos_embed_calls.append(i)
        ids = args[0] if args else kwargs["ids"]
        w.dump(f"dit.cap.pos_embed.call{i}.ids", ids)
        w.dump(f"dit.cap.pos_embed.call{i}.cos", output[0])
        w.dump(f"dit.cap.pos_embed.call{i}.sin", output[1])

    def norm_out_pre(mod, args, kwargs):
        if not capture["on"]:
            return
        w.dump("dit.cap.norm_out.in_hidden", args[0])
        w.dump("dit.cap.norm_out.in_temb", args[1])

    hooks = [
        tr.time_guidance_embed.register_forward_hook(
            cap_out("temb"), with_kwargs=True
        ),
        tr.double_stream_modulation_img.register_forward_hook(
            cap_out("mod_double_img"), with_kwargs=True
        ),
        tr.double_stream_modulation_txt.register_forward_hook(
            cap_out("mod_double_txt"), with_kwargs=True
        ),
        tr.single_stream_modulation.register_forward_hook(
            cap_out("mod_single"), with_kwargs=True
        ),
        tr.pos_embed.register_forward_hook(pos_embed_hook, with_kwargs=True),
        tr.x_embedder.register_forward_hook(cap_out("x_embedder"), with_kwargs=True),
        tr.context_embedder.register_forward_hook(
            cap_out("context_embedder"), with_kwargs=True
        ),
        tr.transformer_blocks[0].register_forward_pre_hook(
            cap_block("double0"), with_kwargs=True
        ),
        tr.transformer_blocks[0].register_forward_hook(
            cap_out("double0"), with_kwargs=True
        ),
        tr.transformer_blocks[4].register_forward_pre_hook(
            cap_block("double4"), with_kwargs=True
        ),
        tr.transformer_blocks[4].register_forward_hook(
            cap_out("double4"), with_kwargs=True
        ),
        tr.single_transformer_blocks[0].register_forward_pre_hook(
            cap_block("single0"), with_kwargs=True
        ),
        tr.single_transformer_blocks[0].register_forward_hook(
            cap_out("single0"), with_kwargs=True
        ),
        tr.single_transformer_blocks[19].register_forward_pre_hook(
            cap_block("single19"), with_kwargs=True
        ),
        tr.single_transformer_blocks[19].register_forward_hook(
            cap_out("single19"), with_kwargs=True
        ),
        tr.norm_out.register_forward_pre_hook(norm_out_pre, with_kwargs=True),
        tr.norm_out.register_forward_hook(cap_out("norm_out"), with_kwargs=True),
        tr.proj_out.register_forward_hook(cap_out("proj_out"), with_kwargs=True),
    ]

    def run_denoise(px: int, seed: int, capture_first_step: bool):
        # pipeline __call__ / prepare_latents, verbatim math
        lat_h = 2 * (px // 16)  # 2 * (px // (vae_scale_factor(8) * 2))
        lat_w = lat_h
        shape = (1, 32 * 4, lat_h // 2, lat_w // 2)
        gen = torch.Generator("cpu").manual_seed(seed)
        latents_4d = randn_tensor(shape, generator=gen, dtype=torch.float32)
        latent_ids = prepare_latent_ids(latents_4d)
        latents = pack_latents(latents_4d)  # (1, tokens, 128)
        text_ids = prepare_text_ids(MAX_SEQ)

        w.dump(f"dit.{px}.noise", latents[0])
        w.dump(f"dit.{px}.latent_ids", latent_ids[0])
        w.dump(f"dit.{px}.text_ids", text_ids[0])
        w.meta(f"dit.{px}.seed", seed)

        sched = make_scheduler()
        mu = set_timesteps(sched, latents.shape[1], NUM_STEPS)
        w.meta(f"dit.{px}.mu", mu)

        for i, t in enumerate(sched.timesteps):
            capture["on"] = capture_first_step and i == 0
            timestep = t.expand(1).to(torch.float32)
            print(f"  {px}px step {i} (t={float(t):.3f})...")
            with torch.no_grad():
                noise_pred = tr(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]
            capture["on"] = False
            noise_pred = noise_pred[:, : latents.size(1)]
            w.dump(f"dit.{px}.step{i}.noise_pred", noise_pred[0])
            latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
            w.dump(f"dit.{px}.step{i}.latents", latents[0])
            w.save()  # checkpoint progress (CPU runs are long)

    for px in sizes:
        run_denoise(px, seed=42 + (0 if px == 256 else 1), capture_first_step=(px == 256))

    for h in hooks:
        h.remove()
    w.save()


# ---------------------------------------------------------------------------
# Stage: VAE (f32, 336 MB)
# ---------------------------------------------------------------------------
def stage_vae(w: FixtureWriter):
    from diffusers import AutoencoderKLFlux2

    print("loading VAE (f32)...")
    vae = AutoencoderKLFlux2.from_pretrained(
        MODEL_DIR, subfolder="vae", torch_dtype=torch.float32
    )
    vae.eval()
    bn_eps = vae.config.batch_norm_eps

    w.dump("vae.bn.running_mean", vae.bn.running_mean)  # (128,)
    w.dump("vae.bn.running_var", vae.bn.running_var)
    w.meta("vae.batch_norm_eps", bn_eps)

    # mid-block attention capture (streaming-attention kernel gate)
    mid_cap = {"on": False, "px": 0}

    def find_mid_attn(module):
        for name, m in module.named_modules():
            if name.endswith("mid_block.attentions.0"):
                return m
        return None

    mid_attn = find_mid_attn(vae.decoder)

    def mid_hook(mod, args, kwargs, output):
        if not mid_cap["on"]:
            return
        px = mid_cap["px"]
        x = args[0] if args else kwargs.get("hidden_states")
        w.dump(f"vae.{px}.mid_attn.in", x)
        w.dump(f"vae.{px}.mid_attn.out", output if torch.is_tensor(output) else output[0])
        mid_cap["on"] = False

    if mid_attn is not None:
        mid_attn.register_forward_hook(mid_hook, with_kwargs=True)
    else:
        print("WARN: mid_block.attentions.0 not found; skipping mid-attn capture")

    def bn_denorm(packed_grid: torch.Tensor) -> torch.Tensor:
        # pipeline __call__ lines 909-913: x * std + mean on patchified (128ch) grid
        mean = vae.bn.running_mean.view(1, -1, 1, 1)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + bn_eps)
        return packed_grid * std + mean

    # 1) pure decoder parity on seeded random latents (256px and 512px)
    for px, seed in [(256, 44), (512, 45)]:
        hw = px // 8
        gen = torch.Generator("cpu").manual_seed(seed)
        lat = torch.randn((1, 32, hw, hw), generator=gen, dtype=torch.float32)
        w.dump(f"vae.{px}.rand_latents", lat[0])
        mid_cap["on"] = px == 256
        mid_cap["px"] = px
        print(f"VAE decode random {px}px...")
        with torch.no_grad():
            pix = vae.decode(lat, return_dict=False)[0]
        w.dump(f"vae.{px}.rand_pixels", pix[0])
        w.save()

    # 2) e2e: decode the real denoised 256px latents from the dit stage
    if "dit.256.step3.latents" in w.manifest["tensors"]:
        packed = w.load("dit.256.step3.latents").unsqueeze(0)  # (1, 256, 128)
        n = packed.shape[1]
        hw = int(n**0.5)
        grid = packed.permute(0, 2, 1).reshape(1, 128, hw, hw)  # unpack (row-major ids)
        grid = bn_denorm(grid)
        lat = unpatchify_latents(grid)  # (1, 32, 32, 32)
        w.dump("vae.e2e256.latents_in", lat[0])
        print("VAE decode e2e 256px...")
        with torch.no_grad():
            pix = vae.decode(lat, return_dict=False)[0]
        w.dump("vae.e2e256.pixels", pix[0])
        try:
            from PIL import Image

            img = ((pix[0] / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
            img = img.permute(1, 2, 0).numpy()
            Image.fromarray(img).save(w.out_dir / "e2e_256px.png")
            print(f"  wrote {w.out_dir / 'e2e_256px.png'}")
        except Exception as e:  # PNG is eyeball-only, never gate on it
            print(f"  PNG save skipped: {e}")
    else:
        print("dit.256.step3.latents not in manifest; skipping e2e decode (run --stage dit)")

    # 3) encoder fixture for Phase 6 (edit): deterministic synthetic image
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 256), torch.linspace(-1, 1, 256), indexing="ij"
    )
    img = torch.stack(
        [xx, yy, torch.sin(3.0 * xx) * torch.cos(2.0 * yy)], dim=0
    ).unsqueeze(0)  # (1, 3, 256, 256) in [-1, 1]
    w.dump("vae.enc256.image", img[0])
    print("VAE encode synthetic 256px...")
    with torch.no_grad():
        latent_dist = vae.encode(img, return_dict=False)[0]
    mode = latent_dist.mode()  # sample_mode="argmax" in _encode_vae_image
    w.dump("vae.enc256.latents_mode", mode[0])  # (32, 32, 32)
    # replicate _encode_vae_image: patchify + bn normalize
    patch = patchify_latents(mode)
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + bn_eps)
    normed = (patch - mean) / std
    w.dump("vae.enc256.image_latents", normed[0])  # (128, 16, 16)

    w.save()


# ---------------------------------------------------------------------------
# Stage: edit (Phase 6) -- 256px generation conditioned on the synthetic
# 256px reference image from the vae stage. Replicates the pipeline edit
# path: _encode_vae_image (mode + patchify + bn NORMALIZE) -> pack -> concat
# to the latent seq every step with T=10 ids -> noise_pred[:, :n_gen].
# Needs DiT f32 (~15.5 GB) + VAE f32 (0.34 GB) in one process.
# ---------------------------------------------------------------------------
def stage_edit(w: FixtureWriter):
    from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
    from diffusers.utils.torch_utils import randn_tensor

    assert "te.p1.prompt_embeds" in w.manifest["tensors"], "run --stage te first"
    assert "vae.enc256.image" in w.manifest["tensors"], "run --stage vae first"
    prompt_embeds = w.load("te.p1.prompt_embeds").unsqueeze(0)  # (1, 512, 7680)
    ref_img = w.load("vae.enc256.image").unsqueeze(0)  # (1, 3, 256, 256) [-1,1]

    print("loading VAE (f32)...")
    vae = AutoencoderKLFlux2.from_pretrained(
        MODEL_DIR, subfolder="vae", torch_dtype=torch.float32
    )
    vae.eval()
    bn_eps = vae.config.batch_norm_eps
    mean = vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + bn_eps)

    # _encode_vae_image: encode -> mode (sample_mode="argmax") -> patchify
    # -> bn normalize; then _pack_latents
    print("VAE encode reference image...")
    with torch.no_grad():
        latent_dist = vae.encode(ref_img, return_dict=False)[0]
    mode = latent_dist.mode()  # (1, 32, 32, 32)
    image_latents_4d = (patchify_latents(mode) - mean) / std  # (1, 128, 16, 16)
    image_latents = pack_latents(image_latents_4d)  # (1, 256, 128)
    # _prepare_image_ids: first (and only) reference -> T = 10
    ref_ids = torch.cartesian_prod(
        torch.tensor([10]), torch.arange(16), torch.arange(16), torch.arange(1)
    ).unsqueeze(0)  # (1, 256, 4)
    w.dump("edit.image_latents", image_latents[0])
    w.dump("edit.image_latent_ids", ref_ids[0])

    # generation latents: 256px -> (1, 128, 16, 16) -> 256 tokens
    seed = 46
    gen = torch.Generator("cpu").manual_seed(seed)
    latents_4d = randn_tensor((1, 128, 16, 16), generator=gen, dtype=torch.float32)
    latent_ids = prepare_latent_ids(latents_4d)  # T=0
    latents = pack_latents(latents_4d)  # (1, 256, 128)
    text_ids = prepare_text_ids(MAX_SEQ)
    w.dump("edit.noise", latents[0])
    w.dump("edit.latent_ids", latent_ids[0])
    w.meta("edit.seed", seed)
    w.meta("edit.prompt_idx", 1)
    w.meta("edit.px", 256)

    sched = make_scheduler()
    # image_seq_len = latents.shape[1] BEFORE the ref concat (pipeline :815)
    mu = set_timesteps(sched, latents.shape[1], NUM_STEPS)
    w.meta("edit.mu", mu)

    print("loading DiT (f32)...")
    tr = Flux2Transformer2DModel.from_pretrained(
        MODEL_DIR, subfolder="transformer", torch_dtype=torch.float32
    )
    tr.eval()

    img_ids = torch.cat([latent_ids, ref_ids], dim=1)  # (1, 512, 4)
    for i, t in enumerate(sched.timesteps):
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(1).to(torch.float32)
        print(f"  edit step {i} (t={float(t):.3f})...")
        with torch.no_grad():
            noise_pred = tr(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]
        noise_pred = noise_pred[:, : latents.size(1)]
        w.dump(f"edit.step{i}.noise_pred", noise_pred[0])
        latents = sched.step(noise_pred, t, latents, return_dict=False)[0]
        w.dump(f"edit.step{i}.latents", latents[0])
        w.save()  # checkpoint progress (CPU runs are long)
    del tr
    gc.collect()

    # decode the final latents (row-major ids -> plain reshape unpack)
    grid = latents.permute(0, 2, 1).reshape(1, 128, 16, 16)
    grid = grid * std + mean
    lat = unpatchify_latents(grid)  # (1, 32, 32, 32)
    print("VAE decode edited image...")
    with torch.no_grad():
        pix = vae.decode(lat, return_dict=False)[0]
    w.dump("edit.pixels", pix[0])
    try:
        from PIL import Image

        img = ((pix[0] / 2 + 0.5).clamp(0, 1) * 255).round().to(torch.uint8)
        Image.fromarray(img.permute(1, 2, 0).numpy()).save(w.out_dir / "edit_256px.png")
        print(f"  wrote {w.out_dir / 'edit_256px.png'}")
    except Exception as e:  # PNG is eyeball-only, never gate on it
        print(f"  PNG save skipped: {e}")
    w.save()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage", required=True, choices=["scheduler", "te", "dit", "vae", "edit"]
    )
    ap.add_argument(
        "--sizes",
        default="256,512",
        help="dit stage: comma-separated pixel sizes to run (default 256,512)",
    )
    args = ap.parse_args()

    print(f"model dir: {MODEL_DIR}")
    assert MODEL_DIR.exists(), f"model dir missing: {MODEL_DIR}"
    w = FixtureWriter(OUT_DIR)

    if args.stage == "scheduler":
        stage_scheduler(w)
    elif args.stage == "te":
        stage_te(w)
    elif args.stage == "dit":
        sizes = [int(s) for s in args.sizes.split(",") if s]
        stage_dit(w, sizes)
    elif args.stage == "vae":
        stage_vae(w)
    elif args.stage == "edit":
        stage_edit(w)
    print("done.")


if __name__ == "__main__":
    main()
