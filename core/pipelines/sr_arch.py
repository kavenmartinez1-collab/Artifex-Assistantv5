"""
Artifex Assistant V5 — Super-resolution network architectures.

Self-contained (basicsr-free) implementations of the Real-ESRGAN inference
networks, so photo restoration does not depend on the `realesrgan`/`basicsr`
packages, which are incompatible with torchvision >= 0.17.

Weights are the official Real-ESRGAN release checkpoints; state-dict key
names match basicsr exactly so the published .pth files load with strict=True.
"""

import os
import urllib.request

import torch
from torch import nn
from torch.nn import functional as F


# Official release checkpoints (BSD-3 licensed weights from xinntao/Real-ESRGAN)
SR_MODELS = {
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.1.0/RealESRGAN_x4plus.pth",
        "arch": "rrdbnet",
        "scale": 4,
        "kwargs": {"num_feat": 64, "num_block": 23, "num_grow_ch": 32},
    },
    "realesrgan-x2plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.1/RealESRGAN_x2plus.pth",
        "arch": "rrdbnet",
        "scale": 2,
        "kwargs": {"num_feat": 64, "num_block": 23, "num_grow_ch": 32},
    },
    "realesr-general-x4v3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/"
               "v0.2.5.0/realesr-general-x4v3.pth",
        "arch": "srvgg",
        "scale": 4,
        "kwargs": {"num_feat": 64, "num_conv": 32, "act_type": "prelu"},
    },
}


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """ESRGAN generator. Scale 2 packs pixels via pixel_unshuffle so the
    trunk always upsamples 4x internally."""

    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64,
                 num_block=23, num_grow_ch=32):
        super().__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, downscale_factor=2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, downscale_factor=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(
            F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style net (realesr-general-x4v3): much faster and lighter
    than RRDBNet, slightly softer output."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
                 upscale=4, act_type="prelu"):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        self.body.append(self._act(act_type, num_feat))
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._act(act_type, num_feat))
        self.body.append(
            nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _act(act_type, num_feat):
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        if act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        return nn.PReLU(num_parameters=num_feat)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


def download_weights(model_name: str, weights_dir: str,
                     status_callback=None) -> str:
    """Download release weights for a named SR model if not already cached."""
    info = SR_MODELS[model_name]
    os.makedirs(weights_dir, exist_ok=True)
    dest = os.path.join(weights_dir, os.path.basename(info["url"]))
    if os.path.isfile(dest):
        return dest
    if status_callback:
        status_callback(f"Downloading {model_name} weights...")
    tmp = dest + ".part"
    urllib.request.urlretrieve(info["url"], tmp)
    os.replace(tmp, dest)
    return dest


def load_sr_model(model_name: str, weights_dir: str, device="cuda",
                  half=True, status_callback=None) -> nn.Module:
    """Build the network, load release weights (strict), move to device."""
    info = SR_MODELS[model_name]
    path = download_weights(model_name, weights_dir, status_callback)
    state = torch.load(path, map_location="cpu", weights_only=True)
    # Release checkpoints wrap the state dict in params_ema/params
    state = state.get("params_ema") or state.get("params") or state

    if info["arch"] == "rrdbnet":
        net = RRDBNet(scale=info["scale"], **info["kwargs"])
    else:
        net = SRVGGNetCompact(upscale=info["scale"], **info["kwargs"])
    net.load_state_dict(state, strict=True)
    net.eval()
    net = net.to(device)
    if half and str(device).startswith("cuda"):
        net = net.half()
    return net


@torch.inference_mode()
def upscale_tiled(net: nn.Module, image, scale: int, device="cuda",
                  tile: int = 512, tile_pad: int = 16,
                  status_callback=None):
    """Upscale a PIL image with tiling so arbitrarily large photos fit in VRAM.

    Args:
        net: SR network (RRDBNet / SRVGGNetCompact)
        image: PIL.Image (RGB)
        scale: network's native scale factor
        tile: tile edge in input pixels (0 = no tiling)
    Returns:
        PIL.Image upscaled by `scale`
    """
    import numpy as np
    from PIL import Image

    param = next(net.parameters())
    img = torch.from_numpy(
        np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0).to(device=param.device, dtype=param.dtype)

    # RRDBNet pixel_unshuffles at scale 2 (by 2) and scale 1 (by 4), so
    # dimensions must divide evenly — reflect-pad, crop after upscaling.
    orig_h, orig_w = img.shape[2], img.shape[3]
    mod = 2 if scale == 2 else 4 if scale == 1 else 1
    pad_h, pad_w = (-orig_h) % mod, (-orig_w) % mod
    if pad_h or pad_w:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

    _, _, h, w = img.shape
    if tile <= 0 or (h <= tile and w <= tile):
        out = net(img)
    else:
        out = img.new_zeros((1, 3, h * scale, w * scale))
        tiles_x = -(-w // tile)
        tiles_y = -(-h // tile)
        for ty in range(tiles_y):
            for tx in range(tiles_x):
                if status_callback:
                    status_callback(
                        f"Upscaling tile {ty * tiles_x + tx + 1}"
                        f"/{tiles_x * tiles_y}...")
                x0, y0 = tx * tile, ty * tile
                x1, y1 = min(x0 + tile, w), min(y0 + tile, h)
                # Padded input window for seam-free stitching
                px0, py0 = max(x0 - tile_pad, 0), max(y0 - tile_pad, 0)
                px1, py1 = min(x1 + tile_pad, w), min(y1 + tile_pad, h)
                patch = net(img[:, :, py0:py1, px0:px1])
                # Crop the padding back out of the upscaled patch
                ox0 = (x0 - px0) * scale
                oy0 = (y0 - py0) * scale
                out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                    patch[:, :, oy0:oy0 + (y1 - y0) * scale,
                          ox0:ox0 + (x1 - x0) * scale]

    out = out[:, :, :orig_h * scale, :orig_w * scale]
    out = out.squeeze(0).float().clamp_(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((out * 255.0).round().astype(np.uint8))
