import torch
from diffusers import WanPipeline
import re
from safetensors.torch import load_file
import os
from huggingface_hub import snapshot_download
import numpy as np
from einops import repeat, reduce


@torch.no_grad()
def encode_video(pipe: WanPipeline, video_frames):
    video_tensor = pipe.video_processor.preprocess_video(video_frames).to(
        dtype=pipe.dtype, device=pipe.device
    )
    posterior = pipe.vae.encode(video_tensor, return_dict=False)[0]
    z = posterior.mode()
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(z.device, z.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(z.device, z.dtype)
    latents = (z - latents_mean) * latents_std
    return latents


@torch.no_grad()
def decode_latents(pipe: WanPipeline, latents):
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    video = pipe.vae.decode(latents, return_dict=False)[0]
    video = pipe.video_processor.postprocess_video(video, output_type="np")
    return video


def name_convert(n: str):
    # blocks.* attention
    m = re.match(
        r"blocks\.(\d+)\.(self_attn|cross_attn)\.(q|k|v|o|norm_k|norm_q)\.(weight|bias)",
        n,
    )
    if m:
        b, kind, comp, suf = m.groups()
        attn = "attn1" if kind == "self_attn" else "attn2"
        if comp in ("q", "k", "v"):
            return f"blocks.{b}.{attn}.to_{comp}.{suf}"
        if comp == "o":
            return f"blocks.{b}.{attn}.to_out.0.{suf}"
        return f"blocks.{b}.{attn}.{comp}.{suf}"

    # blocks.* ffn
    m = re.match(r"blocks\.(\d+)\.ffn\.(0|2)\.(weight|bias)", n)
    if m:
        b, idx, suf = m.groups()
        if idx == "0":
            return f"blocks.{b}.ffn.net.0.proj.{suf}"
        return f"blocks.{b}.ffn.net.2.{suf}"

    # blocks.* norm3/modulation
    m = re.match(r"blocks\.(\d+)\.norm3\.(weight|bias)", n)
    if m:
        b, suf = m.groups()
        return f"blocks.{b}.norm2.{suf}"

    m = re.match(r"blocks\.(\d+)\.modulation$", n)
    if m:
        b = m.group(1)
        return f"blocks.{b}.scale_shift_table"

    # patch_embedding
    if n.startswith("patch_embedding."):
        return n

    # text / time embedding
    m = re.match(r"text_embedding\.(0|2)\.(weight|bias)", n)
    if m:
        idx, suf = m.groups()
        lin = "linear_1" if idx == "0" else "linear_2"
        return f"condition_embedder.text_embedder.{lin}.{suf}"

    m = re.match(r"time_embedding\.(0|2)\.(weight|bias)", n)
    if m:
        idx, suf = m.groups()
        lin = "linear_1" if idx == "0" else "linear_2"
        return f"condition_embedder.time_embedder.{lin}.{suf}"

    m = re.match(r"time_projection\.1\.(weight|bias)", n)
    if m:
        suf = m.group(1)
        return f"condition_embedder.time_proj.{suf}"

    # head
    if n == "head.head.weight":
        return "proj_out.weight"
    if n == "head.head.bias":
        return "proj_out.bias"
    if n == "head.modulation":
        return "scale_shift_table"

    return n


def load_vibt_weight(transformer, repo_path="Yuanshi/Bridge"):
    repo_path = snapshot_download(repo_path)
    tensors = load_file(
        os.path.join(
            repo_path, "stylization/run_20251103_235853/step-20000.safetensors"
        )
    )

    new_tensors = {}

    for key, value in tensors.items():
        key = name_convert(key)
        new_tensors[key] = value

    for name, param in transformer.named_parameters():
        device, dtype = param.device, param.dtype
        if name in new_tensors:
            assert (
                param.shape == new_tensors[name].shape
            ), f"{name}: {param.shape} != {new_tensors[name].shape}"
            param.data = new_tensors[name].to(device=device, dtype=dtype)
