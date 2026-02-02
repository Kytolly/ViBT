import torch
import torch.nn as nn
import re
from diffusers import WanPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import logging
logger = logging.getLogger(__name__)

class WanModel(nn.Module):
    def __init__(self, pretrained_model_path, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        print(f"Loading WanPipeline from {pretrained_model_path}...")
        self.pipe = WanPipeline.from_pretrained(
            pretrained_model_path, 
            torch_dtype=dtype
        ).to(device)
        
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.transformer = self.pipe.transformer
        
        # 默认冻结组件 (Transformer会在train.py中根据需要解冻)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False)

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(path, **kwargs)

    @torch.no_grad()
    def encode(self, pixel_values):
        """Encode video frames to normalized latents."""
        pixel_values = pixel_values.to(dtype=self.vae.dtype, device=self.device)
        
        # VAE Encode
        dist = self.vae.encode(pixel_values).latent_dist
        latents = dist.sample()

        # [核心修复] 将 list 转换为 tensor，并调整 shape 用于广播
        # latents: [B, C, F, H, W]
        # mean/std: [C] -> [1, C, 1, 1, 1]
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        
        latents = (latents - latents_mean) / latents_std
        return latents

    @torch.no_grad()
    def decode(self, latents):
        """Decode normalized latents back to video frames."""
        latents = latents.to(dtype=self.vae.dtype)
        
        # [核心修复] 同样需要转换
        latents_mean = torch.tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        
        latents = latents * latents_std + latents_mean
        
        # VAE Decode
        video = self.vae.decode(latents, return_dict=False)[0]
        return video

    @torch.no_grad()
    def encode_prompt(self, prompts: list[str], max_length: int = 512):
        """
        显式编码文本提示
        """
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)

        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )[0]
        
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        
        return prompt_embeds

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        """Transformer Forward Pass"""
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0]

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


def load_vibt_weight(
    transformer, repo_name="Yuanshi/Bridge", weight_path=None, local_path=None
):
    assert (
        weight_path or local_path
    ) is not None, "Either weight_path or local_path must be provided."

    tensors = load_file(local_path or hf_hub_download(repo_name, weight_path))

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