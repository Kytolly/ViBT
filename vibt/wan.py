import torch
import torch.nn as nn
import re
from diffusers import WanPipeline
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import logging
logger = logging.getLogger(__name__)

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

class WanModel(nn.Module):
    def __init__(self, pretrained_model_path, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        logger.info(f"Loading WanPipeline from {pretrained_model_path}...")
        # 加载完整 Pipeline (VAE, TextEncoder, Transformer)
        self.pipe = WanPipeline.from_pretrained(
            pretrained_model_path, 
            torch_dtype=dtype
        ).to(device)
        
        # 为了方便访问，建立引用
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.transformer = self.pipe.transformer # 这是我们要训练的核心
        
        # 冻结 VAE 和 Text Encoder (ViBT 仅训练 Transformer)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.transformer.requires_grad_(False) # 稍后在 Trainer 中开启 LoRA

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        return cls(path, **kwargs)

    @torch.no_grad()
    def encode(self, pixel_values):
        """
        将视频像素编码为 Latent。
        Args:
            pixel_values: [B, C, F, H, W] 范围 [-1, 1]
        Returns:
            latents: [B, C_out, F_out, H_out, W_out]
        """
        # Wan VAE 通常期望输入范围是 [-1, 1] (如果是基于 SD 的 VAE) 
        # 或者 [0, 1]。Diffusers 的 image_processor 处理逻辑较复杂。
        # 这里的 VAE (AutoencoderKLWan) encode 方法接受 [B, C, F, H, W]
        # 我们假设输入已经是 Tensor 且在 device 上
        
        # 确保数据类型匹配
        pixel_values = pixel_values.to(dtype=self.dtype, device=self.device)
        
        # VAE Encode
        # Wan2.1 的 VAE encode 输出是分布，我们需要采样
        if hasattr(self.vae, 'encode'):
            dist = self.vae.encode(pixel_values).latent_dist
            latents = dist.sample()
        else:
            raise NotImplementedError("Unknown VAE structure")

        # 标准化 Latents (这也是 Diffusers pipeline 内部做的)
        latents = (latents - self.vae.config.latents_mean) / self.vae.config.latents_std
        return latents

    @torch.no_grad()
    def encode_prompt(self, prompts, max_length=512):
        """编码文本提示"""
        # 使用 Pipeline 内部的逻辑 (简化版)
        prompt_embeds = self.pipe._get_text_embeddings(
            prompt=prompts,
            max_sequence_length=max_length
        )
        # prompt_embeds 通常是 tuple (context, negative_context)
        # 训练时我们只需要 context
        return prompt_embeds[0]

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        """
        Transformer 前向传播
        Args:
            hidden_states: Noisy Latents / Bridge State [B, C, F, H, W]
            timestep: Time step tensor [B]
            encoder_hidden_states: Text Embeddings [B, L, D]
        """
        # Wan Transformer 的输入参数
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False
        )[0] # 返回 (sample,)

    def save_pretrained(self, save_directory):
        """保存模型 (主要是 LoRA 权重)"""
        self.pipe.save_pretrained(save_directory)