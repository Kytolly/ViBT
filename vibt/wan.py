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
    def encode_prompt(self, prompts: list[str], max_length: int = 512):
        """
        显式编码文本提示，不依赖 Pipeline 的私有方法。
        """
        # 1. Tokenize
        # Wan2.1 使用 T5 Tokenizer
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # 2. 搬运到 GPU
        text_input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device) # T5 通常需要 mask

        # 3. 编码 (T5 Encoder)
        # output[0] 是 last_hidden_state
        prompt_embeds = self.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False
        )[0]
        
        # 4. 数据类型转换 (确保匹配 Transformer 的 dtype，如 bf16)
        prompt_embeds = prompt_embeds.to(dtype=self.dtype)
        
        return prompt_embeds