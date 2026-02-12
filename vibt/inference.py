import torch
from torch import Tensor
from diffusers import WanPipeline
from diffusers.utils import export_to_video, load_video

from .wan import load_vibt_weight, encode_video
from .scheduler import ViBTScheduler

def validate_with_checkpoint(
    pipe: WanPipeline,
    prompt: str,
    source_latents: Tensor,
    checkpoint_path: str,
    output_path: str,
    noise_scale: float = 1.0,
    shift_gamma: float = 5.0,
    num_inference_steps: int = 28,
    guidance_scale: float = 1.5,
    seed: int = 42,
    device: str = "cuda",
):
    load_vibt_weight(
        pipe.transformer,
        local_path=checkpoint_path,
    )
    pipe.scheduler = ViBTScheduler.from_scheduler(pipe.scheduler)
    pipe.scheduler.set_parameters(noise_scale=noise_scale, shift_gamma=shift_gamma, seed=seed)
    output = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        latents=source_latents,
    ).frames[0]
    export_to_video(output, output_path, fps=15)

