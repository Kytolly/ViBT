import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from vibt.wan import WanModel
from dataset_wrapper import FollowBenchDatasetWrapper 

class ViBTTrainer:
    def __init__(self, args):
        self.device = "cuda"
        self.args = args
        
        print(f"Loading Wan2.1 1.3B from {args.pretrained_model_path}")
        self.model = WanModel.from_pretrained(args.pretrained_model_path)
        self.model.to(self.device)
        
        # 冻结所有参数，仅训练 LoRA
        self.model.requires_grad_(False)
        
        # 注入 LoRA (针对 Transformer)
        # 假设 WanModel.model 是实际的 Transformer (需检查 vibt/wan.py 源码确认属性名)
        # 如果 vibt/wan.py 没有自动加 LoRA，我们需要手动加
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=16, lora_alpha=32, 
            target_modules=["to_q", "to_k", "to_v", "to_out.0"], # Wan 的 Attention 层
            bias="none"
        )
        self.model.model = get_peft_model(self.model.model, lora_config)
        self.model.model.print_trainable_parameters()
        
        self.optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=1e-4)

    def compute_loss(self, source_pixel, target_pixel, prompts):
        # 1. 编码视频到 Latent
        # source_pixel, target_pixel: [B, C, F, H, W]
        with torch.no_grad():
            z_0 = self.model.encode(source_pixel) # Source Latent (Start)
            z_1 = self.model.encode(target_pixel) # Target Latent (End)
            
        # 2. 采样时间步 t ~ U[0, 1]
        B = z_0.shape[0]
        t = torch.rand((B,), device=self.device)
        
        # 3. 构造 Bridge State (线性插值)
        # z_t = (1-t) * z_0 + t * z_1
        t_expand = t.view(B, 1, 1, 1) # 调整维度以支持广播
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        
        # 4. 计算目标速度 (Target Velocity)
        # v = z_1 - z_0
        target_v = z_1 - z_0
        
        # 5. 模型预测
        # ViBT 的输入通常是 (z_t, t, prompt_embeddings)
        # vibt/wan.py 的 forward 应该处理了文本编码
        # 注意：需要确认 forward 接口，假设是 model(x, t, prompt_list)
        # 时间步通常需要缩放到模型原本的训练范围 (例如 0-1000)
        t_input = t * 1000 
        pred_v = self.model(z_t, t_input, prompts)
        
        # 6. Loss
        loss = F.mse_loss(pred_v, target_v)
        return loss

    def train(self, dataloader, epochs):
        self.model.train()
        for epoch in range(epochs):
            pbar = tqdm(dataloader)
            for batch in pbar:
                # 假设 Dataset 返回字典
                src = batch['first_pixel_values'].to(self.device) # Ego
                tgt = batch['third_pixel_values'].to(self.device) # Exo
                txt = batch['prompts']
                
                loss = self.compute_loss(src, tgt, txt)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                pbar.set_description(f"Ep {epoch} Loss: {loss.item():.4f}")
                
            # Save Checkpoint
            self.model.model.save_pretrained(f"output/vibt_epoch_{epoch}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)
    args = parser.parse_args()
    
    # 使用你之前实现的 Dataset
    dataset = CustomTrainDataset(
        json_index_path=args.data_json, 
        followbench_root=args.video_root,
        warped_video_root=args.video_root, # 如果不需要 warp 可填任意值
        height=480, width=832, # Wan2.1 推荐分辨率 (必须是 16 倍数)
        sample_n_frames=33     # Wan2.1 推荐帧数 (1.3B 通常是 4n+1)
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    
    trainer = ViBTTrainer(args)
    trainer.train(dataloader, epochs=10)