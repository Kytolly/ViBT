import os
import json
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_single_video(input_path, output_path, fps=24, frames=120):
    """
    使用 ffmpeg 强制转换 FPS 并精确截断帧数
    -filter:v "fps=24" : 确保每秒 24 帧，处理丢帧/补帧
    -frames:v 120      : 严格只输出前 120 帧
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:v', f'fps=fps={fps}',
        '-frames:v', str(frames),
        '-c:v', 'libx264', '-crf', '18', # 高质量编码
        '-pix_fmt', 'yuv420p',           # 确保 VAE 兼容性
        output_path
    ]
    
    # 运行命令并隐藏输出
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def align_dataset(root_dir, output_dir, index_file):
    index_path = os.path.join(root_dir, index_file)
    with open(index_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    new_dataset = []
    logger_info = f"🚀 开始离线对齐 {len(dataset)} 个样本..."
    print(logger_info)

    # 使用多线程加速处理 (根据 CPU 核心数调整 max_workers)
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for item in dataset:
            # 这里的路径处理要对应您的 index 格式
            for key in ['destyle', 'style']:
                in_p = os.path.join(root_dir, item[key])
                out_p = os.path.join(output_dir, item[key])
                futures.append(executor.submit(process_single_video, in_p, out_p))
            
            # 记录新的索引项
            new_dataset.append(item)

        # 等待所有任务完成
        for _ in tqdm(futures, desc="FFmpeg Processing"):
            pass

    # 导出新的索引文件
    new_index_path = os.path.join(output_dir, "index_for_train_aligned.json")
    with open(new_index_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 处理完成！对齐后的数据已存入: {output_dir}")
    print(f"📝 新索引文件已生成: {new_index_path}")

if __name__ == "__main__":
    SRC_ROOT = "/opt/liblibai-models/user-workspace2/datasets/style1000"
    DST_ROOT = "/opt/liblibai-models/user-workspace2/datasets/style1000_aligned"
    INDEX = "index_for_train.json"

    align_dataset(SRC_ROOT, DST_ROOT, INDEX)