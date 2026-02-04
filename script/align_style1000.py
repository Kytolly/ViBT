import os
import json
import subprocess
from tqdm import tqdm
import logging
DETAILED_FORMAT = (
    '%(asctime)s | '
    '%(levelname)-s | '
    '%(name)s | '
    '%(filename)s:%(lineno)d | '
    '%(funcName)s() | ' 
    # 'PID:%(process)d | TID:%(thread)d | '
    '%(message)s'
)
logging.basicConfig(
    level=logging.INFO,
    format=DETAILED_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

def align_video(input_path, output_path, target_fps=24, target_frames=None):
    """使用 ffmpeg 强制转换 FPS 并裁剪长度"""
    # -r 强制 FPS, -vframes 强制总帧数
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-filter:v', f'fps=fps={target_fps}',
        '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p'
    ]
    if target_frames:
        cmd += ['-frames:v', str(target_frames)]
    cmd.append(output_path)
    
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_dataset(root_dir, index_file):
    with open(os.path.join(root_dir, index_file), 'r') as f:
        dataset = json.load(f)

    logger.info("🚀 开始物理对齐 style1000 数据集...")
    
    for i, item in enumerate(tqdm(dataset)):
        de_path = os.path.join(root_dir, item['destyle'])
        st_path = os.path.join(root_dir, item['style'])
        
        # 目标：统一为 24 FPS
        # 我们先获取 style 视频的长度作为基准
        # (这里建议先用对齐工具将所有对视频处理成相同长度)
        tmp_de = de_path.replace(".mp4", "_aligned.mp4")
        tmp_st = st_path.replace(".mp4", "_aligned.mp4")

        # 1. 统一重采样 destyle 到 24FPS，并与 style 对齐
        # 建议直接将两者都重采样到 24FPS 且截取前 N 帧（取两者最小值）
        align_video(de_path, tmp_de, target_fps=24)
        align_video(st_path, tmp_st, target_fps=24)
        
        # 2. 覆盖原文件或更新索引 (建议覆盖以保持目录整洁)
        os.replace(tmp_de, de_path)
        os.replace(tmp_st, st_path)

if __name__ == "__main__":
    process_dataset("/opt/liblibai-models/user-workspace2/datasets/style1000", "index_for_train.json")