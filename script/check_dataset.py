import os
import sys
import json
import argparse
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# -----------------------------------------------------------------------------
# 环境与导入
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from vibt.env import CONFIG
    from vibt.utils import load_video_to_device
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Worker: 单样本校验 (Pair Check)
# -----------------------------------------------------------------------------
def validate_sample_worker(args):
    """
    校验单个样本（同时检查 Ego 和 Exo）
    Args:
        args: (vid_id, ego_path, exo_path, clip_len, stride)
    """
    vid_id, ego_path, exo_path, clip_len, stride = args
    
    # 1. 检查文件存在性
    if not os.path.exists(ego_path):
        return vid_id, False, f"Missing Ego: {ego_path}"
    if not os.path.exists(exo_path):
        return vid_id, False, f"Missing Exo: {exo_path}"

    # 2. 尝试读取 (Fast Check: 只读 max_frames)
    # 使用 CPU，并且利用 utils.py 的早停机制
    try:
        # Check Ego
        ego_tensor = load_video_to_device(
            ego_path, 
            max_frames=clip_len, 
            sample_stride=stride, 
            device='cpu'
        )
        if ego_tensor is None or ego_tensor.shape[1] == 0:
             return vid_id, False, f"Corrupted Ego: {ego_path}"

        # Check Exo
        exo_tensor = load_video_to_device(
            exo_path, 
            max_frames=clip_len, 
            sample_stride=stride, 
            device='cpu'
        )
        if exo_tensor is None or exo_tensor.shape[1] == 0:
             return vid_id, False, f"Corrupted Exo: {exo_path}"
             
        # (可选) 检查对齐：如果两者读取到的帧数差异巨大，可能意味着源视频长度严重不一致
        # 但由于 load_video_to_device 会自动 padding/truncate 到 clip_len，
        # tensor.shape[1] 总是等于 clip_len，所以这里很难通过 shape 判断原始长度。
        # 只要能读出来，就算通过。

        return vid_id, True, "OK"

    except Exception as e:
        return vid_id, False, f"Exception: {str(e)}"

# -----------------------------------------------------------------------------
# 主程序
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fast Dataset Validator")
    parser.add_argument("--workers", type=int, default=16, help="Worker processes")
    args = parser.parse_args()

    print(f"==================================================")
    print(f"   ViBT Fast Dataset Validator")
    print(f"==================================================")
    
    # 1. 配置加载
    data_root = '/opt/liblibai-models/user-workspace2/users/xqy/project/ViBT/dataset/test_unseen'
    index_path = os.path.join(data_root, "index.json")
    
    clip_len = CONFIG.dataset.clip_len
    stride = CONFIG.dataset.stride
    
    print(f"📂 Root:   {data_root}")
    print(f"🎞️  Config: clip_len={clip_len}, stride={stride}")

    # 2. 读取索引
    with open(index_path, 'r') as f:
        index_data = json.load(f)
    print(f"🔍 Total Samples: {len(index_data)}")

    # 3. 准备任务
    tasks = []
    for vid_id, item in index_data.items():
        ego_p = os.path.join(data_root, item.get('the first view', ''))
        exo_p = os.path.join(data_root, item.get('the third view', ''))
        tasks.append((vid_id, ego_p, exo_p, clip_len, stride))

    # 4. 并行执行
    print(f"🚀 Starting with {args.workers} workers...")
    bad_samples = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(validate_sample_worker, t): t[0] for t in tasks}
        
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Validating"):
            vid_id, is_valid, msg = future.result()
            if not is_valid:
                bad_samples.append(f"{vid_id}\t{msg}")

    total_time = time.time() - start_time
    
    # 5. 报告
    print("\n" + "="*50)
    print(f"   Validation Report ({total_time:.1f}s)")
    print("="*50)
    print(f"✅ Good Samples: {len(tasks) - len(bad_samples)}")
    print(f"❌ Bad Samples:  {len(bad_samples)}")
    
    if bad_samples:
        log_file = "dataset_fast_error_log.txt"
        with open(log_file, "w") as f:
            for line in bad_samples:
                f.write(line + "\n")
        print(f"📝 Bad samples saved to: {log_file}")
        print("💡 You can use 'script/clean_dataset.py' logic to remove them.")
    else:
        print("🎉 Dataset is perfectly clean for training!")

if __name__ == "__main__":
    main()