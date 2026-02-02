import os
import sys
import json
import argparse

# -----------------------------------------------------------------------------
# 1. 环境设置
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from vibt.env import CONFIG
except ImportError as e:
    print(f"❌ Import Error: {e}")
    sys.exit(1)

def parse_error_log(log_path):
    """从日志中提取损坏文件的路径集合"""
    bad_files = set()
    if not os.path.exists(log_path):
        print(f"❌ Log file not found: {log_path}")
        return bad_files

    print(f"📖 Reading log file: {log_path}")
    with open(log_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # 跳过空行和标题行
        if not line or line.startswith("=") or line.startswith("Validation Time") or line.startswith("["):
            continue
        
        # 提取路径 (处理 "path \t REASON: ..." 格式)
        if "\tREASON:" in line:
            path = line.split("\tREASON:")[0].strip()
        else:
            path = line.strip()
            
        if path:
            bad_files.add(path)
            
    return bad_files

def main():
    parser = argparse.ArgumentParser(description="Remove corrupted files from index.json based on error log.")
    parser.add_argument("--log_file", type=str, default="dataset_fast_error_log.txt", help="Path to the error log file")
    parser.add_argument("--output_file", type=str, default="index_clean.json", help="Name of the cleaned index file")
    args = parser.parse_args()

    # 1. 获取坏文件列表
    bad_files_abs = parse_error_log(args.log_file)
    if not bad_files_abs:
        print("✅ No bad files found in log (or log is empty). Exiting.")
        return

    print(f"🔍 Found {len(bad_files_abs)} bad files in log.")

    # 2. 确定索引路径
    data_root = os.path.join(CONFIG.dataset.root, CONFIG.dataset.phase)
    index_path = os.path.join(data_root, CONFIG.dataset.index)
    
    if not os.path.exists(index_path):
        print(f"❌ Index file not found: {index_path}")
        return

    # 3. 加载原始索引
    print(f"📖 Loading original index: {index_path}")
    with open(index_path, 'r', encoding='utf-8') as f:
        original_index = json.load(f)

    total_samples = len(original_index)
    print(f"   Total samples before: {total_samples}")

    # 4. 过滤逻辑
    new_index = {}
    removed_ids = []
    
    for vid_id, item in original_index.items():
        # 获取该样本涉及的所有文件路径 (转换为绝对路径以匹配日志)
        sample_files = []
        if 'the first view' in item:
            sample_files.append(os.path.join(data_root, item['the first view']))
        if 'the third view' in item:
            sample_files.append(os.path.join(data_root, item['the third view']))
        
        # 检查是否命中坏文件
        is_bad_sample = False
        for f_path in sample_files:
            if f_path in bad_files_abs:
                is_bad_sample = True
                break
        
        if is_bad_sample:
            removed_ids.append(vid_id)
        else:
            new_index[vid_id] = item

    # 5. 输出结果
    print(f"\n🗑️  Removed {len(removed_ids)} samples.")
    print(f"✅ Remaining samples: {len(new_index)}")
    
    # 6. 保存新索引
    output_path = os.path.join(data_root, args.output_file)
    print(f"💾 Saving cleaned index to: {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_index, f, indent=4)
        
    print("\n💡 Next Steps:")
    print(f"1. Check the new index file.")
    print(f"2. Update your 'config/video2video.yaml' to use the new index:")
    print(f"   dataset:")
    print(f"     index: \"{args.output_file}\"")

if __name__ == "__main__":
    main()