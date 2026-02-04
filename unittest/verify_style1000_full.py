import os
import json
import logging
import numpy as np
from tqdm import tqdm
from decord import VideoReader, cpu

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DataAudit")

def verify_dataset(root_dir, index_file, clip_len=16, stride=4):
    index_path = os.path.join(root_dir, index_file)
    with open(index_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    logger.info(f"🔍 启动 style1000 深度审计 | 样本总数: {len(dataset)}")
    
    # 计算模型所需的最小总帧数
    min_required_frames = (clip_len - 1) * stride + 1
    errors = []

    for i, item in enumerate(tqdm(dataset)):
        destyle_path = os.path.join(root_dir, item['destyle'])
        style_path = os.path.join(root_dir, item['style'])
        caption = item.get('caption', "").strip()

        # 1. 基础文件存在性
        if not os.path.exists(destyle_path) or not os.path.exists(style_path):
            errors.append(f"Index {i}: 文件缺失 - {item['destyle']} 或 {item['style']}")
            continue

        try:
            # 2. 视频合法性与长度对齐校验
            vr_de = VideoReader(destyle_path, ctx=cpu(0))
            vr_st = VideoReader(style_path, ctx=cpu(0))
            
            len_de = len(vr_de)
            len_st = len(vr_st)

            # 校验 A: 长度是否严格对齐
            if len_de != len_st:
                errors.append(f"Index {i}: 长度不对齐! destyle({len_de}) vs style({len_st})")
            
            # 校验 B: 视频是否满足配置所需的最小长度
            if len_de < min_required_frames:
                errors.append(f"Index {i}: 视频过短! 长度({len_de}) < 配置要求({min_required_frames})")

            # 校验 C: 检查 FPS (非对齐可能导致运动不匹配)
            fps_de = vr_de.get_avg_fps()
            fps_st = vr_st.get_avg_fps()
            if abs(fps_de - fps_st) > 0.1:
                errors.append(f"Index {i}: FPS 不匹配! {fps_de} vs {fps_st}")

            # 3. 标注校验
            if not caption:
                errors.append(f"Index {i}: Caption 为空")

        except Exception as e:
            errors.append(f"Index {i}: 视频损坏或无法解码 - {e}")

    # 汇总报告
    logger.info("\n" + "="*50)    
    logger.info(f"📊 审计报告汇总:")
    if not errors:
        logger.info("🎉 完美！数据完全符合模型训练要求。")
    else:
        logger.error(f"❌ 发现 {len(errors)} 处配置或数据错误:")
        for err in errors: logger.error(f"  - {err}")
        # if len(errors) > 10: logger.error(f"  ... 还有 {len(errors)-10} 处错误未列出")
    logger.info("="*50)

if __name__ == "__main__":
    verify_dataset(
        root_dir="/opt/liblibai-models/user-workspace2/datasets/style1000",
        index_file="index_for_train.json",
        clip_len=16, # 与 yaml 配置一致
        stride=4     # 与 yaml 配置一致
    )