import os
import json
import logging
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IntegrityCheck")

def check_style1000_integrity(root_dir, index_file):
    """
    检查 style1000 数据集的完整性。
    注意：该数据集使用 'destyle' (x0) 和 'style' (x1) 字段。
    """
    index_path = os.path.join(root_dir, index_file)
    
    if not os.path.exists(index_path):
        logger.error(f"❌ 索引文件不存在: {index_path}")
        return

    # 1. 加载索引
    try:
        with open(index_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
    except Exception as e:
        logger.error(f"❌ 无法读取 JSON 文件: {e}")
        return

    if not isinstance(dataset, list):
        logger.error("❌ 索引格式错误：应为包含字典的列表 (List[Dict])。")
        return

    total_samples = len(dataset)
    missing_destyle = []
    missing_style = []
    empty_captions = 0
    
    logger.info(f"🚀 开始校验 {total_samples} 条样本...")

    # 2. 遍历并检查物理文件
    for i, item in enumerate(tqdm(dataset)):
        # 获取路径
        destyle_rel = item.get('destyle')
        style_rel = item.get('style')
        caption = item.get('caption', "").strip()

        # 检查字段完整性
        if not destyle_rel or not style_rel:
            logger.warning(f"⚠️ 样本 {i} 缺失路径字段。")
            continue

        # 构造绝对路径
        destyle_abs = os.path.join(root_dir, destyle_rel)
        style_abs = os.path.join(root_dir, style_rel)

        # 检查文件存在性
        if not os.path.exists(destyle_abs):
            missing_destyle.append(destyle_rel)
        
        if not os.path.exists(style_abs):
            missing_style.append(style_rel)

        # 检查 Caption
        if not caption:
            empty_captions += 1

    # 3. 输出汇总报告
    logger.info("="*50)
    logger.info(f"📊 校验完成汇总报告：")
    logger.info(f"✅ 总计样本数: {total_samples}")
    
    if not missing_destyle and not missing_style:
        logger.info("🎉 所有视频文件均在物理磁盘上对齐存在！")
    else:
        if missing_destyle:
            logger.error(f"❌ 缺失 'destyle' (源视频) 数量: {len(missing_destyle)}")
            # 可选：打印前5个缺失路径
            for p in missing_destyle[:5]: logger.error(f"   - 缺失: {p}")
        
        if missing_style:
            logger.error(f"❌ 缺失 'style' (目标视频) 数量: {len(missing_style)}")
            for p in missing_style[:5]: logger.error(f"   - 缺失: {p}")

    if empty_captions > 0:
        logger.warning(f"⚠️ 缺失 'caption' 的样本数: {empty_captions}")
    else:
        logger.info("✅ 所有样本均包含标注文本。")
    logger.info("="*50)

if __name__ == "__main__":
    # 请根据您的实际环境修改 root_dir
    DATA_ROOT = "/opt/liblibai-models/user-workspace2/datasets/style1000"
    INDEX_FILE = "index_for_train.json"
    
    check_style1000_integrity(DATA_ROOT, INDEX_FILE)