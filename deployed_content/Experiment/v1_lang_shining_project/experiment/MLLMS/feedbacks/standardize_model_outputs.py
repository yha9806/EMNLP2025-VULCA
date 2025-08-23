import os
import shutil
import argparse
import json
import re
from datetime import datetime

# ========== 默认映射表 ==========
DEFAULT_MONTH_MAP = {
    "一月": "January", "正月": "January", "一月_正月": "January",
    "二月": "February",
    "三月": "March",
    "四月": "April",
    "五月": "May", "五月_龙舟竞渡": "May",
    "六月": "June",
    "七月": "July",
    "八月": "August", "八月_仲秋": "August",
    "九月": "September",
    "十月": "october", "十月_开冬农闲": "october",
    "十一月": "November",
    "十二月": "December"
}
DEFAULT_PERSONA_MAP = {
    "阿里斯_索恩博士_Dr_Aris_Thorne": "with_Dr_Aris_Thorne",
    "郭熙_Guo_Xi": "with_Guo_Xi",
    "苏轼_Su_Shi": "with_Su_Shi",
    "约翰_罗斯金_John_Ruskin": "with_John_Ruskin",
    "托马斯修士_Brother_Thomas": "with_Brother_Thomas",
    "埃琳娜_佩特洛娃教授_Professor_Elena_Petrova": "with_Professor_Elena_Petrova",
    "冈仓天心_Okakura_Kakuzo": "with_Okakura_Kakuzō",
    "佐拉妈妈_Mama_Zola": "with_Mama_Zola",
    "(Basic)": "basic",
    "basic": "basic"
}

# ========== 工具函数 ==========
def load_mapping(path, default_map):
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return default_map

def parse_filename(filename_stem, month_map, persona_map):
    # 优先识别标准化命名格式：如 四月(with_Brother_Thomas) 或 四月(basic)
    std_match = re.match(r'^([\u4e00-\u9fa5]+)[_\u4e00-\u9fa5]*\((with_[^\)]+|basic)\)$', filename_stem)
    if std_match:
        month = std_match.group(1)
        persona = std_match.group(2)
        return month, persona
    # 兼容如 八月_仲秋(with_XXX) 这种
    std_match2 = re.match(r'^([\u4e00-\u9fa5]+_[^\(]+)\((with_[^\)]+|basic)\)$', filename_stem)
    if std_match2:
        month = std_match2.group(1)
        persona = std_match2.group(2)
        return month, persona
    # 兼容原始命名
    stem = re.sub(r'^清_清院_十二月令图_', '', filename_stem)
    month = None
    for k in sorted(month_map, key=lambda x: -len(x)):
        if stem.startswith(k):
            month = k
            stem = stem[len(k):]
            break
    persona = None
    for k in persona_map:
        if k in stem:
            persona = k
            break
    if not persona and ("(Basic)" in stem or "basic" in stem):
        persona = "basic"
    return month, persona

def standardize_name(month, persona, month_map, persona_map):
    zh_month = month if month else "未知"
    en_month = month_map.get(month, "Unknown")
    # persona 已是标准化格式时直接用
    if persona and (persona.startswith("with_") or persona == "basic"):
        persona_tag = persona
    else:
        persona_tag = persona_map.get(persona, "unknown_persona")
    if persona_tag == "basic":
        return zh_month + "(basic).txt", en_month
    elif persona_tag.startswith("with_"):
        return zh_month + f"({persona_tag}).txt", en_month
    else:
        return zh_month + f"(unknown).txt", en_month

def main():
    parser = argparse.ArgumentParser(description="标准化多模型输出文件名和目录结构")
    parser.add_argument('--source_dir', required=True, help='原始模型输出根目录')
    parser.add_argument('--target_dir', required=True, help='标准化输出根目录')
    parser.add_argument('--move', action='store_true', help='移动文件（默认复制）')
    parser.add_argument('--month_map', default=None, help='自定义月份映射表json')
    parser.add_argument('--persona_map', default=None, help='自定义persona映射表json')
    args = parser.parse_args()

    month_map = load_mapping(args.month_map, DEFAULT_MONTH_MAP)
    persona_map = load_mapping(args.persona_map, DEFAULT_PERSONA_MAP)

    log_lines = []
    processed, skipped, failed = 0, 0, 0
    for root, _, files in os.walk(args.source_dir):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            src_path = os.path.join(root, fname)
            stem = os.path.splitext(fname)[0]
            month, persona = parse_filename(stem, month_map, persona_map)
            if not month or not persona:
                log_lines.append(f"[WARN] 跳过无法解析: {src_path}")
                skipped += 1
                continue
            std_name, en_month = standardize_name(month, persona, month_map, persona_map)
            target_month_dir = os.path.join(args.target_dir, en_month)
            os.makedirs(target_month_dir, exist_ok=True)
            dst_path = os.path.join(target_month_dir, std_name)
            if os.path.exists(dst_path):
                log_lines.append(f"[SKIP] 目标已存在: {dst_path}")
                skipped += 1
                continue
            try:
                if args.move:
                    shutil.move(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                log_lines.append(f"[OK] {src_path} -> {dst_path}")
                processed += 1
            except Exception as e:
                log_lines.append(f"[FAIL] {src_path} -> {dst_path}: {e}")
                failed += 1
    # 输出日志
    log_path = os.path.join(args.target_dir, f'standardize_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        for line in log_lines:
            f.write(line + '\n')
        f.write(f"\n成功: {processed} 跳过: {skipped} 失败: {failed}\n")
    print(f"标准化完成！成功: {processed} 跳过: {skipped} 失败: {failed}\n日志见: {log_path}")

if __name__ == "__main__":
    main() 