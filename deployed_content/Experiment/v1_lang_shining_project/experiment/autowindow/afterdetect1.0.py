import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern

# ===================== 全局参数 =====================
#LABEL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\labels"
LABEL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\labels"
##ALL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\all"
ALL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\all"
#RESULT_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\afterdetect"
RESULT_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\afterdetect"
OUTPUT_SIZE = (640, 640)
IOU_THRESHOLD = 0.5
CONF_THRESHOLD = 0.8
# 预设窗口尺寸
WINDOW_SIZES = {
    "large": 2560,
    "medium": 1280,
    "small": 640
}

# 多阈值参数
LOW_THRESHOLD = 0.25
HIGH_THRESHOLD = 0.6
OVERLAP_LOW = 0.1      # 步长 = 0.9 * window_size
OVERLAP_MEDIUM = 0.25  # 步长 = 0.75 * window_size
OVERLAP_HIGH = 0.5     # 步长 = 0.5 * window_size

# 绘制滑动轨迹时不同来源的颜色 (BGR)
COLOR_MAPPING = {
    "original": (0, 255, 255),  # 黄色
    "large":    (0, 255, 0),    # 绿色
    "medium":   (255, 0, 0),    # 蓝色
    "small":    (0, 0, 255),    # 红色
    "default":  (0, 255, 255)   # 默认黄色
}

# 类别映射字典
CLASS_MAPPING = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}

# IoU 阈值
IOU_THRESHOLD = 0.5
# 置信度过滤阈值
CONF_THRESHOLD = 0.7
# =====================================================

def compute_iou(box1, box2):
    """计算 IoU, 框格式：[x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area1 = (box1[2]-box1[0]+1) * (box1[3]-box1[1]+1)
    area2 = (box2[2]-box2[0]+1) * (box2[3]-box2[1]+1)
    return inter_area / float(area1+area2-inter_area+1e-8)



def nms_priority(detections, iou_thresh=IOU_THRESHOLD, conf_thresh=CONF_THRESHOLD):
    """
    对同一类别的检测结果进行非极大值抑制（NMS）。
    若检测框同属同一物体（IoU > iou_thresh），则保留来源优先级更高的，优先级顺序：
      original > large > medium > small > default
    并且过滤掉 score < conf_thresh 的检测框。
    detections: 列表，每个元素为 (box, label, score, source)
                box=[x1,y1,x2,y2], label=str, score=float, source=str
    返回过滤后的检测列表。
    """
    # 1. 过滤低置信度
    detections = [d for d in detections if d[2] >= conf_thresh]

    # 2. 定义来源优先级
    priority_map = {
        "original": 4,
        "large": 3,
        "medium": 2,
        "small": 1,
        "default": 0
    }

    filtered = []
    # 按类别分组处理
    labels = list(set([d[1] for d in detections]))
    for label in labels:
        cls_dets = [d for d in detections if d[1] == label]
        # 按优先级降序排序
        cls_dets.sort(key=lambda d: priority_map.get(d[3], 0), reverse=True)

        kept = []
        while cls_dets:
            best = cls_dets.pop(0)  # 优先级最高
            kept.append(best)
            # 与 best 重叠超过 iou_thresh 的都丢弃
            cls_dets = [d for d in cls_dets if compute_iou(best[0], d[0]) < iou_thresh]

        filtered.extend(kept)

    return filtered

def parse_label_filename(filepath):
    """
    解析标签文件名。
    若文件名中包含 "_slice_"，格式为：
      imageName_slice_{win_size}_{patch_x}_{patch_y}.txt
    否则认为是原图检测结果，格式为 imageName.txt
    返回 (image_name, win_size, patch_x, patch_y)
    对于原图检测，win_size="original", patch_x=patch_y=0。
    """
    base = os.path.splitext(os.path.basename(filepath))[0]
    if "_slice_" not in base:
        return base, "original", 0, 0
    parts = base.split("_slice_")
    if len(parts) != 2:
        raise ValueError(f"文件名格式不正确: {filepath}")
    image_name = parts[0]
    subparts = parts[1].split("_")
    if len(subparts) < 3:
        raise ValueError(f"文件名格式不正确: {filepath}")
    try:
        win_size = int(subparts[0])
        patch_x = int(subparts[1])
        patch_y = int(subparts[2])
    except Exception as e:
        raise ValueError(f"解析文件名出错: {filepath}") from e
    return image_name, win_size, patch_x, patch_y

def read_detection_file(filepath):
    """
    读取检测结果文件（YOLO格式），每行格式为：
      class x_center y_center width height
    坐标均为归一化值（相对于 OUTPUT_SIZE）。
    返回列表，每个元素为 (label, score, [x1, y1, x2, y2])，score 固定为1.0。
    """
    detections = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"警告: 文件 {filepath} 行格式错误: {line}")
                continue
            try:
                class_id = int(parts[0])
                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
            except Exception as e:
                print(f"警告: 文件 {filepath} 转换错误: {line}")
                continue
            w, h = OUTPUT_SIZE
            x_center = x_center_norm * w
            y_center = y_center_norm * h
            box_w = width_norm * w
            box_h = height_norm * h
            x1 = x_center - box_w / 2
            y1 = y_center - box_h / 2
            x2 = x_center + box_w / 2
            y2 = y_center + box_h / 2
            label = CLASS_MAPPING.get(class_id, str(class_id))
            detections.append((label, 1.0, [x1, y1, x2, y2]))
    return detections

def map_box_to_original(box, patch_top_left, win_size, resized_size=OUTPUT_SIZE, original_size=None):
    """
    将检测框映射回原图坐标。
    如果 win_size=="original"，则使用 original_size（格式：(width, height)）进行比例映射；
    否则 factor = win_size / OUTPUT_SIZE[0]。
    返回映射后的检测框 [X1, Y1, X2, Y2].
    """
    if win_size == "original":
        if original_size is None:
            raise ValueError("对于原图检测，必须提供 original_size")
        factor_x = original_size[0] / float(resized_size[0])
        factor_y = original_size[1] / float(resized_size[1])
        x1, y1, x2, y2 = box
        X1 = int(x1 * factor_x) + patch_top_left[0]
        Y1 = int(y1 * factor_y) + patch_top_left[1]
        X2 = int(x2 * factor_x) + patch_top_left[0]
        Y2 = int(y2 * factor_y) + patch_top_left[1]
        return [X1, Y1, X2, Y2]
    else:
        factor = win_size / float(resized_size[0])
        x1, y1, x2, y2 = box
        X1 = int(x1 * factor) + patch_top_left[0]
        Y1 = int(y1 * factor) + patch_top_left[1]
        X2 = int(x2 * factor) + patch_top_left[0]
        Y2 = int(y2 * factor) + patch_top_left[1]
        return [X1, Y1, X2, Y2]

def draw_final_detections(image, detections):
    """
    在原图上绘制检测框，每个框格式为 (box, label, score, source)。
    使用不同颜色（来源决定颜色），文字字号 0.8，框线宽 3。
    """
    drawn = image.copy()
    for (box, label, score, source) in detections:
        color = COLOR_MAPPING.get(source, COLOR_MAPPING["default"])
        X1, Y1, X2, Y2 = box
        cv2.rectangle(drawn, (X1, Y1), (X2, Y2), color, 3)
        cv2.putText(drawn, f"{label}:{score:.2f}", (X1, max(Y1-8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return drawn

def draw_sliding_trajectory(image, slice_infos):
    """
    在原图上绘制滑动轨迹，每个切片记录 (source, win_size, x, y)。
    如果 source=="original"，则绘制整幅图边框；
    否则使用对应的颜色，框线宽 3。
    """
    traj = image.copy()
    for (source, win_size, x, y) in slice_infos:
        if win_size == "original":
            h, w = traj.shape[:2]
            cv2.rectangle(traj, (0,0), (w-1, h-1), COLOR_MAPPING["original"], 3)
        else:
            cv2.rectangle(traj, (x, y), (x+win_size, y+win_size), COLOR_MAPPING.get(source, COLOR_MAPPING["default"]), 3)
    return traj

def process_all_labels():
    """
    遍历 LABEL_FOLDER 下所有TXT文件，解析后按 image_name 分组，
    返回字典： { image_name: [ (win_size, patch_x, patch_y, label_file_path) ] }
    """
    files = glob.glob(os.path.join(LABEL_FOLDER, "*.txt"))
    groups = {}
    for f in files:
        try:
            image_name, win_size, patch_x, patch_y = parse_label_filename(f)
        except Exception as e:
            print(e)
            continue
        if image_name not in groups:
            groups[image_name] = []
        groups[image_name].append((win_size, patch_x, patch_y, f))
    return groups

def get_source_from_win_size(win_size):
    """
    根据 win_size 返回检测结果来源标识
    """
    if win_size == "original":
        return "original"
    elif win_size == 2560:
        return "large"
    elif win_size == 1280:
        return "medium"
    elif win_size == 640:
        return "small"
    else:
        return "default"

def nms_priority(detections, iou_thresh=IOU_THRESHOLD, conf_thresh=CONF_THRESHOLD):
    """
    对同一类别的检测结果进行非极大值抑制（NMS）。
    若检测框同属同一物体（IoU > iou_thresh），则保留来源优先级更高的，优先级顺序：
      original > large > medium > small > default
    另外只保留 score >= conf_thresh 的检测框。
    detections: 列表，每个元素为 (box, label, score, source)
    返回过滤后的检测列表。
    """
    # 过滤低于置信度阈值的检测
    detections = [d for d in detections if d[2] >= conf_thresh]
    priority_map = {
        "original": 4,
        "large": 3,
        "medium": 2,
        "small": 1,
        "default": 0
    }
    filtered = []
    for label in set([d[1] for d in detections]):
        cls_dets = [d for d in detections if d[1] == label]
        cls_dets.sort(key=lambda d: priority_map.get(d[3], 0), reverse=True)
        kept = []
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            cls_dets = [d for d in cls_dets if compute_iou(best[0], d[0]) < iou_thresh]
        filtered.extend(kept)
    return filtered

def process_image_detections():
    """
    根据 LABEL_FOLDER 中所有标签文件，将每个检测结果映射回原图，
    并在原图上绘制检测框和滑动轨迹，最终保存结果到 RESULT_FOLDER。
    """
    groups = process_all_labels()
    if not groups:
        print("没有读取到任何标签文件。")
        return

    for image_name, records in tqdm(groups.items(), desc="Processing detection results"):
        # 在 ALL_FOLDER 中查找原图（支持 jpg/png/jpeg）
        exts = ["*.jpg", "*.png", "*.jpeg"]
        original_path = None
        for ext in exts:
            pattern = os.path.join(ALL_FOLDER, image_name + ext[1:])
            candidates = glob.glob(pattern)
            if candidates:
                original_path = candidates[0]
                break
        if original_path is None:
            print(f"未找到原图: {image_name} 在 {ALL_FOLDER}")
            continue

        original_img = cv2.imread(original_path)
        if original_img is None:
            print(f"读取原图失败: {original_path}")
            continue
        original_size = (original_img.shape[1], original_img.shape[0])  # (width, height)

        all_detections = []  # (box, label, score, source)
        slice_info_list = [] # (source, win_size, patch_x, patch_y)
        for (win_size, px, py, label_file) in records:
            dets = read_detection_file(label_file)
            source = get_source_from_win_size(win_size)
            for (label, score, box) in dets:
                mapped_box = map_box_to_original(box, (px, py), win_size, resized_size=OUTPUT_SIZE, original_size=original_size)
                all_detections.append((mapped_box, label, score, source))
            slice_info_list.append((source, win_size, px, py))

        # 进行 NMS
        final_dets = nms_priority(all_detections, iou_thresh=IOU_THRESHOLD, conf_thresh=CONF_THRESHOLD)

        # 绘制检测框（增加线宽和文字字号）
        result_img = draw_final_detections(original_img, final_dets) if final_dets else original_img.copy()
        # 绘制滑动轨迹
        trajectory_img = draw_sliding_trajectory(original_img, slice_info_list)

        os.makedirs(RESULT_FOLDER, exist_ok=True)
        result_path = os.path.join(RESULT_FOLDER, f"{image_name}_detected.jpg")
        traj_path = os.path.join(RESULT_FOLDER, f"{image_name}_trajectory.jpg")
        cv2.imwrite(result_path, result_img)
        cv2.imwrite(traj_path, trajectory_img)
        print(f"Saved detection result: {result_path}")
        print(f"Saved sliding trajectory: {traj_path}")

def main():
    process_image_detections()

if __name__ == "__main__":
    main()
