import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import local_binary_pattern

# ===================== 全局参数 =====================
# 标签文件夹：所有检测结果TXT文件均存放在该文件夹，
# 对于原图直接检测： imageName.txt
# 对于切片检测： imageName_slice_{win_size}_{patch_x}_{patch_y}.txt
#LABEL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\labels"
LABEL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\labels"
# 原图文件夹：存放所有原图，文件名（不含扩展名）与标签文件中解析出的 imageName 一致
#ALL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\all"
ALL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\all"
# 结果输出文件夹：绘制好检测框后的原图保存到此处
#RESULT_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_langshining\hongliguanqutu\afterdetect"
RESULT_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\afterdetect"

# 输出切片统一resize的尺寸（检测时切片统一resize到该尺寸），单位像素
OUTPUT_SIZE = (640, 640)

# 检测框颜色映射：来源→颜色 (BGR)
# 分别为：原图（original）→白色, 大尺寸（2560）→绿色, 中尺寸（1280）→蓝色, 小尺寸（640）→红色, 其他→默认灰色
COLOR_MAPPING = {
    "original": (255, 255, 255),  # 白
    "large":    (0, 255, 0),    # 绿色
    "medium":   (255, 0, 0),    # 蓝色
    "small":    (0, 0, 255),    # 红色
    "default":  (128, 128, 128) # 默认灰色
}

# 类别映射字典，将类别编号转换为名称（这里仅给出部分，其他请补充或自行修改）
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

# NMS 相关阈值
IOU_THRESHOLD = 0.7       # IoU 阈值
CONF_THRESHOLD = 0.7      # 置信度阈值
CONTAIN_THRESHOLD = 0.3   # 包含度阈值（小框被大框覆盖比例超过此值，认为是同一目标）
# =====================================================

def area_of_box(box):
    """返回框的面积，box=[x1,y1,x2,y2]"""
    w = max(0, box[2] - box[0] + 1)
    h = max(0, box[3] - box[1] + 1)
    return w * h

def intersection_area(box1, box2):
    """返回两个框的交集面积"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1 + 1)
    inter_h = max(0, y2 - y1 + 1)
    return inter_w * inter_h

def containment_ratio(small_box, large_box):
    """计算小框被大框覆盖的比例"""
    inter = intersection_area(small_box, large_box)
    area_s = area_of_box(small_box)
    if area_s <= 0:
        return 0
    return inter / float(area_s)

def compute_iou(box1, box2):
    """计算两个框的 IoU, 框格式：[x1,y1,x2,y2]"""
    inter = intersection_area(box1, box2)
    a1 = area_of_box(box1)
    a2 = area_of_box(box2)
    return inter / float(a1 + a2 - inter + 1e-8)

def nms_priority(detections, iou_thresh=IOU_THRESHOLD, conf_thresh=CONF_THRESHOLD, contain_thresh=CONTAIN_THRESHOLD):
    """
    对同一类别的检测结果进行非极大值抑制（NMS）+ 包含度判断。
    若同一类别检测框 IoU > iou_thresh 或小框被大框覆盖比例 > contain_thresh，
    视为同一目标，保留优先级更高的。
    优先级： original(4) > large(3) > medium(2) > small(1) > default(0)
    detections: 列表，每个元素=(box, label, conf, source)
    """
    detections = [d for d in detections if d[2] >= conf_thresh]
    priority_map = {
        "original": 4,
        "large": 3,
        "medium": 2,
        "small": 1,
        "default": 0
    }
    results = []
    labels = list(set(d[1] for d in detections))
    for lbl in labels:
        cls_dets = [d for d in detections if d[1] == lbl]
        cls_dets.sort(key=lambda d: priority_map.get(d[3], 0), reverse=True)
        kept = []
        while cls_dets:
            best = cls_dets.pop(0)
            kept.append(best)
            new_list = []
            for d in cls_dets:
                iou_val = compute_iou(best[0], d[0])
                ratio = 0
                area_best = area_of_box(best[0])
                area_d = area_of_box(d[0])
                if area_best <= area_d:
                    ratio = containment_ratio(best[0], d[0])
                else:
                    ratio = containment_ratio(d[0], best[0])
                if iou_val > iou_thresh or ratio > contain_thresh:
                    # 视为同一目标，保留优先级更高的best，丢弃d
                    continue
                else:
                    new_list.append(d)
            cls_dets = new_list
        results.extend(kept)
    return results

def parse_label_filename(filepath):
    """
    解析标签文件名。
    若文件名中包含 "_slice_"，格式：
      imageName_slice_{win_size}_{px}_{py}.txt
    否则认为是原图检测，格式： imageName.txt
    返回 (image_name, win_size, px, py)，对于原图检测，win_size="original"，px=py=0。
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
        px = int(subparts[1])
        py = int(subparts[2])
    except Exception as e:
        raise ValueError(f"解析文件名出错: {filepath}") from e
    return image_name, win_size, px, py

def read_detection_file(filepath):
    """
    读取检测结果文件（YOLO格式），每行格式为：
      class x_center y_center width height conf
    坐标均为归一化值（相对于 OUTPUT_SIZE）。
    返回列表，每个元素为 (label, conf, [x1, y1, x2, y2])。
    """
    detections = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                print(f"警告: {filepath} 行格式错误: {line}")
                continue
            try:
                class_id = int(parts[0])
                xc = float(parts[1])
                yc = float(parts[2])
                wn = float(parts[3])
                hn = float(parts[4])
                conf = float(parts[5])
            except Exception as e:
                print(f"警告: {filepath} 转换错误: {line}")
                continue
            w, h = OUTPUT_SIZE
            x1 = xc * w - (wn * w) / 2
            y1 = yc * h - (hn * h) / 2
            x2 = xc * w + (wn * w) / 2
            y2 = yc * h + (hn * h) / 2
            label = CLASS_MAPPING.get(class_id, str(class_id))
            detections.append((label, conf, [x1, y1, x2, y2]))
    return detections

def map_box_to_original(box, patch_top_left, win_size, resized_size=OUTPUT_SIZE, original_size=None):
    """
    将检测框映射回原图坐标。
    如果 win_size=="original"，则使用 original_size（格式：(width, height)）计算比例；
    否则 factor = win_size / OUTPUT_SIZE[0].
    返回映射后的检测框 [X1, Y1, X2, Y2].
    """
    if win_size == "original":
        if original_size is None:
            raise ValueError("对于原图检测，需要 original_size")
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
    在原图上绘制检测框。
    detections: 列表，每个元素为 (box, label, conf, source)。
    使用不同颜色（由 source 决定），线宽为5，文字字号为1.5。
    绘制文字时显示 "label:conf"。
    """
    drawn = image.copy()
    for (box, label, conf, source) in detections:
        color = COLOR_MAPPING.get(source, COLOR_MAPPING["default"])
        X1, Y1, X2, Y2 = box
        cv2.rectangle(drawn, (X1, Y1), (X2, Y2), color, 5)
        cv2.putText(drawn, f"{label}:{conf:.2f}", (X1, max(Y1-8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,1.5, color, 3)
    return drawn

def draw_sliding_trajectory(image, slice_infos):
    """
    绘制滑动轨迹, slice_infos: 列表，每个元素为 (source, win_size, px, py)。
    若 win_size=="original", 则在整幅图上绘制一个框；
    否则使用对应颜色绘制每个切片边框，线宽为3。
    """
    traj = image.copy()
    for (source, win_size, px, py) in slice_infos:
        color = COLOR_MAPPING.get(source, COLOR_MAPPING["default"])
        if win_size == "original":
            h, w = traj.shape[:2]
            cv2.rectangle(traj, (0,0), (w-1, h-1), color, 3)
        else:
            cv2.rectangle(traj, (px, py), (px+win_size, py+win_size), color, 3)
    return traj

def process_all_labels():
    """
    遍历 LABEL_FOLDER 下所有 TXT 文件，解析后按 image_name 分组，
    返回字典： { image_name: [ (win_size, px, py, file), ... ] }。
    """
    files = glob.glob(os.path.join(LABEL_FOLDER, "*.txt"))
    groups = {}
    for f in files:
        try:
            image_name, win_size, px, py = parse_label_filename(f)
        except Exception as e:
            print(e)
            continue
        if image_name not in groups:
            groups[image_name] = []
        groups[image_name].append((win_size, px, py, f))
    return groups

def get_source_from_win_size(win_size):
    """根据 win_size 返回检测结果来源"""
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

def process_image_detections():
    """
    主流程：对于每个 image_name，找到原图，从所有标签文件中读取检测结果，
    将检测框映射回原图，执行 NMS 后绘制检测框和滑动轨迹，最终保存结果到 RESULT_FOLDER。
    """
    groups = process_all_labels()
    if not groups:
        print("没有读取到任何标签文件。")
        return

    for image_name, recs in tqdm(groups.items(), desc="Processing detection results"):
        # 查找原图
        exts = ["*.jpg", "*.png", "*.jpeg"]
        original_path = None
        for e in exts:
            pattern = os.path.join(ALL_FOLDER, image_name + e[1:])
            cands = glob.glob(pattern)
            if cands:
                original_path = cands[0]
                break
        if original_path is None:
            print(f"未找到原图: {image_name} in {ALL_FOLDER}")
            continue

        original_img = cv2.imread(original_path)
        if original_img is None:
            print(f"读取原图失败: {original_path}")
            continue
        original_size = (original_img.shape[1], original_img.shape[0])  # (width, height)

        all_detections = []   # 每个元素为 (box, label, conf, source)
        slice_info_list = []  # 每个元素为 (source, win_size, px, py)
        for (win_size, px, py, label_file) in recs:
            dets = read_detection_file(label_file)
            source = get_source_from_win_size(win_size)
            for (label, conf, box) in dets:
                mapped_box = map_box_to_original(box, (px, py), win_size,
                                                   resized_size=OUTPUT_SIZE,
                                                   original_size=original_size)
                all_detections.append((mapped_box, label, conf, source))
            slice_info_list.append((source, win_size, px, py))

        final_dets = nms_priority(all_detections, iou_thresh=IOU_THRESHOLD,
                                  conf_thresh=CONF_THRESHOLD, contain_thresh=CONTAIN_THRESHOLD)

        result_img = draw_final_detections(original_img, final_dets) if final_dets else original_img.copy()
        traj_img = draw_sliding_trajectory(original_img, slice_info_list)

        os.makedirs(RESULT_FOLDER, exist_ok=True)
        det_path = os.path.join(RESULT_FOLDER, f"{image_name}_detected.jpg")
        traj_path = os.path.join(RESULT_FOLDER, f"{image_name}_trajectory.jpg")
        cv2.imwrite(det_path, result_img)
        cv2.imwrite(traj_path, traj_img)
        print(f"Saved detection result: {det_path}")
        print(f"Saved sliding trajectory: {traj_path}")

def main():
    process_image_detections()

if __name__ == "__main__":
    main()
