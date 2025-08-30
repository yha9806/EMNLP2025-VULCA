import os
import cv2
import glob
from tqdm import tqdm

# ================ 修改以下路径为你的实际路径 ====================
LABEL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\labels"
ALL_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\all"
RESULT_FOLDER = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\afterdetect"
SLICE_INDEX_PATH = r"D:\Qiufeng\aiart\Auto_Cropping_12MP\shierlingyutu (2)\slice_index.txt"

OUTPUT_SIZE = (640, 640)
IOU_THRESHOLD = 0.3
CONF_THRESHOLD = 0.7
CONTAIN_THRESHOLD = 0.2

# 不同尺度对应的颜色 (BGR)
COLOR_MAPPING = {
    "large": (0, 0, 255),    # 红色
    "medium": (255, 0, 0),   # 蓝色
    "small": (0, 255, 0),    # 绿色
    "dircrop": (0, 128, 255) # 橙色
}

# ======================= 辅助函数 ==========================
def area_of_box(box):
    w = max(0, box[2] - box[0])
    h = max(0, box[3] - box[1])
    return w * h

def intersection_area(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    return inter_w * inter_h

def containment_ratio(small, large):
    inter = intersection_area(small, large)
    return inter / area_of_box(small)

def compute_iou(box1, box2):
    inter = intersection_area(box1, box2)
    union = area_of_box(box1) + area_of_box(box2) - inter
    return inter / union

def nms_priority(detections, iou_thresh, conf_thresh, contain_thresh):
    detections = [d for d in detections if d[2] >= conf_thresh]
    detections.sort(key=lambda x: x[2], reverse=True)
    kept = []
    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [d for d in detections if compute_iou(best[0], d[0]) <= iou_thresh and
                      containment_ratio(d[0], best[0]) <= contain_thresh]
    return kept

def read_detection_file(filepath):
    detections = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            class_id, xc, yc, wn, hn, conf = map(float, parts)
            x1 = (xc - wn / 2) * OUTPUT_SIZE[0]
            y1 = (yc - hn / 2) * OUTPUT_SIZE[1]
            x2 = (xc + wn / 2) * OUTPUT_SIZE[0]
            y2 = (yc + hn / 2) * OUTPUT_SIZE[1]
            detections.append((int(class_id), conf, [x1, y1, x2, y2]))
    return detections

def load_slice_index(filepath):
    slices = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            mode, win_size, x, y, filename = line.strip().split(',')
            key = filename.replace('.png', '')
            slices[key] = (int(win_size), int(x), int(y), mode)
    return slices

# ========================== 主函数 ============================
def process():
    slices = load_slice_index(SLICE_INDEX_PATH)
    label_files = glob.glob(os.path.join(LABEL_FOLDER, '*.txt'))
    original_img_path = glob.glob(os.path.join(ALL_FOLDER, '*_original.*'))[0]
    original_img = cv2.imread(original_img_path)

    detections_all = []
    for lbl_file in tqdm(label_files, desc="Mapping detections"):
        slice_key = os.path.splitext(os.path.basename(lbl_file))[0]
        if slice_key not in slices:
            continue
        win_size, px, py, mode = slices[slice_key]
        dets = read_detection_file(lbl_file)
        factor = win_size / OUTPUT_SIZE[0]
        for cls, conf, box in dets:
            x1, y1, x2, y2 = [int(b * factor) for b in box]
            mapped_box = [x1 + px, y1 + py, x2 + px, y2 + py]
            detections_all.append((mapped_box, cls, conf, mode))

    final_detections = nms_priority(detections_all, IOU_THRESHOLD, CONF_THRESHOLD, CONTAIN_THRESHOLD)

    for box, cls, conf, mode in final_detections:
        x1, y1, x2, y2 = map(int, box)
        color = COLOR_MAPPING.get(mode, (255, 255, 255))
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(original_img, f"{cls}:{conf:.2f}", (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX,
                    1, color, 2)

    os.makedirs(RESULT_FOLDER, exist_ok=True)
    result_path = os.path.join(RESULT_FOLDER, "final_detection_colored.jpg")
    cv2.imwrite(result_path, original_img)
    print(f"Saved final detection result with color: {result_path}")

if __name__ == '__main__':
    process()
