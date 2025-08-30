import os
import glob
import numpy as np
from tqdm import tqdm

def read_boxes(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls, x1, y1, x2, y2, conf = parts
            boxes.append((int(cls), [int(x1), int(y1), int(x2), int(y2)], float(conf)))
    return boxes

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0

    box1_area = (box1[2]-box1[0])*(box1[3]-box1[1])
    box2_area = (box2[2]-box2[0])*(box2[3]-box2[1])
    return inter_area / float(box1_area + box2_area - inter_area)

def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    matched_gt = set()
    matched_pred = set()

    for i, (cls_pred, pred_box, _) in enumerate(pred_boxes):
        for j, (cls_gt, gt_box, _) in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            if cls_pred == cls_gt and compute_iou(pred_box, gt_box) >= iou_threshold:
                matched_gt.add(j)
                matched_pred.add(i)
                break

    TP = len(matched_pred)
    FP = len(pred_boxes) - TP
    FN = len(gt_boxes) - TP

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    box_acc = TP / len(gt_boxes) if gt_boxes else 0

    return precision, recall, f1_score, box_acc

# ================ 修改为你的实际路径 ===================
GROUND_TRUTH_PATH = r"path_to_your_ground_truth_label.txt"
PREDICTION_FOLDER = r"path_to_your_prediction_labels"

# ======================================================

def evaluate_all():
    gt_boxes = read_boxes(GROUND_TRUTH_PATH)

    pred_files = glob.glob(os.path.join(PREDICTION_FOLDER, '*.txt'))
    results = {}

    for pred_file in tqdm(pred_files, desc="Evaluating results"):
        dimension = os.path.splitext(os.path.basename(pred_file))[0]
        pred_boxes = read_boxes(pred_file)

        precision, recall, f1_score, box_acc = match_boxes(gt_boxes, pred_boxes)

        results[dimension] = {
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1_score,
            "Box Accuracy": box_acc
        }

    print("\nEvaluation Results:")
    for dimension, metrics in results.items():
        print(f"\nDimension: {dimension}")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    evaluate_all()
