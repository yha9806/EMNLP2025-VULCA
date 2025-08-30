import os
import sys
import cv2
import glob
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# 将 U-2-Net 模型代码所在路径添加到 Python 路径中
sys.path.append(r"F:\coproject\aiart\atuowindow\U-2-Net-master\model")
from u2net import U2NETP  # 轻量版模型

# 设置输入和输出目录
input_folder = r"D:\Qiufeng\aiart\img"
output_root = r"D:\Qiufeng\aiart\Auto_U2net_Cropping"

# 预设窗口尺寸（单位：像素），这决定了切片原始尺寸
WINDOW_SIZES = {
    "large": 2560,
    "medium": 1280,
    "small": 640
}

# 输出切片统一调整为 640x640（可修改全局变量以适应模型输入）
OUTPUT_SIZE = (640, 640)

# 自适应滑动窗口的多阈值参数
LOW_THRESHOLD = 0.2  # 平均显著性低于此值，认为区域信息较少
HIGH_THRESHOLD = 0.6  # 平均显著性高于此值，认为区域信息丰富
OVERLAP_LOW = 0.2  # 低显著区域采用20%重叠 => 步长 = window_size * 0.8
OVERLAP_MEDIUM = 0.5  # 中等显著区域采用50%重叠 => 步长 = window_size * 0.5
OVERLAP_HIGH = 0.7  # 高显著区域采用70%重叠 => 步长 = window_size * 0.3

# 模型权重文件路径（请确保该文件存在且对应轻量版 U2-NetP）
MODEL_PATH = r"F:\coproject\aiart\atuowindow\u2netp.pth"


def load_u2net(model_path):
    print("Loading U2-NetP model ...")
    net = U2NETP(3, 1)
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()
    return net


# 全局加载 U2-NetP 模型
u2net = load_u2net(MODEL_PATH)


def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 范围内
    """
    h, w = image.shape[:2]
    short_side = min(h, w)
    scale = 1.0
    if short_side < min_short:
        scale = min_short / short_side
    elif short_side > max_short:
        scale = max_short / short_side

    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from ({w}x{h}) to ({new_w}x{new_h}) with scale factor {scale:.2f}")
    else:
        print("No resizing needed.")
    return image


def compute_saliency_map(image):
    """
    使用预训练的 U2-NetP 生成显著性热图。
    输入 image 为 BGR 格式的 numpy 数组，
    输出 saliency 为归一化后的显著性图（范围 0-1）。
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_image).unsqueeze(0)  # [1, 3, H, W]
    # 将输入放到与模型一致的设备
    device = next(u2net.parameters()).device
    input_tensor = input_tensor.to(device)

    try:
        with torch.no_grad():
            d1, d2, d3, d4, d5, d6, d7 = u2net(input_tensor)
            pred = d1[:, 0, :, :]
            pred = torch.sigmoid(pred)
            pred_np = pred.squeeze().cpu().data.numpy()
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "CUDNN_STATUS_MAPPING_ERROR" in str(e):
            print("CUDA 内存错误，切换到 CPU 模式进行显著性计算...")
            torch.cuda.empty_cache()
            u2net.cpu()
            device = torch.device("cpu")
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                d1, d2, d3, d4, d5, d6, d7 = u2net(input_tensor)
                pred = d1[:, 0, :, :]
                pred = torch.sigmoid(pred)
                pred_np = pred.squeeze().cpu().data.numpy()
        else:
            raise e

    pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min() + 1e-8)
    return pred_np


def adaptive_sliding_window(image, saliency_map, window_size):
    """
    自适应滑动窗口扫描，根据局部平均显著性动态调整步长。
    返回切片列表，每个切片为 (x, y, roi)。
    """
    h, w = image.shape[:2]
    windows = []
    y = 0
    while y + window_size <= h:
        x = 0
        while x + window_size <= w:
            roi = image[y:y + window_size, x:x + window_size]
            # 从 saliency_map 中取对应区域
            sal_roi = saliency_map[y:y + window_size, x:x + window_size]
            avg_sal = np.mean(sal_roi)
            # 根据平均显著性调整步长
            if avg_sal < LOW_THRESHOLD:
                stride = int(window_size * (1 - OVERLAP_LOW))  # 较大步长：低重叠率
            elif avg_sal > HIGH_THRESHOLD:
                stride = int(window_size * (1 - OVERLAP_HIGH))  # 较小步长：高重叠率
            else:
                stride = int(window_size * (1 - OVERLAP_MEDIUM))
            windows.append((x, y, roi))
            x += stride
        y += stride
    return windows


def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图和自适应滑动窗口切片，
    并将结果保存到对应文件夹中。所有切片统一调整为 OUTPUT_SIZE。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return

    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_folder = os.path.join(output_root, base_name)
    os.makedirs(out_folder, exist_ok=True)
    for key in WINDOW_SIZES.keys():
        os.makedirs(os.path.join(out_folder, key), exist_ok=True)

    # 生成全局显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 自适应滑动窗口扫描：利用 saliency_map 指导扫描步长
    for key, win_size in WINDOW_SIZES.items():
        slices = adaptive_sliding_window(image, saliency, win_size)
        subfolder = os.path.join(out_folder, key)
        print(f"Processing {len(slices)} adaptive slices for {key} window size ({win_size}x{win_size}) ...")
        for (x, y, roi) in slices:
            # 统一将切片调整为 OUTPUT_SIZE
            roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            filename = f"slice_{win_size}_{x}_{y}.png"
            save_path = os.path.join(subfolder, filename)
            cv2.imwrite(save_path, roi_resized)


def main():
    img_files = glob.glob(os.path.join(input_folder, "*.[jJ][pP][gG]")) + \
                glob.glob(os.path.join(input_folder, "*.[pP][nN][gG]")) + \
                glob.glob(os.path.join(input_folder, "*.[jJ][pP][eE][gG]"))
    if not img_files:
        print("未在输入文件夹中找到图像文件。")
        return

    for img_path in tqdm(img_files, desc="Processing images"):
        print(f"Processing image: {img_path}")
        process_image(img_path)
        print("Done.\n")


if __name__ == "__main__":
    main()
