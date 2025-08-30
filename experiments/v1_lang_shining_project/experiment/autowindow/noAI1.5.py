import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm
import shutil

# ===================== 全局参数 =====================
input_folder = r"D:\Qiufeng\aiart\langshining"  # 原图所在文件夹
output_root = r"D:\Qiufeng\aiart\Auto_Cropping_langshining"  # 输出总目录

# 预设窗口尺寸（单位：像素）用于原图短边在 [1280,5120] 的情况
WINDOW_SIZES = {
    "large": 2560,
    "medium": 1280,
    "small": 640
}

# 输出切片统一调整为尺寸（可修改全局输出尺寸）
OUTPUT_SIZE = (640, 640)

# 多阈值参数（显著性阈值与对应的重叠率）
LOW_THRESHOLD = 0.25  # 低信息密度阈值
HIGH_THRESHOLD = 0.6  # 高信息密度阈值
OVERLAP_LOW = 0.1  # 低密度区域采用10%重叠 -> stride = 0.9 * window_size
OVERLAP_MEDIUM = 0.25  # 中密度区域采用25%重叠 -> stride = 0.75 * window_size
OVERLAP_HIGH = 0.5  # 高密度区域采用50%重叠 -> stride = 0.5 * window_size

# 绘制滑动轨迹时不同窗口类型的颜色 (BGR格式)
WINDOW_COLORS = {
    "large": (0, 0, 255),  # 红色
    "medium": (255, 0, 0),  # 蓝色
    "small": (0, 255, 0)  # 绿色
}


# =====================================================

def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 范围内：
      - 如果短边在 [1280,5120] 内，则不缩放；
      - 如果短边大于 max_short，则缩小到 max_short；
      - 如果短边小于 1280，则不缩放（后续动态设置窗口尺寸）。
    """
    h, w = image.shape[:2]
    short_side = min(h, w)
    if 1280 <= short_side <= 5120:
        print("Short side in [1280,5120], no resizing performed.")
        return image
    elif short_side > max_short:
        scale = max_short / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from ({w}x{h}) to ({new_w}x{new_h}) with scale factor {scale:.2f}")
        return image
    else:
        print("Short side less than 1280, no resizing performed (using dynamic window sizes).")
        return image


def compute_edge_density(image):
    """
    计算 Sobel 边缘密度，返回归一化后的边缘图（浮点数数组）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx ** 2 + sobely ** 2)
    return edge_map / (edge_map.max() + 1e-8)


def compute_texture_complexity(image, radius=1, n_points=8):
    """
    计算 LBP 纹理复杂度，返回归一化后的纹理图（浮点数数组）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    return lbp / (lbp.max() + 1e-8)


def compute_saliency_map(image):
    """
    生成传统显著性热图：结合边缘密度和纹理复杂度。
    """
    edge = compute_edge_density(image)
    texture = compute_texture_complexity(image)
    return (edge + texture) / 2.0


def adaptive_sliding_window(image, saliency_map, window_size):
    """
    自适应滑动窗口扫描，根据局部平均显著性动态调整步长，
    并确保图像边缘被完全覆盖。
    返回切片列表，每个切片为 (x, y, roi)。
    """
    h, w = image.shape[:2]
    slices = []

    # 计算步长函数
    def get_stride(avg_sal):
        if avg_sal < LOW_THRESHOLD:
            return int(window_size * 0.9)
        elif avg_sal > HIGH_THRESHOLD:
            return int(window_size * 0.5)
        else:
            return int(window_size * 0.75)

    y = 0
    while True:
        if y + window_size >= h:
            y = h - window_size
        x = 0
        while True:
            if x + window_size >= w:
                x = w - window_size
            roi = image[y:y + window_size, x:x + window_size]
            slices.append((x, y, roi))
            # 根据当前窗口区域计算步长
            roi_sal = saliency_map[y:y + window_size, x:x + window_size]
            avg_sal = np.mean(roi_sal)
            stride = get_stride(avg_sal)
            if x + window_size >= w:
                break
            x_next = x + stride
            if x_next + window_size > w:
                x = w - window_size
            else:
                x = x_next
        if y + window_size >= h:
            break
        y_next = y + stride
        if y_next + window_size > h:
            y = h - window_size
        else:
            y = y_next
    return slices


def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图，
    并按不同窗口尺寸进行自适应滑动窗口切片。
    切片保存时，文件名格式为： imageName_slice_{win_size}_{patch_x}_{patch_y}.png
    同时保存4个子文件夹： small, medium, large, all（all 包含所有切片及原图）。
    记录每个切片的位置信息到 index 文件，并绘制滑动轨迹图。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return
    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    short_side = min(h, w)
    # 根据短边尺寸确定窗口尺寸：若短边 < 1280，动态设置；否则使用预设
    if short_side < 1280:
        dynamic_sizes = {
            "large": short_side,
            "medium": int(short_side / 2),
            "small": int(short_side / 4)
        }
        print(f"Dynamic window sizes: {dynamic_sizes}")
    else:
        dynamic_sizes = WINDOW_SIZES
        print(f"Using preset window sizes: {dynamic_sizes}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_folder = os.path.join(output_root, base_name)
    os.makedirs(out_folder, exist_ok=True)
    # 创建子文件夹： small, medium, large, all
    for folder in list(dynamic_sizes.keys()) + ["all"]:
        os.makedirs(os.path.join(out_folder, folder), exist_ok=True)
    # 保存原图到 "all" 文件夹
    cv2.imwrite(os.path.join(out_folder, "all", f"{base_name}.jpg"), image)

    # 保存全局显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 初始化切片位置信息文件
    index_file_path = os.path.join(out_folder, "slice_index.txt")
    with open(index_file_path, "w", encoding="utf-8") as f:
        f.write("Format: window_type, window_size, x, y\n")

    # 用于绘制滑动轨迹（基于原图）
    trajectory_img = image.copy()

    # 针对每个窗口尺寸进行切片处理
    for key, win_size in dynamic_sizes.items():
        slices = adaptive_sliding_window(image, saliency, win_size)
        subfolder = os.path.join(out_folder, key)
        print(f"Processing {len(slices)} adaptive slices for {key} window size ({win_size}x{win_size}) ...")
        for (x, y, roi) in slices:
            # 记录切片信息
            with open(index_file_path, "a", encoding="utf-8") as f:
                f.write(f"{key},{win_size},{x},{y}\n")
            # 绘制滑动轨迹边框（使用预设颜色）
            color = WINDOW_COLORS.get(key, (0, 255, 0))
            cv2.rectangle(trajectory_img, (x, y), (x + win_size, y + win_size), color, 2)
            # 将切片统一 resize 到 OUTPUT_SIZE 后保存
            roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            # 文件名： imageName_slice_{win_size}_{x}_{y}.png
            filename = f"{base_name}_slice_{win_size}_{x}_{y}.png"
            # 保存到对应窗口文件夹
            save_path = os.path.join(subfolder, filename)
            cv2.imwrite(save_path, roi_resized)
            # 同时保存到 "all" 文件夹
            all_save_path = os.path.join(out_folder, "all", filename)
            cv2.imwrite(all_save_path, roi_resized)

    # 保存滑动轨迹图
    trajectory_path = os.path.join(out_folder, "sliding_trajectory.png")
    cv2.imwrite(trajectory_path, trajectory_img)
    print(f"Saved sliding trajectory: {trajectory_path}")


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
