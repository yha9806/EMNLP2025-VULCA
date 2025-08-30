import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# ===================== 全局参数 =====================
input_folder = r"D:\Qiufeng\aiart\img"
output_root = r"D:\Qiufeng\aiart\Auto_Cropping2"

# 预设窗口尺寸（单位：像素）
WINDOW_SIZES = {
    "large": 2560,
    "medium": 1280,
    "small": 640
}

# 输出切片统一调整为尺寸（可在此处修改全局输出尺寸）
OUTPUT_SIZE = (640, 640)


# =====================================================

def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 范围内：
      - 如果短边在 [1280,5120] 之间，不做处理；
      - 如果短边大于 max_short，则缩小至 max_short；
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
        print("Short side less than 1280, no resizing performed (dynamic window sizes will be used).")
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


def sliding_window(image, window_size, overlap_ratio=0.5):
    """
    对图像进行滑动窗口切片，窗口尺寸为 window_size x window_size，返回切片列表，
    每个切片为 (x, y, roi)。
    """
    h, w = image.shape[:2]
    stride = int(window_size * overlap_ratio)
    slices = []
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            roi = image[y:y + window_size, x:x + window_size]
            slices.append((x, y, roi))
    return slices


def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图、记录切片信息并进行滑动窗口切片，
    最后将每个切片 resize 到 OUTPUT_SIZE 保存，同时在原图上绘制滑动轨迹。
    输出目录结构：
      output_root/[图像名]/
          saliency_heatmap.png
          slice_index.txt   <- 记录每个切片的 (win_size, x, y)
          sliding_trajectory.png  <- 原图上绘制所有切片边框
          {large, medium, small}/slice_x_y.png  (如果使用预设尺寸)
    当原图短边 < 1280 时，动态设置窗口尺寸：大窗口 = L, 中窗口 = L/2, 小窗口 = L/4；
    当原图短边在 [1280,5120] 时，使用预设 WINDOW_SIZES。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return

    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    short_side = min(h, w)

    # 动态设置窗口尺寸
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

    # 保存显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 用于记录所有切片位置信息
    index_file_path = os.path.join(out_folder, "slice_index.txt")
    with open(index_file_path, "w", encoding="utf-8") as index_file:
        index_file.write("slice_index.txt: Format: window_size, x, y\n")

    # 复制原图用于绘制滑动轨迹
    trajectory_img = image.copy()

    # 针对每个窗口尺寸进行滑动窗口切片
    for key, win_size in dynamic_sizes.items():
        subfolder = os.path.join(out_folder, key)
        os.makedirs(subfolder, exist_ok=True)
        slices = sliding_window(image, win_size, overlap_ratio=0.5)
        print(f"Processing {len(slices)} slices for {key} window size ({win_size}x{win_size}) ...")
        for (x, y, roi) in slices:
            # 保存切片前，记录位置信息到 index_file
            with open(index_file_path, "a", encoding="utf-8") as index_file:
                index_file.write(f"{win_size},{x},{y}\n")
            # 绘制窗口边框到滑动轨迹图（用绿色矩形）
            cv2.rectangle(trajectory_img, (x, y), (x + win_size, y + win_size), (0, 255, 0), 2)
            # 将切片统一 resize 到 OUTPUT_SIZE 后保存
            roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            filename = f"slice_{win_size}_{x}_{y}.png"
            save_path = os.path.join(subfolder, filename)
            cv2.imwrite(save_path, roi_resized)

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
