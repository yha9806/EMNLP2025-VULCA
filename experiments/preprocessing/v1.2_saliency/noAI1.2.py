import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# 设置输入和输出目录（请根据实际情况调整路径）
input_folder = r"D:\Qiufeng\aiart\img"
output_root = r"D:\Qiufeng\aiart\Auto_Cropping"

# 预设窗口尺寸（单位：像素）用于图像短边在 [1280, 5120] 时
WINDOW_SIZES = {
    "large": 2560,
    "medium": 1280,
    "small": 640
}

# 输出切片统一调整为 640x640（可在此处修改全局输出尺寸）
OUTPUT_SIZE = (640, 640)

# 自适应滑动窗口的多阈值参数（用于后续版本，可保留）
LOW_THRESHOLD = 0.2  # 平均显著性低于此值，认为区域信息较少
HIGH_THRESHOLD = 0.6  # 平均显著性高于此值，认为区域信息丰富
OVERLAP_LOW = 0.2  # 低显著区域采用20%重叠 => 步长 = window_size * 0.8
OVERLAP_MEDIUM = 0.5  # 中等显著区域采用50%重叠 => 步长 = window_size * 0.5
OVERLAP_HIGH = 0.7  # 高显著区域采用70%重叠 => 步长 = window_size * 0.3


def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 范围内。
    如果短边在 [1280,5120] 之间，则不做缩放；
    如果短边大于 max_short，则缩小到 max_short；
    如果短边小于1280，则不缩放（后续使用动态窗口尺寸）。
    """
    h, w = image.shape[:2]
    short_side = min(h, w)
    # 如果短边在 [1280,5120] 内，不做处理
    if 1280 <= short_side <= 5120:
        print("Short side is within [1280,5120], no resizing performed.")
        return image
    elif short_side > max_short:
        scale = max_short / short_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Resized image from ({w}x{h}) to ({new_w}x{new_h}) with scale factor {scale:.2f}")
        return image
    else:
        # 如果短边小于1280，则不缩放，但后续会动态设置窗口尺寸
        print("Short side is less than 1280, no resizing performed (using dynamic window sizes).")
        return image


def compute_edge_density(image):
    """
    计算 Sobel 边缘密度，返回归一化后的边缘图（浮点数数组）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx ** 2 + sobely ** 2)
    edge_map = edge_map / (edge_map.max() + 1e-8)
    return edge_map


def compute_texture_complexity(image, radius=1, n_points=8):
    """
    计算 LBP 纹理复杂度，返回归一化后的纹理图（浮点数数组）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp = lbp / (lbp.max() + 1e-8)
    return lbp


def compute_saliency_map(image):
    """
    生成传统显著性热图：结合边缘密度和纹理复杂度。
    """
    edge_density = compute_edge_density(image)
    texture_complexity = compute_texture_complexity(image)
    saliency = (edge_density + texture_complexity) / 2.0
    return saliency


def sliding_window(image, window_size, overlap_ratio=0.5):
    """
    对图像进行滑动窗口切片，窗口尺寸为 window_size x window_size，无重叠不处理。
    返回切片列表，每个切片信息为 (x, y, roi)。
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
    对单张图像进行自适应缩放、生成全局显著性热图以及滑动窗口切片，
    并将结果保存到对应文件夹中。
    输出目录结构：
      output_root/【图像名（无扩展名）】/
          saliency_heatmap.png
          {large, medium, small}/slice_x_y.png
    所有保存的切片统一调整为 OUTPUT_SIZE。

    当图像短边 < 1280 时，动态设置窗口尺寸：
      大尺寸 = L, 中尺寸 = L/2, 小尺寸 = L/4
    当图像短边在 [1280,5120] 时，采用预设的 WINDOW_SIZES。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return

    # 先进行自适应缩放（如果短边在[1280,5120]则不缩放；如果大于5120则缩放到5120；如果小于1280则不缩放）
    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    short_side = min(h, w)

    # 根据短边大小选择窗口尺寸
    if short_side < 1280:
        dynamic_sizes = {
            "large": short_side,
            "medium": int(short_side / 2),
            "small": int(short_side / 4)
        }
        print(f"Dynamic window sizes set: {dynamic_sizes}")
    else:
        dynamic_sizes = WINDOW_SIZES
        print(f"Using preset window sizes: {dynamic_sizes}")

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_folder = os.path.join(output_root, base_name)
    os.makedirs(out_folder, exist_ok=True)
    # 为每个窗口尺寸创建文件夹
    for key in dynamic_sizes.keys():
        os.makedirs(os.path.join(out_folder, key), exist_ok=True)

    # 生成全局传统显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 对于每个窗口尺寸，进行滑动窗口切片并保存（统一调整输出为 OUTPUT_SIZE）
    for key, win_size in dynamic_sizes.items():
        slices = sliding_window(image, win_size, overlap_ratio=0.5)
        subfolder = os.path.join(out_folder, key)
        print(f"Processing {len(slices)} slices for {key} window size ({win_size}x{win_size}) ...")
        for (x, y, roi) in slices:
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
