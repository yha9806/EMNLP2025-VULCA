import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# 设置输入和输出目录（请根据实际情况调整路径）
input_folder = r"D:\Qiufeng\aiart\img"
output_root = r"D:\Qiufeng\aiart\Direct_Cropping"

# 输出切片统一调整为 640x640（可在此处修改全局输出尺寸）
OUTPUT_SIZE = (640, 640)

def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使得图像短边在 [min_short, max_short] 范围内。
    如果短边小于 min_short，则放大；如果短边大于 max_short，则缩小；否则不缩放。
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

def compute_edge_density(image):
    """
    计算 Sobel 边缘密度，返回归一化后的边缘图（浮点数数组）。
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx**2 + sobely**2)
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

def direct_sliding_window(image, window_size):
    """
    直接切片：以给定窗口尺寸（正方形）对图像进行无重叠切片。
    如果最后边缘区域不足一个完整窗口，则丢弃。
    返回切片列表，每个切片为 (x, y, roi)。
    """
    h, w = image.shape[:2]
    slices = []
    for y in range(0, h - window_size + 1, window_size):
        for x in range(0, w - window_size + 1, window_size):
            roi = image[y:y+window_size, x:x+window_size]
            slices.append((x, y, roi))
    return slices

def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图以及直接无重叠切片，
    并将结果保存到对应文件夹中。
    输出目录结构：
      output_root/【图像名（无扩展名）】/
          saliency_heatmap.png
          slices/slice_x_y.png
    所有保存的切片统一调整为 OUTPUT_SIZE。
    对于图像短边较小（例如仅1000+），窗口大小将采用图像短边。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return

    # 自适应缩放：保证短边在2560到5120之间
    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    out_folder = os.path.join(output_root, base_name)
    os.makedirs(out_folder, exist_ok=True)
    # 创建切片存放文件夹（统一放在一个文件夹里）
    slices_folder = os.path.join(out_folder, "slices")
    os.makedirs(slices_folder, exist_ok=True)

    # 生成全局传统显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 根据图像短边动态确定窗口尺寸（直接切片，无重叠）
    window_size = min(h, w)
    print(f"Using window size = {window_size} (short side of image)")
    slices = direct_sliding_window(image, window_size)
    print(f"Processing {len(slices)} slices (direct tiling) ...")
    for (x, y, roi) in slices:
        # 将切片统一调整为 OUTPUT_SIZE
        roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        filename = f"slice_{window_size}_{x}_{y}.png"
        save_path = os.path.join(slices_folder, filename)
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
