import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# ===================== 全局参数 =====================
input_folder = r"D:\Qiufeng\aiart\langshining"  # 原图所在文件夹
# 输入原图所在文件夹

output_root = r"D:\Qiufeng\aiart\dircrop"
# 输出总目录

# 输出文件夹固定为 "all"，包含两部分：
# ① 原图（未resize）
# ② 切片后的图像（resize到 OUTPUT_SIZE）
OUTPUT_FOLDER = os.path.join(output_root, "all")

# 输出切片统一resize的尺寸（单位像素）
OUTPUT_SIZE = (640, 640)


# =====================================================

def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 范围内。
    若短边小于 min_short 则放大；大于 max_short 则缩小；否则不缩放。
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
    edge_density = compute_edge_density(image)
    texture_complexity = compute_texture_complexity(image)
    return (edge_density + texture_complexity) / 2.0


def direct_sliding_window(image, window_size):
    """
    直接切片：以给定窗口尺寸对图像进行切片。
    若图像边缘不足一个完整窗口，则补齐一个窗口（从末端取窗口，使得覆盖至图像右侧和下侧）。
    返回切片列表，每个切片为 (x, y, roi)。
    """
    h, w = image.shape[:2]
    slices = []
    # 生成 x 坐标列表
    xs = list(range(0, w - window_size + 1, window_size))
    if not xs or xs[-1] != w - window_size:
        xs.append(w - window_size)
    # 生成 y 坐标列表
    ys = list(range(0, h - window_size + 1, window_size))
    if not ys or ys[-1] != h - window_size:
        ys.append(h - window_size)
    for y in ys:
        for x in xs:
            roi = image[y:y + window_size, x:x + window_size]
            slices.append((x, y, roi))
    return slices


def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图以及直接切片（无重叠补齐）。
    保存结果至 OUTPUT_FOLDER 下，文件命名如下：
      - 原图： {image_name}.jpg   （原始图，不做resize）
      - 切片： {image_name}_slice_{window_size}_{x}_{y}.png   （切片统一resize到 OUTPUT_SIZE）
      - 生成一个 index 文件： {image_name}_slice_index.txt，记录每个切片信息
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return

    # 为了后续检测对比，这里先自适应缩放（保证短边在2560-5120内）
    image_resized = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image_resized.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 确保输出文件夹存在（统一为 OUTPUT_FOLDER）
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 保存原图（未经过 OUTPUT_SIZE 的resize），直接保存自适应后的图像
    original_save_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_original.jpg")
    cv2.imwrite(original_save_path, image_resized)
    print(f"Saved original image: {original_save_path}")

    # 保存全局显著性热图
    saliency = compute_saliency_map(image_resized)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 根据图像短边确定窗口尺寸
    window_size = min(h, w)
    print(f"Using window size = {window_size} (short side of image)")

    # 生成直接切片（确保全图覆盖）
    slices = direct_sliding_window(image_resized, window_size)
    print(f"Generated {len(slices)} slices (direct tiling)")

    # 创建一个 index 文件，记录每个切片信息
    index_file = os.path.join(OUTPUT_FOLDER, f"{base_name}_slice_index.txt")
    with open(index_file, "w", encoding="utf-8") as f:
        f.write("window_size,x,y,filename\n")

    # 保存每个切片（统一resize到 OUTPUT_SIZE）并写入 index 文件
    for (x, y, roi) in slices:
        roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
        filename = f"{base_name}_slice_{window_size}_{x}_{y}.png"
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(save_path, roi_resized)
        with open(index_file, "a", encoding="utf-8") as f:
            f.write(f"{window_size},{x},{y},{filename}\n")

    print(f"Saved {len(slices)} slices and index file: {index_file}")


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
