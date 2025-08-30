import os
import cv2
import glob
import numpy as np
from skimage.feature import local_binary_pattern
from tqdm import tqdm

# ===================== 全局参数 =====================
input_folder = r"D:\Qiufeng\aiart\12MP"  # 原图所在文件夹
output_root = r"D:\Qiufeng\aiart\Auto_Cropping_12MP"  # 输出总目录

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
HIGH_THRESHOLD = 0.5  # 高信息密度阈值
OVERLAP_LOW = 0.1  # 低密度区域采用10%重叠 -> stride = 0.9 * window_size
OVERLAP_MEDIUM = 0.25  # 中密度区域采用25%重叠 -> stride = 0.75 * window_size
OVERLAP_HIGH = 0.5  # 高密度区域采用50%重叠 -> stride = 0.5 * window_size

# 绘制滑动轨迹时不同窗口类型的颜色 (BGR格式)
# 这里 preset 中 large、medium、small 按原来设定
WINDOW_COLORS = {
    "large": (0, 0, 255),  # 红色
    "medium": (255, 0, 0),  # 蓝色
    "small": (0, 255, 0)  # 绿色
}
# 输出时 "all" 文件夹统一保存
# =====================================================

# NMS 相关阈值
IOU_THRESHOLD = 0.3  # IoU 阈值
CONF_THRESHOLD = 0.7  # 置信度阈值
CONTAIN_THRESHOLD = 0.2  # 包含度阈值（小框被大框覆盖比例超过此值，认为是同一目标）


# （注：对于不同场景可以调节，这里仅为初始设置）

# ===================== 辅助函数 =====================

def adaptive_resize(image, min_short=2560, max_short=5120):
    """
    自适应缩放图像，使图像短边在 [min_short, max_short] 内：
      - 若短边在 [1280,5120] 内，则不缩放；
      - 若短边 > max_short，则缩小到 max_short；
      - 若短边 < 1280，则不缩放（后续采用动态窗口）。
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
    """计算 Sobel 边缘密度，并归一化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_map = np.sqrt(sobelx ** 2 + sobely ** 2)
    return edge_map / (edge_map.max() + 1e-8)


def compute_texture_complexity(image, radius=1, n_points=8):
    """计算 LBP 纹理复杂度，并归一化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    return lbp / (lbp.max() + 1e-8)


def compute_saliency_map(image):
    """生成传统显著性热图：结合边缘密度和纹理复杂度"""
    edge = compute_edge_density(image)
    texture = compute_texture_complexity(image)
    return (edge + texture) / 2.0


# 自适应滑动窗口（根据 saliency 调整步长）
def adaptive_sliding_window(image, saliency_map, window_size):
    h, w = image.shape[:2]
    slices = []

    def get_stride(avg_sal):
        if avg_sal < LOW_THRESHOLD:
            return int(window_size * (1 - OVERLAP_LOW))
        elif avg_sal > HIGH_THRESHOLD:
            return int(window_size * (1 - OVERLAP_HIGH))
        else:
            return int(window_size * (1 - OVERLAP_MEDIUM))

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


# 固定重叠率的滑动窗口，用于 "dircrop" 模式，重叠率固定为 10%
def sliding_window_fixed_overlap(image, window_size, overlap_ratio=0.1):
    h, w = image.shape[:2]
    slices = []
    stride = int(window_size * (1 - overlap_ratio))
    xs = list(range(0, w - window_size + 1, stride))
    if not xs or xs[-1] != w - window_size:
        xs.append(w - window_size)
    ys = list(range(0, h - window_size + 1, stride))
    if not ys or ys[-1] != h - window_size:
        ys.append(h - window_size)
    for y in ys:
        for x in xs:
            roi = image[y:y + window_size, x:x + window_size]
            slices.append((x, y, roi))
    return slices


# ===================== 直接裁切方案 =====================
def process_image(image_path):
    """
    对单张图像进行自适应缩放、生成全局显著性热图，
    并按不同窗口尺寸进行滑动窗口切片。
    当图像短边 >= 1280 时，除了预设的 large/medium/small，
    额外加入一种直接裁切模式 "dircrop"（窗口尺寸取原图短边，重叠率固定为10%）。
    所有切片统一调整为 OUTPUT_SIZE。
    输出目录结构：
       output_root/{image_name}/
           saliency_heatmap.png
           slices/
                {imageName}_slice_{mode}_{win_size}_{x}_{y}.png
           slice_index.txt  —— 记录每个切片信息，格式：mode,win_size,x,y,filename
           原图复制到 "all" 文件夹内： {image_name}_original.jpg 及所有切片也复制到 "all"。
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"读取图像失败：{image_path}")
        return
    image = adaptive_resize(image, min_short=2560, max_short=5120)
    h, w = image.shape[:2]
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # 输出文件夹
    out_folder = os.path.join(output_root, base_name)
    os.makedirs(out_folder, exist_ok=True)

    # 创建子文件夹： small, medium, large, dircrop, all
    for folder in list(WINDOW_SIZES.keys()) + ["dircrop", "all"]:
        os.makedirs(os.path.join(out_folder, folder), exist_ok=True)

    # 保存原图到 "all" 文件夹（原图不做 resize）
    original_save_path = os.path.join(out_folder, "all", f"{base_name}_original.jpg")
    cv2.imwrite(original_save_path, image)

    # 保存全局显著性热图
    saliency = compute_saliency_map(image)
    saliency_8u = np.uint8(255 * saliency)
    heatmap = cv2.applyColorMap(saliency_8u, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(out_folder, "saliency_heatmap.png")
    cv2.imwrite(heatmap_path, heatmap)
    print(f"Saved saliency heatmap: {heatmap_path}")

    # 初始化 index 文件（记录切片信息）
    index_file_path = os.path.join(out_folder, "slice_index.txt")
    with open(index_file_path, "w", encoding="utf-8") as f:
        f.write("mode,window_size,x,y,filename\n")

    # 用于绘制滑动轨迹（基于原图）
    trajectory_img = image.copy()

    # 先处理预设的模式（large, medium, small）
    for mode, win_size in WINDOW_SIZES.items():
        slices = adaptive_sliding_window(image, saliency, win_size)
        subfolder = os.path.join(out_folder, mode)
        print(f"Processing {len(slices)} adaptive slices for {mode} mode ({win_size}x{win_size}) ...")
        for (x, y, roi) in slices:
            with open(index_file_path, "a", encoding="utf-8") as f:
                f.write(f"{mode},{win_size},{x},{y},{base_name}_slice_{mode}_{win_size}_{x}_{y}.png\n")
            # 绘制边框到轨迹图（使用对应颜色）
            color = WINDOW_COLORS.get(mode, (0, 255, 0))
            cv2.rectangle(trajectory_img, (x, y), (x + win_size, y + win_size), color, 2)
            roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            filename = f"{base_name}_slice_{mode}_{win_size}_{x}_{y}.png"
            save_path = os.path.join(out_folder, mode, filename)
            cv2.imwrite(save_path, roi_resized)
            # 同时保存到 "all" 文件夹
            all_save_path = os.path.join(out_folder, "all", filename)
            cv2.imwrite(all_save_path, roi_resized)

    # 若图像短边 >= 1280，则额外开启 "dircrop" 模式（直接切片，固定重叠率10%）
    if min(h, w) >= 1280:
        # 这里我们采用窗口尺寸为原图短边
        win_size_dir = min(h, w)
        slices_dir = sliding_window_fixed_overlap(image, win_size_dir, overlap_ratio=0.1)
        mode = "dircrop"
        subfolder = os.path.join(out_folder, mode)
        print(f"Processing {len(slices_dir)} direct slices for {mode} mode ({win_size_dir}x{win_size_dir}) ...")
        for (x, y, roi) in slices_dir:
            with open(index_file_path, "a", encoding="utf-8") as f:
                f.write(f"{mode},{win_size_dir},{x},{y},{base_name}_slice_{mode}_{win_size_dir}_{x}_{y}.png\n")
            color = (0, 128, 255)  # 设置 dircrop 模式颜色，例如橙色，可与 original（黄色）区分
            cv2.rectangle(trajectory_img, (x, y), (x + win_size_dir, y + win_size_dir), color, 2)
            roi_resized = cv2.resize(roi, OUTPUT_SIZE, interpolation=cv2.INTER_AREA)
            filename = f"{base_name}_slice_{mode}_{win_size_dir}_{x}_{y}.png"
            save_path = os.path.join(out_folder, mode, filename)
            cv2.imwrite(save_path, roi_resized)
            all_save_path = os.path.join(out_folder, "all", filename)
            cv2.imwrite(all_save_path, roi_resized)

    # 保存滑动轨迹图
    trajectory_path = os.path.join(out_folder, "sliding_trajectory.png")
    cv2.imwrite(trajectory_path, trajectory_img)
    print(f"Saved sliding trajectory: {trajectory_path}")
    print(f"Saved slices and index file at {out_folder}")


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
