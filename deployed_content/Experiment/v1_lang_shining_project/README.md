# 古籍OCR工具

这是一个专门用于古籍文字识别的OCR工具，集成了多种预处理方法和OCR引擎，能够更好地处理古籍中的竖排文本和特殊字体。

## 功能特点

- **多种图像预处理方法**：包括去噪、对比度增强、二值化、形态学操作等，提高OCR识别准确率
- **多引擎集成**：支持CnOCR和PaddleOCR两种引擎，并可集成结果
- **竖排文本支持**：专门针对古籍中常见的竖排文本进行优化
- **排版处理**：支持多栏和单栏排版，按照正确的阅读顺序输出文本
- **命令行接口**：提供灵活的命令行参数，方便批量处理

## 安装依赖

```bash
pip install cnocr paddleocr opencv-python numpy scikit-image matplotlib
```

## 使用方法

### 基本用法

```bash
python ancient_chinese_ocr.py --image 图片路径.jpg --output 输出文件.txt
```

### 参数说明

- `--image`：输入图像路径（必需）
- `--output`：输出文件路径（默认：ocr_result.txt）
- `--vertical`：是否为竖排文本（默认：否）
- `--engine`：OCR引擎选择（可选：cnocr, paddle, ensemble，默认：ensemble）
- `--layout`：排版类型（可选：multi_column, single_column，默认：multi_column）
- `--preprocess`：预处理方法（可选：original, denoise, contrast, binarize, morphology, deskew, combined，默认：combined）

### 示例

1. 识别竖排文本，使用集成引擎：

```bash
python ancient_chinese_ocr.py --image 古籍图片.jpg --output 结果.txt --vertical
```

2. 使用特定预处理方法和OCR引擎：

```bash
python ancient_chinese_ocr.py --image 古籍图片.jpg --preprocess binarize --engine cnocr
```

3. 处理单栏排版的文档：

```bash
python ancient_chinese_ocr.py --image 古籍图片.jpg --layout single_column
```

## 预处理方法说明

- `original`：原始图像，不做预处理
- `denoise`：去噪处理，减少图像噪点
- `contrast`：对比度增强，使文字更加清晰
- `binarize`：二值化处理，将图像转换为黑白两色
- `morphology`：形态学操作，连接断开的笔画
- `deskew`：校正倾斜，修正图像角度
- `combined`：组合处理，依次应用去噪、对比度增强、二值化和形态学操作

## OCR引擎说明

- `cnocr`：使用CnOCR引擎，适合中文识别
- `paddle`：使用PaddleOCR引擎，支持多语言
- `ensemble`：同时使用两种引擎，并集成结果，提高识别准确率

## 排版类型说明

- `multi_column`：多栏排版，适合古籍中常见的多栏布局
- `single_column`：单栏排版，适合现代文档

## 注意事项

1. 首次运行时，程序会自动下载OCR模型，需要保持网络连接
2. 对于大尺寸图像，处理可能需要较长时间
3. 竖排文本识别效果取决于OCR引擎的支持程度
4. 对于特别复杂的古籍版面，可能需要手动调整参数以获得最佳效果

## 进阶用法

对于批量处理多张图片，可以编写简单的批处理脚本：

```python
import os
import subprocess

input_dir = "古籍图片目录"
output_dir = "识别结果目录"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.png', '.jpeg', '.tif', '.tiff')):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".txt")
        
        cmd = [
            "python", "ancient_chinese_ocr.py",
            "--image", input_path,
            "--output", output_path,
            "--vertical",  # 如果是竖排文本
            "--engine", "ensemble",
            "--preprocess", "combined"
        ]
        
        subprocess.run(cmd)
        print(f"处理完成: {filename}")
```

## 贡献与改进

欢迎提交问题和改进建议！如果您有任何问题或建议，请提交issue或pull request。
