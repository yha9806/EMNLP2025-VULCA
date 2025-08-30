# Overleaf上传指南

## 需要上传的文件清单

### 核心LaTeX文件
- [ ] `final.tex` - 主文档文件（设为主文档）
- [ ] `appendix.tex` - 附录
- [ ] `formulas.tex` - 公式定义

### 样式和格式文件
- [ ] `acl.sty` - ACL会议样式
- [ ] `emnlp2024.sty` - EMNLP 2024样式

### 参考文献
- [ ] `references.bib` - 参考文献数据库
- [ ] `acl_natbib.bst` - 参考文献样式

### 图片文件（picture/目录）
- [ ] `paper_structure.png` - 论文结构图
- [ ] `combined_author_semantic_space_visualization_tsne(作者语义空间t-SNE可视化).png` - t-SNE可视化图
- [ ] `composite_figure_tsne_radar.png` - 复合图表
- [ ] `profiling_summary_figure.png` - 概要分析图

## Overleaf设置要求

### 1. 编译器设置
- **编译器**: XeLaTeX（必须，因为文档使用了中文字体）
- **主文档**: final.tex

### 2. 字体说明
文档使用了以下字体配置：
```latex
\setmainfont{Times New Roman}
\setmonofont{Courier New}
\setCJKmainfont{Microsoft YaHei}  # 中文主字体
```

注意：Overleaf可能没有Microsoft YaHei字体，可能需要调整为：
- `Noto Sans CJK SC` 或
- `Source Han Sans SC` 或
- 其他Overleaf支持的中文字体

### 3. 上传步骤

1. **登录Overleaf**
   - 访问 https://www.overleaf.com
   - 登录您的账号

2. **创建新项目**
   - 点击 "New Project"
   - 选择 "Blank Project" 或 "Upload Project"

3. **上传文件**
   - 方法1：逐个上传
     - 点击左侧文件树上方的上传图标
     - 选择文件上传
   
   - 方法2：ZIP上传（推荐）
     - 将所有文件打包成ZIP
     - 使用 "Upload Project" 选项

4. **创建picture文件夹**
   - 在Overleaf中创建 `picture` 文件夹
   - 上传所有图片文件到此文件夹

5. **设置编译选项**
   - 点击左上角的 "Menu"
   - Compiler: 选择 "XeLaTeX"
   - Main document: 选择 "final.tex"

6. **编译项目**
   - 点击 "Recompile" 按钮
   - 等待编译完成

## 可能遇到的问题及解决方案

### 1. 字体缺失错误
如果出现字体错误，修改 `final.tex` 中的字体设置：
```latex
% 替换为Overleaf支持的字体
\setCJKmainfont{Noto Sans CJK SC}
```

### 2. 图片路径问题
确保所有图片都在 `picture/` 文件夹中，路径格式正确。

### 3. 编译超时
如果文档太大导致编译超时：
- 尝试删除辅助文件（.aux, .log等）
- 清理缓存后重新编译

### 4. 中文显示问题
确保：
- 使用XeLaTeX编译器
- 字体设置正确
- 文件编码为UTF-8

## 文件大小信息
- 总大小：约4.5MB
- 最大文件：final.pdf (4.2MB) - 这个是输出文件，不需要上传
- 图片总大小：约300KB

## 本地文件路径
所有文件位于：`I:\EMNLP2025\deployed_content\Paper_writing\`