"""
Overleaf EMNLP项目文件上传自动化脚本
使用Playwright自动化上传文件到现有的EMNLP项目
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError

# 项目文件路径
PROJECT_DIR = Path("I:/EMNLP2025/deployed_content/Paper_writing")

# 需要上传的文件列表
FILES_TO_UPLOAD = [
    # 主要LaTeX文件
    ("final.tex", ""),
    ("appendix.tex", ""),
    ("formulas.tex", ""),
    
    # 样式文件
    ("acl.sty", ""),
    ("emnlp2024.sty", ""),
    
    # 参考文献
    ("references.bib", ""),
    ("acl_natbib.bst", ""),
    
    # 图片文件 - 需要先创建picture文件夹
    ("picture/paper_structure.png", "picture"),
    ("picture/combined_author_semantic_space_visualization_tsne(作者语义空间t-SNE可视化).png", "picture"),
    ("picture/composite_figure_tsne_radar.png", "picture"),
    ("picture/profiling_summary_figure.png", "picture"),
]

def wait_for_element(page, selector, timeout=10000):
    """等待元素出现"""
    try:
        page.wait_for_selector(selector, timeout=timeout)
        return True
    except TimeoutError:
        return False

def upload_files_to_overleaf():
    """使用Playwright自动化上传文件到Overleaf"""
    
    with sync_playwright() as p:
        # 启动浏览器（使用有头模式以便观察和手动介入）
        print("启动浏览器...")
        browser = p.chromium.launch(
            headless=False,
            args=['--start-maximized']
        )
        
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='zh-CN'
        )
        
        page = context.new_page()
        
        # 打开Overleaf
        print("打开Overleaf网站...")
        page.goto("https://www.overleaf.com/login")
        
        print("\n" + "="*60)
        print("请在浏览器中完成以下操作：")
        print("1. 登录您的Overleaf账号")
        print("2. 打开您的EMNLP项目")
        print("3. 确保项目编辑器已完全加载")
        print("="*60)
        input("\n完成后按Enter继续...")
        
        # 检查是否在项目页面
        current_url = page.url
        if "/project/" not in current_url:
            print("错误：当前不在项目页面")
            print(f"当前URL: {current_url}")
            input("请导航到EMNLP项目页面，然后按Enter继续...")
        
        print(f"\n当前项目URL: {page.url}")
        print("开始上传文件...\n")
        
        # 首先创建picture文件夹
        print("检查是否需要创建picture文件夹...")
        try:
            # 查找文件树中是否已有picture文件夹
            if not page.locator('.entity-name:has-text("picture")').count():
                print("创建picture文件夹...")
                
                # 点击新建文件夹按钮
                new_folder_btn = page.locator('button[ng-click="openNewFolderModal()"]').or_(
                    page.locator('button:has-text("New Folder")').or_(
                        page.locator('[aria-label="New folder"]')
                    )
                )
                
                if new_folder_btn.count():
                    new_folder_btn.first.click()
                    time.sleep(1)
                    
                    # 输入文件夹名
                    folder_input = page.locator('input[ng-model="inputs.name"]').or_(
                        page.locator('input[placeholder*="folder"]')
                    )
                    if folder_input.count():
                        folder_input.first.fill("picture")
                        
                        # 确认创建
                        create_btn = page.locator('button:has-text("Create")').or_(
                            page.locator('button[ng-click="create()"]')
                        )
                        if create_btn.count():
                            create_btn.first.click()
                            print("✓ picture文件夹已创建")
                            time.sleep(2)
                else:
                    print("未找到新建文件夹按钮，可能需要手动创建")
            else:
                print("picture文件夹已存在")
        except Exception as e:
            print(f"创建文件夹时出错: {e}")
            print("请手动创建picture文件夹")
            input("完成后按Enter继续...")
        
        # 上传文件
        uploaded_count = 0
        failed_files = []
        
        for file_info in FILES_TO_UPLOAD:
            if isinstance(file_info, tuple):
                file_path, target_folder = file_info
            else:
                file_path = file_info
                target_folder = ""
            
            full_path = PROJECT_DIR / file_path
            
            if not full_path.exists():
                print(f"⚠ 文件不存在: {file_path}")
                failed_files.append(file_path)
                continue
            
            print(f"上传: {file_path}", end=" ... ")
            
            try:
                # 如果需要上传到子文件夹，先选择该文件夹
                if target_folder:
                    folder_element = page.locator(f'.entity-name:has-text("{target_folder}")').first
                    if folder_element.count():
                        folder_element.click()
                        time.sleep(1)
                
                # 查找上传按钮
                upload_button = None
                upload_selectors = [
                    'button[ng-click="openUploadDocModal()"]',
                    'button[aria-label="Upload"]',
                    'button:has-text("Upload")',
                    '[ng-click="openUploadDocModal()"]',
                    '.toolbar-left button[title*="upload" i]',
                    'a[ng-click="openUploadDocModal()"]'
                ]
                
                for selector in upload_selectors:
                    if page.locator(selector).count():
                        upload_button = page.locator(selector).first
                        break
                
                if upload_button:
                    upload_button.click()
                    time.sleep(1)
                    
                    # 处理文件选择
                    file_input = page.locator('input[type="file"]').first
                    if file_input.count():
                        file_input.set_input_files(str(full_path))
                        time.sleep(2)  # 等待上传完成
                        
                        # 关闭上传对话框（如果有）
                        close_btn = page.locator('button:has-text("Done")').or_(
                            page.locator('button[ng-click="done()"]')
                        )
                        if close_btn.count():
                            close_btn.first.click()
                        
                        print("✓")
                        uploaded_count += 1
                    else:
                        print("✗ (未找到文件输入框)")
                        failed_files.append(file_path)
                else:
                    print("✗ (未找到上传按钮)")
                    failed_files.append(file_path)
                    
            except Exception as e:
                print(f"✗ (错误: {str(e)[:50]})")
                failed_files.append(file_path)
            
            time.sleep(1)  # 避免操作太快
        
        # 设置编译选项
        print("\n" + "="*60)
        print("文件上传完成统计：")
        print(f"成功上传: {uploaded_count}/{len(FILES_TO_UPLOAD)} 个文件")
        
        if failed_files:
            print(f"\n失败文件 ({len(failed_files)}个):")
            for f in failed_files:
                print(f"  - {f}")
            print("\n请手动上传这些文件")
        
        print("\n" + "="*60)
        print("请完成以下设置：")
        print("1. 点击左上角 Menu 按钮")
        print("2. 设置 Compiler: XeLaTeX")
        print("3. 设置 Main document: final.tex")
        print("4. 点击 Recompile 编译项目")
        print("="*60)
        
        input("\n完成所有操作后按Enter关闭浏览器...")
        browser.close()
        
        return uploaded_count, failed_files

def main():
    print("="*60)
    print("Overleaf EMNLP项目文件上传工具")
    print("="*60)
    print(f"\n源目录: {PROJECT_DIR}")
    print(f"待上传文件数: {len(FILES_TO_UPLOAD)}")
    
    try:
        uploaded, failed = upload_files_to_overleaf()
        
        print("\n" + "="*60)
        print("上传任务完成！")
        
        if len(failed) == 0:
            print("✓ 所有文件上传成功！")
        else:
            print(f"部分文件需要手动上传 ({len(failed)}个)")
            
    except Exception as e:
        print(f"\n错误: {e}")
        print("请检查网络连接和Overleaf网站状态")

if __name__ == "__main__":
    main()