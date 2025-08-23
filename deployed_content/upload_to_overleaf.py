"""
Overleaf Upload Automation Script
使用Playwright自动化上传LaTeX项目文件到Overleaf

注意：这个脚本需要您手动登录Overleaf并创建一个新项目
"""

import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright
import argparse

# 项目文件路径
PROJECT_DIR = Path("I:/EMNLP2025/deployed_content/Paper_writing")

# 需要上传的文件列表
FILES_TO_UPLOAD = [
    # 主要LaTeX文件
    "final.tex",
    "appendix.tex",
    "formulas.tex",
    
    # 样式文件
    "acl.sty",
    "emnlp2024.sty",
    
    # 参考文献
    "references.bib",
    "acl_natbib.bst",
    
    # 图片文件
    "picture/paper_structure.png",
    "picture/combined_author_semantic_space_visualization_tsne(作者语义空间t-SNE可视化).png",
    "picture/composite_figure_tsne_radar.png",
    "picture/profiling_summary_figure.png",
]

def upload_to_overleaf(email=None, password=None, project_url=None):
    """
    使用Playwright自动化上传文件到Overleaf
    
    Args:
        email: Overleaf账号邮箱（可选，用于自动登录）
        password: Overleaf账号密码（可选，用于自动登录）
        project_url: 已存在的Overleaf项目URL（可选）
    """
    
    with sync_playwright() as p:
        # 启动浏览器（使用有头模式以便手动操作）
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        
        print("正在打开Overleaf...")
        
        if project_url:
            # 如果提供了项目URL，直接打开
            page.goto(project_url)
            print(f"已打开项目: {project_url}")
        else:
            # 否则打开Overleaf主页
            page.goto("https://www.overleaf.com")
            print("已打开Overleaf主页")
            
            if email and password:
                # 尝试自动登录
                print("尝试自动登录...")
                try:
                    # 点击登录按钮
                    page.click("text=Log In")
                    time.sleep(2)
                    
                    # 输入邮箱和密码
                    page.fill('input[name="email"]', email)
                    page.fill('input[name="password"]', password)
                    
                    # 提交登录
                    page.click('button[type="submit"]')
                    time.sleep(5)
                    
                    print("登录成功！")
                except Exception as e:
                    print(f"自动登录失败: {e}")
                    print("请手动登录...")
            else:
                print("\n请手动完成以下步骤：")
                print("1. 登录您的Overleaf账号")
                print("2. 创建一个新项目或打开现有项目")
                print("3. 确保项目编辑器已打开")
                input("\n完成后按Enter继续...")
        
        # 等待项目编辑器加载
        print("\n等待项目编辑器加载...")
        time.sleep(3)
        
        # 检查是否在编辑器页面
        if "/project/" not in page.url:
            print("警告：似乎不在项目编辑器页面")
            print("当前URL:", page.url)
            input("请导航到项目编辑器页面，然后按Enter继续...")
        
        print("\n开始上传文件...")
        
        # 上传每个文件
        for file_path in FILES_TO_UPLOAD:
            full_path = PROJECT_DIR / file_path
            
            if not full_path.exists():
                print(f"警告：文件不存在 - {file_path}")
                continue
            
            print(f"正在上传: {file_path}")
            
            try:
                # 点击上传按钮（通常在左侧文件树上方）
                # Overleaf的上传按钮可能有不同的选择器，这里尝试几种常见的
                upload_button_selectors = [
                    'button[aria-label="Upload"]',
                    'button:has-text("Upload")',
                    '.toolbar-left button[ng-click*="upload"]',
                    'button[ng-click="openUploadDocModal()"]',
                    'a[ng-click="openUploadDocModal()"]'
                ]
                
                clicked = False
                for selector in upload_button_selectors:
                    try:
                        page.click(selector, timeout=2000)
                        clicked = True
                        break
                    except:
                        continue
                
                if not clicked:
                    print("无法找到上传按钮，尝试使用键盘快捷键...")
                    # 尝试使用拖放或其他方法
                    
                time.sleep(1)
                
                # 处理文件选择对话框
                with page.expect_file_chooser() as fc_info:
                    # 触发文件选择器
                    page.click('input[type="file"]', timeout=5000)
                    
                file_chooser = fc_info.value
                file_chooser.set_files(str(full_path))
                
                print(f"✓ 已上传: {file_path}")
                time.sleep(2)  # 等待上传完成
                
            except Exception as e:
                print(f"✗ 上传失败: {file_path} - {e}")
                print("  提示：您可能需要手动上传此文件")
        
        print("\n文件上传完成！")
        print("\n重要提示：")
        print("1. 请检查所有文件是否已正确上传")
        print("2. 将主文档设置为 'final.tex'")
        print("3. 编译器设置为 XeLaTeX")
        print("4. 点击重新编译以生成PDF")
        
        input("\n按Enter关闭浏览器...")
        browser.close()

def main():
    parser = argparse.ArgumentParser(description="上传LaTeX项目到Overleaf")
    parser.add_argument("--email", help="Overleaf账号邮箱（可选）")
    parser.add_argument("--password", help="Overleaf账号密码（可选）")
    parser.add_argument("--project-url", help="现有Overleaf项目URL（可选）")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Overleaf项目上传工具")
    print("=" * 50)
    print(f"\n源目录: {PROJECT_DIR}")
    print(f"文件数量: {len(FILES_TO_UPLOAD)}")
    print("\n文件列表:")
    for f in FILES_TO_UPLOAD:
        print(f"  - {f}")
    print()
    
    upload_to_overleaf(args.email, args.password, args.project_url)

if __name__ == "__main__":
    main()