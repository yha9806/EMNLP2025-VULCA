@echo off
echo 正在配置 rclone for Google Drive...
echo.
echo 请按照以下步骤操作：
echo.
echo 1. 运行此命令后，选择 'n' 创建新配置
echo 2. 输入名称：gdrive
echo 3. 选择存储类型：输入 18 (Google Drive)
echo 4. Client ID 和 Secret：直接按回车（使用默认）
echo 5. Scope：选择 1 (完全访问)
echo 6. Root folder ID：直接按回车
echo 7. Service Account File：直接按回车
echo 8. Edit advanced config：输入 n
echo 9. Use web browser：输入 y
echo 10. 浏览器会打开，登录 Google 账号并授权
echo 11. 完成后选择 q 退出
echo.
pause
C:\Users\MyhrDyzy\bin\gclone.exe config