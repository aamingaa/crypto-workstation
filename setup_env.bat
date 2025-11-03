@echo off
REM =============================================================================
REM Crypto Workstation - Python 环境安装脚本 (Windows)
REM =============================================================================
REM 用途：自动创建虚拟环境并安装所有依赖
REM 使用方法：双击运行或在命令行执行 setup_env.bat
REM =============================================================================

setlocal enabledelayedexpansion
chcp 65001 > nul

echo.
echo ========================================
echo Crypto Workstation 环境安装
echo ========================================
echo.

REM =============================================================================
REM 1. 检查 Python 版本
REM =============================================================================
echo [步骤 1] 检查 Python 版本...
echo.

where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.8 或更高版本
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [信息] 检测到 Python 版本: %PYTHON_VERSION%

python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [错误] Python 版本过低，需要 Python 3.8 或更高版本
    pause
    exit /b 1
)
echo [成功] Python 版本满足要求
echo.

REM =============================================================================
REM 2. 创建虚拟环境
REM =============================================================================
echo [步骤 2] 创建虚拟环境...
echo.

set VENV_DIR=venv

if exist "%VENV_DIR%" (
    echo [警告] 虚拟环境已存在: %VENV_DIR%
    set /p RECREATE="是否删除并重新创建？(y/N): "
    if /i "!RECREATE!"=="y" (
        echo [信息] 删除旧的虚拟环境...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo [信息] 使用现有虚拟环境
    )
)

if not exist "%VENV_DIR%" (
    echo [信息] 创建虚拟环境: %VENV_DIR%
    python -m venv "%VENV_DIR%"
    echo [成功] 虚拟环境创建成功
) else (
    echo [信息] 跳过虚拟环境创建
)
echo.

REM =============================================================================
REM 3. 激活虚拟环境
REM =============================================================================
echo [步骤 3] 激活虚拟环境...
echo.

call "%VENV_DIR%\Scripts\activate.bat"
echo [成功] 虚拟环境已激活
echo.

REM =============================================================================
REM 4. 升级 pip
REM =============================================================================
echo [步骤 4] 升级 pip...
echo.

python -m pip install --upgrade pip
echo [成功] pip 升级完成
echo.

REM =============================================================================
REM 5. 安装 Python 依赖
REM =============================================================================
echo [步骤 5] 安装 Python 依赖...
echo.

if not exist "requirements.txt" (
    echo [错误] 未找到 requirements.txt 文件
    pause
    exit /b 1
)

echo [信息] 从 requirements.txt 安装依赖...
set /p USE_MIRROR="是否使用清华镜像源加速安装？(Y/n): "

if /i "!USE_MIRROR!"=="n" (
    echo [信息] 使用默认镜像源安装...
    pip install -r requirements.txt
) else (
    echo [信息] 使用清华镜像源安装...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo [成功] Python 依赖安装完成
echo.

REM =============================================================================
REM 6. 验证安装
REM =============================================================================
echo [步骤 6] 验证安装...
echo.

echo [信息] 验证关键库是否安装成功...

python -c "import pandas" 2>nul && (echo [成功] pandas ✓) || (echo [错误] pandas ✗)
python -c "import numpy" 2>nul && (echo [成功] numpy ✓) || (echo [错误] numpy ✗)
python -c "import sklearn" 2>nul && (echo [成功] sklearn ✓) || (echo [错误] sklearn ✗)
python -c "import xgboost" 2>nul && (echo [成功] xgboost ✓) || (echo [错误] xgboost ✗)
python -c "import lightgbm" 2>nul && (echo [成功] lightgbm ✓) || (echo [错误] lightgbm ✗)
python -c "import matplotlib" 2>nul && (echo [成功] matplotlib ✓) || (echo [错误] matplotlib ✗)
python -c "import plotly" 2>nul && (echo [成功] plotly ✓) || (echo [错误] plotly ✗)

echo.

REM =============================================================================
REM 7. 完成
REM =============================================================================
echo ========================================
echo 安装完成
echo ========================================
echo.

echo [信息] 当前 Python 环境信息：
python --version
pip --version
where python
echo.

echo [成功] 环境配置完成！
echo.
echo [信息] 激活虚拟环境命令：
echo   venv\Scripts\activate.bat
echo.
echo [信息] 退出虚拟环境命令：
echo   deactivate
echo.
echo [信息] 运行项目：
echo   python run_gp.py
echo.

pause

