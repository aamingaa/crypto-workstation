#!/bin/bash

# =============================================================================
# Crypto Workstation - Python 环境安装脚本
# =============================================================================
# 用途：自动创建虚拟环境并安装所有依赖
# 使用方法：bash setup_env.sh
# =============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# =============================================================================
# 1. 检查 Python 版本
# =============================================================================
print_header "步骤 1: 检查 Python 版本"

if ! command -v python3 &> /dev/null; then
    print_error "未找到 python3，请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
print_info "检测到 Python 版本: $PYTHON_VERSION"

# 检查是否满足最低版本要求（3.8+）
REQUIRED_VERSION="3.8"
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_error "Python 版本过低，需要 Python 3.8 或更高版本"
    exit 1
fi
print_success "Python 版本满足要求"

# =============================================================================
# 2. 创建虚拟环境
# =============================================================================
print_header "步骤 2: 创建虚拟环境"

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    print_warning "虚拟环境已存在: $VENV_DIR"
    read -p "是否删除并重新创建？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除旧的虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        print_info "使用现有虚拟环境"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    print_info "创建虚拟环境: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    print_success "虚拟环境创建成功"
else
    print_info "跳过虚拟环境创建"
fi

# =============================================================================
# 3. 激活虚拟环境
# =============================================================================
print_header "步骤 3: 激活虚拟环境"

print_info "激活虚拟环境..."
source "$VENV_DIR/bin/activate"
print_success "虚拟环境已激活"

# =============================================================================
# 4. 升级 pip
# =============================================================================
print_header "步骤 4: 升级 pip"

print_info "升级 pip 到最新版本..."
pip install --upgrade pip
print_success "pip 升级完成"

# =============================================================================
# 5. 检查系统依赖
# =============================================================================
print_header "步骤 5: 检查系统依赖"

# 检查 TA-Lib
if command -v brew &> /dev/null; then
    print_info "检测到 Homebrew，检查 TA-Lib..."
    
    if brew list ta-lib &> /dev/null; then
        print_success "TA-Lib 已安装"
    else
        print_warning "TA-Lib 未安装"
        read -p "是否使用 Homebrew 安装 TA-Lib？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "安装 TA-Lib..."
            brew install ta-lib
            print_success "TA-Lib 安装完成"
        else
            print_warning "跳过 TA-Lib 安装，稍后安装 Python TA-Lib 包可能会失败"
        fi
    fi
else
    print_warning "未检测到 Homebrew"
    print_info "如需安装 TA-Lib，请参考: https://github.com/mrjbq7/ta-lib"
fi

# =============================================================================
# 6. 安装 Python 依赖
# =============================================================================
print_header "步骤 6: 安装 Python 依赖"

if [ -f "requirements.txt" ]; then
    print_info "从 requirements.txt 安装依赖..."
    
    # 询问是否使用国内镜像
    read -p "是否使用清华镜像源加速安装？(Y/n): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_info "使用清华镜像源安装..."
        PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    else
        print_info "使用默认镜像源安装..."
        PIP_INDEX=""
    fi
    
    # 安装依赖
    if [ -n "$PIP_INDEX" ]; then
        pip install -r requirements.txt -i "$PIP_INDEX"
    else
        pip install -r requirements.txt
    fi
    
    print_success "Python 依赖安装完成"
else
    print_error "未找到 requirements.txt 文件"
    exit 1
fi

# =============================================================================
# 7. 验证安装
# =============================================================================
print_header "步骤 7: 验证安装"

print_info "验证关键库是否安装成功..."

REQUIRED_PACKAGES=("pandas" "numpy" "sklearn" "xgboost" "lightgbm" "matplotlib" "plotly")
ALL_SUCCESS=true

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        print_success "$package ✓"
    else
        print_error "$package ✗"
        ALL_SUCCESS=false
    fi
done

# =============================================================================
# 8. 完成
# =============================================================================
print_header "安装完成"

if [ "$ALL_SUCCESS" = true ]; then
    print_success "所有依赖安装成功！"
    echo ""
    print_info "激活虚拟环境命令："
    echo -e "  ${GREEN}source venv/bin/activate${NC}"
    echo ""
    print_info "退出虚拟环境命令："
    echo -e "  ${GREEN}deactivate${NC}"
    echo ""
    print_info "运行项目："
    echo -e "  ${GREEN}python run_gp.py${NC}"
    echo ""
else
    print_warning "部分依赖安装失败，请检查错误信息"
fi

# 显示 Python 环境信息
print_info "当前 Python 环境信息："
echo "  Python 版本: $(python --version)"
echo "  pip 版本: $(pip --version)"
echo "  虚拟环境路径: $(which python)"
echo ""

print_success "环境配置完成！"

