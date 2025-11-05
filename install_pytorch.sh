#!/bin/bash

# PyTorch 安装脚本
# 根据系统自动选择合适的PyTorch版本

echo "========================================"
echo "PyTorch 自动安装脚本"
echo "========================================"

# 检测操作系统
OS="$(uname -s)"
echo ""
echo "检测到操作系统: $OS"

# 检测是否有CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到 NVIDIA GPU"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "CUDA 版本: $CUDA_VERSION"
    HAS_CUDA=true
else
    echo "❌ 未检测到 NVIDIA GPU，将安装CPU版本"
    HAS_CUDA=false
fi

echo ""
echo "========================================"
echo "开始安装..."
echo "========================================"

# 根据系统和CUDA情况安装
if [ "$OS" = "Darwin" ]; then
    # macOS
    echo ""
    echo "检测到 macOS 系统"
    
    # 检测是否是Apple Silicon (M1/M2)
    ARCH="$(uname -m)"
    if [ "$ARCH" = "arm64" ]; then
        echo "✅ 检测到 Apple Silicon (M1/M2)"
        echo "安装支持 MPS 加速的 PyTorch..."
        pip install torch torchvision torchaudio
    else
        echo "检测到 Intel Mac"
        echo "安装 CPU 版本 PyTorch..."
        pip install torch torchvision torchaudio
    fi
    
elif [ "$OS" = "Linux" ]; then
    # Linux
    if [ "$HAS_CUDA" = true ]; then
        echo ""
        echo "安装 CUDA 版本 PyTorch..."
        
        # 根据CUDA版本选择
        if [[ "$CUDA_VERSION" == "12."* ]]; then
            echo "安装 CUDA 12.x 版本..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        elif [[ "$CUDA_VERSION" == "11."* ]]; then
            echo "安装 CUDA 11.x 版本..."
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        else
            echo "CUDA版本未知，安装默认版本..."
            pip install torch torchvision torchaudio
        fi
    else
        echo ""
        echo "安装 CPU 版本 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
else
    # Windows (Git Bash/WSL)
    echo ""
    echo "Windows 系统，请手动选择安装方式："
    echo ""
    echo "方式1 - CUDA 版本 (如果有NVIDIA GPU):"
    echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    echo ""
    echo "方式2 - CPU 版本:"
    echo "  pip install torch torchvision torchaudio"
    echo ""
    exit 0
fi

echo ""
echo "========================================"
echo "验证安装..."
echo "========================================"

# 验证安装
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'当前 GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print(f'MPS (Apple Silicon) 可用: True')
else:
    print('使用 CPU')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ PyTorch 安装成功！"
    echo "========================================"
    echo ""
    echo "下一步："
    echo "1. 安装其他依赖: pip install -r requirements.txt"
    echo "2. 快速测试: python tft_quick_start.py"
    echo "3. 完整训练: python tft_main.py"
else
    echo ""
    echo "========================================"
    echo "❌ 安装失败，请检查错误信息"
    echo "========================================"
    exit 1
fi

