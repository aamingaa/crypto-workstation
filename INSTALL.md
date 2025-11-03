# 安装指南

本文档说明如何快速设置 Crypto Workstation 的 Python 环境。

## 📋 前置要求

- **Python 3.8+** （推荐 Python 3.9 或 3.10）
- **pip** （Python 包管理器）
- **macOS/Linux**: Homebrew（可选，用于安装 TA-Lib）
- **Windows**: 无特殊要求

## 🚀 快速安装

### macOS / Linux

```bash
# 方法 1: 运行安装脚本（推荐）
bash setup_env.sh

# 方法 2: 手动安装
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows

```batch
REM 方法 1: 运行安装脚本（推荐）
setup_env.bat

REM 方法 2: 手动安装
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 📦 安装脚本功能

自动化安装脚本 (`setup_env.sh` / `setup_env.bat`) 会自动完成以下步骤：

1. ✅ 检查 Python 版本（需要 3.8+）
2. ✅ 创建虚拟环境 (venv)
3. ✅ 激活虚拟环境
4. ✅ 升级 pip 到最新版本
5. ✅ 检查系统依赖（如 TA-Lib）
6. ✅ 安装所有 Python 依赖包
7. ✅ 验证关键库是否安装成功
8. ✅ 显示环境信息

## 🔧 TA-Lib 安装说明

TA-Lib 是一个技术分析库，需要先安装系统级依赖。

### macOS

```bash
brew install ta-lib
pip install TA-Lib
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

### Windows

下载预编译的 wheel 文件：
https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

```batch
pip install TA_Lib‑0.4.xx‑cpxx‑cpxxm‑win_amd64.whl
```

## 🌐 使用国内镜像加速

如果下载速度慢，可以使用清华镜像源：

```bash
# 安装时使用镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或设置为默认镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

安装脚本会自动询问是否使用镜像源。

## ✨ 激活和使用环境

### 激活虚拟环境

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```batch
venv\Scripts\activate.bat
```

### 退出虚拟环境

```bash
deactivate
```

### 运行项目

```bash
# 确保虚拟环境已激活
python run_gp.py

# 或运行主程序
python gp_crypto_next/main_gp_new.py
```

## 📝 依赖说明

主要依赖包括：

| 类别 | 包名 | 说明 |
|------|------|------|
| 数据处理 | pandas, numpy | 核心数据处理 |
| 机器学习 | scikit-learn, xgboost, lightgbm | 机器学习模型 |
| 深度学习 | keras | 神经网络 |
| 技术分析 | ta, TA-Lib | 技术指标计算 |
| 可视化 | matplotlib, plotly, seaborn | 数据可视化 |
| 加密货币 | cryptofeed, tardis-dev, binance | 数据源 |
| 异步网络 | aiohttp, nest-asyncio | 异步处理 |
| 工具 | loguru, tqdm, pyyaml | 日志和工具 |

完整的依赖列表请查看 `requirements.txt` 文件。

## 🐛 常见问题

### 1. 找不到 Python 命令

确保 Python 已安装并添加到系统 PATH。

```bash
# 检查 Python 版本
python --version  # Windows
python3 --version # macOS/Linux
```

### 2. pip 安装包失败

- 检查网络连接
- 尝试使用国内镜像源
- 确保 pip 已升级：`pip install --upgrade pip`

### 3. TA-Lib 安装失败

TA-Lib 需要先安装系统级依赖，请参考上面的 TA-Lib 安装说明。

### 4. 权限错误

**macOS/Linux:**
```bash
# 如果脚本无法执行，添加执行权限
chmod +x setup_env.sh
```

**Windows:**
以管理员身份运行命令提示符或 PowerShell。

### 5. 虚拟环境激活失败

确保在项目根目录下执行命令，虚拟环境应该在 `venv/` 目录中。

## 📚 更多信息

- 项目主 README: [README.md](README.md)
- 性能优化说明: [性能优化说明.md](性能优化说明.md)
- GP 优化指南: [gp_crypto_next/OPTIMIZATION_GUIDE.md](gp_crypto_next/OPTIMIZATION_GUIDE.md)

## 🆘 获取帮助

如果遇到问题：

1. 检查 Python 版本是否 >= 3.8
2. 确保虚拟环境已正确激活
3. 查看安装脚本的输出信息
4. 检查错误日志

---

**祝使用愉快！** 🚀

