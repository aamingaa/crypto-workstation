# 安装指南

本文档说明如何快速设置 Crypto Workstation 的 Python 环境。

## 📋 前置要求

- **Linux 系统** （Ubuntu/Debian 或 CentOS/RHEL/Fedora）
- **Python 3.8** （推荐版本，脚本会自动安装）
- **sudo 权限**（用于安装系统软件包）

## 🚀 快速安装（推荐）

### 使用自动化脚本

```bash
# 一键安装（会自动检测并安装 Python 3.8）
bash setup_env.sh
```

该脚本会自动完成：
1. ✅ 检测系统是否有 Python 3.8
2. ✅ 如果没有，自动安装 Python 3.8
3. ✅ 创建虚拟环境
4. ✅ 安装所有依赖包
5. ✅ 验证安装结果

### 手动安装

如果你已经有 Python 3.8：

```bash
# Ubuntu/Debian - 确保有 Python 3.8
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip

# CentOS/RHEL - 确保有 Python 3.8
sudo yum install -y python38 python38-pip python38-devel

# 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 📦 安装脚本功能

自动化安装脚本 (`setup_env.sh`) 会自动完成以下步骤：

1. ✅ 检测系统是否有 Python 3.8，如没有则自动安装
2. ✅ 支持 Ubuntu/Debian (apt) 和 CentOS/RHEL (yum)
3. ✅ 创建 Python 3.8 虚拟环境 (venv)
4. ✅ 激活虚拟环境
5. ✅ 升级 pip 到最新版本
6. ✅ 可选使用清华镜像源加速下载
7. ✅ 安装所有 Python 依赖包
8. ✅ 验证关键库是否安装成功
9. ✅ 显示完整的环境信息

## 🔧 TA-Lib 安装说明

TA-Lib 是一个技术分析库，需要先安装系统级依赖。

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install build-essential wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# 激活虚拟环境后安装 Python 包
source venv/bin/activate
pip install TA-Lib
```

### CentOS/RHEL

```bash
sudo yum install gcc gcc-c++ make wget
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install

# 激活虚拟环境后安装 Python 包
source venv/bin/activate
pip install TA-Lib
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

```bash
source venv/bin/activate
```

激活后，命令提示符前会显示 `(venv)`。

### 退出虚拟环境

```bash
deactivate
```

### 运行项目

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 运行主程序
python run_gp.py

# 或运行 GP 主程序
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

如果脚本提示找不到 Python，可以手动安装：

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.8 python3.8-venv python3.8-dev python3-pip

# CentOS/RHEL
sudo yum install -y python38 python38-pip python38-devel

# 验证安装
python3.8 --version
```

### 2. pip 安装包失败

- 检查网络连接
- 尝试使用国内镜像源（脚本会自动询问）
- 确保 pip 已升级：`pip install --upgrade pip`
- 检查是否在虚拟环境中：命令提示符前应显示 `(venv)`

### 3. TA-Lib 安装失败

TA-Lib 需要先安装系统级依赖和编译工具：

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CentOS/RHEL
sudo yum install gcc gcc-c++ make
```

然后按照上面的 TA-Lib 安装说明操作。

### 4. 权限错误

```bash
# 如果脚本无法执行，添加执行权限
chmod +x setup_env.sh

# 如果安装包时提示权限错误，确保：
# 1. 使用 sudo 安装系统包
# 2. 在虚拟环境中安装 Python 包（不需要 sudo）
```

### 5. 虚拟环境激活失败

```bash
# 确保在项目根目录下执行
cd /path/to/crypto-workstation
source venv/bin/activate

# 如果虚拟环境损坏，删除重建
rm -rf venv
python3.8 -m venv venv
source venv/bin/activate
```

### 6. Python 版本不匹配

如果系统默认 Python 版本不是 3.8：

```bash
# 显式使用 python3.8 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 验证虚拟环境中的 Python 版本
python --version  # 应该显示 Python 3.8.x
```

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

