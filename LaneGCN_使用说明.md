# LaneGCN 项目使用说明

## 📋 项目概述

LaneGCN (Learning Lane Graph Representations for Motion Forecasting) 是一个用于运动预测的车道图表示学习项目。该项目在 Argoverse Motion Forecasting Competition 中获得了第一名。

**论文**: [Learning Lane Graph Representations for Motion Forecasting](https://arxiv.org/pdf/2007.13732)

## 🖥️ 环境要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 RTX 4090 或更高)
- **内存**: 至少 16GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+)
- **Python**: 3.7
- **CUDA**: 11.6 (通过conda环境提供)

## 🚀 快速开始

### 1. 环境配置

#### 1.1 创建和激活conda环境
```bash
# 创建Python 3.7环境
conda create --name lanegcn python=3.7 -y

# 初始化conda (如果遇到激活问题)
conda init bash

# 重新加载shell配置
source ~/.bashrc

# 激活环境
conda activate lanegcn
```

#### 1.2 验证环境激活
```bash
# 确认环境已激活 (应该看到(lanegcn)前缀)
python --version  # 应该显示Python 3.7.x

# 如果仍然无法激活，尝试以下方法：
# 方法1: 使用完整路径
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lanegcn

# 方法2: 重新初始化conda
conda init bash
exec bash
conda activate lanegcn
```

#### 1.3 安装PyTorch (支持RTX 4090)
```bash
# 安装PyTorch 1.12.1 + CUDA 11.6
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

#### 1.4 安装Argoverse API
```bash
# 安装Argoverse API
pip install git+https://github.com/argoai/argoverse-api.git
```

#### 1.5 安装其他依赖
```bash
# 安装项目依赖包
pip install scikit-image IPython tqdm ipdb scikit-learn
```

#### 1.6 验证环境
```bash
# 测试PyTorch和GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"无GPU\"}')"

# 测试项目模块
python -c "import lanegcn, layers, data, utils; print('所有模块导入成功')"
```

### 2. 数据准备

#### 2.1 下载数据 (推荐使用修复版脚本)
```bash
# 使用修复版数据下载脚本
bash get_data_fixed.sh
```

#### 2.2 手动下载数据 (如果脚本失败)
```bash
# 创建数据集目录
mkdir -p dataset && cd dataset

# 下载HD Maps
wget -O hd_maps.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/hd_maps.tar.gz"
tar xf hd_maps.tar.gz

# 下载运动预测数据
wget -O forecasting_train_v1.1.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_train_v1.1.tar.gz"
wget -O forecasting_val_v1.1.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_val_v1.1.tar.gz"
wget -O forecasting_test_v1.1.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_test_v1.1.tar.gz"

# 解压数据
tar xvf forecasting_train_v1.1.tar.gz
tar xvf forecasting_val_v1.1.tar.gz
tar xvf forecasting_test_v1.1.tar.gz

# 返回项目根目录
cd ..

# 复制地图文件到Python包目录
PY_SITE_PACKAGE_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -r dataset/map_files $PY_SITE_PACKAGE_PATH
```

#### 2.3 数据预处理
```bash
# 运行数据预处理 (需要几个小时)
python preprocess_data.py -m lanegcn

# 或者下载预处理的训练数据 (推荐)
cd dataset
wget -O train_crs_dist6_angle90.p "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/train_crs_dist6_angle90.p"
wget -O val_crs_dist6_angle90.p "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/val_crs_dist6_angle90.p"
wget -O test_test.p "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/test_test.p"
cd ..
```

### 3. 模型训练

#### 3.1 环境检查（推荐先运行）
```bash
# 运行完整环境检查
python check_env.py

# 或运行简化训练测试
python test_lanegcn.py
```

#### 3.2 单GPU训练
```bash
# 方式1：便捷启动脚本（推荐）
python start_training.py

# 方式2：直接启动训练（已修复Horovod兼容性问题）
python train.py -m lanegcn

# 方式3：从检查点恢复训练
python train.py -m lanegcn --resume 1.000.ckpt  # 恢复第1轮
python train.py -m lanegcn --resume 2.000.ckpt  # 恢复第2轮

# 注意：训练会自动创建日志目录并保存模型检查点
# 训练时间较长，建议使用 screen 或 tmux 在后台运行
```

#### 3.3 多GPU训练 (可选)
```bash
# 安装Horovod (可选，用于多GPU训练)
pip install mpi4py
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4

# 4个GPU训练
horovodrun -np 4 -H localhost:4 python train.py -m lanegcn
```

#### 3.4 训练参数说明
- `-m lanegcn`: 指定模型名称
- 训练时间: 在4个RTX 5000上约需8小时
- 模型保存: 训练过程中会自动保存检查点

### 4. 模型测试

#### 4.1 下载预训练模型
```bash
# 下载官方预训练模型
wget -O 36.000.ckpt "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/36.000.ckpt"
```

#### 4.2 测试验证集
```bash
# 在验证集上测试
python test.py -m lanegcn --weight=36.000.ckpt --split=val
```

#### 4.3 测试测试集
```bash
# 在测试集上生成提交文件
python test.py -m lanegcn --weight=36.000.ckpt --split=test
```

## 📁 项目结构

```
LaneGCN/
├── lanegcn.py          # 主模型文件
├── layers.py           # 网络层定义
├── data.py             # 数据加载器
├── utils.py            # 工具函数
├── train.py            # 训练脚本
├── test.py             # 测试脚本
├── preprocess_data.py  # 数据预处理脚本
├── get_data.sh         # 原始数据下载脚本
├── get_data_fixed.sh   # 修复版数据下载脚本
├── test_lanegcn.py     # 环境测试脚本  
├── check_env.py        # 完整环境检查脚本
├── start_training.py   # 便捷训练启动脚本
├── resume_training.py  # 训练恢复脚本
├── dataset/            # 数据集目录
│   ├── preprocess/     # 预处理数据文件
│   ├── map_files/      # 地图文件
│   ├── train/          # 训练数据
│   ├── val/            # 验证数据
│   └── test/           # 测试数据
└── LaneGCN_使用说明.md  # 本说明文档
```

## 🔧 常见问题解决

### 1. CUDA版本不兼容
**问题**: 系统CUDA版本过高，项目要求CUDA 10.2
**解决**: 使用conda环境安装PyTorch 1.12.1 + CUDA 11.6，向下兼容

### 2. 数据下载失败
**问题**: 原始AWS S3数据源被破坏
**解决**: 使用修复版脚本 `get_data_fixed.sh` 或手动下载

### 3. 内存不足
**问题**: 训练时GPU内存不足
**解决**: 
- 减少batch size
- 使用梯度累积
- 使用多GPU训练

### 4. 模块导入错误
**问题**: `ModuleNotFoundError: No module named 'argoverse'`
**解决**: 重新安装Argoverse API
```bash
pip install git+https://github.com/argoai/argoverse-api.git --force-reinstall
```

### 5. 训练脚本Horovod错误
**问题**: `ValueError: Horovod has not been initialized; use hvd.init().` 或 `NameError: name 'comm' is not defined`
**解决**: 
- 项目的 `train.py` 已全面修复Horovod兼容性问题
- 修复了所有MPI通信相关代码（hvd.rank, hvd.size, comm.allgather等）
- 现在完全支持单GPU训练，无需配置Horovod
- 训练开始时会显示 "Horovod not available, using single GPU training"
- 如仍遇到问题，请先运行 `python test_lanegcn.py` 验证环境

### 6. 训练中断和恢复
**问题**: 训练过程中出现 "Killed" 或意外中断
**原因**: 
- 系统资源管理策略
- SSH连接断开
- 云服务器自动重启

**解决方案**:
```bash
# 方法1：从最新检查点恢复（推荐）
python train.py -m lanegcn --resume 1.000.ckpt

# 方法2：使用恢复脚本
python resume_training.py

# 方法3：使用screen防止中断
screen -S training
python train.py -m lanegcn --resume 1.000.ckpt
# 按 Ctrl+A, D 分离会话
```

**检查训练进度**:
```bash
# 查看保存的检查点
ls -la results/lanegcn/*.ckpt

# 查看GPU使用情况
nvidia-smi

# 重新连接screen会话
screen -r training
```

### 7. Conda环境激活问题
**问题**: `CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'`
**解决步骤**:
```bash
# 步骤1: 初始化conda
conda init bash

# 步骤2: 重新加载shell配置
source ~/.bashrc

# 步骤3: 激活环境
conda activate lanegcn
```

**如果上述方法不工作，尝试**:
```bash
# 方法1: 使用完整路径
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lanegcn

# 方法2: 重新启动shell
exec bash
conda activate lanegcn

# 方法3: 检查conda安装路径
which conda
# 如果路径不同，使用实际路径
source /实际路径/etc/profile.d/conda.sh
conda activate lanegcn
```

## 📊 性能指标

### 官方结果
- **ADE**: 0.71
- **FDE**: 1.09
- **Competition Rank**: 1st

### 训练配置
- **官方配置**: 4x RTX 5000, ~8小时
- **单GPU RTX 4090D**: ~30-32小时 (实测)
- **训练速度**: 2-4 it/s (取决于系统负载)
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **总样本数**: 205,942
- **每轮批次数**: 6,436

## 🎯 使用建议

1. **首次使用**: 
   - 先运行 `python test_lanegcn.py` 验证环境配置
   - 建议下载预处理的训练数据，避免长时间预处理

2. **训练步骤**:
   ```bash
   # 第一步：激活环境
   conda activate lanegcn
   
   # 第二步：验证环境（重要！）
   python test_lanegcn.py
   
   # 第三步：开始训练
   python train.py -m lanegcn
   ```

3. **后台运行**: 训练时间较长（数小时），建议使用screen或tmux：
   ```bash
   # 创建screen会话
   screen -S lanegcn_training
   
   # 在screen中运行训练
   python train.py -m lanegcn
   
   # 按 Ctrl+A, D 分离会话
   # 重新连接: screen -r lanegcn_training
   ```

4. **监控训练**: 查看保存目录中的日志文件和检查点

## 📞 技术支持

- **GitHub**: [LaneGCN Repository](https://github.com/uber-research/LaneGCN)
- **论文**: [ECCV 2020 Paper](https://arxiv.org/pdf/2007.13732)
- **问题反馈**: 在GitHub上提交Issue

---

**注意**: 本说明文档针对RTX 4090D + CUDA 12.4环境进行了优化，确保在您的硬件上能够正常运行。 