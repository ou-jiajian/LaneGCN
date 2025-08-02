# LaneGCN: 基于车道图表示的轨迹预测

> [!CAUTION]
> 原始AWS S3数据源已被破坏，文件可能已损坏。我们已修改相关文件，并注释掉了从该存储桶的检索操作。
> 请谨慎使用，建议使用我们提供的替代数据源。

[论文](https://arxiv.org/pdf/2007.13732) | [幻灯片](http://www.cs.toronto.edu/~byang/slides/LaneGCN.pdf) | [项目页面]() | [**ECCV 2020 Oral** 视频](https://yun.sfo2.digitaloceanspaces.com/public/lanegcn/video.mp4)

**作者**: Ming Liang, Bin Yang, Rui Hu, Yun Chen, Renjie Liao, Song Feng, Raquel Urtasun

**成就**: [Argoverse轨迹预测竞赛](https://evalai.cloudcv.org/web/challenges/challenge-page/454/leaderboard/1279) **第一名**

![架构图](misc/arch.png)

## 📋 目录
- [🚀 快速开始](#-快速开始)
- [🔧 环境配置](#-环境配置)
- [📊 数据准备](#-数据准备)
- [🏋️ 模型训练](#️-模型训练)
- [🧪 模型测试](#-模型测试)
- [📈 性能结果](#-性能结果)
- [❓ 常见问题](#-常见问题)
- [📄 许可证](#-许可证)
- [📚 引用](#-引用)

## 🚀 快速开始

### 一键环境配置（推荐）
```bash
# 1. 创建并激活conda环境
conda create --name lanegcn python=3.7
conda activate lanegcn

# 2. 安装PyTorch（适配RTX 4090D）
conda install pytorch==1.12.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia

# 3. 安装Argoverse API
pip install git+https://github.com/argoai/argoverse-api.git

# 4. 安装其他依赖
pip install scikit-image IPython tqdm ipdb scikit-learn
```

### 一键数据下载
```bash
# 使用修复版数据下载脚本
bash get_data_fixed.sh
```

### 一键环境检查
```bash
# 检查环境配置
python check_env.py

# 快速测试模型
python test_lanegcn.py
```

### 一键开始训练
```bash
# 使用便捷训练脚本
python start_training.py

# 或直接运行
python train.py -m lanegcn
```

## 🔧 环境配置

### 系统要求
- **Python**: 3.7
- **CUDA**: 10.2+ (推荐11.6用于RTX 4090D)
- **GPU**: 支持CUDA的NVIDIA显卡
- **内存**: 建议16GB+

### 详细安装步骤

#### 1. 创建Conda环境
```bash
conda create --name lanegcn python=3.7
conda activate lanegcn
```

#### 2. 安装PyTorch
**RTX 4090D用户（推荐）**:
```bash
conda install pytorch==1.12.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

**其他显卡**:
```bash
conda install pytorch==1.5.1 torchvision cudatoolkit=10.2 -c pytorch
```

#### 3. 安装Argoverse API
```bash
# 方法1：直接安装
pip install git+https://github.com/argoai/argoverse-api.git

# 方法2：如果遇到sklearn错误
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install git+https://github.com/argoai/argoverse-api.git
```

#### 4. 安装其他依赖
```bash
pip install scikit-image IPython tqdm ipdb scikit-learn
```

#### 5. 安装Horovod（可选，用于多GPU训练）
```bash
# 单GPU用户（代码兼容性）
pip install horovod

# 多GPU用户
pip install mpi4py
HOROVOD_GPU_OPERATIONS=NCCL pip install horovod==0.19.4
```

## 📊 数据准备

### 自动下载（推荐）
```bash
# 使用修复版脚本，自动下载所有数据
bash get_data_fixed.sh
```

### 手动下载
如果自动下载失败，可以手动下载：

1. **HD地图数据**:
   ```bash
   wget -O dataset/hd_maps.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/hd_maps.tar.gz"
   cd dataset && tar xf hd_maps.tar.gz
   ```

2. **轨迹预测数据**:
   ```bash
   cd dataset
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_train_v1.1.tar.gz"
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_val_v1.1.tar.gz"
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/forecasting_test_v1.1.tar.gz"
   tar xf forecasting_train_v1.1.tar.gz
   tar xf forecasting_val_v1.1.tar.gz
   tar xf forecasting_test_v1.1.tar.gz
   ```

3. **预处理数据（推荐）**:
   ```bash
   cd dataset
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/train_crs_dist6_angle90.p"
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/val_crs_dist6_angle90.p"
   wget "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/test_test.p"
   mkdir -p preprocess
   mv *.p preprocess/
   ```

## 🏋️ 模型训练

### 单GPU训练（推荐）
```bash
# 使用便捷脚本
python start_training.py

# 或直接运行
python train.py -m lanegcn
```

### 多GPU训练
```bash
# 单节点4GPU
horovodrun -np 4 -H localhost:4 python train.py -m lanegcn

# 2节点，每节点4GPU
horovodrun -np 8 -H serverA:4,serverB:4 python train.py -m lanegcn
```

### 训练恢复
```bash
# 从最新检查点恢复
python resume_training.py

# 或指定检查点
python train.py -m lanegcn --resume 10.000.ckpt
```

### 训练时间估算
- **RTX 4090D**: 约4-6小时（单GPU）
- **RTX 5000**: 约8小时（4GPU）
- **检查点保存**: 每1000步保存一次

## 🧪 模型测试

### 下载预训练模型
```bash
wget -O 36.000.ckpt "http://yun.sfo2.digitaloceanspaces.com/public/lanegcn/36.000.ckpt"
```

### 测试集推理（提交用）
```bash
python test.py -m lanegcn --weight=36.000.ckpt --split=test
```

### 验证集推理（指标评估）
```bash
python test.py -m lanegcn --weight=36.000.ckpt --split=val
```

## 📈 性能结果

### 定性结果
**标签（红色）预测（绿色）其他智能体（蓝色）**

<p>
<img src="misc/5304.gif" width = "30.333%"  align="left" />
<img src="misc/25035.gif" width = "30.333%" align="center"  />
 <img src="misc/19406.gif" width = "30.333%" align="right"   />
</p>

### 定量结果
![定量结果](misc/res_quan.png)

## ❓ 常见问题

### 环境问题
**Q: Conda激活失败**
```bash
# 解决方案
conda init bash
source ~/.bashrc
conda activate lanegcn
```

**Q: CUDA版本不兼容**
```bash
# RTX 4090D用户使用
conda install pytorch==1.12.1 torchvision pytorch-cuda=11.6 -c pytorch -c nvidia
```

### 数据问题
**Q: 数据下载失败**
- 使用 `get_data_fixed.sh` 脚本
- 检查网络连接
- 使用VPN或代理

**Q: 预处理数据缺失**
```bash
# 创建必要的目录结构
mkdir -p dataset/{train,val,test}/data
touch dataset/{train,val,test}/data/dummy.csv
```

### 训练问题
**Q: Horovod初始化错误**
- 已修复，支持单GPU训练
- 使用 `python train.py -m lanegcn` 即可

**Q: 训练中断**
```bash
# 使用screen后台运行
screen -S lanegcn
python train.py -m lanegcn
# Ctrl+A+D 分离

# 恢复训练
python resume_training.py
```

**Q: 内存不足**
- 减少batch_size
- 使用梯度累积
- 检查GPU内存使用情况

### 模块导入问题
**Q: No module named 'argoverse'**
```bash
pip install git+https://github.com/argoai/argoverse-api.git --force-reinstall
```

**Q: No module named 'lanegcn'**
- 确保在项目根目录运行
- 检查Python路径设置

## 📄 许可证
查看 [LICENSE](LICENSE) 文件

## 📚 引用
如果您使用了我们的代码，请考虑引用以下论文：
```bibtex
@InProceedings{liang2020learning,
  title={Learning lane graph representations for motion forecasting},
  author={Liang, Ming and Yang, Bin and Hu, Rui and Chen, Yun and Liao, Renjie and Feng, Song and Urtasun, Raquel},
  booktitle = {ECCV},
  year={2020}
}
```

## 🤝 贡献
如果您有任何问题或建议，请：
1. 查看 [LaneGCN_使用说明.md](LaneGCN_使用说明.md) 获取详细帮助
2. 运行 `python check_env.py` 检查环境
3. 在GitHub上提交Issue

## 📞 联系方式
如有代码相关问题，请联系 [@chenyuntc](https://github.com/chenyuntc)

---

**注意**: 本项目已针对RTX 4090D等现代显卡进行了优化，支持单GPU训练，并提供了完整的中文文档和便捷工具。 