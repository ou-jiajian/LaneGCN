#!/usr/bin/env python3
"""
LaneGCN 简化测试脚本
用于测试环境是否配置正确，无需复杂的Horovod设置
"""

import os
import sys
import torch
import numpy as np
from importlib import import_module

def test_environment():
    """测试环境配置"""
    print("=== LaneGCN 环境测试 ===")
    
    # 测试PyTorch和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试项目模块导入
    try:
        import lanegcn
        import layers
        import data
        import utils
        print("✓ 所有项目模块导入成功")
    except ImportError as e:
        print(f"✗ 模块导入失败: {e}")
        return False
    
    # 测试模型创建
    try:
        model = import_module("lanegcn")
        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()
        print("✓ 模型创建成功")
        print(f"  批次大小: {config.get('batch_size', 'N/A')}")
        print(f"  学习率: {config.get('lr', 'N/A')}")
        print(f"  训练轮数: {config.get('num_epochs', 'N/A')}")
    except Exception as e:
        print(f"✗ 模型创建失败: {e}")
        return False
    
    # 测试数据加载
    try:
        # 检查预处理数据是否存在
        preprocess_files = [
            'dataset/preprocess/train_crs_dist6_angle90.p',
            'dataset/preprocess/val_crs_dist6_angle90.p',
            'dataset/preprocess/test_test.p'
        ]
        
        for file_path in preprocess_files:
            if os.path.exists(file_path):
                size_gb = os.path.getsize(file_path) / (1024**3)
                print(f"✓ {os.path.basename(file_path)}: {size_gb:.1f} GB")
            else:
                print(f"✗ {file_path} 不存在")
                return False
        
        # 尝试创建数据集
        dataset = Dataset(config["val_split"], config, train=False)
        print(f"✓ 数据集创建成功，包含 {len(dataset)} 个样本")
        
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    return True

def test_simple_training():
    """测试简单的训练步骤"""
    print("\n=== 简单训练测试 ===")
    
    try:
        # 导入模型
        model = import_module("lanegcn")
        config, Dataset, collate_fn, net, loss, post_process, opt = model.get_model()
        
        # 修改配置以支持简单测试
        config["horovod"] = False
        config["batch_size"] = 1  # 减少批次大小
        
        # 创建数据集和数据加载器
        from torch.utils.data import DataLoader
        
        dataset = Dataset(config["val_split"], config, train=False)
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,  # 减少workers
            collate_fn=collate_fn,
            shuffle=False
        )
        
        print(f"✓ 数据加载器创建成功")
        
        # 获取一个批次的数据
        data_iter = iter(data_loader)
        data = next(data_iter)
        data = dict(data)
        
        print(f"✓ 数据批次获取成功")
        print(f"  批次包含的键: {list(data.keys())}")
        
        # 前向传播测试
        net.eval()
        with torch.no_grad():
            output = net(data)
            print(f"✓ 前向传播成功")
            print(f"  输出包含的键: {list(output.keys()) if isinstance(output, dict) else 'tensor'}")
        
        # 损失计算测试
        loss_out = loss(output, data)
        print(f"✓ 损失计算成功")
        print(f"  损失值: {loss_out['loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ 训练测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("LaneGCN 项目测试工具")
    print("=" * 50)
    
    # 环境测试
    env_ok = test_environment()
    
    if not env_ok:
        print("\n❌ 环境配置有问题，请检查安装")
        return
    
    # 训练测试
    train_ok = test_simple_training()
    
    if train_ok:
        print("\n🎉 所有测试通过！")
        print("您的LaneGCN环境配置正确，可以进行训练。")
        print("\n建议的下一步:")
        print("1. 运行 python train.py -m lanegcn 开始完整训练")
        print("2. 如遇问题，请检查GPU内存和数据路径")
    else:
        print("\n❌ 训练测试失败，请检查配置")

if __name__ == "__main__":
    main()