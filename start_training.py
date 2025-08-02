#!/usr/bin/env python3
"""
LaneGCN 训练启动脚本
提供便捷的训练启动和监控功能
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def check_environment():
    """检查基本环境"""
    print("🔍 检查训练环境...")
    
    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA不可用，无法进行GPU训练")
            return False
        print(f"✅ GPU可用: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查数据文件
    data_files = [
        'dataset/preprocess/train_crs_dist6_angle90.p',
        'dataset/preprocess/val_crs_dist6_angle90.p'
    ]
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"❌ 数据文件不存在: {file_path}")
            return False
        size_gb = os.path.getsize(file_path) / (1024**3)
        print(f"✅ {os.path.basename(file_path)}: {size_gb:.1f} GB")
    
    # 检查项目模块
    try:
        import lanegcn
        print("✅ LaneGCN模块可用")
    except ImportError:
        print("❌ LaneGCN模块导入失败")
        return False
    
    return True

def start_training():
    """启动训练"""
    print("\n🚀 启动LaneGCN训练...")
    
    # 创建日志目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"training_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 启动训练
    cmd = [sys.executable, "train.py", "-m", "lanegcn"]
    
    print(f"📝 训练日志将保存到: {log_dir}/")
    print(f"🎯 训练命令: {' '.join(cmd)}")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        # 启动训练进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # 实时显示输出并保存日志
        log_file_path = os.path.join(log_dir, "training.log")
        with open(log_file_path, 'w') as log_file:
            for line in process.stdout:
                print(line.rstrip())  # 实时显示
                log_file.write(line)  # 保存到日志
                log_file.flush()      # 立即写入
        
        process.wait()
        
        if process.returncode == 0:
            print("\n🎉 训练完成！")
        else:
            print(f"\n❌ 训练异常结束，退出码: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️  训练被用户中断")
        process.terminate()
    except Exception as e:
        print(f"\n❌ 训练启动失败: {e}")

def main():
    """主函数"""
    print("LaneGCN 训练启动器")
    print("=" * 40)
    
    # 环境检查
    if not check_environment():
        print("\n❌ 环境检查失败，请先修复环境问题")
        print("\n建议操作:")
        print("1. 运行 conda activate lanegcn")
        print("2. 运行 python test_lanegcn.py 进行详细检查")
        return
    
    print("\n✅ 环境检查通过")
    
    # 确认开始训练
    try:
        response = input("\n是否开始训练？[y/N]: ").lower().strip()
        if response != 'y':
            print("训练已取消")
            return
    except KeyboardInterrupt:
        print("\n训练已取消")
        return
    
    # 开始训练
    start_training()

if __name__ == "__main__":
    main()