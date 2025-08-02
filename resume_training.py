#!/usr/bin/env python3
"""
LaneGCN 训练恢复脚本
从最新检查点自动恢复训练
"""

import os
import sys
import glob
import subprocess
from datetime import datetime

def find_latest_checkpoint():
    """查找最新的检查点文件"""
    ckpt_pattern = "results/lanegcn/*.ckpt"
    ckpt_files = glob.glob(ckpt_pattern)
    
    if not ckpt_files:
        print("❌ 未找到检查点文件")
        return None
    
    # 按修改时间排序，获取最新的
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return latest_ckpt

def resume_training():
    """恢复训练"""
    print("🔄 LaneGCN 训练恢复工具")
    print("=" * 50)
    
    # 查找最新检查点
    latest_ckpt = find_latest_checkpoint()
    if not latest_ckpt:
        print("请先进行初始训练")
        return
    
    print(f"📁 找到检查点: {latest_ckpt}")
    
    # 获取检查点信息
    try:
        import torch
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        epoch = ckpt.get('epoch', 0)
        print(f"📊 检查点信息:")
        print(f"   轮数: {epoch:.6f}")
        print(f"   文件大小: {os.path.getsize(latest_ckpt) / 1024**2:.1f} MB")
        
        # 计算进度
        total_epochs = 36
        progress = (epoch / total_epochs) * 100
        remaining_epochs = total_epochs - epoch
        
        print(f"📈 训练进度: {progress:.1f}%")
        print(f"⏳ 剩余轮数: {remaining_epochs:.1f}")
        
    except Exception as e:
        print(f"⚠️  无法读取检查点信息: {e}")
    
    # 确认恢复
    print("\\n" + "="*50)
    response = input("是否从此检查点恢复训练？[y/N]: ").lower().strip()
    if response != 'y':
        print("训练恢复已取消")
        return
    
    # 构建恢复命令
    cmd = [sys.executable, "train.py", "-m", "lanegcn", "--resume", latest_ckpt]
    
    print(f"🚀 恢复训练命令: {' '.join(cmd)}")
    print(f"📅 恢复时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    print("💡 提示: 建议在screen会话中运行以避免中断")
    print("   创建screen: screen -S lanegcn_resume")
    print("   分离会话: Ctrl+A, D")
    print("   重新连接: screen -r lanegcn_resume")
    print("=" * 50)
    
    try:
        # 启动恢复训练
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\\n⏹️  训练被用户中断")
    except subprocess.CalledProcessError as e:
        print(f"\\n❌ 训练恢复失败: {e}")
    except Exception as e:
        print(f"\\n❌ 发生错误: {e}")

if __name__ == "__main__":
    resume_training()