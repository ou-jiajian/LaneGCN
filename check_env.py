#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LaneGCN 环境检查脚本
用于验证环境配置是否正确
"""

import sys
import os

def print_header(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")

def print_success(msg):
    print(f"✓ {msg}")

def print_error(msg):
    print(f"✗ {msg}")

def print_warning(msg):
    print(f"⚠ {msg}")

def check_python_version():
    """检查Python版本"""
    print_header("Python版本检查")
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 7:
        print_success("Python版本符合要求 (3.7)")
        return True
    else:
        print_error(f"Python版本不符合要求，需要Python 3.7，当前为{version.major}.{version.minor}")
        return False

def check_conda_env():
    """检查conda环境"""
    print_header("Conda环境检查")
    
    # 检查是否在conda环境中
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print_success(f"当前在conda环境中: {conda_env}")
        if conda_env == 'lanegcn':
            print_success("环境名称正确")
            return True
        else:
            print_warning(f"建议使用lanegcn环境，当前为{conda_env}")
            return True
    else:
        print_error("未检测到conda环境")
        return False

def check_pytorch():
    """检查PyTorch"""
    print_header("PyTorch检查")
    
    try:
        import torch
        print_success(f"PyTorch版本: {torch.__version__}")
        
        # 检查CUDA
        if torch.cuda.is_available():
            print_success("CUDA可用")
            print(f"  CUDA版本: {torch.version.cuda}")
            print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
            # 测试GPU计算
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.mm(x, x.t())
                print_success("GPU计算测试通过")
            except Exception as e:
                print_error(f"GPU计算测试失败: {e}")
                return False
        else:
            print_warning("CUDA不可用，将使用CPU训练")
        
        return True
    except ImportError as e:
        print_error(f"PyTorch导入失败: {e}")
        return False

def check_argoverse():
    """检查Argoverse API"""
    print_header("Argoverse API检查")
    
    try:
        import argoverse
        print_success("Argoverse API导入成功")
        print(f"  安装路径: {argoverse.__file__}")
        return True
    except ImportError as e:
        print_error(f"Argoverse API导入失败: {e}")
        return False

def check_project_modules():
    """检查项目模块"""
    print_header("项目模块检查")
    
    modules = ['lanegcn', 'layers', 'data', 'utils']
    all_success = True
    
    for module in modules:
        try:
            __import__(module)
            print_success(f"{module}.py 导入成功")
        except ImportError as e:
            print_error(f"{module}.py 导入失败: {e}")
            all_success = False
    
    return all_success

def check_dependencies():
    """检查其他依赖"""
    print_header("其他依赖检查")
    
    dependencies = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('cv2', 'OpenCV'),
        ('skimage', 'Scikit-Image'),
        ('IPython', 'IPython'),
        ('tqdm', 'tqdm')
    ]
    
    all_success = True
    for module, name in dependencies:
        try:
            __import__(module)
            print_success(f"{name} 导入成功")
        except ImportError as e:
            print_error(f"{name} 导入失败: {e}")
            all_success = False
    
    return all_success

def check_data_directory():
    """检查数据目录"""
    print_header("数据目录检查")
    
    if os.path.exists('dataset'):
        print_success("dataset目录存在")
        
        # 检查子目录
        subdirs = ['map_files', 'train', 'val', 'test']
        for subdir in subdirs:
            path = os.path.join('dataset', subdir)
            if os.path.exists(path):
                print_success(f"  {subdir}/ 目录存在")
            else:
                print_warning(f"  {subdir}/ 目录不存在")
    else:
        print_warning("dataset目录不存在，需要下载数据")
    
    return True

def main():
    """主函数"""
    print("LaneGCN 环境检查工具")
    print("="*50)
    
    checks = [
        check_python_version,
        check_conda_env,
        check_pytorch,
        check_argoverse,
        check_project_modules,
        check_dependencies,
        check_data_directory
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print_error(f"检查过程中出错: {e}")
            results.append(False)
    
    # 总结
    print_header("检查总结")
    passed = sum(results)
    total = len(results)
    
    print(f"通过检查: {passed}/{total}")
    
    if passed == total:
        print_success("🎉 所有检查通过！环境配置正确，可以开始使用LaneGCN项目。")
        print("\n建议的下一步操作:")
        print("1. 运行 bash get_data_fixed.sh 下载数据")
        print("2. 运行 python train.py -m lanegcn 开始训练")
    else:
        print_error("⚠️ 部分检查失败，请根据上述错误信息修复环境配置。")
        print("\n常见解决方案:")
        print("1. 确保已激活lanegcn环境: conda activate lanegcn")
        print("2. 重新安装PyTorch: pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116")
        print("3. 重新安装Argoverse: pip install git+https://github.com/argoai/argoverse-api.git --force-reinstall")

if __name__ == "__main__":
    main() 