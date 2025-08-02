#!/bin/bash

echo "=== LaneGCN 数据下载脚本 (修复版) ==="
echo "注意：原始AWS S3数据源已被破坏，使用替代数据源"

# 创建数据集目录
mkdir -p dataset && cd dataset

echo "步骤1: 下载Argoverse HD Maps..."
# 使用替代数据源
if [ ! -f "hd_maps.tar.gz" ]; then
    echo "正在从替代源下载HD Maps..."
    wget -O hd_maps.tar.gz "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/hd_maps.tar.gz"
    if [ $? -ne 0 ]; then
        echo "警告：HD Maps下载失败，跳过此步骤"
    fi
fi

if [ -f "hd_maps.tar.gz" ]; then
    echo "解压HD Maps..."
    tar xf hd_maps.tar.gz
    
    # 复制地图文件到argoverse API目录
    PY_SITE_PACKAGE_PATH=$(python -c 'import site; print(site.getsitepackages()[0])')
    echo "复制地图文件到: $PY_SITE_PACKAGE_PATH"
    if [ -d "map_files" ]; then
        cp -r map_files $PY_SITE_PACKAGE_PATH
        echo "✓ 地图文件复制完成"
    else
        echo "警告：map_files目录不存在"
    fi
fi

echo ""
echo "步骤2: 下载Argoverse Motion Forecasting数据..."
# 使用替代数据源
DATA_FILES=("forecasting_train_v1.1.tar.gz" "forecasting_val_v1.1.tar.gz" "forecasting_test_v1.1.tar.gz")

for file in "${DATA_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "正在下载 $file..."
        wget -O "$file" "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/$file"
        if [ $? -ne 0 ]; then
            echo "警告：$file 下载失败"
            continue
        fi
    fi
    
    if [ -f "$file" ]; then
        echo "解压 $file..."
        tar xvf "$file"
        echo "✓ $file 解压完成"
    fi
done

echo ""
echo "步骤3: 数据预处理..."
cd ..

# 检查是否有预处理脚本
if [ -f "preprocess_data.py" ]; then
    echo "运行数据预处理（这可能需要几个小时）..."
    python preprocess_data.py -m lanegcn
else
    echo "警告：preprocess_data.py 不存在，跳过预处理步骤"
fi

echo ""
echo "步骤4: 下载预处理的训练数据（推荐）..."
cd dataset

# 下载预处理的训练数据
PREPROCESSED_FILES=("train_crs_dist6_angle90.p" "val_crs_dist6_angle90.p" "test_test.p")

for file in "${PREPROCESSED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "正在下载预处理数据: $file..."
        wget -O "$file" "https://yun.sfo2.cdn.digitaloceanspaces.com/public/lanegcn/$file"
        if [ $? -eq 0 ]; then
            echo "✓ $file 下载完成"
        else
            echo "警告：$file 下载失败"
        fi
    else
        echo "✓ $file 已存在"
    fi
done

cd ..
echo ""
echo "=== 数据下载完成 ==="
echo "如果某些文件下载失败，请检查网络连接或手动下载" 