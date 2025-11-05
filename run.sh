#!/bin/bash

# GPU碰撞检测系统 - 快速启动脚本

echo "======================================================================="
echo "GPU Collision Detection System - Quick Launcher"
echo "======================================================================="
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "虚拟环境未找到。正在创建..."
    python3 -m venv venv
    echo "✓ 虚拟环境已创建"
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source venv/bin/activate

# 检查依赖
echo ""
echo "检查依赖..."
if ! python -c "import cupy" 2>/dev/null; then
    echo "CuPy未安装。正在安装依赖..."
    echo "请确保已安装CUDA！"
    echo ""
    
    # 检测CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1/p')
        echo "检测到CUDA版本: $CUDA_VERSION.x"
        
        if [ "$CUDA_VERSION" = "11" ]; then
            pip install cupy-cuda11x
        elif [ "$CUDA_VERSION" = "12" ]; then
            pip install cupy-cuda12x
        else
            echo "不支持的CUDA版本: $CUDA_VERSION"
            echo "请手动安装CuPy"
            exit 1
        fi
    else
        echo "未检测到CUDA！请先安装CUDA Toolkit。"
        exit 1
    fi
    
    pip install numpy scipy matplotlib opencv-python
    echo "✓ 依赖安装完成"
else
    echo "✓ CuPy已安装"
fi

# 菜单
echo ""
echo "======================================================================="
echo "请选择要运行的程序:"
echo "======================================================================="
echo "1. 系统验证测试 (推荐首次运行)"
echo "2. 重力下落场景 (主示例)"
echo "3. 性能基准测试"
echo "4. 退出"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "运行系统验证测试..."
        python tests/simple_test.py
        ;;
    2)
        echo ""
        echo "运行重力下落场景..."
        echo "这将需要5-10分钟，并生成视频文件。"
        echo ""
        read -p "按Enter继续，或Ctrl+C取消..."
        python examples/gravity_fall.py
        ;;
    3)
        echo ""
        echo "运行性能基准测试..."
        echo "这将需要10-20分钟。"
        echo ""
        read -p "按Enter继续，或Ctrl+C取消..."
        python tests/benchmark.py
        ;;
    4)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "======================================================================="
echo "完成！"
echo "======================================================================="
echo ""
echo "输出文件位于: output/"
echo "查看README.md了解更多信息"
echo ""
