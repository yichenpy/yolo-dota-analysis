# YOLO11 航空图像目标检测工具集

基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) 的航空遥感目标检测训练与分析工具集，包含两个独立子项目。

## 子项目

### [yolo_dota_project/](./yolo_dota_project/)

YOLO11 OBB（旋转框）在 DOTA / DOTAv1.5 / DIOR 航空数据集上的训练与分析框架，主要特性：

- 支持 baseline、自定义架构（P2 检测头）、断点续训三种训练模式
- 自定义模块：CBAM 通道空间注意力、残差多尺度特征融合（ResidualCBFuse）
- 独立分析脚本：目标尺寸分布、漏检/虚检分析、检测头特征图分析

### [yolo11_analysis_app/](./yolo11_analysis_app/)

基于 Streamlit 的本地 YOLO11 检测分析 Web 应用，主要特性：

- 数据集统计分析（尺寸分布、类别分布、小/中/大目标占比）
- 漏检/虚检分析（自定义 IoU 匹配，分类别 TP/FP/FN 统计）
- 指标分析（Precision / Recall / mAP@0.5 / mAP@0.5:0.95，PR 曲线，混淆矩阵）
- 模型结构与特征图可视化
- 完整分析历史快照管理

## 环境依赖

```
Python 3.10 或 3.11
PyTorch >= 2.1（推荐 CUDA 版本）
ultralytics >= 8.3
```

## 快速开始

```bash
# 训练项目
cd yolo_dota_project
pip install -r requirements.txt
python train.py --data datasets/DOTA-split-lite/data.yaml --model yolo11s-obb.pt

# 分析 App
cd yolo11_analysis_app
pip install -r requirements.txt
streamlit run app.py
```

## 许可证

MIT License
