# YOLO11 Detection Analysis App

基于 Streamlit + Ultralytics YOLO11 的本地目标检测分析工具，支持自定义数据集和训练权重，不依赖 Ultralytics 官方 `val()` 黑盒。

## 功能概览

| 模块 | 主要内容 |
|------|----------|
| **数据集分析** | 图像宽高、长宽比、目标尺寸分布（原图/缩放后）、按类别统计、小/中/大目标占比 |
| **预测结果分析** | 预测框明细、类别分布、置信度分布、单图可视化 |
| **漏检/虚检分析** | 自定义 IoU 匹配、TP/FP/FN 统计、分类别漏检率/虚检率、单图 GT/TP/FN/FP 可视化 |
| **指标分析** | Precision / Recall / mAP@0.5 / mAP@0.5:0.95、PR 曲线、AP 柱状图、混淆矩阵、CSV/JSON/PNG 导出 |
| **历史快照** | 保存/加载完整分析快照，支持多次运行对比 |
| **模型分析** | 层级结构摘要（Backbone/Neck/Head）、特征图提取、激活热力图、指定通道可视化 |

## 安装

```bash
cd yolo11_analysis_app
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

> 如需 GPU 推理，请按本机 CUDA 版本安装对应的 `torch`，参考 [PyTorch 官网](https://pytorch.org/get-started/locally/)。

## 运行

```bash
cd yolo11_analysis_app
streamlit run app.py
```

启动后浏览器自动打开，默认地址 `http://localhost:8501`。

## 使用方法

1. **配置输入**（左侧栏）：
   - 权重文件路径（`best.pt`）
   - `data.yaml` 路径
   - 图像目录 / 标签目录
   - 可选：验证/测试集图像和标签目录

2. **设置推理参数**：`imgsz`、`conf`、`iou`、`device`、`max_det`

3. **设置匹配参数**：匹配 IoU 阈值、小/中/大目标尺寸阈值

4. **切换顶部标签页**查看各模块分析结果

5. **保存分析快照**：在左侧栏填写备注后点击"保存当前分析快照"

6. **查看历史**：在左侧栏切换到"历史运行"，选择快照复用

## 数据格式支持

- **标签格式**：YOLO 标准格式 `class cx cy w h`（水平框）和 OBB 格式 `class x1 y1 x2 y2 x3 y3 x4 y4`（旋转框）
- **数据集配置**：Ultralytics 标准 `data.yaml`（含 `path`、`train`、`val`、`names` 字段）

## 与官方 `val()` 的区别

本工具自行实现 GT/预测匹配与 AP 计算，主要优势：

- 可调整 IoU 匹配阈值进行自定义分析
- 提供图片级、类别级、尺寸级的细粒度漏检/虚检拆解
- 分析结果可完整保存为快照，便于跨实验比较

## 项目结构

```
yolo11_analysis_app/
├── app.py                   # Streamlit 主入口
├── requirements.txt
├── configs/
│   └── example_analysis.yaml   # 参数配置示例
└── yolo11_analysis/
    ├── schemas.py           # 数据类定义
    ├── io.py                # 数据集加载
    ├── inference.py         # 推理封装
    ├── matching.py          # GT-预测匹配
    ├── error_analysis.py    # 漏检/虚检分析
    ├── metrics.py           # AP / 混淆矩阵
    ├── model_analysis.py    # 模型结构与特征图
    ├── geometry.py          # 多边形几何工具
    ├── visualization.py     # Matplotlib 图表
    ├── history.py           # 历史快照管理
    └── pages/               # 各标签页渲染逻辑
```

## 注意事项

- 超大图像或超大验证集会显著增加内存/显存占用，建议先用小批量数据验证配置
- 历史快照保存完整分析对象；模型特征图仍按需即时计算，不纳入快照
- 推理时若 GPU OOM，自动回退到 CPU

## 许可证

MIT License
