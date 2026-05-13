# yolo_dota_project

基于 Ultralytics YOLO11 OBB 的航空遥感图像旋转框检测训练与分析框架，支持 DOTA、DOTAv1.5、DIOR 等数据集。

## 特性

- **三种训练模式**：直接训练 / 自定义架构 + 部分权重加载 / 断点续训
- **P2 检测头实验**：增加高分辨率小目标检测分支，适用于航空图像中的密集小目标
- **自定义模块**（`custom_modules.py`）：
  - `ProjectCBAM`：通道 + 空间双重注意力，动态适配特征维度
  - `ResidualCBFuse`：可学习残差融合，将浅层细节引入深层语义
- **独立分析脚本**：目标尺寸分布统计、漏检/虚检分析、检测头特征图分析

## 目录结构

```
yolo_dota_project/
├── train.py              # 训练入口（支持 baseline / P2 / resume）
├── custom_modules.py     # 自定义神经网络模块
├── test.py               # 快速测试脚本
├── requirements.txt      # 依赖列表
├── models/               # 自定义模型 YAML 配置
│   ├── yolo11n/s/m/l/x-obb-p2.yaml   # P2 检测头（各尺度）
│   ├── yolo11s-obb-cbam-no-p2.yaml   # CBAM 消融实验
│   ├── yolo11s-obb-p2-residual-p3.yaml  # 残差融合实验
│   └── ...
├── analysis/             # 独立分析脚本
│   ├── common_obb.py               # 共享 OBB 几何、数据集工具
│   ├── analyze_errors.py           # 漏检/虚检分析
│   ├── analyze_object_sizes.py     # 目标尺寸分布分析
│   ├── analyze_detection_layers.py # 检测头特征图分析
│   └── plot_run_metrics_comparison.py  # 多次训练对比
└── datasets/             # 数据集目录（仅含 data.yaml，图像需自行下载）
    ├── DOTA/data.yaml
    ├── DOTA-split-lite/data.yaml
    ├── DOTAv1.5/data.yaml
    └── split.py          # 数据集切分工具
```

## 环境要求

```bash
pip install -r requirements.txt
```

- Python 3.10 / 3.11
- PyTorch >= 2.1（GPU 训练需 CUDA 版本）
- ultralytics >= 8.3

验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

## 数据集

本项目支持以下数据集，需自行下载并放置到 `datasets/` 目录：

| 数据集 | 说明 | 下载地址 |
|--------|------|----------|
| DOTA v1.0 | 15 类航空目标，旋转框标注 | [DOTA 官网](https://captain-whu.github.io/DOTA/) |
| DOTAv1.5 | DOTA v1.0 扩展，增加小目标 | [DOTA 官网](https://captain-whu.github.io/DOTA/) |
| DIOR | 20 类遥感目标（水平框） | [百度网盘](https://pan.baidu.com/s/1iLKT0JQoKXEJTGNxt5lSMg) |

> 下载后按 `datasets/DOTA/data.yaml` 中的路径结构放置图像和标签。
> 可使用 `datasets/split.py` 对大图进行切分，生成 `DOTA-split-lite` 格式。

## 训练

### 1. 直接训练（baseline）

```bash
python train.py \
  --data datasets/DOTA-split-lite/data.yaml \
  --model yolo11s-obb.pt \
  --name yolo11s_obb_baseline
```

### 2. P2 检测头训练

```bash
python train.py \
  --data datasets/DOTA-split-lite/data.yaml \
  --cfg models/yolo11s-obb-p2.yaml \
  --weights yolo11s-obb.pt \
  --name yolo11s_obb_p2 \
  --batch 4 \
  --device auto
```

### 3. 断点续训

```bash
python train.py \
  --resume \
  --resume-from dota_runs/yolo11s_obb_p2/weights/last.pt
```

> P2 检测头显存占用约为 baseline 的 2 倍，建议使用 `--batch 4` 或更小。

## `train.py` 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据集 YAML 路径 | 自动搜索 |
| `--model` | 预训练权重路径（直接训练） | — |
| `--cfg` | 自定义模型 YAML | — |
| `--weights` | cfg 模式下的部分加载权重 | — |
| `--scale` | 通用 YAML 的模型尺度 `n/s/m/l/x` | 从文件名推断 |
| `--imgsz` | 输入图像尺寸 | 1024 |
| `--epochs` | 训练轮数 | 100 |
| `--batch` | batch size | 8 |
| `--device` | `auto` / `cpu` / `0` / `0,1` | auto |
| `--workers` | 数据加载线程数 | 8 |
| `--cache` | 缓存模式 `False` / `ram` / `disk` | False |
| `--resume` | 开启断点续训 | — |
| `--resume-from` | checkpoint 路径 | — |
| `--wandb-mode` | `online` / `offline` / `disabled` | online |

## 分析脚本

### 目标尺寸分布

```bash
python analysis/analyze_object_sizes.py \
  --dataset datasets/DOTA/data.yaml \
  --splits train val \
  --size-metric equivalent_side
```

### 漏检/虚检分析

```bash
# 方式一：直接推理
python analysis/analyze_errors.py \
  --dataset datasets/DOTA/data.yaml \
  --split val \
  --weights path/to/best.pt \
  --device 0 \
  --save-details

# 方式二：读取已导出的预测文件
python analysis/analyze_errors.py \
  --dataset datasets/DOTA/data.yaml \
  --split val \
  --predictions path/to/predictions/labels \
  --prediction-format txt
```

### 检测头特征图分析

```bash
python analysis/analyze_detection_layers.py \
  --dataset datasets/DOTA/data.yaml \
  --weights path/to/best.pt \
  --split val \
  --device 0
```

## 模型 YAML 说明

| 文件 | 描述 |
|------|------|
| `models/yolo11{n,s,m,l,x}-obb-p2.yaml` | P2 检测头（推荐，含尺度后缀） |
| `models/yolo11-obb-p2.yaml` | 通用 P2（需手动指定 `--scale`） |
| `models/yolo11s-obb-cbam-no-p2.yaml` | 仅 CBAM，无 P2 头（消融） |
| `models/yolo11s-obb-p2-residual-p3.yaml` | P2 + 残差融合（完整实验） |
| `models/yolo11s-obb-p2-branch-only.yaml` | 仅 P2 分支（消融） |
| `models/yolo11s-obb-p2-fusion-only.yaml` | 仅融合层（消融） |

> 建议使用带尺度后缀的 YAML（如 `yolo11s-obb-p2.yaml`），避免 Ultralytics 默认推断为 `scale='n'`。

## 许可证

MIT License
