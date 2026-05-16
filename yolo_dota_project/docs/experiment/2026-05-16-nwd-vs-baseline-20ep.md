# NWD vs Baseline 20-epoch 对比实验

> SOP 类型：算法与模型实验 | 实验日期：2026-05-16

## 1. 实验目的

短期（20 epoch）验证 NWD 损失（C=64, α=0.5）是否能在 DOTA-split-lite 上带来：

- 整体 mAP@0.5 / mAP@0.5:0.95 提升
- **特别关注**：`small-vehicle` 等小目标类 AP 是否提升（NWD 的设计目标）

属于"先看苗头"的短训练对比，而非最终性能评估。

## 2. 实验环境与配置

### 2.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | 待用户补充 |
| 训练时长 | 每个 run 约 20 min（待用户补充） |

### 2.2 软件依赖

| 库 | 版本 |
|----|------|
| Python | E:\miniconda3\envs\cuda（3.x） |
| PyTorch | 2.5.1 |
| ultralytics | 8.3.93 |
| CUDA | 12.x |

### 2.3 数据集

| 数据集 | data.yaml | 验证集图像数 | 验证集实例数 | 类别数 |
|--------|-----------|--------------|--------------|--------|
| DOTA-split-lite | `datasets/DOTA-split-lite/data.yaml` | 3503 | 123,897 | 16 |

### 2.4 训练超参（两次完全一致除 NWD 外）

```yaml
model: yolo11s-obb.pt
imgsz: 1024
epochs: 20
batch: 8
optimizer: AdamW
lr0: 0.001
weight_decay: 5e-4
mosaic: 1.0
close_mosaic: 10
amp: True
seed: 0
```

**NWD 配置（仅 yolo11s_nwd_20ep）**：

```yaml
use_nwd: true
nwd_c: 64.0
nwd_weight: 0.5
```

## 3. 实验结果

### 3.1 整体指标对比

| 指标 | Baseline | NWD | Δ |
|------|----------|-----|---|
| mAP@0.5 (all) | 0.741 | 0.748 | **+0.7%** |
| mAP@0.5:0.95 (all) | 0.578 | 0.583 | +0.5% |
| Precision (all) | 0.763 | 0.778 | +1.5% |
| Recall (all) | 0.707 | 0.699 | −0.8% |

**总体微弱正向，但在统计噪声范围内**。

### 3.2 分类别 mAP@0.5 对比（按 Δ 排序）

| 类别 | 实例数 | Baseline | NWD | Δ mAP50 | Δ mAP50-95 | 类别属性 |
|------|--------|----------|-----|---------|-------------|----------|
| **helicopter** | 128 | 0.568 | 0.677 | **+10.9** | +9.6 | 稀有类，方差大 |
| **ground-track-field** | 214 | 0.663 | 0.718 | **+5.5** | +4.8 | 稀有类 |
| plane | 4478 | 0.960 | 0.979 | +1.9 | +1.6 | 中大目标，多 |
| harbor | 4185 | 0.865 | 0.875 | +1.0 | +1.8 | 中目标 |
| ship | 21784 | 0.885 | 0.893 | +0.8 | +0.2 | 中小目标 |
| tennis-court | 1515 | 0.951 | 0.952 | +0.1 | +0.1 | 持平 |
| **small-vehicle** | **73504** | **0.670** | **0.670** | **0.0** | −0.2 | **小目标主力，反常持平** |
| basketball-court | 287 | 0.750 | 0.745 | −0.5 | −0.1 | 持平 |
| storage-tank | 4819 | 0.844 | 0.839 | −0.5 | −1.4 | 中目标 |
| swimming-pool | 997 | 0.833 | 0.827 | −0.6 | +0.2 | 持平 |
| baseball-diamond | 354 | 0.899 | 0.891 | −0.8 | −2.4 | 持平 |
| soccer-ball-field | 243 | 0.690 | 0.681 | −0.9 | −0.5 | 持平 |
| roundabout | 287 | 0.754 | 0.743 | −1.1 | −1.7 | 持平 |
| large-vehicle | 10284 | 0.841 | 0.829 | −1.2 | −0.9 | 微输 |
| **bridge** | 790 | 0.668 | 0.639 | **−2.9** | **−2.5** | **极端长条，明显回退** |
| container-crane | 28 | 0.008 | 0.002 | −0.6 | −0.3 | 模型未学会，忽略 |

### 3.3 损失曲线观察（来自 wandb）

- `train/box_loss`：NWD 看起来"大幅下降"（0.78 → 0.55），但这是**损失尺度变化导致的视觉假象**（NWD 项的 `1-NWD` 量级小于 `1-probiou`），不是回归更准的体现
- `train/cls_loss`：两者几乎重合
- `train/dfl_loss`：NWD 略低，可忽略
- `train/angle_loss`：两者几乎重合
- `val/box_loss`：同 train，假象
- `val/cls/dfl/angle`：差距极小

### 3.4 是否符合预期

| 假设 | 实际 | 结论 |
|------|------|------|
| 整体 mAP 有提升 | +0.7% mAP50 | ✓ 弱支持 |
| small-vehicle 类 AP 显著提升 | **0.0 提升** | ✗ **反预期** |
| 没有明显退化的类 | bridge 退 2.9 mAP50 | ✗ **反预期** |

## 4. 深度分析

### 4.1 符合预期的部分

- **helicopter / plane / harbor 提升**：这些类的目标尺寸跨度大、形状规则，NWD 的"平滑梯度"在中等尺寸目标上发挥了作用。helicopter +10.9 因样本少（128 instances）方差大，需要谨慎解读，但 plane +1.9（4478 instances）是可靠的正向信号。
- **整体 mAP 微弱正向**：与论文在 COCO/AI-TOD 上的"加 NWD 后 +1~3 mAP" 大致一致（论文 +9 mAP 是在 AI-TOD 这种全微小目标数据集上，DOTA 不可能复制这种增益）。

### 4.2 不符合预期 (Bad Case 分析)

#### Bad Case 1：small-vehicle 完全不动

**现象**：73,504 实例的大类，NWD 提升为 0。

**根因**：`C=64` 对小车太大。

- 小车 sqrt(area) 估计在 12-20 px 范围
- W₂ 距离对小车的"远距离"是 5-10 px，NWD = exp(-10/64) ≈ 0.86，已经很高
- `1 - NWD ≈ 0.14`，远小于 ProbIoU 对小车的损失（通常 0.3-0.5）
- 加权 `0.5 * 0.14 + 0.5 * 0.4 = 0.27` ≈ 0.5 * 原 ProbIoU 损失
- 实际效果：NWD 项在小车上**贡献被稀释**，等于变相降低了小车的回归权重

**改进方向**：降低 C 到 20-30 范围，让 NWD 在小车上真正进入"敏感区间"。

#### Bad Case 2：bridge 退 2.9 mAP50

**现象**：790 实例的桥类，NWD 让 mAP50 从 0.668 降到 0.639。

**根因**：Wasserstein 距离对**极端宽高比**的几何敏感性不足。

- 桥的 OBB 典型形状是 w=200, h=10
- Σ = diag(200²/12, 10²/12) = diag(3333, 8.3)
- trace 被长边主导（3333 vs 8.3），短边变化被稀释
- 短边定位错误的损失贡献严重不足
- ProbIoU 用 Bhattacharyya 距离涉及 det(Σ)，对极端长宽比敏感度更高，原本能正确惩罚桥的短边误差

**改进方向**：

- 短期：降低 NWD 权重 α（如 0.3）让 ProbIoU 主导
- 长期：在 NWD 项中引入宽高比修正（类似 CIoU 对 IoU 的修正）
- 或者：bridge 类排除 NWD（class-conditional NWD）

#### Bad Case 3：所有"持平"类（−1 ~ +1）

约 8 个类在噪声范围内。可能原因：

- 这些类原本 ProbIoU 已经处理得很好（如 tennis-court、storage-tank），NWD 加入只是冗余
- 20 epoch 还在收敛中，差异不显著
- α=0.5 折中导致两个损失互相稀释

### 4.3 关键技术洞察

1. **C 是数据集敏感的核心超参**：不是越小越好或越大越好，而是要匹配目标尺寸分布的中位数。C=64 估计偏向了中等目标（plane, harbor），错过了 small-vehicle。

2. **NWD 不是"小目标专用"的银弹**：论文的"+9 mAP"成绩在 AI-TOD 上是因为**整个数据集都是微小目标**，C=12.8 对所有目标都有效。在 DOTA 这种**尺度跨度大**的数据集，固定单一 C 永远是妥协。

3. **Wasserstein 对长条目标先天不友好**：bridge 的回退印证了这一点，应该作为后续设计 class-conditional 损失的依据。

### 4.4 下一步改进计划

按 ROI 排序：

- [ ] **优先：C=24 重训**，验证是否能救 small-vehicle
- [ ] 跑 `analyze_object_sizes.py` 取得精确尺寸分布，按各类中位数评估 C 的合理性
- [ ] 若 C=24 仍不行：α 降到 0.3，看是否能止住 bridge 的退化
- [ ] 若 C=24 仍不行：考虑 class-conditional NWD（仅 small-vehicle、ship 启用）
- [ ] 备选方案：完全换 KFIoU，看 OBB 专用损失是否更适合
- [ ] 与 P2 头组合实验

## 5. 复现说明

### 5.1 Baseline 命令

```powershell
cd E:\cy\yolo_dota_project
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTA-split-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_baseline_20ep `
  --epochs 20 --batch 8
```

### 5.2 NWD 命令

```powershell
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTA-split-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_nwd_20ep `
  --epochs 20 --batch 8 `
  --use-nwd --nwd-c 64 --nwd-weight 0.5
```

### 5.3 权重保存

- Baseline: `dota_runs/yolo11s_baseline_20ep/weights/best.pt`
- NWD: `dota_runs/yolo11s_nwd_20ep/weights/best.pt`
- 训练配置：各自目录下的 `resolved_train_config.json`

## 6. 数据可信度评估

| 类别 | 实例数 | 评估可靠度 |
|------|--------|------------|
| small-vehicle | 73,504 | 高 |
| ship | 21,784 | 高 |
| large-vehicle | 10,284 | 高 |
| storage-tank | 4,819 | 高 |
| plane | 4,478 | 高 |
| harbor | 4,185 | 高 |
| ship / tennis-court | 1-2k | 中 |
| swimming-pool | 997 | 中 |
| bridge | 790 | 中 |
| baseball-diamond | 354 | 低 |
| roundabout / basketball-court | 287 | 低 |
| soccer-ball-field | 243 | 低 |
| ground-track-field | 214 | 低 |
| helicopter | 128 | 低（单次结果方差大） |
| container-crane | 28 | 不可用 |

**结论**：bridge 的 −2.9 和 small-vehicle 的 0.0 是**可信信号**；helicopter 的 +10.9 需要多次种子验证才能确认。
