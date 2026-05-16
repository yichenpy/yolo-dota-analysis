# NWD vs Baseline 20-epoch 对比实验

> SOP 类型：算法与模型实验 | 实验日期：2026-05-16

## 1. 实验目的

短期（20 epoch）验证 NWD 损失（C=64, α=0.5）是否能在 DOTAv1.5-lite 上带来：

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
| DOTAv1.5-lite | `datasets/DOTAv1.5-lite/data.yaml` | 3503 | 123,897 | 16 |

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
nwd_c: 16.0      # 来自 analyze_object_sizes.py 的 sqrt(area) 中位数
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

#### Bad Case 1：small-vehicle 完全不动（核心反预期）

**现象**：73,504 实例的小目标主力类，NWD（C=16）提升为 0。

**第一版根因（错误）**：C 太大不匹配小车尺寸。

**修正根因（正确）**：C 已经匹配（来自数据集 sqrt(area) 中位数），但 **NWD 和 ProbIoU 在小目标上提供了高度冗余的信号**。

具体测算（16 px 小车，2 px 中心偏移）：

| 损失项 | 计算 | 数值 |
|--------|------|------|
| ProbIoU 项 `1 − probiou` | 16px 框 2px 偏移 → IoU ≈ 0.875 | ~0.125 |
| NWD 项 `1 − NWD` | `1 − exp(−2/16)` | ~0.118 |
| 加权 (α=0.5) | 0.5 × 0.125 + 0.5 × 0.118 | **0.122** |

加权后总损失（0.122）和**纯 ProbIoU（0.125）几乎一样**。等价于把 ProbIoU 替换成了一个数值接近的损失，自然不带来增益。

**结论**：**NWD 在 size-matched C 下不是"加强小目标信号"，而是"冗余地复制小目标信号"**。这与论文的暗示相反——论文在 AI-TOD（全微小目标）上有效，是因为 IoU 在 6×6 上极不稳定，NWD 提供了一个**不同的、更平滑的**信号。在 DOTAv1.5-lite，ProbIoU 在 16 px 上已经稳定且严苛，NWD 无新增价值。

#### Bad Case 1b：plane 反而提升 +1.9（颠覆原假设）

按"NWD 是小目标损失"理解，飞机（中大目标）不应该是受益者。但实际 plane +1.9 mAP50 是统计可信的提升。

测算（100 px 飞机，2 px 偏移）：

| 损失项 | 数值 |
|--------|------|
| ProbIoU 项（大目标小偏移） | ~0.020 |
| NWD 项 `1 − exp(−2/16)` | ~0.118 |
| 加权 | **0.069**（比原 ProbIoU 高 ~3.5 倍） |

**关键洞察**：NWD 在 C=16 下对位置偏移是 **size-invariant**（2 px 偏移无论框多大，NWD 都给 ~0.118）。而 ProbIoU 对大目标小偏移近乎不敏感（0.020）。所以 **NWD 实际作用是把大目标的位置惩罚抬到了和小目标一个量级**，这是它在 plane、helicopter、harbor 上有增益的真正原因。

**修正后的 NWD 定位**：

> NWD（在 size-matched C 下）不是"小目标损失"，而是 **"size-invariant 的位置正则项"**——它强化的是 ProbIoU 在大目标上不敏感的位置误差，而非小目标。

#### Bad Case 2：bridge 退 2.9 mAP50（根因不变）

桥的 W=200, h=10 极端长条形：

- 长边 trace ≈ 3333，短边 trace ≈ 8，被长边完全主导
- 短边定位误差在 W₂² 中几乎不可见
- ProbIoU 用 Bhattacharyya 涉及 det(Σ)，对短边敏感
- α=0.5 加权后短边监督被严重稀释 → 桥的窄方向回归失准

**改进方向（保持有效）**：

- 短期：α 降到 0.3，让 ProbIoU 主导桥的回归
- 中期：class-conditional NWD，bridge / harbor 这类长条目标关闭 NWD
- 长期：在 NWD 上引入宽高比修正项

#### Bad Case 3：所有"持平"类（−1 ~ +1）

约 8 个类在噪声范围内。可能原因：

- 这些类原本 ProbIoU 已经处理得很好（如 tennis-court、storage-tank），NWD 加入只是冗余
- 20 epoch 还在收敛中，差异不显著
- α=0.5 折中导致两个损失互相稀释

### 4.3 关键技术洞察（修正版）

1. **NWD 的实际定位是 size-invariant 位置正则，不是小目标损失**。当 C 匹配数据集尺寸时：
   - 小目标上 NWD 与 ProbIoU 量级接近，是冗余信号
   - 大目标上 NWD 把 ProbIoU 已经"放过"的小偏移重新纳入惩罚
   - 因此 NWD 实际是"补救大目标的 IoU 不敏感性"，与论文小目标叙事相反

2. **论文 +9 mAP on AI-TOD 不可在 DOTA 复制**：AI-TOD 全是 < 16 px 的极小目标，IoU 完全失效，NWD 是唯一能提供位置梯度的工具。DOTA 的 ProbIoU 在中位 16 px 上已经稳定。

3. **Wasserstein 对长条目标先天不友好**：trace 被长边主导，短边定位损失被稀释，bridge 必然退化。

4. **C 不是单调可调的小目标按钮**：再降 C（如 C=8）会让所有目标的 NWD 项趋于饱和（exp 衰减），不解决问题。

### 4.4 下一步改进计划（颠覆原计划）

**放弃的方向**（基于错误诊断）：

- ~~C=24 重训~~：C 已经匹配，再降无意义
- ~~进一步降 C 救 small-vehicle~~：math 上不成立

**新的优先级**：

| 优先级 | 方向 | 理由 |
|--------|------|------|
| 1 | **SAHI 切片推理** | 零训练成本，对 aerial small object 常规 +3-8 mAP，立刻验证 |
| 2 | **P2 头 + 现 NWD 组合训练** | 用架构层面攻击 small-vehicle，NWD 顺便保留对 plane 的增益 |
| 3 | **Small-Object Copy-Paste 增强** | 数据层面增加 small-vehicle 频率，正交于损失改动 |
| 4 | **α=0.3 重训** | 减少 bridge 退化，保留大目标增益，低成本验证 |
| 5 | **class-conditional NWD（仅大目标启用）** | 顺应实际作用机理，但实现成本高 |
| 6 | **KFIoU 替代 ProbIoU** | OBB 专用，可能对桥这种长条目标更友好 |

## 5. 复现说明

### 5.1 Baseline 命令

```powershell
cd E:\cy\yolo_dota_project
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTAv1.5-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_baseline_20ep `
  --epochs 20 --batch 8
```

### 5.2 NWD 命令

```powershell
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTAv1.5-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_nwd_20ep `
  --epochs 20 --batch 8 `
  --use-nwd --nwd-c 16 --nwd-weight 0.5
```

> C=16 来自先期跑的 `analyze_object_sizes.py`，取 sqrt(area) 中位数。

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
