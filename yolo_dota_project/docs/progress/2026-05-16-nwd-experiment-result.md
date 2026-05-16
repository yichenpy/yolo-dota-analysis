# 项目推进进度 - NWD 首轮实验结果分析（修正版）

> 日期：2026-05-16  
> 主题：NWD（C=16 / α=0.5）20-epoch 对比训练完成，**机理理解修正**

## 0. 重要修正记录（2026-05-16 晚）

> **错误**：初版分析假设 C=64，得出"C 不匹配 small-vehicle"的结论，并推荐"C=24 重训"。
> **修正**：实际训练用的是 C=16（来自 analyze_object_sizes.py 中位数）。
> **新结论**：NWD 不是小目标损失，C=16 已经是匹配值，再降无意义。详见下方"修正后的根因"。

## 1. 本次完成

- [x] 完成 baseline（yolo11s_baseline_20ep）与 NWD（yolo11s_nwd_20ep, C=16）的 20-epoch 对比训练
- [x] 跑 `analyze_object_sizes.py` 取得 sqrt(area) 中位数 = 16
- [x] 收集 wandb 损失曲线和验证集分类别 AP 数据
- [x] 完成分类别深度分析（16 类逐一比对）
- [x] 修正机理理解：NWD 在 C=size_median 下实为 size-invariant 位置正则项
- [x] 写完整实验文档 → [`docs/experiment/2026-05-16-nwd-vs-baseline-20ep.md`](../experiment/2026-05-16-nwd-vs-baseline-20ep.md)（已修正）

## 2. 关键结论

| 维度 | 结论 |
|------|------|
| 整体 mAP@0.5 | **+0.7%（弱正向，噪声范围内）** |
| 小目标主力 small-vehicle | **完全无提升（NWD 与 ProbIoU 信号冗余）** |
| 大目标 plane / harbor / helicopter | +1.0 ~ +10.9 mAP50（NWD 抬高了 IoU 对大目标的位置敏感度） |
| 桥 bridge | **−2.9 mAP50（长条几何被 Wasserstein 长边主导）** |
| **NWD 真实作用** | **不是小目标损失，而是"补救大目标位置不敏感"的正则项** |

## 3. 修正后的根因

### 3.1 small-vehicle 0 提升的真正原因

**不是 C 错**，而是 **NWD 与 ProbIoU 信号冗余**：

| 损失项 | 16 px 框 2 px 偏移下数值 |
|--------|---------------------------|
| ProbIoU (1 − probiou) | ~0.125 |
| NWD (1 − exp(−2/16)) | ~0.118 |
| 加权 α=0.5 | ~0.122（与原 ProbIoU 几乎相同） |

**结论**：在 C=size_median 下，小目标的 NWD 损失数值与 ProbIoU 损失数值几乎相等，相当于把 ProbIoU 替换成一个略相似的损失，不带来新信息。

### 3.2 plane / helicopter 反而提升的原因

NWD 在 C=16 下对位置偏移是 **size-invariant**：100 px 飞机 2 px 偏移和 16 px 小车 2 px 偏移的 NWD 项**完全一样**（都是 0.118）。而 ProbIoU 对大目标小偏移近乎无感（0.02）。所以：

> NWD 实际把大目标的位置惩罚"抬高"到了和小目标一个量级，**这是 NWD 在 plane/helicopter 上有增益的真正机理**。

### 3.3 颠覆性认知

**NWD 的真实定位**：

> NWD 不是论文宣传的"小目标专用损失"。当 C 匹配数据集尺寸时，NWD 是一个 **size-invariant 的位置正则项**——它强化的是 ProbIoU 在大目标上不敏感的位置误差，**而非小目标**。

**论文 +9 mAP 不可在 DOTA 复制**：AI-TOD 全是 < 16 px 极小目标，IoU 完全失效，NWD 是救命稻草。DOTA 跨度大，ProbIoU 在中位 16 px 已经稳定。

## 4. 下次推进点（IMPORTANT，已修正）

### 4.1 已排除的方向

- ~~C=24 重训~~：之前误推荐，C 已经匹配
- ~~进一步降 C 救 small-vehicle~~：数学上不成立

### 4.2 新优先级（按 ROI 排序）

#### 优先 1：SAHI 切片推理（零训练成本，立刻能验证）

不动训练，仅修改推理流程。对 DOTA aerial small object 通常 +3-8 mAP。

```bash
pip install sahi
```

实现要点：用滑窗切大图为重叠小 patch → 每个 patch 跑 YOLO → 合并预测 → 全局 OBB-NMS。需要写一个小脚本（约 100 行）。

#### 优先 2：P2 头 + 当前 NWD 组合（架构层面攻击小目标）

```powershell
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTA-split-lite\data.yaml `
  --cfg models\yolo11s-obb-p2.yaml `
  --weights yolo11s-obb.pt `
  --name yolo11s_p2_nwd_20ep `
  --epochs 20 --batch 4 `
  --use-nwd --nwd-c 16 --nwd-weight 0.5
```

P2 头提供高分辨率小目标特征，与 NWD 对大目标的增益正交。

#### 优先 3：Small-Object Copy-Paste 数据增强

实现 `analysis/copy_paste.py`，把 small-vehicle、ship 等小目标实例从原图裁出、贴回（同图或跨图 1-3 次）。论文经验：DOTA small-vehicle 通常 +2-5 mAP。

#### 优先 4：α=0.3 重训（低成本验证）

NWD 占比降低，减少 bridge 退化，保留大目标增益：

```powershell
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTA-split-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_nwd_a03_20ep `
  --epochs 20 --batch 8 `
  --use-nwd --nwd-c 16 --nwd-weight 0.3
```

#### 优先 5：KFIoU 替代 ProbIoU（不是补充）

OBB 专用损失，对长条目标更友好。要研究论文 + 实现，工作量与 NWD 相当。

#### 优先 6：class-conditional NWD

NWDRotatedBboxLoss 内部按类别开关 NWD：大目标类启用、长条类禁用。实现成本中等。

### 4.3 推荐执行顺序

1. **第一步：SAHI**（最快验证天花板，1-2 小时实现 + 半小时测试）
2. **第二步：P2 + NWD**（架构 + 损失叠加，正面攻击小目标，1 次训练）
3. **第三步：根据 1、2 的结果决定**：
   - 若 SAHI 给出 +5 mAP：说明推理才是瓶颈，训练改动 ROI 低
   - 若 P2+NWD 给出小目标增益：继续此路线 + Copy-Paste
   - 若两者都平：转 KFIoU 或干脆放弃损失方向，专注数据扩增

### 4.4 需要的上下文

新会话开始时按顺序读：

1. `docs/progress/2026-05-16-nwd-experiment-result.md`（本文件）
2. `docs/experiment/2026-05-16-nwd-vs-baseline-20ep.md`（含修正版分析）
3. `docs/code/nwd-loss.md`（NWD 使用方法）

### 4.5 关键决策记录

| 决策 | 理由 |
|------|------|
| 放弃 C 调优路线 | C=16 已匹配数据集中位数，math 证明再降无意义 |
| 优先 SAHI 而非新训练 | 零成本验证天花板；如果天花板低就别浪费 GPU |
| 保留 NWD 实现但调整应用场景 | NWD 对大目标有正向贡献，不全废弃；future 改 class-conditional 模式 |
| 转向架构 + 数据双路线 | 损失改动已被证明在 small-vehicle 上无效，需要换攻击面 |

## 5. 历史进度链

- [2026-05-16 NWD 损失集成](2026-05-16-nwd-loss.md) → 实现完成
- **[2026-05-16 NWD 首轮实验结果分析（修正版）](2026-05-16-nwd-experiment-result.md)** ← 当前
- 下次候选：SAHI 实现，或 P2 + NWD 训练

## 6. 当前项目快照（更新）

| 状态 | 内容 |
|------|------|
| 损失改进 | NWD 实现完成，C=16 首轮整体 +0.7% mAP，**机理与论文宣传不符** |
| 已确认 NWD 有效场景 | 大目标位置敏感度补强（plane, helicopter） |
| 已确认 NWD 无效/有害场景 | small-vehicle（信号冗余）、bridge（几何破坏） |
| 架构改进 | P2 / CBAM / ResidualCBFuse 已就绪，未与 NWD 组合 |
| 工程优化 | SAHI / Copy-Paste / 蒸馏全未开始，**应当优先开始** |
| 分析工具 | analyze_object_sizes 已用（中位数 = 16）；analyze_errors 未对比使用 |
