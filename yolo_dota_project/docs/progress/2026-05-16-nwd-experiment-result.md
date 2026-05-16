# 项目推进进度 - NWD 首轮实验结果分析

> 日期：2026-05-16  
> 主题：NWD（C=64, α=0.5）20-epoch 对比训练完成，得出关键结论

## 1. 本次完成

- [x] 完成 baseline（yolo11s_baseline_20ep）与 NWD（yolo11s_nwd_20ep）的 20-epoch 对比训练
- [x] 收集 wandb 损失曲线和验证集分类别 AP 数据
- [x] 完成分类别深度分析（16 类逐一比对）
- [x] 识别两个关键反预期信号（small-vehicle 持平、bridge 回退）
- [x] 提出根因解释（C 值不匹配 + Wasserstein 对长条目标不友好）
- [x] 写完整实验文档 → [`docs/experiment/2026-05-16-nwd-vs-baseline-20ep.md`](../experiment/2026-05-16-nwd-vs-baseline-20ep.md)

## 2. 关键结论

| 维度 | 结论 |
|------|------|
| 整体 mAP@0.5 | **+0.7%（弱正向，噪声范围内）** |
| 小目标主力 small-vehicle | **完全无提升（反预期）** |
| 桥 bridge | **−2.9 mAP50（明显回退）** |
| 中目标 plane / harbor | +1-2 mAP50（可信） |
| 稀有类 helicopter | +10.9 mAP50（方差大，需复验） |
| **train/box_loss 看似大幅下降** | **是损失尺度变化的视觉假象，非真实改善** |

**核心判断：C=64, α=0.5 这一组配置在 DOTA-split-lite 上没有命中小目标痛点**。

## 3. 已识别的两个 Bad Case 根因

### Bad Case 1：small-vehicle 0 提升

`C=64` 对小车（sqrt(area) ~ 12-20 px）过大，NWD 项在小车上恒近 1，损失贡献被稀释。

→ **必须把 C 降到与 small-vehicle 尺寸匹配的 20-30 范围**。

### Bad Case 2：bridge 退 2.9

Wasserstein 距离对极端长条形目标（如 w=200, h=10）短边敏感度不足，trace 被长边主导。这是数学层面的局限，不是 bug。

→ **要么降 α 让 ProbIoU 主导，要么实现 class-conditional NWD 排除桥**。

## 4. 下次推进点（IMPORTANT）

### 4.1 推荐执行顺序

**第一优先：试 C=24 重训**

```powershell
cd E:\cy\yolo_dota_project
& "E:\miniconda3\envs\cuda\python.exe" train.py `
  --data datasets\DOTA-split-lite\data.yaml `
  --model yolo11s-obb.pt `
  --name yolo11s_nwd_c24_20ep `
  --epochs 20 --batch 8 `
  --use-nwd --nwd-c 24 --nwd-weight 0.5
```

**判断标准**：

- 若 small-vehicle mAP50 涨 ≥ 1.5（从 0.670 → 0.685+）：NWD 路线有效，继续优化
- 若 small-vehicle 仍持平：说明 NWD 在这个数据集上**根本不是瓶颈**，换路线
- 若 small-vehicle 涨但 bridge 进一步退：说明需要 class-conditional NWD

**第二步：跑数据集尺寸分布分析（应该早做的）**

```powershell
& "E:\miniconda3\envs\cuda\python.exe" analysis\analyze_object_sizes.py `
  --dataset datasets\DOTA-split-lite\data.yaml --splits train val
```

输出会给出每个类的 sqrt(area) 分布，**精确决定 C 的最优区间**。

**第三步（如 C=24 无效，分支决策）**

| 现象 | 推荐方向 |
|------|----------|
| small-vehicle 不动但 helicopter 类小样本类波动大 | 多种子复验，剔除噪声 |
| small-vehicle 不动且数据集 sqrt(area) 普遍 > 50 | NWD 不适合，转 KFIoU |
| bridge 持续回退 | 实现 class-conditional NWD（排除桥） |
| 总体没改善 | 放弃损失方向，转 SAHI 切片推理（零训练成本） |

### 4.2 需要的上下文

新会话开始时，按顺序读：

1. `docs/progress/2026-05-16-nwd-experiment-result.md`（本文件）
2. `docs/experiment/2026-05-16-nwd-vs-baseline-20ep.md`（第一轮详细数据）
3. `docs/code/nwd-loss.md`（怎么改参数）

### 4.3 关键决策记录

| 决策 | 理由 |
|------|------|
| 没有立刻跑 C=24 而是先记录分析 | SOP 要求实验结束必写文档；下次会话用新 C 是干净的实验起点 |
| 不直接跑 100 epoch | 20 epoch 已经能看出小目标是否被 NWD 影响；100 epoch 在错的 C 下浪费时间 |
| 优先排查 C 而非 α | C 是数据集敏感参数，α 只是权重；C 错了什么 α 都救不了 |
| bridge 退化不算 blocker | 整体 +0.7 mAP，bridge 占比小；先解决核心问题再回头修 bridge |

## 5. 风险与遗留问题

### 5.1 已知风险

1. **20 epoch 收敛不充分**：所有曲线都还在上升，差异可能被低估
2. **种子方差**：仅一次实验，helicopter +10.9 这种小样本类高度不可信
3. **C=24 也可能不对**：DOTA 内部目标尺寸跨度太大，单一 C 始终是妥协

### 5.2 未解疑问

- 如果 C=24 让 small-vehicle 涨但其他类大量退化怎么办？→ 走 class-conditional 路线
- DOTA-split-lite 是否本身已经做了切片放大了小目标？→ 看 analyze_object_sizes 输出
- NWD 在 AI-TOD 论文里 +9 mAP 是因为全是小目标，能否人工构造一个 "DOTA 小目标 only" 子集来验证 NWD 上限？

## 6. 当前项目快照

| 状态 | 内容 |
|------|------|
| 损失改进路线 | NWD 实现完成，C=64 首轮无小目标增益，待 C=24 验证 |
| 架构改进 | P2 头 / CBAM / ResidualCBFuse 已实现，未与 NWD 组合 |
| 工程优化 | 未开始：SAHI / Copy-Paste / 高分辨率教师蒸馏 |
| 分析工具 | analyze_object_sizes / analyze_errors 已就绪但本次未用 |

## 7. 历史进度链

- [2026-05-16 NWD 损失集成](2026-05-16-nwd-loss.md) → 实现完成
- **[2026-05-16 NWD 首轮实验结果分析](2026-05-16-nwd-experiment-result.md)** ← 当前
- 下次：NWD C=24 实验结果（或换路线决策）
