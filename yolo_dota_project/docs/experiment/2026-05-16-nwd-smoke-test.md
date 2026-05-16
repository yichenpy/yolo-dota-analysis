# NWD 损失烟雾测试

> SOP 类型：算法与模型实验 | 实验日期：2026-05-16

## 1. 实验目的

验证 `nwd_loss.py` 的核心实现正确性，确保：

1. NWD 数学公式实现无误（重合时 = 1，远距离 ≈ 0）
2. 梯度可以正常回传（包括 IoU=0 的极端情况）
3. monkey-patch 机制可以安全地启用/更新/卸载
4. 与 Ultralytics 8.3.93 `_get_covariance_matrix` 接口兼容

这是单元/烟雾级测试，不是性能实验（无需 GPU 训练数据）。

## 2. 实验环境与配置

### 2.1 硬件环境

测试在 CPU 上跑（仅验证数学，无需 GPU）。

### 2.2 软件依赖

| 库 | 版本 |
|----|------|
| Python | 3.x (E:\miniconda3\envs\cuda) |
| PyTorch | 2.5.1 |
| ultralytics | 8.3.93 |
| CUDA | 12.x（验证 cuda.is_available()=True，但本测试不用 GPU） |

### 2.3 测试脚本

`tests/test_nwd_smoke.py`，共 9 个测试用例。

## 3. 实验结果

### 3.1 测试运行结果

```
[OK] identity_box: NWD(B, B) ~ 1.0 for 4 shape variants
[OK] far_box: NWD = 0.000000
[OK] close_box: NWD = 0.918688
[OK] smoothness_vs_iou: NWD=0.984496 for 1px shift (size-invariant, smooth)
[OK] gradient_flow: loss=0.1618, grad=[-0.0116, -0.0058, -0.0005, -0.0002, -0.0024]
[OK] no_overlap_gradient: |grad|=2.66e-04 (IoU would give 0)
[OK] batch_consistency: [0.9641, 0.9187, 0.8950]
[OK] integration_compute_nwd_obb: same=0.999995, far=0.000145
[OK] integration_patch_lifecycle: enable / update / disable all work

PASSED 9/9 tests
```

### 3.2 关键数值验证

| 用例 | 输入 | 预期 | 实际 | 状态 |
|------|------|------|------|------|
| 自重合 | NWD(B, B) | ≈ 1.0 | 0.999995 ~ 1.0 | ✓ |
| 远距离 | center 移动 700px | ≈ 0 | < 1e-4 | ✓ |
| 1px 平移 | 6×6 tiny 框 | exp(-1/64) ≈ 0.9845 | 0.9845 | ✓ |
| 1px 平移 | 100×100 大框 | exp(-1/64) ≈ 0.9845 | 0.9845 | ✓ |
| 无重叠梯度 | (100,100,10,10) vs (300,300,10,10) | grad ≠ 0 | 2.66e-04 | ✓ |
| Batch 一致性 | 3 个框 vs 一个个算 | 完全相等 | < 1e-5 误差 | ✓ |

### 3.3 性质验证

| 性质 | 验证方式 | 结果 |
|------|----------|------|
| Wasserstein 距离对称性 | NWD(A, B) = NWD(B, A) | 隐含通过（公式对称） |
| 自重合时为 1（非严格相等） | NWD(B, B) | 0.999995 ≈ 1（受 EPS 影响微小偏差，可接受） |
| **平移尺度不变性** | 同 shift 下 tiny 和 large 框 NWD 相等 | 0.984 == 0.984 |
| 无重叠时梯度非零 | grad on disjoint boxes | 2.66e-04 ≠ 0 |
| Patch 幂等性 | 调两次 enable 不报错 | ✓ |
| Patch 可恢复 | disable 后恢复原 __init__ | ✓ |

## 4. 深度分析

### 4.1 符合预期

- **数学正确性 ✓**：NWD 闭式解的实现与理论值（exp(-1/64) ≈ 0.9845）数值精确匹配，说明 Wasserstein-2 公式与协方差矩阵的处理都正确。
- **梯度连续性 ✓**：在 `test_no_overlap_gradient` 中，两个完全不重叠的小框（IoU = 0）仍然产生 2.66e-04 的梯度，这正是 NWD 相对 IoU 的核心优势——梯度不会在无重叠时消失。
- **Patch 安全性 ✓**：lifecycle 测试通过，可以安全 enable → 更新参数 → disable，不会污染全局状态。

### 4.2 不符合预期 / Bad Case

#### 用例：`test_math_size_sensitivity`（初版）

**问题**：第一次写的测试错误地假设"tiny 框的 1px shift 应该比 large 框 NWD 更小"，实际测试结果是**相等**（都是 0.9845）。

**根因分析**：
- W₂² 在纯平移情况下 = ||μ_a - μ_b||² + trace_sum - cross_term。当两框形状完全相同时，trace_sum 和 cross_term 相互抵消，只剩中心距离平方。
- 因此 NWD 在**纯平移**下确实与目标尺寸**无关**，只与 C 和 shift 大小有关。

**结论**：这是 NWD 的**正确行为**而非 bug。NWD 论文的主张不是"对小目标更严苛"，而是"对小目标更平滑"——IoU 在 6×6 上 1px shift 会从 1.0 掉到 0.71，NWD 在 6×6 和 100×100 上 1px shift 都是 0.9845。修正测试断言为验证"size-invariant + smooth"两个性质，重新通过。

**启发**：这印证了 NWD 设计的精妙——通过把框抽象为高斯分布，把"对位置敏感度"与"目标尺寸"解耦。这正是论文 Section 3.2 的核心论证。

### 4.3 已知限制（待训练实验验证）

1. **C = 64 是否适合 DOTA-split-lite 尚未确认**：目前只是基于 sqrt(area) 中位数的估计，需要先跑 `analyze_object_sizes.py` 取得精确分布
2. **nwd_weight = 0.5 是否最优未知**：论文经验值，但 DOTA 大目标比例不低，可能 0.3 更合适
3. **AMP 混合精度下数值稳定性未验证**：测试均在 fp32 下进行
4. **DDP 多卡训练下 patch 传播未验证**：当前测试为单进程

### 4.4 下一步计划

- [ ] 在 DOTA-split-lite 上跑 1-epoch baseline 与 NWD（C=64, α=0.5）的对比训练，验证 loss 曲线正常收敛
- [ ] 跑完整 100 epoch 训练 baseline、NWD-on，对比 mAP@0.5 / mAP@0.5:0.95 / 各类 AP
- [ ] 若 NWD 有提升，做小规模超参搜索：C ∈ {32, 64, 100}，α ∈ {0.3, 0.5, 0.7}
- [ ] 与 P2 头组合实验：baseline P2 vs P2 + NWD

## 5. 复现说明

```powershell
# 在 cuda conda 环境下运行
& "E:\miniconda3\envs\cuda\python.exe" "E:\cy\yolo_dota_project\tests\test_nwd_smoke.py"
```

预期输出 `PASSED 9/9 tests`。

## 6. 测试覆盖矩阵

| 测试 | 数学层 | Ultralytics 集成层 |
|------|--------|--------------------|
| identity_box | ✓ | — |
| far_box | ✓ | — |
| close_box | ✓ | — |
| smoothness_vs_iou | ✓ | — |
| gradient_flow | ✓ | — |
| no_overlap_gradient | ✓ | — |
| batch_consistency | ✓ | — |
| integration_compute_nwd_obb | — | ✓（用 `_get_covariance_matrix`） |
| integration_patch_lifecycle | — | ✓（enable/disable） |

**未覆盖（留给后续训练实验）**：

- 完整的 `NWDRotatedBboxLoss.forward` 路径（包含 DFL 项）
- 与 `v8OBBLoss.__call__` 的实际集成
- 单 epoch 训练收敛性
- 多卡 DDP 下 patch 传播
- AMP 混合精度数值稳定
