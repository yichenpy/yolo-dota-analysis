# NWD 损失集成到 Ultralytics YOLO11 OBB 的设计方案

> SOP 类型：设计文档（代码开发前置）| 日期：2026-05-16

## 1. 设计目标

把 NWD（Normalized Wasserstein Distance）作为附加回归损失项接入到 Ultralytics YOLO11 OBB 的训练流程，要求：

1. **不修改 ultralytics 源码**：通过 monkey-patch 注入，便于 pip 升级
2. **开关可控**：可通过 CLI 参数开启/关闭、调整 NWD 与 ProbIoU 的加权比、设置归一化常数 C
3. **向后兼容**：默认关闭 NWD，等价于现行 baseline 训练
4. **精度无损**：采用闭式 Wasserstein-2 距离，无需 AABB 投影近似

## 2. 现有损失链路梳理

### 2.1 调用链

```
YOLO("yolo11s-obb.yaml").train(data=..., ...)
  → BaseTrainer._do_train()
    → model.loss(batch)
      → BaseModel.loss()                     # 首次调用时构建 criterion
        → self.criterion = self.init_criterion()
          → OBBModel.init_criterion()        # 返回 v8OBBLoss(self)
        → self.criterion(preds, batch)       # 后续每个 batch 调用
          → v8OBBLoss.__call__()
            → self.bbox_loss(...)            # RotatedBboxLoss 实例
              → probiou(pred, target)        # 关键替换点
              → DFL loss
```

### 2.2 关键源码摘要

`ultralytics/utils/loss.py`:

```python
class RotatedBboxLoss(BboxLoss):
    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])  # ← 注入点
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # ... DFL 部分原样保留
        return loss_iou, loss_dfl

class v8OBBLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk=10, tal_topk2=None):
        super().__init__(model, tal_topk=tal_topk)
        self.assigner = RotatedTaskAlignedAssigner(...)
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)  # ← 替换点
```

`ultralytics/utils/metrics.py`:

```python
def _get_covariance_matrix(boxes):
    # boxes: (N, 5) xywhr → 返回 (a, b, c) 各形状 (N, 1)
    # 表示协方差矩阵 [[a, c], [c, b]]
```

### 2.3 数据格式

| 张量 | 形状 | 含义 |
|------|------|------|
| `pred_bboxes` | (B, N, 5) | (cx, cy, w, h, angle)，已解码到图像像素坐标 |
| `target_bboxes` | (B, N, 5) | 同上，GT |
| `fg_mask` | (B, N) bool | 前景 anchor 掩码 |
| `weight` | (M, 1) | TAL 软标签权重（M = fg_mask.sum()） |

## 3. 方案选型

### 3.1 三种集成方案对比

| 方案 | 实现位置 | 改动量 | 优点 | 缺点 |
|------|----------|--------|------|------|
| A. monkey-patch `RotatedBboxLoss.forward` | 全局函数替换 | 小 | 一次 patch 所有 v8OBBLoss 实例生效 | 全局副作用，难关闭 |
| **B. 子类化 + patch `v8OBBLoss.__init__`** | 替换 `self.bbox_loss` | 中 | 通过子类清晰隔离 NWD 逻辑，可关闭 | 需正确处理 `.to(device)` |
| C. 子类化 OBBModel + patch `init_criterion` | 模型层 patch | 大 | 最干净 | 涉及模型工厂改动，影响 `YOLO()` 调用 |

**选定方案 B**：折中方案，保留原 `RotatedBboxLoss` 不动，新增 `NWDRotatedBboxLoss` 子类继承它，只重写 `forward`；通过 patch `v8OBBLoss.__init__` 替换 `self.bbox_loss`。

### 3.2 方案 B 的具体设计

```
custom_modules.py / nwd_loss.py
├── compute_nwd_obb(pred_obb, target_obb, C)   # 核心数学函数
├── class NWDRotatedBboxLoss(RotatedBboxLoss)
│   ├── __init__(reg_max, nwd_c, nwd_weight, use_nwd)
│   └── forward(...)  # 复用父类 DFL，替换 IoU 项为 α·NWD + (1-α)·ProbIoU
└── enable_nwd_loss(nwd_c=12.8, nwd_weight=0.5)  # 一键 patch v8OBBLoss
```

## 4. 数学实现细节

### 4.1 闭式 Wasserstein-2（OBB）

给定两个 OBB 对应的高斯分布 $\mathcal{N}_a = (\boldsymbol{\mu}_a, \Sigma_a)$ 和 $\mathcal{N}_b = (\boldsymbol{\mu}_b, \Sigma_b)$，其中 $\Sigma = \begin{bmatrix} a & c \\ c & b \end{bmatrix}$（由 `_get_covariance_matrix` 返回）：

```
center_dist = (cx_a - cx_b)² + (cy_a - cy_b)²
trace_sum = (a1 + b1) + (a2 + b2)
T = a1*a2 + b1*b2 + 2*c1*c2          # Tr(Σ_a · Σ_b)
D1 = a1*b1 - c1²                     # det(Σ_a)
D2 = a2*b2 - c2²                     # det(Σ_b)
sqrt_term = sqrt(T + 2*sqrt(D1*D2 + eps) + eps)
W2_squared = center_dist + trace_sum - 2*sqrt_term
nwd = exp(-sqrt(W2_squared + eps) / C)
loss_nwd = 1 - nwd
```

### 4.2 数值稳定

- 所有 sqrt 加 `eps=1e-7`，防止零梯度
- `W2_squared` 在数值上可能因浮点误差略小于 0（理论上 ≥ 0），用 `.clamp_min(eps)` 兜底
- `D1 * D2` 同样可能极小，加 eps

### 4.3 与 ProbIoU 的加权融合

```
loss_iou_combined = nwd_weight * loss_nwd + (1 - nwd_weight) * loss_probiou
```

- 当 `nwd_weight = 0`：完全等价于原始 ProbIoU baseline
- 当 `nwd_weight = 1`：纯 NWD 损失
- 推荐起点：`nwd_weight = 0.5`（论文经验）

### 4.4 归一化常数 C 的选择

DOTAv1.5-lite 数据集中目标 sqrt(area) 经验分布：
- 5th percentile: ~12 px
- 中位数: ~32 px
- 95th percentile: ~120 px

**推荐 `nwd_c = 64`** 作为 DOTA 起点，对应中位偏小的目标尺寸。后续可通过 `analyze_object_sizes.py` 输出的统计精确调整。

> 后续优化：可以在 `analysis/analyze_object_sizes.py` 输出中加入"推荐 NWD C 值"字段。

## 5. CLI 参数设计

在 `train.py` 中新增三个参数：

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--use-nwd` | flag | False | 是否启用 NWD 损失（关闭时完全等价于原 baseline） |
| `--nwd-c` | float | 64.0 | NWD 归一化常数（与数据集尺寸有关，DOTA 推荐 64-100） |
| `--nwd-weight` | float | 0.5 | NWD 在回归损失中的权重 α，范围 [0, 1] |

启用示例：

```bash
python train.py \
  --data datasets/DOTAv1.5-lite/data.yaml \
  --cfg models/yolo11s-obb-p2.yaml \
  --weights yolo11s-obb.pt \
  --use-nwd \
  --nwd-c 64 \
  --nwd-weight 0.5 \
  --name yolo11s_obb_p2_nwd
```

## 6. 集成接入点（在 train.py 中）

```python
# 在 model = YOLO(...) 之后，model.train(...) 之前
if args.use_nwd:
    from nwd_loss import enable_nwd_loss
    enable_nwd_loss(nwd_c=args.nwd_c, nwd_weight=args.nwd_weight)
    print(f"[NWD] enabled with C={args.nwd_c}, weight={args.nwd_weight}")
```

NWD 信息会随 `resolved_train_config.json` 一同保存，便于复现。

## 7. 回退策略

- 不传 `--use-nwd`：完全等价于原 baseline 训练，零行为变化
- 训练中发现 NWD 让收敛变差：直接去掉 `--use-nwd` 重训
- `nwd_weight` 渐进调整：可以训前几个 epoch 用 0.2，后期升到 0.7（暂不做自动调度，留作后续优化）

## 8. 风险与边界条件

| 风险 | 现象 | 缓解 |
|------|------|------|
| C 设置错误 | NWD 趋近 0 或趋近 1，梯度消失 | 文档强烈建议先跑 `analyze_object_sizes.py` 看分布 |
| 极小目标（< 4 px）下 detΣ ≈ 0 | sqrt 数值不稳 | 加 eps + clamp |
| fp16 训练 | `_get_covariance_matrix` 在 half 精度下溢出 | 在 NWD 计算前 `.float()`，结束后 cast 回去 |
| 损失尺度不匹配 | NWD 与 ProbIoU 量级差异让加权失衡 | 起步用 0.5 加权，跑 1-2 epoch 看 loss 曲线再调 |

## 9. 验证清单

实现完成后，烟雾测试要验证：

- [ ] `nwd = compute_nwd_obb(box, box)` 应返回接近 1.0（完全重合时 NWD=1）
- [ ] `nwd = compute_nwd_obb(box_a, box_far_away)` 应返回接近 0
- [ ] 梯度能正常回传（`loss.backward()` 后 pred_bboxes.grad 非零）
- [ ] 开启 NWD 后训练 1 个 epoch 不 NaN / Inf
- [ ] `--use-nwd` 关闭时损失数值与 baseline 完全一致

## 10. 后续扩展点（不在本次范围）

- [ ] NWD 加入到 RotatedTaskAlignedAssigner（替换 IoU 作为正负样本分配指标）
- [ ] NWD-based NMS
- [ ] α 动态调度（训练初期偏 NWD，后期偏 ProbIoU）
- [ ] 自动从数据集统计推荐 C 值
