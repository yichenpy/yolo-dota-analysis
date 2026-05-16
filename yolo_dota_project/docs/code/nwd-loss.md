# NWD 损失模块开发文档

> SOP 类型：代码开发 | 开发日期：2026-05-16

## 1. 需求与逻辑思路

### 1.1 功能描述

为 Ultralytics YOLO11 OBB 训练流程注入 NWD（Normalized Wasserstein Distance）回归损失，与原 ProbIoU 损失加权融合，目的是提升小目标 OBB 检测精度。要求**不修改 ultralytics 源码**，**默认行为零变化**。

### 1.2 设计思路

- 数学：闭式 Wasserstein-2 距离（旋转 2D 高斯），见 `docs/paper/nwd.md` 第 3.5 节。复杂度 O(N)，无矩阵开方。
- 工程：子类化 `RotatedBboxLoss` → `NWDRotatedBboxLoss`；monkey-patch `v8OBBLoss.__init__` 替换 `self.bbox_loss`。
- CLI：`train.py` 新增 `--use-nwd / --nwd-c / --nwd-weight` 三个参数。

详细设计见 `docs/design/nwd-integration.md`。

### 1.3 复杂度

- 时间：每 anchor 一次 NWD 计算，与 probiou 同量级 O(N)
- 空间：无额外缓存，原地张量运算

## 2. 核心方法与技术栈

| 项目 | 选择 | 理由 |
|------|------|------|
| 主框架 | PyTorch + Ultralytics 8.3+ | 与现有项目一致 |
| 注入方式 | Monkey-patch + 子类继承 | 不改 ultralytics 源码，pip 升级无影响 |
| 协方差获取 | `ultralytics.utils.metrics._get_covariance_matrix` | 复用官方实现，与 probiou 一致 |
| 数值精度 | 计算时强制 fp32 (`.float()`) | OBB 极小目标下 detΣ 在 fp16 下数值不稳 |

## 3. 文件清单

| 文件 | 作用 |
|------|------|
| `nwd_loss.py` | 核心实现：`compute_nwd_obb` / `NWDRotatedBboxLoss` / `enable_nwd_loss` / `disable_nwd_loss` |
| `train.py` | 新增 CLI 参数与启用钩子 |
| `docs/paper/nwd.md` | 论文阅读笔记与公式推导 |
| `docs/design/nwd-integration.md` | 集成设计方案 |
| `docs/code/nwd-loss.md` | 本文档 |

## 4. 使用说明

### 4.1 环境配置

无新增依赖，沿用现有环境：

```bash
pip install -r requirements.txt
```

### 4.2 运行指南

**启用 NWD 训练**（推荐先在 baseline 上对比）：

```bash
python train.py \
  --data datasets/DOTAv1.5-lite/data.yaml \
  --model yolo11s-obb.pt \
  --name yolo11s_obb_baseline_nwd \
  --use-nwd \
  --nwd-c 64 \
  --nwd-weight 0.5
```

**与 P2 头叠加**：

```bash
python train.py \
  --data datasets/DOTAv1.5-lite/data.yaml \
  --cfg models/yolo11s-obb-p2.yaml \
  --weights yolo11s-obb.pt \
  --name yolo11s_obb_p2_nwd \
  --batch 4 \
  --use-nwd \
  --nwd-c 64 \
  --nwd-weight 0.5
```

**关闭 NWD（baseline）**：去掉 `--use-nwd` 即可，行为与原版完全一致。

### 4.3 编程接口

```python
from nwd_loss import enable_nwd_loss, compute_nwd_obb

# 1. 启用 NWD 损失（在 model.train(...) 之前调用）
enable_nwd_loss(nwd_c=64.0, nwd_weight=0.5)

# 2. 直接调用 NWD 计算（用于离线分析）
import torch
pred = torch.tensor([[100., 100., 30., 20., 0.1]])    # (N, 5)
target = torch.tensor([[102., 99., 32., 21., 0.1]])
nwd = compute_nwd_obb(pred, target, c_constant=64.0)  # (N, 1)
print(f"NWD = {nwd.item():.4f}")  # ~0.95
```

## 5. 关键参数

| 参数 | 类型 | 默认 | 范围 | 说明 |
|------|------|------|------|------|
| `--use-nwd` | flag | False | — | 开关；不传则零行为变化 |
| `--nwd-c` | float | 64.0 | (0, +∞) | 归一化常数 C，单位像素，与数据集目标平均尺寸相关 |
| `--nwd-weight` | float | 0.5 | [0, 1] | α 权重；0 = 纯 ProbIoU，1 = 纯 NWD |

### 5.1 C 值参考表

| 数据集 / Split | 建议 C 值 | 依据 |
|----------------|-----------|------|
| AI-TOD | 12.8 | 论文原始值（平均目标 12.8 px） |
| DOTAv1.5-lite（1024 切片） | **64** | sqrt(area) 中位数偏小 |
| DOTAv1.5-lite | 64 | 同上 |
| 未切片原图 DOTA | 100~120 | 目标尺寸跨度大 |
| DIOR（水平框） | 50 | 平均目标约 50 px |

> 推荐先跑 `python analysis/analyze_object_sizes.py --dataset ...` 看分布再决定。

### 5.2 nwd_weight 调参建议

| 场景 | α 推荐 |
|------|--------|
| 小目标比例 > 60% | 0.6 ~ 0.7 |
| 小目标比例 30%~60% | 0.4 ~ 0.5 |
| 小目标比例 < 30% | 0.2 ~ 0.3 |
| 训练 NaN / 不收敛 | 临时降到 0.3 排查 |

## 6. 异常处理与避坑指南

### 6.1 已知问题与解决方案

| 问题 | 触发条件 | 现象 | 解决方案 |
|------|----------|------|----------|
| ImportError on `_get_covariance_matrix` | ultralytics < 8.1 | `cannot import name` | 升级 ultralytics 到 ≥ 8.3 |
| Loss 出现 NaN | 极小目标 + fp16 | 前几个 step loss 飞 | 检查 EPS、确保 `.float()` 被调用；降 nwd_weight |
| NWD 始终接近 1（损失不下降） | C 设置过大 | nwd ≈ 1，loss_nwd ≈ 0 | 减小 C（如从 100 → 32） |
| NWD 始终接近 0（梯度爆炸） | C 设置过小 | nwd ≈ 0，loss_nwd ≈ 1 | 增大 C |
| Resume 时未恢复 NWD | resume 模式不读 resolved_train_config | NWD patch 没生效 | 手动加 `--use-nwd` 重新启用 |
| 多卡 DDP 报 patch 失效 | 子进程内重新 import ultralytics | NWD 仅主进程生效 | 在 main() 最开头 call `enable_nwd_loss()`，确保所有 worker 入口都经过 |

### 6.2 验证步骤

烟雾测试代码（也可参考 `docs/experiment/2026-05-16-nwd-smoke.md`）：

```python
import torch
from nwd_loss import compute_nwd_obb

# 1. 完全重合 → NWD ≈ 1
box = torch.tensor([[100., 100., 30., 20., 0.5]])
assert abs(compute_nwd_obb(box, box, c_constant=64.0).item() - 1.0) < 1e-4

# 2. 远距离 → NWD ≈ 0
box_far = torch.tensor([[500., 500., 30., 20., 0.5]])
assert compute_nwd_obb(box, box_far, c_constant=64.0).item() < 0.01

# 3. 梯度正常回传
pred = torch.tensor([[100., 100., 30., 20., 0.5]], requires_grad=True)
target = torch.tensor([[105., 102., 32., 21., 0.5]])
loss = (1.0 - compute_nwd_obb(pred, target, c_constant=64.0)).sum()
loss.backward()
assert pred.grad is not None and not torch.isnan(pred.grad).any()
```

### 6.3 调试日志

启用 NWD 时会打印：

```
[nwd] enabled: C=64.0, alpha=0.5
```

如果没有看到这行，说明 `--use-nwd` 没传或者参数解析失败。

训练 log 中 `box_loss` 字段就是融合后的 loss_iou（NWD + ProbIoU 加权和），无法单独看 NWD 分量。如需单独观察，可在 `NWDRotatedBboxLoss.forward` 末尾加 print 临时调试。

## 7. 与现有 ProbIoU 的关系

| 维度 | ProbIoU（原版） | NWDRotatedBboxLoss |
|------|-----------------|---------------------|
| 距离度量 | Bhattacharyya | Wasserstein-2 |
| 数学根基 | 概率密度重叠 | 最优传输距离 |
| 无重叠时梯度 | 可能饱和 | 线性下降 |
| 极小目标稳定性 | 中 | 强 |
| 用法 | 默认开启 | `--use-nwd` 开启，与 ProbIoU 加权 |
| 计算开销 | 中（含 log/exp） | 低（仅 sqrt） |

> 二者不互斥，NWD 是在 ProbIoU 基础上的加权补充。

## 8. 后续维护点

- ultralytics 升级后需要验证 `_get_covariance_matrix` 签名是否变化
- `RotatedBboxLoss.__init__` / `forward` 签名变化时需要同步更新 `NWDRotatedBboxLoss`
- 若引入新数据集，建议先用 `analyze_object_sizes.py` 重新评估 C 值
