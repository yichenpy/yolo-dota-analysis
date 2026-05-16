# SAHI 切片推理 vs 标准推理对比实验

> SOP 类型：算法与模型实验 | 实验日期：2026-05-16（实施待用户在服务器完成）  
> 状态：**模板已准备，等待服务器实际执行后回填结果**

## 1. 实验目的

在 DOTAv1.5-lite 上，验证 SAHI 切片推理对小目标 mAP 的真实增益。具体回答：

1. SAHI 在 baseline 权重上相比标准 model.val() 能带来多少 mAP 提升？
2. SAHI 在 NWD 权重上是否仍有同样幅度的增益（即两个增益是否正交）？
3. small-vehicle、ship、bridge 等关键类的 AP 变化方向？
4. SAHI 推理实际开销是否可接受？

## 2. 实验环境与配置

### 2.1 硬件环境

| 项目 | 配置 |
|------|------|
| 服务器 | Linux（路径前缀 `/root/cy/...`） |
| GPU | 待用户补充 |
| 显存 | 待用户补充 |

### 2.2 软件依赖

| 库 | 版本 |
|----|------|
| Python | （服务器环境） |
| PyTorch | 2.5.1（与本地一致） |
| ultralytics | 8.3.93+ |
| CUDA | 12.x |

### 2.3 输入

| 项目 | 路径 |
|------|------|
| Baseline 权重 | `runs/obb/test/yolo11s_baseline_20ep/weights/best.pt` |
| NWD 权重 | `runs/obb/test/yolo11s_nwd_20ep/weights/best.pt` |
| 验证集 | `datasets/DOTAv1.5-lite/images/val/`（3503 张） |
| 数据配置 | `datasets/DOTAv1.5-lite/data.yaml` |

### 2.4 SAHI 推理参数

```yaml
slice_size: 512
overlap: 0.2
perform_standard_pred: true   # 同时跑整图，捕获大目标
conf_thres: 0.001             # 评估时设低保留全谱预测
nms_iou: 0.5
```

## 3. 完整执行步骤（服务器）

> **重要更新**：之前的"200 张样本 + analyze_errors"得到了 miss_rate=93% 的伪指标。
> 根因：analyze_errors 默认遍历全部 3503 val 图，未覆盖的 3303 张图的 GT 全算 missed。
> 修复：新增 `--predictions-only` flag + `standard_inference.py` 对照脚本，做到 apples-to-apples 对比。
> 详情见 [`docs/code/fair-comparison.md`](../code/fair-comparison.md)。

### 3.1 拉取最新代码到服务器

```bash
cd /root/cy
git pull origin main
```

确认这三个文件最新：

- `yolo_dota_project/analysis/sahi_inference.py`（带 rotated_nms 跨版本兼容）
- `yolo_dota_project/analysis/standard_inference.py`（**新**）
- `yolo_dota_project/analysis/analyze_errors.py`（带 `--predictions-only`）

### 3.2 推荐：200 张样本 SAHI vs Standard 公平对比（约 40 分钟）

```bash
cd /root/cy/yolo_dota_project

# 步骤 1：SAHI 推理 200 张（约 30 min）
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_sample200 \
  --slice-size 512 --overlap 0.2 --device 0 \
  --max-images 200

# 步骤 2：Standard 推理同样 200 张（约 1 min，自动按 SAHI 输出目录过滤）
python analysis/standard_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_sample200_std \
  --from-sahi-dir analysis/outputs/sahi/baseline_sample200 \
  --device 0

# 步骤 3：公平评估 SAHI（带 --predictions-only）
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/baseline_sample200 \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --predictions-only \
  --output-dir analysis/outputs/errors/baseline_sahi_200_fair \
  --skip-official-val

# 步骤 4：公平评估 Standard（同 200 张）
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/baseline_sample200_std \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --predictions-only \
  --output-dir analysis/outputs/errors/baseline_std_200_fair \
  --skip-official-val
```

把两个 `summary.json` 都贴给我即可。

**决策点**（看 miss_rate 和 missed_by_class.small_vehicle 的差异）：

- SAHI miss_rate 比 Standard 低 ≥ 5 个点 → SAHI 有效，值得跑全集
- SAHI miss_rate 持平或更高 → SAHI 在 DOTAv1.5-lite 上无效，转 P2 头路线
- SAHI 整体好但 false 暴增 → 调高 conf_thres 或调 NMS 阈值再试

### 3.3 完整推理（如果 3.2 看起来有戏，约 9 小时/run）

```bash
# Baseline + SAHI（完整）
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_20ep \
  --slice-size 512 --overlap 0.2 --device 0

python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --split val \
  --predictions analysis/outputs/sahi/baseline_20ep \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/baseline_sahi

# NWD + SAHI（完整）
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_nwd_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/nwd_20ep \
  --slice-size 512 --overlap 0.2 --device 0

python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --split val \
  --predictions analysis/outputs/sahi/nwd_20ep \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/nwd_sahi
```

### 3.4 推荐：用 ultralytics 官方 model.val() 而非 analyze_errors 评估

`analyze_errors.py` 提供自定义诊断指标，但与官方 mAP 数值可能略不同。如果要跟之前 `model.val()` 输出的 mAP 直接对比，更可靠的方式是写一个小脚本把 SAHI txt 加载成 Ultralytics 兼容的 prediction 格式后用 `model.val(predictions=...)` 评估。

如果嫌麻烦，**直接使用 analyze_errors.py 的 summary.json 中"official_val_metrics"分段**（如果生成的话）作为对比基准。

## 4. 实验结果

### 4.1 整体指标对比（待填）

| 组合 | mAP@0.5 | mAP@0.5:0.95 | P | R | 推理耗时 |
|------|---------|--------------|---|---|----------|
| **A. Baseline 标准推理** | 0.741 | 0.578 | 0.763 | 0.707 | ~40 s（model.val） |
| B. Baseline + SAHI | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |
| **C. NWD 标准推理** | 0.748 | 0.583 | 0.778 | 0.699 | ~75 s（model.val） |
| D. NWD + SAHI | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |

### 4.2 关键对比

| 对比 | 期望 | 实际 |
|------|------|------|
| Δ(B − A): 纯 SAHI 增益 | +2~5 mAP50（如果 SAHI 适用） | _待填_ |
| Δ(D − C): SAHI 叠加 NWD 增益 | 类似 B−A，证明正交 | _待填_ |
| Δ(D − A): 联合方案 vs 基线 | 累积增益 | _待填_ |

### 4.3 关键类别 mAP@0.5 对比（待填）

| 类别 | 实例数 | A(Baseline) | B(B+SAHI) | C(NWD) | D(NWD+SAHI) |
|------|--------|-------------|-----------|--------|-------------|
| small-vehicle | 73504 | 0.670 | _ | 0.670 | _ |
| ship | 21784 | 0.885 | _ | 0.893 | _ |
| plane | 4478 | 0.960 | _ | 0.979 | _ |
| bridge | 790 | 0.668 | _ | 0.639 | _ |
| harbor | 4185 | 0.865 | _ | 0.875 | _ |

### 4.4 推理耗时（待填）

| 组合 | 单图耗时 | 全 val 总时长 |
|------|----------|----------------|
| Standard | ~0.05 s | ~40 s |
| SAHI (slice=512, overlap=0.2) | _待填_ | _待填_ |

## 5. 深度分析（待填）

### 5.1 SAHI 是否对 DOTAv1.5-lite 有效？

_待用户回填实验结果后填写。重点分析：_

- _如果 small-vehicle 涨 ≥ 2，证明 SAHI 在已预切片数据集上仍有效_
- _如果总体涨但 small-vehicle 不涨，可能 SAHI 帮的是其他类_
- _如果完全无增益，说明 1024 预切片已经把小目标"放大"够了，再切无意义_

### 5.2 SAHI × NWD 是否正交？

_待回填。如果 Δ(D−C) ≈ Δ(B−A)，说明两个改进各自独立加成；如果 Δ(D−C) < Δ(B−A)，说明有重叠收益（NWD 已经吃掉了部分 SAHI 增益）。_

### 5.3 时间成本是否可接受？

_待回填。如果 8 小时全集推理对你可以接受，SAHI 可以作为最终评估流程；否则需要：_

- _加速：去掉 perform_standard_pred，或加大 slice_size_
- _或限定到一个固定子集做对比（如分层抽样 500 张）_

### 5.4 Bad Case 分析（待填）

_待回填。重点：_

- _SAHI 是否在 bridge 上回退（切片可能把桥切两半）？_
- _是否有跨 patch 重复检测（NMS 没合并干净）？_

## 6. 复现说明

完整命令见第 3 节。所有产出在：

- 预测 txt: `analysis/outputs/sahi/<run>/`
- 推理 meta: `analysis/outputs/sahi/<run>/sahi_meta.json`
- 评估结果: `analysis/outputs/errors/<run>/summary.json`

## 7. 结论与下一步（待填）

_待用户回填后填写。可能的结论分支：_

- **如果 SAHI +3+ mAP**：把 SAHI 集成进 `analyze_errors.py`，作为默认评估方式；用 NWD+SAHI 跑完整 100 epoch
- **如果 SAHI +0~1 mAP**：DOTAv1.5-lite 已预切片让 SAHI 价值有限，转 P2 头 + NWD 路线
- **如果 SAHI 退化**：分析 Bad Case，可能需要更小 slice 或调 NMS 阈值
