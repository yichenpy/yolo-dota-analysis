# SAHI 切片推理脚本开发文档

> SOP 类型：代码开发 | 开发日期：2026-05-16

## 1. 需求与逻辑思路

### 1.1 功能描述

实现一个独立的 SAHI 风格切片推理脚本，输入 YOLO11 OBB 权重 + 数据集，输出 YOLO OBB 预测 txt，直接喂给 `analyze_errors.py` 做精度评估。

### 1.2 设计思路

**自实现切片**（不依赖 `sahi` 库），原因：

- `sahi` < 0.12 对 OBB 支持仍有 known issues（github discussion #987）
- 自实现可直接使用 ultralytics 自己的 `nms_rotated`，对 OBB 数学最干净
- 避免新依赖污染 ultralytics 环境
- 代码透明，约 220 行，易于调试

### 1.3 算法流程

```
对每张原图 I (H×W)：
  1. 切片：滑窗 slice_size × slice_size，相邻重叠 overlap_ratio
     → patches = [(patch_i, x0_i, y0_i), ...]
  2. 对每个 patch 跑 model.predict() → 得到局部 OBB 预测
  3. 局部 OBB 中心 += (x0_i, y0_i) → 还原到原图坐标
  4. （可选）整图也跑一次 model.predict(imgsz=1024)，捕获大目标
  5. 全局 NMS：按类别 ID 做坐标 offset 实现 per-class NMS
     → 用 ultralytics.utils.ops.nms_rotated
  6. 转 OBB → 4 corner points → 写 YOLO OBB txt
```

### 1.4 复杂度

- 时间：每张原图 = N_patches × per_patch_inference + 1 × full_image_inference + NMS
  - 1024×1024 输入 + slice=512 + overlap=0.2 → 3×3 = 9 patches + 1 full = 10 inferences
  - GPU 实测 ~9 秒/张（含 NMS 时间限制告警）
- 空间：单图 batch=1，显存占用与原始 model.predict 一致

## 2. 核心方法与技术栈

| 项目 | 选择 | 理由 |
|------|------|------|
| 主框架 | Ultralytics 8.3+ | 复用 model.predict() / nms_rotated |
| 不依赖 sahi 库 | 自实现 | OBB 支持稳定性 + 代码透明 |
| OBB NMS | ultralytics.utils.ops.nms_rotated | 基于 probiou，与训练对齐 |
| Per-class NMS | 坐标 offset trick | 单次 NMS 调用处理所有类，比循环每类一次快 |

## 3. 文件清单

| 文件 | 作用 |
|------|------|
| `analysis/sahi_inference.py` | 主脚本（CLI 入口） |
| `docs/paper/sahi.md` | SAHI 论文阅读笔记 |
| `docs/design/sahi-inference.md` | 集成设计方案 |
| `docs/code/sahi-inference.md` | 本文档 |
| `analysis/outputs/sahi/<run_name>/*.txt` | 输出预测（每图一份） |
| `analysis/outputs/sahi/<run_name>/sahi_meta.json` | 推理参数与统计 |

## 4. 使用说明

### 4.1 环境配置

无需新依赖。原有 `cuda` conda 环境（ultralytics + torch）即可：

```powershell
& "E:\miniconda3\envs\cuda\python.exe" -c "import ultralytics; print(ultralytics.__version__)"
# 应输出 8.3.93 或更新
```

### 4.2 快速验证（小批量烟雾测试）

```powershell
cd E:\cy\yolo_dota_project
& "E:\miniconda3\envs\cuda\python.exe" analysis\sahi_inference.py `
  --weights yolo11s-obb.pt `
  --dataset datasets\DOTAv1.5-lite\data.yaml `
  --output-dir analysis\outputs\sahi\smoke_test `
  --slice-size 512 --overlap 0.2 `
  --device 0 `
  --max-images 3
```

预期：3 张图约 30 秒，输出 3 个 .txt + 1 个 sahi_meta.json。

### 4.3 完整推理 + 评估流程

#### Step 1：对 baseline 权重跑 SAHI 推理

```powershell
# 注意：本地 dota_runs/ 下没有 20-epoch 权重；以下命令在 Linux 服务器上运行
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_20ep \
  --slice-size 512 --overlap 0.2 \
  --device 0
```

预计时间：3503 张 × ~9 秒 ≈ 9 小时。建议先用 `--max-images 200` 看苗头：

```bash
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_sample200 \
  --slice-size 512 --overlap 0.2 \
  --device 0 \
  --max-images 200
```

#### Step 2：用 SAHI 预测跑 mAP 评估

```bash
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --split val \
  --predictions analysis/outputs/sahi/baseline_20ep \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/baseline_sahi
```

输出在 `analysis/outputs/errors/baseline_sahi/summary.json`，里面有总 mAP / 分类别 / 按 size。

#### Step 3：对 NWD 权重同样操作，得到对比

```bash
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_nwd_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/nwd_20ep \
  --device 0

python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --split val \
  --predictions analysis/outputs/sahi/nwd_20ep \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/nwd_sahi
```

## 5. 关键参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--weights` | path | （必填） | YOLO11 OBB 权重路径 |
| `--dataset` | path | （必填） | data.yaml 路径 |
| `--split` | str | val | 数据集 split |
| `--output-dir` | path | （必填） | 预测 txt 与 meta.json 输出目录 |
| `--slice-size` | int | 512 | 单 patch 边长（h=w） |
| `--overlap` | float | 0.2 | 相邻 patch 重叠比例 |
| `--conf-thres` | float | 0.001 | 检测置信度阈值（评估时设低，让 mAP 完整） |
| `--nms-iou` | float | 0.5 | 全局 NMS IoU 阈值 |
| `--no-standard-pred` | flag | False | 关闭整图推理，加速但可能漏大目标 |
| `--imgsz` | int | 1024 | 整图推理 imgsz |
| `--device` | str | "0" | cuda:0 / cpu |
| `--max-images` | int | None | 限制处理图像数（调试用） |
| `--max-det-per-slice` | int | 300 | 单 patch 最大检测数 |

### 5.1 关键参数调优建议

| 场景 | 推荐配置 |
|------|----------|
| **DOTAv1.5-lite (1024 预切片)** | `--slice-size 512 --overlap 0.2`（推荐起点） |
| 加速 2x（小目标可能略降） | 加 `--no-standard-pred` |
| 更激进切片（小目标更友好） | `--slice-size 384 --overlap 0.3` |
| 大图未切片（如原 DOTA） | `--slice-size 1024 --overlap 0.2 --no-standard-pred` |
| GPU 显存紧张 | 调小 `--max-det-per-slice` 到 100 |

## 6. 异常处理与避坑指南

### 6.1 已知问题与解决方案

| 问题 | 触发条件 | 现象 | 解决方案 |
|------|----------|------|----------|
| `WARNING ⚠️ NMS time limit X.Xs exceeded` | conf_thres 太低，候选过多 | 训练日志中出现警告，但不影响结果 | 可忽略；如果担心，调高 `--max-det-per-slice` 或 `--conf-thres 0.01` |
| 推理慢于预期 | 每图 9 patches + 1 full = 10 次推理 | 全 val 8-10 小时 | 用 `--max-images 200` 先跑样本；或 `--no-standard-pred` |
| 输出 txt 全为空 | 模型在该数据集上未训练好 | 0 predictions per image | 检查权重 vs 数据集匹配；用 model.val() 单独验证权重可用 |
| 显存 OOM | `--slice-size` 太大 + per-image batch | CUDA OOM | 减小 `--slice-size` 或确认 `--device 0` 可用 |
| 坐标超出原图边界 | NMS 之前的预测在 patch 边缘 | analyze_errors 报警 | 已自动 clip 在写文件前；如仍有警告可加额外 clamp |
| `prediction-layout` 不匹配 | 默认就是 class_xyxyxyxy_conf | analyze_errors 解析错 | 确认 `analyze_errors --prediction-layout class_xyxyxyxy_conf` |

### 6.2 调试步骤

1. **冒烟测试** (`--max-images 3`)：验证脚本可跑通
2. **小样本评估** (`--max-images 100` + analyze_errors)：得到初步 mAP，与 baseline 对比看苗头
3. **完整推理**：如果小样本显示 +2 mAP 以上，再花时间跑全集
4. **如果无增益**：换参数（slice=384、overlap=0.3）；或确认本数据集 SAHI 是否适用

### 6.3 性能基准（参考）

| 配置 | 单图时间 (4090) | 全 val (3503 张) |
|------|------------------|-------------------|
| slice=512, overlap=0.2, +standard | ~9 s | ~8.7 h |
| slice=512, overlap=0.2, no-standard | ~5 s | ~4.9 h |
| slice=640, overlap=0.2, +standard | ~6 s | ~5.8 h |
| slice=384, overlap=0.3, +standard | ~14 s | ~13.6 h |

## 7. 与现有 analyze_errors 的协同

SAHI 输出的 txt 完全兼容 `analyze_errors.py`：

```bash
# SAHI 推理
python analysis/sahi_inference.py --weights X --dataset Y --output-dir Z

# 误差分析（同一套接口）
python analysis/analyze_errors.py \
  --dataset Y --split val \
  --predictions Z \                              # SAHI 输出目录
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/run_name
```

这样可以拿到与之前 NWD vs Baseline 对比一致的分类别 / 按 size 分析。

## 8. 后续维护点

- ultralytics 升级后需要验证 `nms_rotated` 签名（当前: `(boxes, scores, threshold, use_triu)`）
- 若引入 multi-batch 切片推理（一次性把多个 patch 喂进模型），可提速 2-3 倍
- 考虑加入 `--export-coco` 选项输出 COCO json 格式给 pycocotools
- 考虑把 SAHI 集成进 `analyze_errors.py` 的 `--use-sahi` flag，避免两步运行
