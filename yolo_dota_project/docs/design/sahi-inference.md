# SAHI 切片推理集成设计方案

> SOP 类型：设计文档（代码开发前置）| 日期：2026-05-16

## 1. 设计目标

把 SAHI 切片推理接入到现有 YOLO11 OBB 训练成果上，**不重新训练**，验证 DOTAv1.5-lite 上 baseline 与 NWD 权重的小目标推理增益天花板。

要求：

1. **不修改 ultralytics 源码**：使用官方 `sahi` 库 + AutoDetectionModel
2. **复用现有分析链路**：输出 YOLO OBB txt 格式，直接喂给 `analyze_errors.py`
3. **可选参数化**：slice_size、overlap、conf、postprocess 阈值都可 CLI 控制
4. **零训练成本**：直接拿 `dota_runs/yolo11s_baseline_20ep/weights/best.pt` 跑

## 2. 输入输出契约

### 2.1 输入

| 项目 | 路径 / 内容 |
|------|-------------|
| 模型权重 | `dota_runs/yolo11s_baseline_20ep/weights/best.pt`（或 nwd_20ep 同位置） |
| 验证图像 | `datasets/DOTAv1.5-lite/images/val/*.png`（或 jpg） |
| 数据配置 | `datasets/DOTAv1.5-lite/data.yaml`（用于 names） |

### 2.2 输出

为每张验证图像生成一个 YOLO OBB 格式预测 txt：

```
<output_dir>/
  P0001.txt       # 对应 images/val/P0001.png
  P0002.txt
  ...
  sahi_meta.json  # 推理参数、耗时、统计
```

每个 .txt 每行一个预测：

```
class_id  x1 y1 x2 y2 x3 y3 x4 y4  conf
```

坐标为**原图像素坐标**（不归一化），与 `analyze_errors.py --prediction-layout class_xyxyxyxy_conf` 兼容。

## 3. 调用链与集成点

```
sahi_inference.py main()
├── 加载 data.yaml → 获取 val 图像目录
├── sahi.AutoDetectionModel.from_pretrained(model_type="ultralytics", model_path=...)
│   ↑ sahi >= 0.11.20 原生支持 YOLO11 OBB
├── 遍历每张图：
│   ├── sahi.predict.get_sliced_prediction(image, model, slice_h, slice_w, overlap_h, overlap_w, postprocess_type="NMS")
│   │   ↑ OBB 时 sahi 自动 force postprocess_type="NMS"
│   ├── 提取 PredictionResult.object_prediction_list
│   ├── 对每个 prediction，从 OBB rotated rectangle 转成 4 corner points
│   └── 写入 <image_stem>.txt
└── 输出 sahi_meta.json：参数、总耗时、平均每张耗时、预测总数
```

## 4. 关键技术细节

### 4.1 sahi 加载 Ultralytics OBB 模型

```python
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="dota_runs/yolo11s_baseline_20ep/weights/best.pt",
    confidence_threshold=0.001,   # 评估时设很低，让 mAP 完整
    device="cuda:0",
)
```

### 4.2 切片推理

```python
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image

result = get_sliced_prediction(
    read_image(image_path),
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred=True,            # 同时跑整图，捕获大目标
    postprocess_type="NMS",                # OBB 强制 NMS
    postprocess_match_metric="IOS",        # Intersection over Smaller
    postprocess_match_threshold=0.5,
    verbose=0,
)
```

### 4.3 OBB 结果提取

sahi 的 `ObjectPrediction` 对 OBB 在 `.bbox` 上存的是 axis-aligned 框，但 `mask` 或 `.orig_obb` 会包含旋转信息。需要查实际 API 行为：

**预期方案**：访问 `obj.bbox.minx/miny/maxx/maxy` 拿 AABB 作为 fallback；如果 sahi 提供 `obj.rbox` 或 `obj.rotated_box` 属性就用它。

**Plan B（如果 sahi 没暴露旋转信息）**：直接在 sahi 库基础上自己写切片循环：
- 用 ultralytics 模型在每个 slice 上跑预测
- 手动平移 OBB 坐标到原图
- 用 ultralytics 的 `rotated_nms`（基于 probiou）合并

为了稳健，**首版同时实现 Plan B**，避免 sahi OBB 提取细节的不确定性。Plan B 代码自包含，约 150 行。

### 4.4 OBB → 4 corner points 转换

给定 OBB (cx, cy, w, h, angle):

```python
def obb_to_corners(cx, cy, w, h, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    w2, h2 = w / 2.0, h / 2.0
    # 4 个角点（相对中心）
    corners = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]
    # 旋转 + 平移
    return [(cx + x * cos_a - y * sin_a, cy + x * sin_a + y * cos_a) for x, y in corners]
```

### 4.5 跨 patch 合并（Plan B 用）

```python
import torch
from ultralytics.utils.ops import nms_rotated

# all_preds: list of (cx, cy, w, h, angle, conf, class) over all patches
boxes = torch.tensor([[p[0], p[1], p[2], p[3], p[4]] for p in all_preds])  # (N, 5)
scores = torch.tensor([p[5] for p in all_preds])
classes = torch.tensor([p[6] for p in all_preds])
keep = nms_rotated(boxes, scores, iou_threshold=0.5)
```

## 5. CLI 参数设计

`analysis/sahi_inference.py`:

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--weights` | path | （必填） | YOLO11 OBB 权重 |
| `--dataset` | path | （必填） | data.yaml |
| `--split` | str | val | 用哪个 split |
| `--output-dir` | path | analysis/outputs/sahi/ | 预测 txt 输出目录 |
| `--slice-size` | int | 512 | 切片大小（h=w） |
| `--overlap` | float | 0.2 | 重叠比例（h 和 w 都用这个） |
| `--conf-thres` | float | 0.001 | 检测置信度阈值 |
| `--nms-iou` | float | 0.5 | 合并 NMS 的 IoU 阈值 |
| `--no-standard-pred` | flag | False | 关闭整图推理（仅切片） |
| `--imgsz` | int | 1024 | 整图推理的 imgsz（仅 perform_standard_pred 时用） |
| `--device` | str | 0 | cuda:0 / cpu |
| `--max-images` | int | None | 限制处理图像数（调试用） |
| `--use-plan-b` | flag | False | 强制使用 Plan B（自实现切片，绕过 sahi） |

## 6. 端到端使用示例

```powershell
# 用 baseline 权重跑 SAHI
& "E:\miniconda3\envs\cuda\python.exe" yolo_dota_project\analysis\sahi_inference.py `
  --weights yolo_dota_project\dota_runs\yolo11s_baseline_20ep\weights\best.pt `
  --dataset yolo_dota_project\datasets\DOTAv1.5-lite\data.yaml `
  --output-dir yolo_dota_project\analysis\outputs\sahi\baseline_sahi `
  --slice-size 512 --overlap 0.2 `
  --device 0

# 用 SAHI 预测跑误差分析（拿到对比 mAP）
& "E:\miniconda3\envs\cuda\python.exe" yolo_dota_project\analysis\analyze_errors.py `
  --dataset yolo_dota_project\datasets\DOTAv1.5-lite\data.yaml `
  --split val `
  --predictions yolo_dota_project\analysis\outputs\sahi\baseline_sahi `
  --prediction-format txt `
  --prediction-layout class_xyxyxyxy_conf `
  --output-dir yolo_dota_project\analysis\outputs\errors\baseline_sahi
```

## 7. 风险与边界条件

| 风险 | 现象 | 缓解 |
|------|------|------|
| sahi OBB 结果坐标精度损失 | 后续 NMS 合并后框不准 | 同时实现 Plan B 做交叉验证 |
| 推理慢 5-10 倍 | 全 val 跑 1 小时+ | 加 `--max-images 100` 做快速验证 |
| 显存峰值 | slice_size × batch 超显存 | 单图 batch=1 推理（slice 内部 batch 由 sahi 控制） |
| 跨 patch 同一目标合并失败 | 同物体两个高分预测都保留 | 调高 NMS IoU 阈值（如 0.3） |
| 大目标被切断 | bridge 类被 patch 切成两半 | 启用 `perform_standard_pred=True`，整图也跑一次 |
| sahi 版本不兼容 | < 0.11.20 不支持 OBB | requirements 加约束 |

## 8. 评估方案

实验需要对 4 个组合各跑一次评估，得到完整 4×指标表：

| 组合 | 权重 | 推理方式 |
|------|------|----------|
| A. baseline 标准推理 | baseline_20ep | model.val() |
| B. baseline + SAHI | baseline_20ep | get_sliced_prediction |
| C. NWD 标准推理 | nwd_20ep | model.val() |
| D. NWD + SAHI | nwd_20ep | get_sliced_prediction |

**关键对比**：

- B vs A：SAHI 在 baseline 上带来多少增益（纯 SAHI 效果）
- D vs C：SAHI 在 NWD 上带来多少增益（验证是否叠加）
- D vs A：组合方案 vs 原始 baseline 的最终增益

A 和 C 我们已经有结果（在 docs/experiment/2026-05-16-nwd-vs-baseline-20ep.md）。本次只需跑 B 和 D。

## 9. 后续扩展（不在本次范围）

- [ ] 把 SAHI 直接集成进 `analyze_errors.py`（用 `--use-sahi` flag）
- [ ] SAHI fine-tuning 模式（重训时也用切片增强）
- [ ] 自适应 SAHI（根据图像内容动态决定 slice_size）
- [ ] Multi-scale TTA + SAHI 组合
