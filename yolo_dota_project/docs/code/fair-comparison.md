# SAHI vs Standard 公平对比工具链

> SOP 类型：代码开发 | 开发日期：2026-05-16

## 1. 需求与逻辑思路

### 1.1 问题背景

`sahi_inference.py` 在小样本（如 `--max-images 200`）上跑出预测后，直接喂给 `analyze_errors.py` 会得到**伪指标**：`analyze_errors` 默认遍历**全部** val 图，没有预测文件的图会被算作 0 prediction → 该图所有 GT 都进 missed，miss_rate 被推高到 93%+。

这次 200 张样本评估就是这个情况，根本没法判断 SAHI 是否有效。

### 1.2 解决方案

两个独立但配套的小工具：

1. **`analyze_errors.py --predictions-only`**：评估时跳过没有预测文件的图
2. **`analysis/standard_inference.py --from-sahi-dir`**：对同一批 SAHI 已覆盖的图跑标准推理对照

组合起来就是 apples-to-apples 对比。

## 2. 改动清单

| 文件 | 类型 | 改动 |
|------|------|------|
| `analysis/analyze_errors.py` | 修改 | 新增 `--predictions-only` flag |
| `analysis/standard_inference.py` | 新增 | 镜像 sahi_inference 接口的标准推理脚本 |

### 2.1 `--predictions-only` 的语义

启用时，在 `load_predictions()` 后立即过滤 `image_paths`，只保留 `prediction_map[image]` 非空的图。

```python
if args.predictions_only:
    image_paths = [p for p in image_paths if prediction_map.get(p)]
```

副作用：

- 后续所有按图遍历的步骤都用这个缩小后的集合
- `summary.json` 的 `image_count`、`total_gt` 也会反映过滤后的数值
- 不影响 `--max-images`（两个 flag 可叠加）

### 2.2 `standard_inference.py` 与 `sahi_inference.py` 的对齐

| 项目 | sahi_inference | standard_inference |
|------|----------------|---------------------|
| 输入接口 | --weights / --dataset / --output-dir | 一致 |
| 输出格式 | YOLO OBB txt | **完全一致** |
| 切片 | 是 | 否（整图 model.predict） |
| --from-sahi-dir | — | 新增：只跑 SAHI 已覆盖的图 |
| OBB NMS | rotated_nms 跨版本包装 | 走 ultralytics 内置 NMS（model.predict 内部） |

两者的预测 txt 可以直接互换喂给 `analyze_errors.py`，不需要任何转换。

## 3. 完整使用流程（公平对比）

### 3.1 服务器执行步骤

```bash
cd /root/cy && git pull origin main && cd yolo_dota_project

# 1. SAHI 推理 200 张（已经做过，跳过）
# python analysis/sahi_inference.py \
#   --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
#   --dataset datasets/DOTAv1.5-lite/data.yaml \
#   --output-dir analysis/outputs/sahi/baseline_sample200 \
#   --slice-size 512 --overlap 0.2 --device 0 \
#   --max-images 200

# 2. 标准推理对照（同一批 200 张）
python analysis/standard_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_sample200_std \
  --from-sahi-dir analysis/outputs/sahi/baseline_sample200 \
  --device 0

# 3. 公平评估 SAHI（只算 200 张）
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/baseline_sample200 \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --predictions-only \
  --output-dir analysis/outputs/errors/baseline_sahi_200_fair \
  --skip-official-val

# 4. 公平评估 Standard（同 200 张）
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/baseline_sample200_std \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --predictions-only \
  --output-dir analysis/outputs/errors/baseline_std_200_fair \
  --skip-official-val
```

### 3.2 对比方法

打开两个 `summary.json`，看：

| 字段 | 含义 |
|------|------|
| `image_count` | 应该都是 200 |
| `total_gt` | 应该完全相同 |
| `matched_pairs` | 越高越好 |
| `missed_count` | 越低越好 |
| `false_count` | 越低越好（注意 SAHI 通常 FP 多） |
| `miss_rate` | 关键指标 |
| `missed_by_class.small_vehicle` | 小目标核心指标 |
| `missed_by_class.bridge` | 长条目标 |

## 4. 烟雾测试结果（本地 yolo11s-obb.pt，2 张图）

| 指标 | Standard | SAHI |
|------|----------|------|
| images | 2 | 2 |
| gt | 96 | 96 |
| predictions | 128 | 280 |
| matched | 92 | **96** |
| missed | 4 | **0** |
| miss_rate | 0.042 | **0.000** |
| false | 36 | 184 |

注意：

- 这只是 COCO 预训练权重在 2 张图上的趋势，**不能外推到全 val**
- 但已经反映 SAHI 典型行为：recall 提升 + FP 增加
- mAP 是否净增益取决于 PR 曲线积分；用 200 张全集才能判断

## 5. 避坑指南

| 问题 | 解决 |
|------|------|
| `--from-sahi-dir` 报 "No images matched any .txt" | 确认 SAHI 输出目录里的 .txt stem 与 val 图像 stem 完全相同（不带扩展名） |
| `--predictions-only` 后 image_count = 0 | 检查 prediction 目录路径正确性、prediction-layout 配置 |
| 两个 summary 的 total_gt 不一致 | 不应该发生；如果有，说明两次评估覆盖图集不同，重新跑 |
| Standard 推理也跑 1024 imgsz 太慢 | 默认 `--imgsz 1024` 已经是训练 imgsz，与训练一致；如要加速可降到 640 |

## 6. 后续扩展

- [ ] 把 SAHI vs Standard 自动化成一个 `compare.py` 脚本，一键出表
- [ ] 加入 mAP 列（目前 summary.json 没有，要手动算或加 model.val 集成）
- [ ] 支持多种 slice_size 的 sweep（512 vs 384 vs 640）
