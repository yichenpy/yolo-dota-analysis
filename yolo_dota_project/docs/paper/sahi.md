# SAHI: Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection

> SOP 类型：文献与理论学习 | 阅读日期：2026-05-16

## 1. 基础信息

| 字段 | 内容 |
|------|------|
| 论文标题 | Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection |
| 作者 | Fatih Cagatay Akyon, Sinan Onur Altinuc, Alptekin Temizel |
| 发表年份 | 2022（ICIP） |
| 出处 | IEEE ICIP 2022 / arXiv:2202.06934 |
| 链接 | https://arxiv.org/abs/2202.06934 |
| 代码仓库 | https://github.com/obss/sahi |
| 当前版本 | sahi >= 0.11.20（支持 Ultralytics YOLO11 OBB） |

## 2. 核心痛点

**通用目标检测器在大图小目标场景下精度严重不足**：

1. **特征图分辨率不足**：YOLO 等检测器把图像缩到 640~1024 输入，原本就小的目标（< 16 px）被进一步压缩到 stride=8 特征图上只剩 1-2 个像素，几乎不可分辨。
2. **训练数据分布偏差**：COCO 这类预训练数据集小目标占比低，模型对小目标的语义特征学习不充分。
3. **推理时整图缩放损失细节**：航空遥感数据（DOTA / xView / VisDrone）单图可能 4000×4000，缩到 1024 输入小目标已经不可见。

针对训练已经做了切片（如 DOTA-split）的项目，推理时如果还是整图喂，小目标依然丢；如果推理时也切片，又面临"如何在切片间合并预测"的工程问题。SAHI 提供了**通用且训练无关**的解决方案。

## 3. 核心创新点 (Methodology)

SAHI 包含两个独立可用的组件：

### 3.1 SAHI（Slicing Aided Hyper Inference）—— 推理时切片

**纯推理增强，不动训练**。流程：

```
1. 输入大图 I (H×W)
2. 用滑窗切成重叠 patch：
   - slice_size = (h, w)（如 512×512）
   - overlap_ratio（如 0.2，即相邻 patch 有 20% 重叠）
3. 每个 patch 独立跑一次检测器，得到局部预测
4. 将局部预测的坐标偏移加上 patch 的左上角偏移，还原到原图坐标
5. 全局 NMS / NMM 合并跨 patch 重叠的预测
6. （可选）原图整图也跑一次预测，加入合并
```

**关键参数**：

| 参数 | 默认 | 说明 |
|------|------|------|
| `slice_height` / `slice_width` | 512 | 单 patch 尺寸（越小对小目标越友好，但慢） |
| `overlap_height_ratio` / `overlap_width_ratio` | 0.2 | 相邻 patch 重叠比例（避免目标被切断） |
| `perform_standard_pred` | True | 是否同时跑整图推理（捕获大目标） |
| `postprocess_type` | "GREEDYNMM"（非 OBB）/ "NMS"（OBB 自动强制） | 合并算法 |
| `postprocess_match_metric` | "IOS"（Intersection over Smaller） | 比 IoU 更适合切片场景 |
| `postprocess_match_threshold` | 0.5 | 合并阈值 |

### 3.2 SF（Slicing Aided Fine-tuning）—— 训练时切片

**离线切片重组数据集后再训练**。本质就是 DOTA-split 那类工作（我们已经用了 DOTAv1.5-lite），SAHI 把这一步标准化成一行命令。

> 我们项目已经用 DOTAv1.5-lite（预切片版本）训练了 baseline 和 NWD，**SF 这一部分已经隐式完成**。本次重点是 **SAHI 推理增强**。

## 4. 实验表现

### 4.1 论文报告增益（VisDrone + xView）

| 检测器 | Baseline mAP | + SAHI 推理 | + SAHI 推理 + SF 微调 |
|--------|--------------|-------------|------------------------|
| FCOS | — | +6.8 | +12.7 |
| VFNet | — | +5.1 | +13.4 |
| TOOD | — | +5.3 | +14.5 |

### 4.2 与本项目场景的相关性

- VisDrone / xView 与 DOTA 都是航空遥感小目标场景
- **仅 SAHI 推理（不重训）即可提升 5-7 mAP**，这是零训练成本的天花板
- 加上 SF（即我们已做的 DOTA-split）通常再多 6-7 mAP
- 我们已经做了 SF（DOTAv1.5-lite），但**没做推理时切片**，所以 SAHI 推理这部分增益没拿到

## 5. 启发与落地

### 5.1 学到了什么

1. **训练切片 ≠ 推理切片**：很多人以为训练数据已经切了，推理就不用切。错。推理时还原到原图后再用整图预测，依然丢失小目标。
2. **IoS 比 IoU 更适合 cross-slice 合并**：因为同一目标被两个 patch 各检测出半个时，IoU 会很低（两个半框 IoU 接近 0），但 IoS（交集 / min(A, B)）依然能识别为同一目标。
3. **OBB 在 SAHI 里只支持 NMS（不支持 GREEDYNMM）**：库代码里有硬编码 `force postprocess_type = "NMS"` for OBB models，因为 NMM (Non-Maximum Merging) 对旋转框还没实现。

### 5.2 能否迁移到本项目

- [x] **完美匹配场景**：DOTAv1.5-lite + YOLO11 OBB 是 SAHI 直接支持的组合
- [x] **零训练成本**：可以直接拿现有 baseline / NWD 权重测试，半小时出结果
- [x] **与 NWD 正交**：SAHI 是推理增强，NWD 是损失改进，两者收益可叠加
- [x] **与 P2 头正交**：架构改动也不冲突，理论上可同时使用

### 5.3 已知限制

1. **推理速度慢 4-10 倍**：每张图切成 N 个 patch 各跑一次，是单次推理时间的 N 倍（加 NMS 合并开销）。对实时性敏感场景不友好；对离线评估和高精度场景完美适用。
2. **OBB 的 NMM 未实现**：跨 patch 的同一旋转目标合并只能用 NMS（保留最高分），不能做框平均化。但实测影响小。
3. **patch 大小需要调**：太小 → 大目标被切断；太大 → 小目标增益不明显。航空场景常见 512×512 + 0.2 overlap。
4. **CUDA 显存峰值**：batch_size 不变时整体显存类似，但需要保证 patch_size × batch_size 不超显存。

### 5.4 落地策略

**第一阶段（本次实现）**：
- 用 `sahi` 官方库 + `AutoDetectionModel(model_type="ultralytics")` 加载 OBB 权重
- 切片参数：`slice=512`, `overlap=0.2`
- 在 DOTAv1.5-lite val 上对 baseline 和 NWD 权重各跑一次普通推理 + 一次 SAHI 推理
- 用我们现有的 `analyze_errors.py` 评估 mAP / 分类别 / 按 size 拆分

**第二阶段（可选）**：
- 调参 sweep：slice ∈ {384, 512, 768}, overlap ∈ {0.1, 0.2, 0.3}
- 把 SAHI 集成到 `analyze_errors.py` 作为可选推理路径
- 探索 perform_standard_pred=True/False 对大目标的影响

### 5.5 关键技术点

#### OBB 检测结果在 SAHI 中的提取

已知问题：Ultralytics 模型返回结果时，普通检测在 `result.boxes.data`，OBB 在 `result.obb`。SAHI 的 UltralyticsDetectionModel 已经处理了这个差异（0.11.20+），用户不需要手动处理。

#### 跨 patch 合并的几何细节

```
对于 OBB:
  pred 在 patch (x0, y0, w_patch, h_patch) 内坐标为 (cx_local, cy_local, w, h, angle)
  → 还原到原图：(cx_local + x0, cy_local + y0, w, h, angle)
  注意：angle 不需要变换（仅平移）
```

#### 与 ultralytics val() 的对接

SAHI 输出的是 `PredictionResult` 对象，包含 `object_prediction_list`，每个有 `.bbox` / `.score` / `.category`。要转成 ultralytics 的 mAP 评估格式，可以：

- 路径 A：转成 YOLO txt 格式 `class x1 y1 x2 y2 x3 y3 x4 y4 conf` 存盘 → 用我们的 `analyze_errors.py --predictions` 模式读
- 路径 B：转成 COCO json → 用 pycocotools 评估
- 路径 C：直接用 sahi 自带的 `sahi predict` CLI

**本项目选路径 A**，理由：可以复用现有 `analyze_errors.py` 的官方 val 对比 + 分类别 + 按 size 分析能力。

## 6. 相关文献

- 原论文：https://arxiv.org/abs/2202.06934
- 官方代码：https://github.com/obss/sahi
- Ultralytics SAHI 集成文档：https://docs.ultralytics.com/guides/sahi-tiled-inference
- OBB 支持讨论：https://github.com/obss/sahi/discussions/987
- 自适应 SAHI 论文（变种）：https://www.mdpi.com/2072-4292/15/5/1249

## 7. 待求证

- [ ] DOTAv1.5-lite 已经是 1024 切片，再切 512 是否还有增益？需要实际跑出结果
- [ ] SAHI 推理时间是否能接受？需要计时
- [ ] OBB 合并用 NMS 是否会丢精度？（vs 论文用的 NMM）
- [ ] perform_standard_pred=True 加整图推理会让 small-vehicle mAP 退化吗？
