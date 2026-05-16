# 项目推进进度 - SAHI 切片推理实现完成

> 日期：2026-05-16  
> 主题：自实现 SAHI 风格切片推理，等待服务器执行对比实验

## 1. 本次完成

- [x] 修正历史文档中所有 `DOTA-split-lite` → `DOTAv1.5-lite`
- [x] SAHI 论文精读 → [`docs/paper/sahi.md`](../paper/sahi.md)
- [x] 调研 sahi 官方库对 OBB 的支持现状（0.11.20+ 名义支持但有 known issues）
- [x] 决策走自实现路线（Plan B），不依赖 sahi 库
- [x] SAHI 集成设计文档 → [`docs/design/sahi-inference.md`](../design/sahi-inference.md)
- [x] 实现 `analysis/sahi_inference.py`（约 220 行，自包含）
- [x] 本地烟雾测试通过（3 张 val 图 9 秒/张）
- [x] 代码开发文档 → [`docs/code/sahi-inference.md`](../code/sahi-inference.md)
- [x] 实验文档模板 → [`docs/experiment/2026-05-16-sahi-vs-standard.md`](../experiment/2026-05-16-sahi-vs-standard.md)（待回填）

## 2. 关键产出

| 文件 | 类型 | 状态 |
|------|------|------|
| `analysis/sahi_inference.py` | 代码（220 行） | 已实现，烟雾测试通过 |
| `docs/paper/sahi.md` | 论文笔记 | 完成 |
| `docs/design/sahi-inference.md` | 设计文档 | 完成 |
| `docs/code/sahi-inference.md` | 代码文档 | 完成 |
| `docs/experiment/2026-05-16-sahi-vs-standard.md` | 实验文档 | **模板已写，等服务器跑结果回填** |

## 3. 未完成 / 阻塞项

### 阻塞：实际对比实验需要在服务器执行

**根因**：20-epoch baseline 和 NWD 训练在 Linux 服务器（`/root/cy/...`）上完成，权重未拷贝到本地 Windows。本地 `yolo_dota_project/dota_runs/` 仅有更老的实验权重，与最新 20-epoch 对比无法直接对应。

**解决路径**：用户在服务器拉取最新代码（包含 sahi_inference.py），按实验文档第 3 节执行命令。

## 4. 下次推进点（IMPORTANT）

### 4.1 用户需要做的事

**第一步：服务器拉取代码**

```bash
cd /root/cy
git pull origin main
ls yolo_dota_project/analysis/sahi_inference.py  # 确认存在
```

**第二步：先跑 200 张样本探路（约 30 分钟）**

```bash
cd /root/cy/yolo_dota_project

python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_sample200 \
  --slice-size 512 --overlap 0.2 --device 0 \
  --max-images 200

python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --split val \
  --predictions analysis/outputs/sahi/baseline_sample200 \
  --prediction-format txt \
  --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/baseline_sahi_sample200
```

**第三步：根据样本结果决策**

| 200 张样本 mAP@0.5 表现 | 推荐下一步 |
|--------------------------|------------|
| 比之前 baseline 高 ≥ 2 | 跑全 val 完整推理 + NWD 对照（约 18 小时） |
| 持平或下降 | SAHI 在 DOTAv1.5-lite 无效，转 P2 头方向 |
| 涨 0~2 | 加大 overlap (0.3) 或减小 slice (384) 再试一次 |

**第四步：把结果回填到 `docs/experiment/2026-05-16-sahi-vs-standard.md`**

回填后通知我（粘贴 summary.json 的关键字段即可），我帮你做深度分析。

### 4.2 需要的上下文（下次会话）

下次开会话时先读：

1. `docs/progress/2026-05-16-sahi-implementation.md`（本文件）
2. `docs/experiment/2026-05-16-sahi-vs-standard.md`（带回填结果）
3. 如果决定继续 SAHI 路线：`docs/code/sahi-inference.md` 参数调优表

### 4.3 关键决策记录

| 决策 | 理由 |
|------|------|
| 不用 sahi 官方库 | OBB 支持有 known issues；自实现 220 行更可控；不增加依赖 |
| 自实现用 ultralytics.nms_rotated | 与训练时的 OBB NMS 完全一致 |
| Per-class NMS 用坐标 offset | 比循环每类 NMS 快 N 倍 |
| 默认 `perform_standard_pred=True` | 兼顾大目标，对应论文推荐配置 |
| 实验未本地跑 | 权重在服务器，强行用本地 yolo11s-obb.pt 跑会得到无意义结果 |

## 5. 风险与遗留问题

### 5.1 已知风险

1. **SAHI 在 DOTAv1.5-lite 上可能无增益**：因为该数据集已经把原 DOTA 大图预切片到 1024，再切 512 是"切片的切片"，价值可能远小于论文场景（VisDrone 是原图 1080p+）
2. **推理 9 倍慢**：全 val 约 9 小时；如果服务器 GPU 紧张需排队
3. **跨 patch NMS 可能漏合并**：当同一目标被切到多个 patch 都检测到时，NMS IoU 阈值 0.5 可能太严，建议初版用默认观察

### 5.2 未解疑问

- DOTAv1.5-lite 的预切片是 1024×1024 还是其他尺寸？这影响 SAHI slice_size 推荐值
  - 已知：val 有 3503 张，按原 DOTA val ~458 大图 × 平均 ~8 个 1024 切片 ≈ 3500 大致吻合
  - 暂定按 1024 假设，slice 512 + overlap 0.2

## 6. 当前项目快照（更新）

| 状态 | 内容 |
|------|------|
| 损失改进 | NWD 实现完成，C=16 整体 +0.7% mAP，小目标无增益 |
| **推理改进** | **SAHI 实现完成（本地烟雾测试通过），等服务器对比实验** |
| 架构改进 | P2 / CBAM / ResidualCBFuse 已实现，未与 NWD/SAHI 组合 |
| 工程优化 | Copy-Paste 增强 / 蒸馏未开始 |
| 分析工具 | analyze_object_sizes / analyze_errors 已用 |

## 7. 历史进度链

- [2026-05-16 NWD 损失集成](2026-05-16-nwd-loss.md) → 实现完成
- [2026-05-16 NWD 首轮实验结果分析（修正版）](2026-05-16-nwd-experiment-result.md) → C=16 整体微弱正向
- **[2026-05-16 SAHI 切片推理实现](2026-05-16-sahi-implementation.md)** ← 当前
- 下次候选：SAHI 实验结果回填 → 决策（继续 SAHI / 转 P2 / 转 Copy-Paste）
