# 项目推进进度 - SAHI 200 张 sweep 完成

> 日期：2026-05-16  
> 主题：SAHI 配置 sweep（5 组）跑完，确定最优配置，等待全 val 决战

## 1. 本次完成

- [x] 在固定 200 张 val 子集上跑 5 组对比（Standard + 4 组 SAHI 变体）
- [x] 借助 `--predictions-only` + `standard_inference.py --from-sahi-dir` 实现 apples-to-apples 对比
- [x] 找到最优 SAHI 配置：**slice=512, overlap=0.2, nms-iou=0.65**
- [x] 完整实验文档 → [`docs/experiment/2026-05-16-sahi-sweep-200.md`](../experiment/2026-05-16-sahi-sweep-200.md)

## 2. 关键结论

### 2.1 SAHI 在 DOTAv1.5-lite 上是有效的

| 指标 | Standard | 最优 SAHI (C) | Δ |
|------|----------|----------------|---|
| recall | 82.0% | **91.3%** | **+9.3pp** |
| miss_rate | 18.0% | **8.7%** | **−9.3pp** |
| small-vehicle missed | 1442 | **487** | **−66.2%** |

**这是首个真正打中 small-vehicle 痛点的方案**。前一轮 NWD 在 small-vehicle 上 +0/-0；SAHI 直接砍掉 66% 漏检。

### 2.2 SAHI 的代价

| 维度 | 代价 |
|------|------|
| swimming-pool 漏检 | 50 → 191（4 倍恶化） |
| 22-62 px 中等目标 | miss_rate 退 5-7pp |
| 推理速度 | 9× 慢于标准推理 |
| FP 数量 | 5× 增长（仅 conf<0.01 长尾，实际部署影响小） |

净增益 = 多救 889 个目标，代价 = 失去约 100-150 个中等目标 → **净 +700-800 真阳性 / 200 张**

### 2.3 不再需要进一步调 SAHI 参数

| 已尝试 | 结论 |
|--------|------|
| nms-iou 0.5 → 0.65 | ✅ 全 size 段都改善 |
| slice 512 → 768 | ❌ 全面退化 |
| overlap 0.2 → 0.3 | ❌ 几乎持平 |

剩下可探索的小幅改进（nms-iou 0.7 / class-conditional）边际收益有限，**应该直接跑全 val 落实生产配置**。

## 3. 未完成 / 阻塞项

- [ ] 全 3503 val 上跑 Run C 配置（约 9 小时）
- [ ] 同样配置在 NWD 权重上跑（验证 SAHI × NWD 是否正交叠加）
- [ ] 把 mAP 数字加入 summary.json（目前只有 recall / miss_rate）

## 4. 下次推进点（IMPORTANT）

### 4.1 服务器执行（约 18 小时 GPU 时间）

```bash
cd /root/cy && git pull origin main && cd yolo_dota_project

# 1. Baseline 全 val + SAHI（Run C 配置）
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_baseline_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/baseline_full_iou065 \
  --slice-size 512 --overlap 0.2 --nms-iou 0.65 --device 0

# 2. NWD 全 val + SAHI（同配置）
python analysis/sahi_inference.py \
  --weights runs/obb/test/yolo11s_nwd_20ep/weights/best.pt \
  --dataset datasets/DOTAv1.5-lite/data.yaml \
  --output-dir analysis/outputs/sahi/nwd_full_iou065 \
  --slice-size 512 --overlap 0.2 --nms-iou 0.65 --device 0

# 3. 评估（全 val，不加 --predictions-only）
python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/baseline_full_iou065 \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/baseline_full_iou065 \
  --skip-official-val

python analysis/analyze_errors.py \
  --dataset datasets/DOTAv1.5-lite/data.yaml --split val \
  --predictions analysis/outputs/sahi/nwd_full_iou065 \
  --prediction-format txt --prediction-layout class_xyxyxyxy_conf \
  --output-dir analysis/outputs/errors/nwd_full_iou065 \
  --skip-official-val
```

### 4.2 跑完后做什么

把两个 summary.json 贴给我，我做：

1. **2×2 大表对比**：Baseline 标准 / Baseline+SAHI / NWD 标准 / NWD+SAHI
2. **SAHI 边际收益**：(B+SAHI − B) vs (NWD+SAHI − NWD)，看 NWD 是否吃掉了 SAHI 部分增益
3. **分类别完整 AP 表**：16 类全部
4. **最终决策**：要不要把 SAHI 作为标准评估流程，是否值得做 class-conditional SAHI

### 4.3 关键决策记录

| 决策 | 理由 |
|------|------|
| 不再调 SAHI 参数 | sweep 已充分，C 是局部最优；继续微调边际 < 1pp 不值得 |
| 不立刻做 class-conditional SAHI | 需先看全 val 整体结果；如果中大目标退化在全 val 上没有放大，class-conditional 优先级降低 |
| NWD 权重也要跑一次 SAHI | 验证两个改进是否正交叠加（如果叠加，得到 baseline+NWD+SAHI 最终方案） |
| 不重训 SAHI fine-tuning | DOTAv1.5-lite 已是预切片，再切等于"切片再切片"，论文中此情景增益有限 |

## 5. 风险与遗留问题

### 5.1 已知风险

1. **swimming-pool 全 val 上可能更严重退化**（200 张是抽样，未来某些 split 可能集中泳池）
2. **NWD 权重在 SAHI 下可能不再有优势**：NWD 主要受益大目标，SAHI 改善小目标，可能"井水不犯河水"也可能"互相抵消"
3. **summary.json 不含 mAP**：需要后续加入或用别的工具评估

### 5.2 未解疑问

- SAHI 增益在全 val 上能维持 +9.3pp recall 吗？（200 张可能恰好对 SAHI 友好）
- 加上 class-conditional SAHI 能否同时获得 small-vehicle 增益且不输 swimming-pool？

## 6. 当前项目快照（更新）

| 状态 | 内容 |
|------|------|
| 损失改进 | NWD 完成，+0.7% mAP，**对 small-vehicle 无效** |
| **推理改进** | **SAHI 200 张 sweep 完成，Run C 配置 recall +9.3pp，small-vehicle −66% 漏检** |
| 待跑 | Baseline+SAHI 全 val、NWD+SAHI 全 val |
| 架构改进 | P2 / CBAM 已有但未与 SAHI 组合实验 |
| 数据增强 | 未开始（SAHI 已经解决了大部分痛点，优先级降低） |

## 7. 历史进度链

- [2026-05-16 NWD 损失集成](2026-05-16-nwd-loss.md)
- [2026-05-16 NWD 首轮实验结果（修正版）](2026-05-16-nwd-experiment-result.md) → +0.7% mAP，small-vehicle 持平
- [2026-05-16 SAHI 切片推理实现](2026-05-16-sahi-implementation.md)
- **[2026-05-16 SAHI 200 张 sweep 完成](2026-05-16-sahi-sweep-result.md)** ← 当前
- 下次：SAHI 全 val 实验结果分析 → 最终决策
