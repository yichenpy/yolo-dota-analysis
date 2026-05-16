# NWD: A Normalized Gaussian Wasserstein Distance for Tiny Object Detection

> SOP 类型：文献与理论学习 | 阅读日期：2026-05-16

## 1. 基础信息

| 字段 | 内容 |
|------|------|
| 论文标题 | A Normalized Gaussian Wasserstein Distance for Tiny Object Detection |
| 作者 | Jinwang Wang, Chang Xu, Wen Yang, Lei Yu（武汉大学） |
| 发表年份 | 2021（arXiv 预印本）；2022 扩展版发表于 ISPRS Journal of Photogrammetry and Remote Sensing |
| 出处 | arXiv:2110.13389 |
| 链接 | https://arxiv.org/abs/2110.13389 |
| 代码仓库 | https://github.com/jwwangchn/NWD |
| 扩展版（航空遥感专用） | "Detecting tiny objects in aerial images: A normalized Wasserstein distance and a new benchmark", ISPRS JPRS 2022 |

## 2. 核心痛点

**IoU 度量对微小目标的位置偏移过度敏感。**

具体表现：

1. **离散性问题**：对于 6×6 的微小目标，1 个像素的位置偏移会让 IoU 从 0.65 直接掉到 0.39，而对 36×36 的中等目标，相同偏移 IoU 仅从 0.92 掉到 0.88。
2. **梯度断裂**：当预测框和 GT 完全不重叠时，IoU = 0，损失函数的梯度无法回传位置信息（DIoU/GIoU 部分缓解但仍不理想）。
3. **NMS 抑制不公**：基于 IoU 的 NMS 在小目标上很难找到合适阈值。
4. **正负样本分配不平衡**：anchor 与小 GT 的 IoU 很难达到 0.5+，导致小目标正样本极少。

DOTA / AI-TOD 等航空遥感场景中，目标平均像素只有几十甚至个位数（AI-TOD 平均 12.8px），上述问题被放大。

## 3. 核心创新点 (Methodology)

### 3.1 关键思路

把边界框视为 **2D 高斯分布**，用 **Wasserstein 距离**（最优传输距离）衡量两个框的相似度，再归一化映射到 [0, 1] 范围作为 IoU 的替代。

### 3.2 边界框 → 高斯分布

给定水平框 $B = (cx, cy, w, h)$，对应内切椭圆的概率密度函数为 2D 高斯分布：

$$
\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}), \quad
\boldsymbol{\mu} = \begin{bmatrix} cx \\ cy \end{bmatrix}, \quad
\boldsymbol{\Sigma} = \begin{bmatrix} \frac{w^2}{4} & 0 \\ 0 & \frac{h^2}{4} \end{bmatrix}
$$

**直观理解**：高斯分布的均值就是框的中心，标准差就是框的半宽/半高（$\sigma_x = w/2$，$\sigma_y = h/2$）。

> **重要约定差异**：Ultralytics 的 `_get_covariance_matrix` 使用 $\sigma^2 = w^2/12$（均匀分布方差，假设框内像素均匀分布），而非 NWD 论文的 $\sigma^2 = w^2/4$。两种约定下 W₂² 仍是合法的距离，但同一框对的 W₂² 在两种约定下差 3 倍左右，因此 **C 的经验值需要按 ultralytics 约定重新校准**。本项目沿用 Ultralytics 约定（与 ProbIoU 一致），所有 C 推荐值已基于此约定给出。

对旋转框 OBB $B = (cx, cy, w, h, \theta)$：

$$
\boldsymbol{\Sigma} = R(\theta) \begin{bmatrix} \frac{w^2}{4} & 0 \\ 0 & \frac{h^2}{4} \end{bmatrix} R(\theta)^T,
\quad R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

### 3.3 Wasserstein-2 距离

两个 2D 高斯之间的 W₂ 距离平方（一般式）：

$$
W_2^2(\mathcal{N}_a, \mathcal{N}_b) = \|\boldsymbol{\mu}_a - \boldsymbol{\mu}_b\|_2^2 + \mathrm{Tr}\left( \boldsymbol{\Sigma}_a + \boldsymbol{\Sigma}_b - 2(\boldsymbol{\Sigma}_a^{1/2} \boldsymbol{\Sigma}_b \boldsymbol{\Sigma}_a^{1/2})^{1/2} \right)
$$

**对水平框**（两个 Σ 都是对角阵），简化为闭式解：

$$
W_2^2(B_a, B_b) = \underbrace{(cx_a - cx_b)^2 + (cy_a - cy_b)^2}_{\text{中心距离}} + \underbrace{\left(\frac{w_a - w_b}{2}\right)^2 + \left(\frac{h_a - h_b}{2}\right)^2}_{\text{尺寸差异}}
$$

等价于把框写成 4D 向量 $v = (cx, cy, w/2, h/2)$ 后的 L2 距离平方：

$$
W_2^2(B_a, B_b) = \|v_a - v_b\|_2^2
$$

### 3.4 归一化为 NWD

W₂ 距离是无界的（[0, ∞)），无法直接当作"相似度"使用。论文用指数归一化：

$$
\boxed{\mathrm{NWD}(B_a, B_b) = \exp\left( -\frac{\sqrt{W_2^2(B_a, B_b)}}{C} \right) \in (0, 1]}
$$

- 当两框完全重合时，$W_2^2 = 0$，NWD = 1
- 当距离越大，NWD → 0

**归一化常数 C** 的选取：通常取为数据集中目标尺寸的平均绝对值（单位：像素）。论文经验值：

| 数据集 | 平均目标大小 | 推荐 C |
|--------|--------------|--------|
| AI-TOD（tiny） | ~12.8 px | **12.8** |
| VisDrone | ~26 px | 26 |
| DOTA / DOTAv1.5 | 跨度大（10-300+ px） | 经验取 **~80-100** ，或按 split 子集统计 |

### 3.5 NWD 作为损失函数

直接用 1 - NWD 作为损失项：

$$
\mathcal{L}_{\mathrm{NWD}} = 1 - \mathrm{NWD}(B_{\mathrm{pred}}, B_{\mathrm{gt}}) = 1 - \exp\left( -\frac{\sqrt{W_2^2}}{C} \right)
$$

### 3.6 与 IoU 损失的加权融合（推荐做法）

实际工程中常把 NWD 和原 IoU 损失加权使用，兼顾大小目标：

$$
\mathcal{L}_{\mathrm{box}} = \alpha \cdot \mathcal{L}_{\mathrm{NWD}} + (1 - \alpha) \cdot \mathcal{L}_{\mathrm{IoU}}
$$

经验值：$\alpha = 0.5 \sim 0.7$（小目标比例越高，α 越大）。

### 3.7 三处嵌入点

论文强调 NWD 可以同时替换三处用 IoU 的地方，提升幅度叠加：

1. **正负样本分配（Label Assignment）**：anchor 与 GT 的匹配从 IoU 阈值改为 NWD 阈值
2. **NMS**：去重时用 NWD 替代 IoU
3. **回归损失**：用 L_NWD 替代 IoU 损失

## 4. 实验表现

### AI-TOD 数据集（微小目标专用，平均 12.8 px）

| 方法 | mAP | AP_vt | AP_t | AP_s | AP_m |
|------|-----|-------|------|------|------|
| Faster R-CNN baseline | 11.1 | 0.0 | 7.6 | 22.0 | 32.1 |
| + NWD（三处替换） | **20.7** | **8.5** | **17.7** | **31.4** | **40.1** |

**提升约 +9.6 mAP，对极小目标（AP_vt）从 0% 提升到 8.5%**。

### DOTA 数据集（航空遥感，扩展版 ISPRS 2022）

不同检测器加 NWD 普遍获得 1.5-3 mAP 的提升，小目标类（如 small-vehicle、ship）提升尤其明显。

## 5. 启发与落地

### 5.1 学到了什么

1. **梯度连续性比 IoU 严苛性更重要**：哪怕两框不重叠，Wasserstein 距离仍能给出连续梯度，这对小目标早期训练特别关键。
2. **几何相似度可以解耦**：W₂² 把"中心距离"和"尺寸差异"清晰分开两项，工程上方便调权。
3. **指数归一化是好习惯**：把无界距离压到 (0, 1]，让损失尺度可控。

### 5.2 能否迁移到本项目

- [x] **适用场景吻合**：本项目主攻 DOTA，存在大量 small-vehicle、ship、helicopter 等小目标，正是 NWD 设计的目标场景
- [x] **与现有方案协同**：
  - 与 **P2 检测头**互补：P2 提供高分辨率特征，NWD 提供更好的小目标位置损失梯度
  - 与 **CBAM** 互补：CBAM 改善特征表达，NWD 改善位置回归
  - 与 **残差融合**互补：互不冲突
- [x] **改动成本可控**：只需替换/加权 BboxLoss 中的回归项，不动模型结构

### 5.3 落地策略

**第一阶段（本次实现）**：
- 仅在 **回归损失** 中加入 NWD，与现有 ProbIoU 加权融合
- 提供开关参数 `--use-nwd`、权重参数 `--nwd-iou-ratio`、归一化常数 `--nwd-c`
- 不动正负样本分配和 NMS（保持 Ultralytics TAL 分配器和默认 NMS）

**第二阶段（后续可选）**：
- 在 TAL 分配器中加入 NWD 项（更复杂，需重写 assigner）
- NMS 阶段引入 NWD 阈值

### 5.4 关键技术点

**OBB 场景的 Wasserstein 距离闭式解**：

经过审阅 Ultralytics 源码（`ultralytics.utils.metrics._get_covariance_matrix`），发现内部已经把 OBB 表示成 2D 协方差矩阵 $\Sigma = \begin{bmatrix} a & c \\ c & b \end{bmatrix}$。这让 Wasserstein-2 距离可以走**闭式解**，不需要矩阵开方运算。

利用恒等式：

$$
\mathrm{Tr}\!\left(\sqrt{M}\right) = \sqrt{\mathrm{Tr}(M) + 2\sqrt{\det(M)}}, \quad M \in \mathbb{R}^{2\times 2} \text{ SPD}
$$

以及循环性 $\mathrm{Tr}(\Sigma_b^{1/2}\Sigma_a\Sigma_b^{1/2}) = \mathrm{Tr}(\Sigma_a\Sigma_b)$ 和 $\det(\Sigma_b^{1/2}\Sigma_a\Sigma_b^{1/2}) = \det(\Sigma_a)\det(\Sigma_b)$，得到 OBB 闭式 W₂²：

$$
\boxed{
W_2^2 = \underbrace{(cx_a-cx_b)^2 + (cy_a-cy_b)^2}_{\text{中心项}} + \underbrace{(a_1+b_1) + (a_2+b_2)}_{\mathrm{Tr}(\Sigma_a)+\mathrm{Tr}(\Sigma_b)} - 2\sqrt{T + 2\sqrt{D_1 D_2}}
}
$$

其中：
- $T = a_1 a_2 + b_1 b_2 + 2c_1 c_2 = \mathrm{Tr}(\Sigma_a \Sigma_b)$
- $D_i = a_i b_i - c_i^2 = \det(\Sigma_i)$

**优势**：
- 完全可微，仅含基础算子（加减乘除 + sqrt）
- 与 Ultralytics 现有的 `_get_covariance_matrix` 完美复用
- 旋转信息无损保留
- 计算开销与 ProbIoU 相当

**本项目采用此闭式解**，无需再考虑 AABB 投影的精度折损。

### 5.5 与 Ultralytics 现有 ProbIoU 的区别

Ultralytics 在 YOLOv8/v11 OBB 中已经使用 **ProbIoU**（同样基于 Gaussian 建模，但用 Bhattacharyya 距离而非 Wasserstein）。

| 项目 | ProbIoU | NWD |
|------|---------|-----|
| 距离度量 | Bhattacharyya | Wasserstein-2 |
| 无重叠时梯度 | 可能饱和 | 线性下降 |
| 小目标稳定性 | 中 | 强 |
| 计算复杂度 | 高（含特殊函数） | 低（仅 L2） |

**结论**：NWD 与 ProbIoU 加权融合，能在小目标上获得更好梯度，在大目标上保留 ProbIoU 的几何严苛性。

## 6. 相关文献

- 原论文 arXiv：https://arxiv.org/abs/2110.13389
- ISPRS 扩展版（航空专用）：https://www.sciencedirect.com/science/article/abs/pii/S0924271622001599
- 官方代码：https://github.com/jwwangchn/NWD
- ProbIoU（Ultralytics OBB 已用）："Gaussian Bounding Boxes and Probabilistic Intersection-over-Union for Object Detection", Llerena et al., 2021
- KFIoU（替代方案）："The KFIoU Loss for Rotated Object Detection", Yang et al., ICLR 2023

## 7. 待求证

- [ ] DOTA 上 C 的最优经验值（论文未明确，需后续在我们数据集上小规模搜索 C ∈ {32, 64, 100, 128}）
- [ ] α 权重的最优值（论文常用 0.5，但 DOTA 大目标多，可能 0.3 更合适）
- [ ] OBB 场景下方案 A vs 方案 B 的实际差距（首版先用 A）
