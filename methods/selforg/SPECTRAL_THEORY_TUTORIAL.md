# 谱图论：从零基础到 SPARC 论文的完整理论路径

> **目标读者**：对线性代数有基本印象（矩阵乘法、向量），但不熟悉特征值分解和谱图论的研究者。
>
> **阅读路线**：第 1-3 章是基础，第 4-6 章是核心理论，第 7-8 章直接对接 SPARC 论文写作。
>
> 全文用「直觉先行，公式跟进」的方式组织——每个概念先给你一个物理图像，再给数学定义。

---

## 目录

1. [特征值与特征向量：为什么它们重要](#1-特征值与特征向量为什么它们重要)
2. [对称矩阵的特殊性质：实特征值与正交基](#2-对称矩阵的特殊性质实特征值与正交基)
3. [从图到矩阵：三种核心矩阵](#3-从图到矩阵三种核心矩阵)
4. [Perron-Frobenius 定理：正矩阵的主特征向量](#4-perron-frobenius-定理正矩阵的主特征向量)
5. [谱间隙：共识的强度信号](#5-谱间隙共识的强度信号)
6. [Fiedler 向量：图的最优二分与分歧发现](#6-fiedler-向量图的最优二分与分歧发现)
7. [DeGroot 模型：意见动力学与共识收敛](#7-degroot-模型意见动力学与共识收敛)
8. [对接 SPARC：从数学到论文语言](#8-对接-sparc从数学到论文语言)
9. [附录：证明与推导细节](#9-附录证明与推导细节)

---

## 1. 特征值与特征向量：为什么它们重要

### 1.1 一句话直觉

一个矩阵 $A$ 可以看作一个"变换机器"——你喂进去一个向量，它吐出来另一个向量。**特征向量**就是那些被这台机器变换后**方向不变**的特殊向量，**特征值**就是它被拉伸或压缩的倍数。

### 1.2 正式定义

给定 $n \times n$ 矩阵 $A$，若存在非零向量 $v$ 和标量 $\lambda$ 满足：

$$
A v = \lambda v
$$
则 $v$ 是 $A$ 的**特征向量**（eigenvector），$\lambda$ 是对应的**特征值**（eigenvalue）。

### 1.3 几何图像

想象二维平面上的一个线性变换（比如拉伸、旋转、剪切）。大多数向量经过变换后方向会改变。但有些特殊方向上的向量，变换后仍然指向同一方向（或反方向），只是长度变了——这些就是特征向量。

```
变换前：  →    （单位向量 v 指向右方）
变换后：  ———→  （变成 3 倍长，但方向不变）

此时 λ = 3，v 是特征向量
```

### 1.4 为什么关心特征值？

特征值分解的核心力量在于：**把一个复杂矩阵拆解成最简单的"拉伸"操作的叠加**。

如果 $A$ 有 $n$ 个线性无关的特征向量 $v_1, v_2, \ldots, v_n$，对应特征值 \lambda_1, \lambda_2, \ldots, \lambda_n，那么：

$$
A = \sum_{i=1}^{n} \lambda_i \, v_i \, v_i^\top
$$
（当 $A$ 对称且特征向量正交归一化时成立，详见第 2 章。）

这意味着：**矩阵的所有信息都编码在特征值和特征向量中**。最大的特征值捕捉矩阵最主要的"模式"，次大的捕捉次要模式，以此类推。

### 1.5 一个具体计算例子

$$A = \begin{pmatrix} 4 & 1 \\ 2 & 3 \end{pmatrix}$$

求特征值：解 $\det(A - \lambda I) = 0$

$$\det \begin{pmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{pmatrix} = (4-\lambda)(3-\lambda) - 2 = \lambda^2 - 7\lambda + 10 = 0$$

$$\lambda_1 = 5, \quad \lambda_2 = 2$$

对 $\lambda_1 = 5$：

$$(A - 5I)v = 0 \implies \begin{pmatrix} -1 & 1 \\ 2 & -2 \end{pmatrix} v = 0 \implies v_1 = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

对 $\lambda_2 = 2$：

$$(A - 2I)v = 0 \implies \begin{pmatrix} 2 & 1 \\ 2 & 1 \end{pmatrix} v = 0 \implies v_2 = \begin{pmatrix} 1 \\ -2 \end{pmatrix}$$

**解读**：矩阵 $A$ 的"主方向"是 $(1,1)$，沿此方向拉伸 5 倍；"次方向"是 $(1,-2)$，沿此方向拉伸 2 倍。

---

## 2. 对称矩阵的特殊性质：实特征值与正交基

### 2.1 为什么对称矩阵特别重要？

你的共识矩阵 $A = \exp(S/\tau)$ 是对称的（因为 $S_{ij} = S_{ji}$）。对称矩阵拥有非常好的数学性质，这是 SPARC 全部理论推导的基石。

### 2.2 谱定理（Spectral Theorem）

**定理**：如果 $A$ 是 $n \times n$ **实对称矩阵**，则：

1. $A$ 的所有特征值都是**实数**（不会出现复数）
2. $A$ 有 $n$ 个特征向量，它们**两两正交**（互相垂直）
3. 这些特征向量构成 $\mathbb{R}^n$ 的一组**正交基**

**直觉**：对称矩阵只做"拉伸"，不做"旋转"。所以它的特征方向总是互相垂直的，像坐标轴一样干净。

### 2.3 谱分解

将特征向量归一化（长度为 1）后，对称矩阵可以写成：

$$A = \lambda_1 v_1 v_1^\top + \lambda_2 v_2 v_2^\top + \cdots + \lambda_n v_n v_n^\top$$

其中 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$（按降序排列），$v_i^\top v_j = \delta_{ij}$。

**这就是"谱"（spectrum）的来源**——$\{\lambda_1, \lambda_2, \ldots, \lambda_n\}$ 这组特征值就像光谱一样，完整描述了矩阵的特性。

### 2.4 从全谱到低秩近似

**关键洞察**：如果 $\lambda_1 \gg \lambda_2$（第一个特征值远大于第二个），那么：

$$A \approx \lambda_1 v_1 v_1^\top$$

矩阵近似为**秩 1**——所有信息集中在一个方向上。

**对应 SPARC 的直觉**：当谱间隙很大时，共识矩阵近似秩 1，所有 agent 的意见在同一个方向上对齐——这就是"强共识"。

### 2.5 numpy 对应

```python
import numpy as np

# np.linalg.eigh 专用于对称矩阵
# 返回特征值（升序）和特征向量（列向量）
eigenvalues, eigenvectors = np.linalg.eigh(A)

# eigenvalues[-1]  = 最大特征值 λ₁
# eigenvectors[:, -1] = 对应的特征向量 v₁（Perron 向量）
# eigenvalues[-2]  = 第二大特征值 λ₂
# eigenvectors[:, -2] = 对应的特征向量 v₂（Fiedler 向量）
```

> **注意**：`eigh` 返回升序排列，所以最大特征值在最后。这就是你代码中 `eigenvectors[:, -1]` 和 `eigenvectors[:, -2]` 的原因。

---

## 3. 从图到矩阵：三种核心矩阵

### 3.1 图的基本概念

**图** $G = (V, E)$ 由节点集 $V$ 和边集 $E$ 组成。在你的场景中：

- **节点** = LLM agent（比如 4 个 agent）
- **边的权重** = agent 之间回复的语义相似度

### 3.2 邻接矩阵（Adjacency Matrix）

最直接的图矩阵。对**无向加权图**：

$$W_{ij} = \text{节点 } i \text{ 和节点 } j \text{ 之间的边权}$$

$W$ 是对称的（$W_{ij} = W_{ji}$）。

**例子**：4 个 agent，两两之间的余弦相似度构成的矩阵：

```
     Agent 0  Agent 1  Agent 2  Agent 3
A0 [  1.00     0.85     0.30     0.32  ]
A1 [  0.85     1.00     0.28     0.35  ]
A2 [  0.30     0.28     1.00     0.90  ]
A3 [  0.32     0.35     0.90     1.00  ]
```

目测就能看出：Agent 0-1 相似，Agent 2-3 相似，两组之间差异大。

### 3.3 度矩阵（Degree Matrix）

对角矩阵，第 $i$ 个对角元素是节点 $i$ 的"度"（所有连接边权之和）：

$$D_{ii} = \sum_{j} W_{ij}$$

### 3.4 拉普拉斯矩阵（Laplacian Matrix）

$$L = D - W$$

拉普拉斯矩阵是谱图论的核心对象。它有以下关键性质：

1. **半正定**：所有特征值 $\geq 0$
2. **最小特征值恒为 0**，对应的特征向量是全 1 向量 $\mathbf{1}$（如果图连通）
3. **第二小特征值**（algebraic connectivity）衡量图的连通程度

### 3.5 你的共识矩阵：一种特殊的邻接矩阵

你的代码构造的是：

$$S = V V^\top \quad (\text{余弦相似度矩阵，因为 } V \text{ 已 L2 归一化})$$

$$A = \exp(S / \tau)$$

这个 $A$ 是一个**所有元素都为正**的对称矩阵——它是邻接矩阵的一种，但每条边都有正权重（因为 $\exp(\cdot) > 0$）。这个"全正"性质恰好满足 Perron-Frobenius 定理的条件（见第 4 章）。

**$\tau$ 的作用**：

| $\tau$ | $\exp(S_{ij}/\tau)$ 的行为 | 效果 |
|--------|---------------------------|------|
| $\tau \to \infty$ | 所有元素趋近 $e^0 = 1$ | 矩阵变得均匀，所有 agent "平等" |
| $\tau = 1$ | 适度放大差异 | 平衡状态 |
| $\tau \to 0^+$ | 高相似度的权重指数爆炸 | 只有最相似的 pair 有影响力，矩阵变"尖锐" |

---

## 4. Perron-Frobenius 定理：正矩阵的主特征向量

### 4.1 直觉：网页排名

Google 的 PageRank 算法就是 Perron-Frobenius 定理的经典应用。想象互联网是一张图，网页是节点，超链接是边。问题：哪些网页最"重要"？

答案：构造转移矩阵，求主特征向量——分量最大的网页最重要。

你的 SPARC 做的是同一件事，只不过：
- "网页" → LLM agent 的回复
- "超链接" → 语义相似度
- "重要性" → 可靠度/贡献度

### 4.2 定理陈述

**Perron-Frobenius 定理**：设 $A$ 是 $n \times n$ 的**正矩阵**（所有元素 $A_{ij} > 0$），则：

1. $A$ 有一个**最大特征值** $\lambda_1 > 0$，且 $\lambda_1$ **严格大于**所有其他特征值的绝对值
2. 对应的特征向量 $v_1$ 的**所有分量都为正**（可以取为正的）
3. $\lambda_1$ 是**简单的**（代数重数为 1）——即该特征值唯一

### 4.3 为什么你的共识矩阵满足条件？

$A_{ij} = \exp(S_{ij}/\tau)$，由于 $\exp(\cdot) > 0$ 恒成立，$A$ 的所有元素严格为正。

因此 Perron-Frobenius 定理直接适用，保证：
- 存在唯一的最大特征值
- 对应的主特征向量 $v_1$（Perron 向量）所有分量为正
- 代码中 `np.abs(eigenvectors[:, -1])` 取绝对值是为了处理 `eigh` 可能返回负号的数值问题，理论上分量本就应该同号

### 4.4 Perron 向量的直觉含义

Perron 向量 $v_1$ 的第 $i$ 个分量 $v_1[i]$ 衡量的是：**节点 $i$ 在整个相互强化网络中的"中心性"**。

具体地说，如果你反复用矩阵 $A$ 左乘一个随机初始向量 $x$：

$$x, \; Ax, \; A^2 x, \; A^3 x, \; \ldots$$

归一化后这个序列会收敛到 $v_1$。这叫做**幂迭代法**（power iteration），也是 PageRank 的实际计算方式。

**在你的场景中**：$v_1[i]$ 大 $\Leftrightarrow$ agent $i$ 的回复与许多其他回复都相似，而且那些与它相似的回复本身也互相相似（传递性的"声望"）。这比简单的"离质心近"（SelfOrg 的做法）更精细：它捕捉了**全局一致性结构**，而不只是局部距离。

### 4.5 与你代码的对应

```python
# soo_contribution.py
def estimate_contributions_soo(embeddings, tau_consensus=1.0):
    S = embeddings @ embeddings.T          # 余弦相似度
    A = np.exp(S / tau_consensus)          # 共识矩阵（正矩阵）
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    c = np.abs(eigenvectors[:, -1])        # Perron 向量 = 贡献度
    return c
```

这就是 Perron-Frobenius 定理的直接应用。

---

## 5. 谱间隙：共识的强度信号

### 5.1 定义

**谱间隙**（spectral gap）是最大特征值和第二大特征值之间的差距。在 SPARC 中，使用归一化的**谱间隙比**：

$$r = \frac{\lambda_1 - \lambda_2}{\lambda_1}$$

其中 $\lambda_1 > \lambda_2$ 是共识矩阵 $A$ 的前两大特征值。

### 5.2 直觉：把特征谱想象成"投票结构"

回忆谱分解 $A = \lambda_1 v_1 v_1^\top + \lambda_2 v_2 v_2^\top + \cdots$

- $\lambda_1 v_1 v_1^\top$ 是"主旋律"——所有 agent 都同意的部分
- $\lambda_2 v_2 v_2^\top$ 是"最大的杂音"——最显著的分歧

**谱间隙大**（$r$ 接近 1）：

```
特征值谱：  ████████████  ██  █  █
            λ₁ (巨大)    λ₂ (很小)

含义：主旋律压倒一切，杂音微弱
     → 强共识，agent 们高度一致
     → 不需要辩论，直接投票即可
```

**谱间隙小**（$r$ 接近 0）：

```
特征值谱：  ████████████  ██████████  ██  █
            λ₁           λ₂ (也很大)

含义：主旋律和杂音势均力敌
     → 存在两个竞争的"意见集群"
     → 需要辩论来解决分歧
```

### 5.3 与 Markov 链混合时间的类比

这个类比是 SPARC 论文理论框架的核心，值得详细理解。

**Markov 链**：想象一个随机游走者在图上跳来跳去。在每个节点，它按照边权比例随机选择下一步。

**转移矩阵**：$P = D^{-1} W$（每行归一化为概率分布）。

**混合时间**：随机游走者从任何起点出发，需要多少步才能"忘记"起点、达到稳态分布？

**关键定理**：混合时间 $t_{\text{mix}} \sim \frac{1}{1 - |\lambda_2(P)|}$

其中 $\lambda_2(P)$ 是转移矩阵的第二大特征值（模）。

| 谱间隙 | $\lvert\lambda_2\rvert$ | 混合时间 | 含义 |
|--------|------------|----------|------|
| 大 | 小（远离 1） | 短 | 快速收敛到共识 |
| 小 | 大（接近 1） | 长 | 收敛缓慢，存在瓶颈 |

**SPARC 的桥接**：虽然 SPARC 的共识矩阵不直接是 Markov 转移矩阵，但基于相同的数学原理——谱间隙度量的是"系统中不同意见多快能混合到一起"。间隙大 = 混合快 = 辩论没什么可加速的；间隙小 = 混合慢 = 辩论有价值。

### 5.4 数值例子

**场景 1：强共识**

4 个 agent 的回复几乎相同，余弦相似度矩阵：

```
S = [1.0  0.95  0.93  0.94]
    [0.95 1.0   0.92  0.96]
    [0.93 0.92  1.0   0.91]
    [0.94 0.96  0.91  1.0 ]
```

$A = \exp(S/1.0)$ 的特征值：$[10.31, \; 0.98, \; 0.87, \; 0.84]$

谱间隙比：$r = (10.31 - 0.98) / 10.31 = 0.905$ —— 非常大，强共识。

**场景 2：两极分化**

Agent 0-1 说"答案是 42"，Agent 2-3 说"答案是 17"：

```
S = [1.0  0.95  0.20  0.22]
    [0.95 1.0   0.18  0.25]
    [0.20 0.18  1.0   0.92]
    [0.22 0.25  0.92  1.0 ]
```

$A = \exp(S/1.0)$ 的特征值：$[6.82, \; 5.15, \; 1.02, \; 0.91]$

谱间隙比：$r = (6.82 - 5.15) / 6.82 = 0.245$ —— 较小，两个集群在竞争。

### 5.5 SPARC 的决策逻辑

```python
# sparc_math_main.py 中的实际逻辑
if gap >= self.gap_high:      # 默认 0.35
    break                     # 跳过辩论
elif gap < self.gap_low:      # 默认 0.15
    # 使用 Fiedler 跨阵营辩论（重火力）
else:
    # 使用标准图辩论（轻量级）
```

---

## 6. Fiedler 向量：图的最优二分与分歧发现

### 6.1 谁是 Fiedler？

Miroslav Fiedler（1926-2015），捷克数学家，1973 年证明了图拉普拉斯矩阵第二小特征值（他称之为"algebraic connectivity"）与图连通性之间的深刻联系。他的名字成为了这个概念的标准术语。

### 6.2 经典定义（图拉普拉斯的第二特征向量）

在标准谱图论中，**Fiedler 向量**指的是图拉普拉斯矩阵 $L = D - W$ 的**第二小特征值**（$\lambda_2$，最小的 $\lambda_1 = 0$）对应的特征向量。

但在你的 SPARC 中，情况有一个镜像对称：

| | 拉普拉斯 $L = D - W$ | 共识矩阵 $A = \exp(S/\tau)$ |
|---|---|---|
| 主特征方向 | 最小特征值 $\lambda_1 = 0$（共识） | 最大特征值 $\lambda_1$（共识） |
| 分歧特征方向 | 第二小特征值 $\lambda_2$ | **第二大**特征值 $\lambda_2$ |
| Fiedler 向量 | $L$ 的 $\lambda_2$ 对应特征向量 | $A$ 的 $\lambda_2$ 对应特征向量 |

原因：$L$ 和 $W$（或 $A$）的特征值是"反着来的"——$L$ 的小特征值对应 $W$ 的大特征值。所以 SPARC 代码中取 `eigenvectors[:, -2]`（第二大特征值对应的特征向量）在数学上完全对应经典 Fiedler 向量的角色。

### 6.3 Fiedler 向量的核心性质：最优二分

**定理**（Fiedler, 1973；非形式化表述）：Fiedler 向量给出图的**近似最优二分**——将节点按 Fiedler 向量分量的符号分成两组，得到的切割接近最小化归一化切割（Normalized Cut）的最优解。

**归一化切割**：

$$\text{Ncut}(A, B) = \frac{\text{cut}(A, B)}{\text{vol}(A)} + \frac{\text{cut}(A, B)}{\text{vol}(B)}$$

其中 $\text{cut}(A, B)$ 是两组之间的边权总和，$\text{vol}(A)$ 是组 $A$ 内部的总度数。

**直觉**：Ncut 最小化意味着"组间联系尽量少，组内联系尽量多"——这恰好就是"找到两个意见阵营"。

### 6.4 图解 Fiedler 向量

继续用"两极分化"的例子（Agent 0-1 vs Agent 2-3）：

```
Fiedler 向量 v₂ = [0.52,  0.48,  -0.51,  -0.49]
                    ^^^^^^^^^^^^    ^^^^^^^^^^^^^^
                    正分量 (Camp A)  负分量 (Camp B)
```

**Fiedler 向量的符号直接给出两个阵营的划分**：

```
Camp A (v₂ ≥ 0): Agent 0, Agent 1  → 说"答案是 42"
Camp B (v₂ < 0): Agent 2, Agent 3  → 说"答案是 17"
```

**分量的绝对值**还给出了"多大程度上属于该阵营"——|0.52| > |0.48|，说明 Agent 0 比 Agent 1 更"典型"地代表 Camp A。

### 6.5 为什么 Fiedler 二分是"最优"的？

这可以用变分法理解。Fiedler 向量是以下优化问题的解：

$$v_2 = \arg\min_{v \perp v_1, \; \|v\|=1} v^\top L v$$

其中 $L$ 是拉普拉斯矩阵，约束 $v \perp v_1$ 排除了平凡解。

$v^\top L v$ 展开后等于：

$$v^\top L v = \sum_{(i,j) \in E} W_{ij} (v_i - v_j)^2$$

这就是**加权的"同组一致性"度量**——相似的节点（$W_{ij}$ 大）被赋予不同的值（$v_i \neq v_j$）时惩罚更大。

所以 Fiedler 向量找到的是：**在保持正交于共识方向的前提下，节点间差异最小的那个方向**——这就是主要分歧轴，因为沿其他方向的差异更大。

### 6.6 对应 SPARC 的跨阵营辩论

```python
# spectral_consensus.py
camp_a = [i for i in range(n) if fiedler[i] >= 0]   # 阵营 A
camp_b = [i for i in range(n) if fiedler[i] < 0]    # 阵营 B

# 每个 agent 看到对方阵营冠军的回答
def cross_camp_peers(node, diag, prev_responses):
    in_a = node in diag.camp_a
    opp_camp = diag.camp_b if in_a else diag.camp_a
    opp_champ = max(opp_camp, key=lambda i: perron[i])  # 对方最强者
    peers = [prev_responses[opp_champ]]
    ...
```

**为什么这比标准 DAG 辩论更好？**

标准 SOO 辩论的信息流是"强者 → 弱者"。但如果所有强者都在同一阵营（都说"答案是 42"），弱者只会被同一方向的声音淹没——这就是**回声室效应**。

Fiedler 辩论保证：**每个 agent 都会看到最大分歧另一端的最强论点**。这打破了回声室，因为信息流沿着分歧轴的"瓶颈"被强制注入。

### 6.7 从谱聚类到 Fiedler 的视角

谱聚类（Spectral Clustering）是机器学习中的经典方法，其核心步骤就是：
1. 构造相似度图
2. 计算拉普拉斯矩阵
3. 取前 $k$ 个最小特征向量
4. 在特征向量空间中做 k-means

SPARC 的 Fiedler 二分可以理解为 **$k=2$ 的谱聚类**，但不需要跑 k-means——Fiedler 向量的符号直接给出聚类结果。这在论文中可以引用谱聚类的经典文献（Shi & Malik 2000, Ng et al. 2001）作为理论背景。

---

## 7. DeGroot 模型：意见动力学与共识收敛

### 7.1 模型描述

DeGroot 模型（1974）是社会学中研究**群体如何达成共识**的经典模型。

设 $N$ 个人各自持有一个意见（实数）$x_i(0)$。每一步，每个人将自己的意见更新为邻居意见的加权平均：

$$x_i(t+1) = \sum_{j=1}^{N} W_{ij} \, x_j(t)$$

其中 $W$ 是**行随机矩阵**（每行和为 1）。

用向量记法：$\mathbf{x}(t+1) = W \, \mathbf{x}(t)$

### 7.2 收敛条件

$$\mathbf{x}(t) = W^t \, \mathbf{x}(0)$$

如果 $W$ 的谱间隙大（第二大特征值模 $|\lambda_2(W)| < 1$），则：

$$W^t \to \mathbf{1} \pi^\top \quad \text{as } t \to \infty$$

其中 $\pi$ 是稳态分布。也就是：**所有人最终收敛到同一个意见**。

**收敛速度**：

$$\|x(t) - x^*\| \leq C \cdot |\lambda_2|^t$$

$|\lambda_2|$ 越小（谱间隙越大），收敛越快。

### 7.3 与 SPARC 的桥接

LLM 多智能体辩论可以类比为一种"意见动力学"：

| DeGroot 模型 | SPARC 多智能体辩论 |
|---|---|
| 每个人持有一个实数意见 | 每个 agent 持有一段文本回复 |
| 加权平均更新 | 读同伴回复后修改自己的回答 |
| 转移矩阵 $W$ | 共识矩阵 $A = \exp(S/\tau)$ |
| 稳态 $\pi$ | 最终共识答案 |
| $\lvert\lambda_2\rvert$ 决定收敛速度 | **谱间隙决定辩论是否有效** |

**这个类比是你论文的理论支柱之一**。SPARC 的洞察是：与其盲目辩论固定轮数，不如**先看谱间隙**——

- 谱间隙大 → DeGroot 告诉我们系统已经（或即将）收敛 → 辩论无法加速 → 跳过
- 谱间隙小 → 系统远离收敛，存在瓶颈 → 辩论（特别是沿瓶颈注入信息的 Fiedler 辩论）有价值

### 7.4 为什么不能直接套用 DeGroot？

真实的 LLM 辩论与 DeGroot 有几个关键区别，论文中需要诚实讨论：

1. **非线性**：LLM 对输入的处理是高度非线性的，不是简单的加权平均
2. **离散答案**：数学推理的最终输出是离散的答案（如"42"），不是连续的
3. **非确定性**：LLM 输出有随机性（temperature > 0）
4. **语义空间**：相似度是在 embedding 空间而非意见空间中计算的

因此，DeGroot 提供的是**启发式类比和理论动机**，而非严格证明。在论文中应该明确说"inspired by"或"analogous to"，而不是"proved by"。

---

## 8. 对接 SPARC：从数学到论文语言

### 8.1 你需要讲清楚的三个理论故事

**故事 1：Perron 向量 → 贡献度估计**

> **论文语言（供参考）**：
>
> We construct a consensus matrix $A$ where $A_{ij} = \exp(\text{sim}(r_i, r_j)/\tau)$, with $\text{sim}$ being cosine similarity between L2-normalized response embeddings. Since all entries of $A$ are strictly positive, the Perron-Frobenius theorem guarantees the existence of a unique dominant eigenvector $v_1$ with all-positive components. We interpret $v_1[i]$ as the *reliability* of agent $i$: it captures not merely pairwise similarity to the centroid (as in prior work), but a *transitive, network-wide* measure of consensus centrality — analogous to eigenvector centrality in social network analysis or PageRank in web graphs.

**故事 2：谱间隙 → 自适应计算分配**

> **论文语言（供参考）**：
>
> The spectral gap ratio $r = (\lambda_1 - \lambda_2)/\lambda_1$ serves as an endogenous *difficulty signal* that governs compute allocation. Drawing on the analogy with mixing times in Markov chains — where the spectral gap of the transition matrix determines how quickly a random walk converges to its stationary distribution — we posit that a large spectral gap indicates the agent population has already converged (or will converge rapidly), making further debate rounds wasteful. Conversely, a small spectral gap reveals competing opinion clusters whose resolution requires targeted intervention. This transforms fixed-budget debate into *adaptive inference-time compute scaling* without any external difficulty classifier.

**故事 3：Fiedler 向量 → 跨阵营辩论**

> **论文语言（供参考）**：
>
> The second eigenvector of the consensus matrix — the *Fiedler vector* in spectral graph theory — reveals the dominant axis of disagreement among agents. Its sign partitions agents into two camps aligned along this axis, corresponding to the approximate minimum normalized cut of the consensus graph (Shi & Malik, 2000). Standard graph-guided debate routes information from high-reliability to low-reliability agents, but this topology is blind to disagreement structure: if all high-reliability agents happen to share the same (possibly incorrect) view, echo-chamber convergence results. Fiedler-guided debate instead pairs each agent with the champion of the opposing camp, injecting information precisely at the bottleneck identified by spectral analysis. This maximizes cross-cluster information transfer in a single debate round.

### 8.2 Related Work 中需要引用的谱图论经典文献

在你的论文中，以下经典文献应该出现在 Related Work 或 Theoretical Background 中：

| 文献 | 贡献 | 你的引用理由 |
|------|------|------------|
| Fiedler (1973). "Algebraic connectivity of graphs" | Fiedler 向量、代数连通度 | 你的分歧检测的数学基础 |
| Shi & Malik (2000). "Normalized cuts and image segmentation" | 归一化切割与谱聚类 | Fiedler 二分等价于近似最优 Ncut 的理论依据 |
| Ng, Jordan & Weiss (2001). "On spectral clustering" | 谱聚类的理论分析 | 补充谱方法用于聚类的通用理论 |
| Perron (1907), Frobenius (1912) | 正矩阵主特征值唯一性 | 你的贡献度估计的理论保证 |
| DeGroot (1974). "Reaching a consensus" | 意见动力学的加权平均模型 | 你的"谱间隙 ↔ 收敛速度"类比的理论来源 |
| Chung (1997). *Spectral Graph Theory* | 教科书级参考 | 谱图论的通用参考 |
| von Luxburg (2007). "A tutorial on spectral clustering" | 最易读的谱聚类教程 | Reviewer 友好的背景参考 |

### 8.3 你可以在论文中做的理论贡献声明

基于以上理论，SPARC 可以声明以下层次的理论贡献：

**可以强声明的**（有数学支撑）：

1. 共识矩阵的 Perron 向量提供了比余弦质心（SelfOrg）更全局的贡献度估计，因为它捕捉了传递性的网络中心性
2. 谱间隙是一个理论上合理的共识强度指标（类比 Markov 链混合时间）
3. Fiedler 二分是近似最优归一化切割（有 Shi & Malik 的理论保证）

**可以中度声明的**（有直觉支撑 + 实验验证）：

4. 谱间隙与问题难度正相关（需要实验验证：简单题初始谱间隙大、难题初始谱间隙小）
5. 跨阵营辩论比标准 DAG 辩论更有效地打破回声室（需要消融实验）

**需要谨慎表述的**（类比层面）：

6. DeGroot 模型的收敛理论"启发"了 SPARC 的设计，但 LLM 辩论的非线性和离散性意味着这不是严格的理论保证

### 8.4 数学符号约定建议

为论文统一符号，建议如下：

| 符号 | 含义 | 首次出现位置 |
|------|------|------------|
| $N$ | agent 数量 | Problem Setup |
| $r_i$ | agent $i$ 的文本回复 | Problem Setup |
| $e_i$ | $r_i$ 的 L2-归一化 embedding 向量 | Method §3.1 |
| $S$ | 相似度矩阵，$S_{ij} = e_i^\top e_j$ | Method §3.1 |
| $A$ | 共识矩阵，$A_{ij} = \exp(S_{ij}/\tau)$ | Method §3.2 |
| $\tau$ | 共识温度参数 | Method §3.2 |
| $\lambda_k$ | $A$ 的第 $k$ 大特征值 | Method §3.3 |
| $v_k$ | $\lambda_k$ 对应的特征向量 | Method §3.3 |
| $v_1$ | Perron 向量（贡献度/可靠度） | Method §3.3 |
| $v_2$ | Fiedler 向量（分歧轴） | Method §3.4 |
| $r$ | 谱间隙比 $(\lambda_1 - \lambda_2)/\lambda_1$ | Method §3.5 |
| $\theta_H, \theta_L, \theta_V$ | 谱间隙决策阈值（high/low/verify） | Method §4.1 |
| $\mathcal{C}_+, \mathcal{C}_-$ | Fiedler 向量定义的两个阵营 | Method §4.2 |

---

## 9. 附录：证明与推导细节

### 9.1 为什么 $\exp(S/\tau)$ 的所有元素为正？

$S_{ij}$ 是余弦相似度，取值范围 $[-1, 1]$。因此 $S_{ij}/\tau \in [-1/\tau, 1/\tau]$。

由于 $\exp(x) > 0$ 对所有实数 $x$ 成立，$A_{ij} = \exp(S_{ij}/\tau) > 0$。

### 9.2 Rayleigh 商与特征值的变分刻画

对称矩阵 $A$ 的特征值可以用 Rayleigh 商刻画：

$$\lambda_1 = \max_{\|x\|=1} x^\top A x, \quad \text{取到最大值时 } x = v_1$$

$$\lambda_2 = \max_{\|x\|=1, \; x \perp v_1} x^\top A x, \quad \text{取到最大值时 } x = v_2$$

**含义**：$v_1$ 是使得二次型 $x^\top A x$ 最大的方向（"最大共识方向"），$v_2$ 是在与 $v_1$ 正交的子空间中使二次型最大的方向（"最大分歧方向"）。

### 9.3 Fiedler 向量与 Ncut 的关系（草案级推导）

对图拉普拉斯 $L = D - W$，归一化切割问题可以写成：

$$\min_{y} \frac{y^\top L y}{y^\top D y} \quad \text{s.t. } y \in \{-1, +1\}^n, \; y^\top D \mathbf{1} = 0$$

这是一个 NP-hard 的组合优化问题。将 $y$ 松弛到连续域 $y \in \mathbb{R}^n$：

$$\min_{y} \frac{y^\top L y}{y^\top D y} \quad \text{s.t. } y \perp D\mathbf{1}$$

这个广义特征值问题 $Ly = \lambda Dy$ 的最小非零特征值对应的特征向量就是 Fiedler 向量。

将连续解 $y$ 离散化回 $\{-1, +1\}$（按符号），就得到了近似最优的归一化切割。

**对你的共识矩阵 $A$**：由于 $A$ 是正矩阵（不是拉普拉斯），对应关系是"反过来"的——$A$ 的第二大特征向量对应的是类似的"最优二分"，只是优化方向相反（最大化组内一致性等价于最小化组间不一致性）。

### 9.4 温度 $\tau$ 对特征谱的影响

当 $\tau \to \infty$ 时，$A \to \mathbf{1}\mathbf{1}^\top$（全 1 矩阵），特征值为 $[n, 0, 0, \ldots, 0]$，谱间隙比 $r = 1$。这意味着无穷大温度下"一切皆共识"——但这是虚假共识，因为矩阵本身已不含信息。

当 $\tau \to 0^+$ 时，$A$ 趋近于在 $S_{ij}$ 最大的位置集中权重，矩阵变得稀疏，特征谱反映真实的聚类结构。

因此 $\tau$ 的选择需要平衡：太大则谱间隙虚高、太小则矩阵数值不稳定。你的默认值 $\tau = 1.0$ 是一个合理的中间点。

### 9.5 Perron 向量 vs 简单质心距离：一个反例

考虑 4 个 agent 的回复嵌入构成如下相似度结构：

```
Agent 0: 与 Agent 1, 2, 3 都有中等相似度 0.5
Agent 1: 与 Agent 2 高度相似 0.95，与 Agent 3 高度相似 0.93，与 Agent 0 中等 0.5
Agent 2: 与 Agent 1 高度相似 0.95，与 Agent 3 高度相似 0.90，与 Agent 0 中等 0.5
Agent 3: 与 Agent 1 高度相似 0.93，与 Agent 2 高度相似 0.90，与 Agent 0 中等 0.5
```

**质心法**（SelfOrg）：Agent 0 与质心的距离不一定最远（因为质心被 1/2/3 的高相似度拉向它们的方向，但 Agent 0 与质心的角度可能接近，取决于嵌入维度）。

**Perron 向量**（SOO）：Agent 1 > Agent 2 > Agent 3 >> Agent 0，因为 1-2-3 之间互相强化形成了"核心共识集团"，Agent 0 虽然与所有人都有些相似，但不在核心集团中。

**Perron 向量更准确地反映了"谁代表真正的共识"。**

---

## 延伸阅读推荐

**入门级**（适合快速建立直觉）：
- von Luxburg (2007). "A Tutorial on Spectral Clustering" — 最好的谱聚类入门材料，20 页讲透核心思想

**教科书**（需要深入理论时查阅）：
- Chung (1997). *Spectral Graph Theory* — 经典教科书
- Spielman (2019). "Spectral and Algebraic Graph Theory" (在线讲义) — 更现代的处理方式

**与你工作最相关的**：
- DeGroot (1974). "Reaching a Consensus" — 意见动力学的开山之作
- Golub & Jackson (2010). "Naive Learning in Social Networks and the Wisdom of Crowds" — DeGroot 模型在社会网络中的现代分析，直接讨论了谱间隙与收敛的关系

---

## 10. SPARC 完整算法伪代码

本节给出与代码实现 (`sparc_math_main.py`, `spectral_consensus.py`, `graph_formation.py`, `soo_contribution.py`) 完全对齐的伪代码。为便于论文写作，先列出所有子程序，最后给出主算法。

> **符号约定**（与第 8.4 节一致）：$N$ = agent 数；$e_i$ = agent $i$ 回复的 L2-归一化 embedding；$\tau_c$ = 共识温度；$\tau_g$ = 通信图相似度阈值；$k$ = 每节点最大入边数；$\theta_H, \theta_L, \theta_V$ = 谱间隙阈值（high / low / verify）；$T$ = 总轮数预算；$T_{\text{cross}}, T_{\text{std}}$ = Fiedler / 标准辩论轮数上限。

---

### 10.1 子程序 1：谱诊断 (Spectral Diagnosis)

对应 `spectral_consensus.py::spectral_diagnosis()`

```
function SPECTRAL_DIAGNOSIS(E, τ_c):
  ──────────────────────────────────────────────────────
  输入: E ∈ ℝ^{N×d}  — L2-归一化的 embedding 矩阵
        τ_c           — 共识温度
  输出: SpectralDiag  — 包含下列字段的结构体
  ──────────────────────────────────────────────────────

  // 1. 相似度矩阵（余弦相似度，因 E 已 L2-归一化）
  S ← E E⊤                               // S_{ij} = cos(e_i, e_j)

  // 2. 共识矩阵（正矩阵，满足 Perron-Frobenius 条件）
  A ← exp(S / τ_c)                       // A_{ij} > 0  ∀ i,j

  // 3. 对称矩阵特征分解
  {λ_n}, {u_n} ← EIGH(A)                 // 升序: λ_1 ≤ ... ≤ λ_N
                                          // u_n 为列向量，两两正交

  // 4. 提取谱信号
  λ₁ ← λ_N                               // 最大特征值
  λ₂ ← λ_{N-1}                           // 第二大特征值
  v₁ ← |u_N|                             // Perron 向量（取绝对值保证非负）
  v₂ ← u_{N-1}                           // Fiedler 向量（保留符号）

  // 5. 谱间隙比
  r ← (λ₁ - λ₂) / max(|λ₁|, ε)         // ε = 1e-12 防除零

  // 6. Fiedler 阵营划分
  C₊ ← { i | v₂[i] ≥ 0 }               // 阵营 A
  C₋ ← { i | v₂[i] < 0 }               // 阵营 B

  // 退化处理：若某阵营为空，按 Perron 分量中位数二分
  if C₊ = ∅ or C₋ = ∅ then
    order ← ARGSORT_DESC(v₁)
    C₊ ← order[0 : ⌈N/2⌉]
    C₋ ← order[⌈N/2⌉ : N]

  return SpectralDiag(v₁, v₂, r, λ₁, λ₂, S, A, C₊, C₋)
```

**理论依据**：
- **v₁**：由 Perron-Frobenius 定理保证唯一且全正（§4）
- **v₂**：Fiedler 二分近似最优归一化切割（§6.3–6.5）
- **r**：类比 Markov 链混合时间间隙（§5.3）

---

### 10.2 子程序 2：通信图构建 (Communication Graph Formation)

对应 `graph_formation.py::form_graph()`

```
function FORM_GRAPH(E, c, τ_g, k):
  ──────────────────────────────────────────────────────
  输入: E ∈ ℝ^{N×d}  — embedding 矩阵
        c ∈ ℝ^N      — 贡献度向量（如 Perron 向量 v₁）
        τ_g           — 相似度阈值
        k             — 每节点最大入边数（可选）
  输出: G = (Adj, π)  — 邻接矩阵 + 拓扑序
  ──────────────────────────────────────────────────────

  // 1. 两两余弦相似度，清除自环
  S ← COSINE_SIM(E)
  S[i,i] ← 0   ∀i

  // 2. 按规则建边：Adj[n,m] = 1 表示 m → n（m 影响 n）
  Adj ← 0^{N×N}
  for n = 0 to N-1 do
    candidates ← { m ≠ n | S[n,m] ≥ τ_g }                // 相似度门槛
    if k ≠ null and |candidates| > k then
      candidates ← TOP_K_BY(candidates, S[n,·], k)        // 保留最相似的 k 个
    for m in candidates do
      if c[m] > c[n] then                                 // 仅强者 → 弱者
        Adj[n,m] ← 1

  // 3. 断环：消除 DAG 中残余的环
  while FIND_CYCLE(Adj) ≠ null do
    cycle ← FIND_CYCLE(Adj)
    w ← argmin_{i ∈ cycle} c[i]                           // 环中最弱者
    succ ← NEXT_IN_CYCLE(cycle, w)
    Adj[succ, w] ← 0                                      // 删去最弱者的出边

  // 4. 拓扑排序（贡献度高者优先处理）
  π ← TOPOLOGICAL_SORT(Adj, tie_break = DESC(c))

  return (Adj, π, S)
```

**信息流含义**：在拓扑序 $\pi$ 中排在前面的 agent（贡献度高、无前驱依赖）先执行。后续 agent 在 LLM 调用时能看到前驱的最新回复，实现"可靠 agent 优先发言，弱 agent 在听到可靠意见后再修正"。

---

### 10.3 子程序 3：标准图辩论 (Standard Graph-Guided Debate)

对应 `sparc_math_main.py::_standard_debate_round()`

```
function STANDARD_DEBATE(responses, E, diag, Q_prepared, contexts):
  ──────────────────────────────────────────────────────
  输入: responses[0..N-1]  — 上一轮各 agent 的回复文本
        E                   — 上一轮 embedding 矩阵
        diag                — 上一轮谱诊断结果
        Q_prepared          — 含 few-shot 的问题 prompt
        contexts[0..N-1]    — 各 agent 的对话历史
  输出: new_responses[0..N-1]
  ──────────────────────────────────────────────────────

  c ← diag.v₁                                        // Perron 向量作为贡献度
  (Adj, π, _) ← FORM_GRAPH(E, c, τ_g, k)            // 构建本轮通信图
  anchors ← ANCHOR_INDICES(c)                         // 高贡献锚点

  new_responses ← [""] × N

  for node in π do                                    // 按拓扑序遍历
    predecessors ← { m | Adj[node, m] = 1 }          // node 的前驱

    // 收集 peer 文本：优先用本轮已生成的新回复
    peer_texts ← []
    for pred in predecessors do
      if new_responses[pred] ≠ "" then
        peer_texts.append(new_responses[pred])
      else
        peer_texts.append(responses[pred])

    // 无前驱节点：注入锚点回复作为辩论参考
    if predecessors = ∅ then
      for a in anchors do
        if a ≠ node and responses[a] ∉ peer_texts then
          peer_texts.append(responses[a])

    // 构造 prompt 并调用 LLM
    if peer_texts ≠ [] then
      user_msg ← DEBATE_PROMPT(Q_prepared, peer_texts)
    else
      user_msg ← Q_prepared
    UPDATE_CONTEXT(contexts[node], user_msg)
    new_responses[node] ← LLM(contexts[node])

  return new_responses
```

---

### 10.4 子程序 4：Fiedler 跨阵营辩论 (Fiedler Cross-Camp Debate)

对应 `sparc_math_main.py::_fiedler_debate_round()` + `spectral_consensus.py::cross_camp_peers()`

```
function FIEDLER_DEBATE(responses, diag, Q_prepared, contexts):
  ──────────────────────────────────────────────────────
  输入: responses[0..N-1]  — 上一轮各 agent 的回复文本
        diag                — 上一轮谱诊断结果（含 C₊, C₋, v₁）
        Q_prepared          — 含 few-shot 的问题 prompt
        contexts[0..N-1]    — 各 agent 的对话历史
  输出: new_responses[0..N-1]
  ──────────────────────────────────────────────────────

  // 1. 确定两阵营的冠军（Perron 分量最大者）
  champ₊ ← argmax_{i ∈ C₊} diag.v₁[i]              // 阵营 A 的最强代表
  champ₋ ← argmax_{i ∈ C₋} diag.v₁[i]              // 阵营 B 的最强代表

  new_responses ← [""] × N

  // 2. 每个 agent 看到对方阵营冠军 + 己方阵营冠军（若非自己）
  for node = 0 to N-1 do
    peer_texts ← []

    if node ∈ C₊ then
      // node 在阵营 A → 注入阵营 B 冠军（跨阵营信号）
      peer_texts.append(responses[champ₋])
      // 附加己方冠军作为锚定（跳过自身）
      if champ₊ ≠ node then
        peer_texts.append(responses[champ₊])
    else
      // node 在阵营 B → 注入阵营 A 冠军
      peer_texts.append(responses[champ₊])
      if champ₋ ≠ node then
        peer_texts.append(responses[champ₋])

    // 去重
    peer_texts ← DEDUPLICATE(peer_texts)

    user_msg ← DEBATE_PROMPT(Q_prepared, peer_texts)
    UPDATE_CONTEXT(contexts[node], user_msg)
    new_responses[node] ← LLM(contexts[node])

  return new_responses
```

**与标准辩论的核心差异**：
- 标准辩论的信息流由 DAG 拓扑决定（"强者 → 弱者"），可能形成单向回声室
- Fiedler 辩论的信息流沿**分歧轴的瓶颈**注入，每个 agent **必定**看到对方阵营的最强论点
- 不依赖通信图，所有 agent 并行执行

---

### 10.5 子程序 5：验证级联 (Verification Cascade)

对应 `sparc_math_main.py::_verification_round()`

```
function VERIFICATION(responses, diag, Q_prepared, contexts, K_verify):
  ──────────────────────────────────────────────────────
  输入: responses[0..N-1]  — 当前各 agent 回复
        diag                — 当前谱诊断结果
        Q_prepared          — 含 few-shot 的问题 prompt
        contexts[0..N-1]    — 各 agent 的对话历史
        K_verify            — 候选答案数量上限
  输出: new_responses[0..N-1]
  ──────────────────────────────────────────────────────

  c ← diag.v₁

  // 1. 抽取各 agent 的数学答案
  extracted[i] ← EXTRACT_MATH_ANSWER(responses[i])   ∀i

  // 2. 按等价关系分组
  groups ← []                        // groups[g] = {agent 下标集合}
  group_answers ← []                 // 每组的规范化答案
  for i = 0 to N-1 do
    if extracted[i] = null then continue
    placed ← false
    for g = 0 to |groups|-1 do
      if IS_EQUIV(extracted[i], group_answers[g]) then
        groups[g] ← groups[g] ∪ {i}
        placed ← true; break
    if not placed then
      groups.append({i})
      group_answers.append(extracted[i])

  // 3. 若答案不足两种，验证无意义，原样返回
  if |group_answers| < 2 then
    return responses

  // 4. 按 (组大小, 组内贡献度之和) 降序排序，取前 K 个候选
  score(g) ← (|groups[g]|,  Σ_{j ∈ groups[g]} c[j])
  ranked ← SORT_DESC_BY(0..|groups|-1, score)
  candidates ← [ STRIP(group_answers[g]) for g in ranked[0 : K_verify] ]

  // 5. 构造验证 prompt（区别于辩论 prompt）
  verify_prompt ← VERIFY_PROMPT(Q_prepared, candidates)
  //   "Re-solve from scratch. Compare with candidates: [c₁], [c₂].
  //    State your final answer. If all candidates are wrong,
  //    provide the correct answer instead."

  // 6. 所有 agent 并行执行验证
  new_responses ← [""] × N
  for node = 0 to N-1 do
    UPDATE_CONTEXT(contexts[node], verify_prompt)
    new_responses[node] ← LLM(contexts[node])

  return new_responses
```

**设计要点**：验证轮与辩论轮在结构上根本不同——辩论是"看别人的推理过程"，验证是"看明确的候选答案然后从头重新推导"。这将推理模式从开放式探索（exploration）切换为聚焦式判别（discrimination）。

---

### 10.6 子程序 6：最终聚合 (Final Aggregation)

对应 `math_answer_utils.py::plurality_answer_by_contribution()` 及锚点回退

```
function FINAL_AGGREGATION(responses, c):
  ──────────────────────────────────────────────────────
  输入: responses[0..N-1]  — 最终轮各 agent 回复
        c ∈ ℝ^N            — 贡献度向量
  输出: final_answer        — 字符串
  ──────────────────────────────────────────────────────

  // ---------- 方式 1：贡献度加权多数投票 ----------

  extracted[i] ← EXTRACT_MATH_ANSWER(responses[i])   ∀i

  // 按等价关系分组（同子程序 5 的步骤 2）
  groups, group_answers ← GROUP_BY_EQUIVALENCE(extracted)

  // 每组的加权票数 = 组内成员贡献度之和
  for each group g do
    weight(g) ← Σ_{j ∈ groups[g]} c[j]

  // 选票数最多的组（平票时取加权票最高者）
  winner ← argmax_g ( |groups[g]|, weight(g) )

  if winner exists then
    return group_answers[winner]

  // ---------- 方式 2：锚点回退 ----------

  anchors ← ANCHOR_INDICES(c)         // 按贡献度降序的前若干 agent
  for a in anchors do
    ans ← EXTRACT_MATH_ANSWER(responses[a])
    if ans ≠ null then return ans
  for a in anchors do
    if responses[a] ≠ "" then return responses[a]

  return ""
```

---

### 10.7 子程序 7：谱间隙停滞检测 (Gap Stagnation Detection)

对应 `spectral_consensus.py::gap_is_stagnant()`

```
function GAP_STAGNANT(gap_history, δ_min, lookback):
  ──────────────────────────────────────────────────────
  输入: gap_history  — 历轮谱间隙比序列 [r⁰, r¹, ...]
        δ_min        — 最小改善量阈值（默认 0.02）
        lookback     — 回看窗口大小（默认 2）
  输出: bool
  ──────────────────────────────────────────────────────

  if |gap_history| < lookback + 1 then
    return false

  baseline ← gap_history[-(lookback + 1)]
  recent_best ← max(gap_history[-lookback :])

  return (recent_best - baseline) < δ_min
```

**含义**：如果最近 `lookback` 轮的谱间隙比最佳值相比基线提升不足 $\delta_{\min}$，判定辩论已停滞，继续辩论无法改善共识，应提前退出或切换到验证级联。

---

### 10.8 主算法：SPARC (SPectral Adaptive Reasoning Consensus)

对应 `sparc_math_main.py::SPARC_Math_Main.inference()`

```
algorithm SPARC
──────────────────────────────────────────────────────────
输入:
  Q                — 用户问题
  N                — agent 数量
  T                — 总轮数预算
  T_cross          — Fiedler 跨阵营辩论最大轮数
  T_std            — 标准图辩论最大轮数
  θ_H, θ_L, θ_V   — 谱间隙阈值 (high, low, verify)
  τ_c              — 共识温度
  τ_g, k           — 通信图参数
  δ_min, lookback  — 停滞检测参数
  K_verify         — 验证候选答案数
  N_consensus      — 早期共识所需最低票数
输出:
  final_answer     — 最终数学答案字符串
──────────────────────────────────────────────────────────

// ═══════════════════════════════════════════════════════
//  Phase 1: 多样化独立初始化
// ═══════════════════════════════════════════════════════

Q_prepared ← FEW_SHOT_EXAMPLES + "Problem: " + Q + " Answer:"

for i = 0 to N-1 do
  contexts[i] ← [
    { role: system, content: "It's a debate. Explain your reasons..." },
    { role: user,   content: Q_prepared }
  ]
  responses[i] ← LLM(contexts[i])

// 早期共识检查（初始轮，均匀权重）
(voted, count) ← PLURALITY_VOTE(responses, weights = 1⃗)
if voted ≠ null and count ≥ N_consensus then
  return voted                                   ◁ EXIT: 初始即共识

// ═══════════════════════════════════════════════════════
//  Phase 2: 谱诊断
// ═══════════════════════════════════════════════════════

E ← EMBED(responses)                             // L2-归一化 embedding
diag ← SPECTRAL_DIAGNOSIS(E, τ_c)                // §10.1
r ← diag.r                                       // 谱间隙比
gap_history ← [r]

// ═══════════════════════════════════════════════════════
//  Phase 3: 自适应辩论循环
// ═══════════════════════════════════════════════════════

n_cross ← 0                                      // 已用 Fiedler 轮数
n_std ← 0                                        // 已用标准辩论轮数
n_total ← 0                                      // 已用总轮数
budget ← T - 1                                   // 剩余预算（Phase 1 消耗 1 轮）

while budget > 0 do

  // ── 决策 1：共识是否已足够强？──
  if r ≥ θ_H then
    break                                        ◁ 强共识，跳过所有辩论

  // ── 决策 2：选择辩论策略 ──
  if r < θ_L and n_cross < T_cross then
    //  深度分歧 → Fiedler 跨阵营辩论（重火力）
    responses ← FIEDLER_DEBATE(responses, diag, Q_prepared, contexts)
    mode ← "fiedler_cross"
    n_cross ← n_cross + 1

  else if n_std < T_std then
    //  中度分歧 → 标准图辩论（轻量级）
    responses ← STANDARD_DEBATE(responses, E, diag, Q_prepared, contexts)
    mode ← "standard_graph"
    n_std ← n_std + 1

  else
    break                                        ◁ 两种辩论预算均耗尽

  n_total ← n_total + 1
  budget ← budget - 1

  // ── 辩论后重新诊断 ──
  E ← EMBED(responses)
  diag ← SPECTRAL_DIAGNOSIS(E, τ_c)
  r ← diag.r
  gap_history.append(r)

  // ── 辩论后早期共识检查 ──
  (voted, count) ← PLURALITY_VOTE(responses, weights = diag.v₁)
  if voted ≠ null and count ≥ N_consensus then
    return voted                                 ◁ EXIT: 辩论后达成共识

  // ── 辩论停滞检测 ──
  if GAP_STAGNANT(gap_history, δ_min, lookback) then
    break                                        ◁ 谱间隙不再改善，停止

end while

// ═══════════════════════════════════════════════════════
//  Phase 4: 验证级联（若仍不确定）
// ═══════════════════════════════════════════════════════

if r < θ_V then
  responses ← VERIFICATION(responses, diag, Q_prepared, contexts, K_verify)
  E ← EMBED(responses)
  diag ← SPECTRAL_DIAGNOSIS(E, τ_c)
  n_total ← n_total + 1

// ═══════════════════════════════════════════════════════
//  Phase 5: 最终聚合
// ═══════════════════════════════════════════════════════

return FINAL_AGGREGATION(responses, diag.v₁)
```

---

### 10.9 执行路径分析

根据谱间隙的不同值，SPARC 的实际执行路径大不相同：

```
                         初始谱间隙 r⁰
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         r⁰ ≥ θ_H       θ_L ≤ r⁰ < θ_H    r⁰ < θ_L
         (强共识)         (中度分歧)         (深度分歧)
              │               │               │
         直接聚合         标准图辩论       Fiedler 跨阵营辩论
         (N 次调用)       (1-2 轮)          (1-2 轮)
                              │               │
                         重新诊断 ──────→ 重新诊断
                              │               │
                         r ≥ θ_H?          r ≥ θ_H?
                        ┌──┴──┐          ┌──┴──┐
                       是    否          是    否
                        │     │          │     │
                      聚合  继续/停滞?  聚合  继续/停滞?
                              │               │
                              └───────┬───────┘
                                      │
                                 r < θ_V?
                              ┌───┴───┐
                             是      否
                              │       │
                          验证级联   直接聚合
                              │
                            聚合
```

**LLM 调用次数**：

| 场景 | 调用次数 | 说明 |
|------|---------|------|
| 简单题 (r⁰ ≥ θ_H) | $N$ | 仅初始化，零辩论 |
| 初始共识 (≥ N_consensus 票相同) | $N$ | 甚至不做谱诊断 |
| 中等题 (1 轮标准辩论后收敛) | $2N$ | 初始化 + 1 轮辩论 |
| 较难题 (2 轮辩论后收敛) | $3N$ | 初始化 + 2 轮辩论 |
| 困难题 (辩论 + 验证) | $(T+1) \times N$ | 满预算辩论 + 1 轮验证 |

对比 SelfOrg / SOO_Math 的固定 $T \times N$ 次调用，SPARC 在简单题上最多节省 $(T-1) \times N$ 次调用。

---

### 10.10 附：SOO_Math 简化版伪代码（对比参考）

SOO_Math 是 SPARC 的前身，使用 Perron 向量但**不使用** Fiedler 向量和谱间隙自适应。列出以供对比消融：

```
algorithm SOO_MATH
──────────────────────────────────────────────────────────
输入: Q, N, T, τ_c, τ_g, k, γ (早停阈值, 可选), N_consensus
输出: final_answer
──────────────────────────────────────────────────────────

// Phase 1: 独立初始化（同 SPARC Phase 1）
for i = 0 to N-1 do
  responses[i] ← LLM(contexts[i])

// 早期共识检查（均匀权重）
(voted, count) ← PLURALITY_VOTE(responses, 1⃗)
if voted ≠ null and count ≥ N_consensus then return voted

// Phase 2: 嵌入 + Perron 贡献度 + 构图
E ← EMBED(responses)
c ← SPECTRAL_DIAGNOSIS(E, τ_c).v₁                // 仅用 Perron 向量
G ← FORM_GRAPH(E, c, τ_g, k)
anchors ← ANCHOR_INDICES(c)

// Phase 3: 固定轮数辩论（无自适应）
for round = 1 to T-1 do

  // 可选：相似度早停
  if γ ≠ null and min_{i≠j} G.S[i,j] ≥ γ then break

  // 标准图辩论（同 SPARC 的 STANDARD_DEBATE）
  prev ← responses
  responses ← [""] × N
  for node in G.π do                              // 拓扑序
    preds ← PREDECESSORS(G.Adj, node)
    peer_texts ← [responses[p] or prev[p] for p in preds]
    if preds = ∅ then
      peer_texts ← [prev[a] for a in anchors if a ≠ node]
    responses[node] ← LLM(DEBATE_PROMPT(Q_prepared, peer_texts))

  E ← EMBED(responses)
  c ← SPECTRAL_DIAGNOSIS(E, τ_c).v₁
  G ← FORM_GRAPH(E, c, τ_g, k)
  anchors ← ANCHOR_INDICES(c)

  // 辩论后共识检查
  (voted, count) ← PLURALITY_VOTE(responses, c)
  if voted ≠ null and count ≥ N_consensus then return voted

// Phase 4: 最终聚合（同 SPARC Phase 5）
return FINAL_AGGREGATION(responses, c)
```

**SOO_Math vs SPARC 的差异一览**：

| | SOO_Math | SPARC |
|---|---|---|
| 贡献度 | Perron 向量 v₁ | Perron 向量 v₁（相同） |
| 辩论策略 | 仅标准图辩论 | 标准图辩论 + Fiedler 跨阵营辩论 |
| 轮数控制 | 固定 T 轮 | 谱间隙 r 自适应控制 |
| 停止条件 | 相似度阈值 γ（可选） | 谱间隙 θ_H + 停滞检测 |
| 验证机制 | 无 | 验证级联（θ_V 触发） |
| 分歧感知 | 无 | Fiedler 向量阵营划分 |

---

## 11. 论文撰写框架：统一叙事与完整 Storyline

### 11.1 核心矛盾：两条平行的研究线

现有多智能体协作研究沿两条相互独立的路线发展：

**路线 A：辩论范式（Debate Paradigm）**

关注"agents 之间怎么讨论"——广播式多轮辩论、投票 vs 共识、谄媚（sycophancy）问题。

代表工作：
- Du et al. (ICML 2024) — Multi-Agent Debate (MAD) 开山之作
- DMAD (ICLR 2025) — 多样化推理策略打破思维定势
- Debate or Vote (NeurIPS 2025 Spotlight) — 证明多数投票解释了 MAD 大部分增益
- CONSENSAGENT (ACL 2025) — 缓解辩论中的谄媚行为
- Voting or Consensus (ACL 2025) — 系统性比较投票 vs 共识协议

**路线 B：通信拓扑范式（Communication Topology Paradigm）**

关注"谁该说给谁听"——动态图构建、贡献度路由、信息流方向。

代表工作：
- DyLAN (Liu et al., 2024) — 动态 LLM Agent 网络
- SelfOrg (Tastan et al., ICLR 2026) — Response-conditioned DAG + Shapley 近似贡献度
- MacNet (Qian et al., 2025) — 分层通信网络
- AgentVerse (Chen et al., 2024) — 多 agent 协作平台

**两条路线的盲区**：

| 路线 A（辩论） | 路线 B（通信拓扑） |
|---|---|
| 所有人看所有人（广播），无信息路由 | 有路由，但不分析分歧在哪里 |
| 不知道何时该辩论、何时该停 | 固定轮数，不自适应问题难度 |
| 不区分"哪些 agent 持不同意见" | DAG 按"强→弱"路由，对回声室视而不见 |

### 11.2 我们的立场：不选边站，站在交叉点

SPARC 的核心观察是：**"谁说给谁听"（路由）和"怎么讨论、讨论多少"（辩论策略与计算预算）是同一个问题的两面**。把它们分开处理会导致次优——拓扑方法不知道何时停，辩论方法不知道该让谁说话。

而共识矩阵 $A = \exp(S/\tau)$ 的特征谱**同时编码了这两方面的信息**：

```
                    共识矩阵 A 的特征谱
                          │
          ┌───────────────┼───────────────┐
          │               │               │
       Perron 向量      Fiedler 向量      谱间隙比
        v₁ (§4)         v₂ (§6)          r (§5)
          │               │               │
     全局可靠度估计    分歧结构发现      共识强度度量
     (改进路由)       (改进辩论拓扑)    (改进计算调度)
          │               │               │
     谁更可靠？       分歧在哪里？     需要辩论吗？
     → 加权投票       → 跨阵营配对     → 自适应轮数
     → DAG 路由       → 打破回声室     → 何时验证
          │               │               │
          └───────────────┼───────────────┘
                          │
              统一的谱治理框架 (SPARC)
```

### 11.3 推荐的 Introduction 结构

**P1 — 背景与两条路线**

> Multi-agent collaboration powered by large language models has emerged as a promising paradigm for complex reasoning. Two parallel research threads have developed: *debate-based methods* (Du et al., 2024; Liang et al., 2024) improve answer quality through iterative multi-round discussion among agents, while *topology-based methods* (Liu et al., 2024; Tastan et al., 2026) optimize the communication structure by routing information through contribution-weighted directed graphs.

**P2 — 揭示两条路线各自的盲区**

> However, these paradigms address different aspects of the same underlying problem but have evolved largely in isolation. Debate methods control *what agents say to each other* but treat all agents symmetrically — broadcasting all responses without considering who should listen to whom. Recent findings even question whether debate itself contributes beyond simple majority voting (Choi et al., NeurIPS 2025). Topology methods control *information routing* but are blind to the *structure of disagreement*: they route from high-contribution to low-contribution agents along a fixed DAG, but cannot detect when agents split into competing opinion camps, nor adapt the number of interaction rounds to problem difficulty.

**P3 — 我们的洞察与方法**

> We observe that these seemingly separate concerns — reliability estimation, disagreement detection, and compute allocation — are all simultaneously encoded in a single mathematical object: the eigenspectrum of the *consensus matrix* constructed from agent response embeddings. Specifically, (1) the **Perron eigenvector** captures transitive, network-wide consensus centrality, providing a more principled reliability estimate than local centroid distance (Tastan et al., 2026); (2) the **Fiedler eigenvector** reveals the dominant axis of disagreement, partitioning agents into opposing camps and enabling targeted cross-camp debate that breaks echo-chamber convergence; and (3) the **spectral gap ratio** measures consensus strength, serving as an endogenous difficulty signal for adaptive inference-time compute allocation — analogous to the role of the spectral gap in governing mixing times of Markov chains (DeGroot, 1974).

**P4 — 方法概述与结果预览**

> We instantiate this insight in **SPARC** (SPectral Adaptive Reasoning Consensus), a framework that unifies topology governance and debate strategy under spectral analysis. When the spectral gap is large, SPARC skips debate entirely; when deep disagreement is detected (small gap), it deploys Fiedler-guided cross-camp debate; when debate stagnates, it triggers a structurally distinct verification cascade. Experiments on [数据集列表] show that SPARC achieves [结果概述], while using [token 效率数据] fewer LLM calls on easy problems compared to fixed-round baselines.

### 11.4 推荐的 Related Work 组织

```
2. Related Work

2.1 Multi-Agent Debate and Consensus
    - Du et al. (ICML 2024): MAD 基础框架
    - DMAD (ICLR 2025): 多样化推理策略
    - Debate or Vote (NeurIPS 2025 Spotlight): 辩论≈鞅，投票是主要增益来源
    - CONSENSAGENT (ACL 2025): 多智能体共识中的谄媚问题
    - Voting or Consensus (ACL 2025): 投票 vs 共识协议的系统比较
    - Reaching Agreement / Aegean (arXiv 2025): 形式化共识协议
    [我们的区别：上述工作不分析分歧结构，不做自适应计算分配]

2.2 Communication Topology in Multi-Agent Systems
    - DyLAN (Liu et al., 2024): 动态 agent 网络
    - SelfOrg (Tastan et al., ICLR 2026): Response-conditioned DAG
    - MacNet, AgentVerse, G-Designer
    [我们的区别：上述工作不利用特征谱，不区分辩论策略，固定轮数]

2.3 Spectral Methods and Opinion Dynamics
    - Perron-Frobenius (1907/1912): 正矩阵主特征向量
    - Fiedler (1973): 代数连通度与图二分
    - Shi & Malik (2000): 归一化切割与谱聚类
    - DeGroot (1974): 意见动力学与收敛的谱间隙条件
    [我们的桥梁：首次将这些经典工具引入 LLM 多智能体协作]

2.4 Inference-Time Compute Scaling
    - Self-Consistency (Wang et al., 2023)
    - Best-of-N, Process Reward Models
    [我们的位置：SPARC 的谱间隙是一种无需外部奖励模型的
     自适应计算缩放信号]
```

### 11.5 推荐的 Method 章节结构

```
3. Method: SPARC

3.1 Problem Setup and Notation
    - N 个 agent，查询 Q，响应 r_i，embedding e_i
    - 目标：在最少 LLM 调用下得到正确答案

3.2 Consensus Matrix Construction
    - S = E E⊤（余弦相似度）
    - A = exp(S/τ)（正矩阵，满足 Perron-Frobenius）
    - τ 的作用与选择

3.3 Spectral Diagnosis: Three Signals from One Matrix
    3.3.1 Perron Eigenvector → Agent Reliability
          - 理论保证（Perron-Frobenius）
          - 与 SelfOrg 质心法的比较（含反例）
    3.3.2 Spectral Gap Ratio → Consensus Strength
          - 定义与 Markov 混合时间类比
          - 作为难度信号的直觉
    3.3.3 Fiedler Eigenvector → Disagreement Structure
          - 阵营划分与归一化切割
          - 与标准 DAG 路由的区别

3.4 Adaptive Inference Pipeline
    3.4.1 Phase 1: Diverse Initialization
    3.4.2 Phase 2: Spectral Triage (θ_H / θ_L 分流)
    3.4.3 Phase 3: Debate Strategy Selection
          - Standard graph debate (中度分歧)
          - Fiedler cross-camp debate (深度分歧)
    3.4.4 Phase 4: Verification Cascade (低置信度)
    3.4.5 Phase 5: Weighted Plurality Aggregation

3.5 Complexity Analysis
    - 简单/中等/困难题的 LLM 调用次数
    - 谱诊断本身的额外开销（O(N³), N 通常 4-6，可忽略）
```

### 11.6 关键实验设计建议

论文的实验部分应回答以下问题：

**Q1：SPARC 整体性能如何？**（主实验表格）

| 方法 | 类型 | MATH | GSM8K | GSM-Hard | AQUA | AIME |
|------|------|------|-------|----------|------|------|
| Single Agent / CoT | 无协作 | | | | | |
| Self-Consistency | 纯投票 | | | | | |
| Majority Voting (N 个独立) | 纯投票 | | | | | |
| MAD (Du et al.) | 辩论 | | | | | |
| SelfOrg (Tastan et al.) | 通信拓扑 | | | | | |
| DyLAN | 通信拓扑 | | | | | |
| **SOO_Math (ours, w/o SPARC)** | 谱拓扑 | | | | | |
| **SPARC (ours, full)** | 谱统一 | | | | | |

**Q2：每个谱组件贡献了什么？**（消融实验）

| 变体 | Perron | Fiedler | 谱间隙自适应 | 验证级联 |
|------|--------|---------|------------|---------|
| SelfOrg (baseline) | - | - | - | - |
| + Perron 贡献度 (=SOO) | Y | - | - | - |
| + 自适应轮数 | Y | - | Y | - |
| + Fiedler 辩论 | Y | Y | Y | - |
| + 验证级联 (=full SPARC) | Y | Y | Y | Y |

**Q3：谱间隙真的是好的难度信号吗？**（分析实验）
- 按题目难度分桶，统计初始谱间隙的分布
- 画 散点图：初始谱间隙 vs 最终正确率
- 预期：强正相关

**Q4：计算效率如何？**（token 效率）
- 简单/中/难题各自的平均 LLM 调用次数
- 与固定 T 轮方法的 token 消耗对比
- 预期：简单题大幅节省，难题略多

**Q5：Fiedler 辩论真的打破了回声室吗？**（case study）
- 找到标准辩论收敛到错误答案、Fiedler 辩论纠正的案例
- 展示 Fiedler 向量划分的两个阵营与正确/错误答案的对应关系

### 11.7 论文贡献总结（Contributions 段落草稿）

> Our contributions are threefold:
>
> 1. **A unified spectral framework for multi-agent collaboration.** We show that the eigenspectrum of the consensus matrix simultaneously provides three complementary governance signals — reliability estimation (Perron vector), disagreement detection (Fiedler vector), and consensus measurement (spectral gap) — bridging the previously disjoint research threads of debate protocols and communication topology optimization.
>
> 2. **SPARC: an adaptive inference pipeline.** We propose SPARC (SPectral Adaptive Reasoning Consensus), which uses the spectral gap as an endogenous difficulty signal to adaptively allocate compute, deploys Fiedler-guided cross-camp debate to break echo chambers, and triggers verification cascades when debate stagnates. SPARC achieves [X]% improvement over [baseline] while using [Y]% fewer LLM calls on easy problems.
>
> 3. **Comprehensive empirical analysis.** We conduct ablation studies isolating the contribution of each spectral component, validate the spectral gap as a difficulty proxy, and demonstrate through case studies that Fiedler-guided debate corrects failures where standard topology-based debate converges to incorrect consensus.

### 11.8 Limitation 与 Future Work 建议

论文中应诚实讨论的局限性：

1. **DeGroot 类比的非严格性**：LLM 辩论是高度非线性的，谱间隙作为收敛预测器的理论保证仅在线性意见动力学下成立。我们提供的是启发式动机 + 实验验证，不是严格证明。

2. **二分局限**：Fiedler 向量只能做二分（两个阵营）。当存在三个或更多竞争答案时，可能需要更多特征向量（扩展到 k-way 谱聚类）。这是自然的未来方向。

3. **Embedding 质量依赖**：整个谱分析建立在 embedding 能够反映语义相似度的假设上。如果 embedding 模型不好（例如对数学表达式的编码不精确），谱信号可能失真。

4. **小 N 的统计限制**：当 agent 数量 $N$ 很小（如 4 个）时，$4 \times 4$ 矩阵的特征谱信息量有限。增大 $N$ 可能提升谱诊断的可靠性，但也增加 LLM 调用成本。

5. **阈值敏感性**：$\theta_H, \theta_L, \theta_V$ 的选择目前是超参数。未来可以探索学习这些阈值（例如基于验证集或 online learning）。
