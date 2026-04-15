# SOO（soo）算法实现逻辑（仓库实现版）

本文面向本仓库 `methods/selforg/` 的当前实现，解释 **SOO（Self-Organized Orchestration）** 在本项目中的实际流程、关键数据结构（嵌入/相似度矩阵/通信图）、以及它与 `selforg` 的**唯一实现差异点**：贡献度（reliability / contribution）如何计算。  
最后给出与代码结构一致的伪代码，写法参考 `methods/dylan/DYLAN_ALGORITHM.md`。

---

## 1. SOO 在这个仓库里是什么？

入口类是 `methods/selforg/soo_main.py::SOO_Main`，它 **继承并复用** `methods/selforg/selforg_main.py::SelfOrg_Main` 的全部主循环逻辑，只覆盖一个 hook：

- `SOO_Main._estimate_contributions()`：改用 `methods/selforg/soo_contribution.py::estimate_contributions_soo()`

也就是说：

- **轮次流程、LLM 调用方式、通信图构建（form_graph）、拓扑序执行、最终输出选择** 都与 `selforg` 完全一致；
- **唯一变化**是“每个 agent 的贡献度/可靠度分数 \(c_i\)”的估计方法。

在 `methods/__init__.py` 里映射为：

- `soo` → `SOO_Main`

---

## 2. SelfOrg vs. SOO（差异点对照）

两者共用的骨架（来自 `SelfOrg_Main.inference()`）：

- Round 0：各 agent 独立作答 → 计算嵌入 → 估计贡献度 → 成通信图（DAG）→ 选本轮“最终答案”候选
- Round 1..T-1：按上一轮图的拓扑序逐个 agent “读前驱→改写” → 重新嵌入/贡献度/成图/选最终
- 输出：返回最后一轮 `final_idx` 对应的回复文本

唯一区别：贡献度估计

- **SelfOrg（`methods/selforg/contribution.py`）**：用“接近群体均值”的程度
  - \(c_i=\cos(r_i,\ \frac{1}{N}\sum_j r_j)\)
- **SOO（`methods/selforg/soo_contribution.py`）**：用“共识矩阵”的主特征向量（Perron–Frobenius 思路）
  - 先用嵌入构建共识矩阵，再用最大特征值对应的特征向量作为可靠度分数

---

## 3. SOO 的核心：Consensus Matrix + 主特征向量（实现细节）

文件：`methods/selforg/soo_contribution.py`

设本轮有 \(N\) 个 agent 回复，嵌入向量为 \(V\in\mathbb{R}^{N\times d}\)（第 \(i\) 行为 \(v_i\)）。

实现按以下步骤计算贡献度 \(c\in\mathbb{R}^N\)：

### 3.1 相似度矩阵 \(S\)

```text
S = embeddings @ embeddings.T
```

代码直接用内积 \(S_{ij}=v_i^\top v_j\)。这等价于余弦相似度的前提是：`embed_texts()` 输出的向量已做 **L2 归一化**（实现里用注释写了 “embeddings should be L2-normalised”）。

### 3.2 共识矩阵 \(A\)

```text
A = exp(S / tau_consensus)
```

- `tau_consensus` 由 `SOO_Main.__init__()` 从配置读取：`consensus_tau`（默认 1.0；仓库 `config_main.yaml` 默认为 0.3）。
- \( \tau \) 越小，指数放大越强，矩阵更“尖锐”，更偏向强相似对的影响。

> 注意：这里 **不会**像 `form_graph()` 那样把对角线清零；因此 \(A_{ii}=\exp(S_{ii}/\tau)\) 通常是 \(\exp(1/\tau)\)（若嵌入单位范数）。

### 3.3 主特征向量作为贡献度 \(c\)

```text
eigenvalues, eigenvectors = eigh(A)
c = abs(eigenvectors[:, -1])
```

实现使用 `np.linalg.eigh(A)`（对称矩阵的特征分解），并取 **最大特征值**对应的特征向量（`-1` 列），再取绝对值作为每个 agent 的贡献度 \(c_i\)。

直觉：

- 若某条回复与很多回复都相似（在共识矩阵里与其他节点“互相强化”），其在主特征向量上的分量会更大；
- 这是一种类似“谱中心性 / reliability” 的估计：**更能代表整体共识的回答更可靠**。

---

## 4. SOO 的整套推理流程（与代码一致）

关键入口：`methods/selforg/selforg_main.py::SelfOrg_Main.inference()`  
SOO 只替换其中的 `contributions = self._estimate_contributions(embeddings)`。

设：

- \(N\)：agent 数量（由 `infer_agent_keys(dataset_name, config)` 得到）
- \(T\)：最大轮数 `max_rounds`（最小为 1）
- `tau` / `top_k`：用于 `form_graph()` 的成图超参（与 SOO 的 `consensus_tau` 不同）

### 4.1 Round 0：独立作答 → 嵌入 → SOO 贡献度 → 成图 → 选本轮最优

1) agent 独立作答（无 peer）：
   - `responses[i] = call_llm(system=role_i, user=build_user_prompt(query, None))`
2) 计算嵌入：`embeddings = embed_texts(responses, embedding_model, device)`
3) SOO 贡献度：`contributions = estimate_contributions_soo(embeddings, consensus_tau)`
4) 成通信图：`graph = form_graph(embeddings, contributions, tau, top_k)`
5) 选当前最优：`final_idx = select_final_response(embeddings, contributions)`
   - 该选择逻辑仍是 SelfOrg 的“**贡献度加权质心最近**”（详见 `methods/selforg/contribution.py`）

### 4.2 Round 1..T-1：按拓扑序“读前驱→改写” → 重算（嵌入/贡献度/图/最优）

每轮（\(_t\ge1\)）：

0) （可选）early stop：若配置了 `early_stop_gamma`，检查上一轮图的相似度矩阵最小值是否 \(\ge \gamma\)（实现见 `graph_formation.check_early_stop`）
1) `prev_responses = responses`，新一轮 `responses = [""] * N`
2) `best_prev = prev_responses[final_idx]`
3) 获取执行顺序：`topo = graph.topological_order`
4) 对每个 `node in topo`：
   - `predecessors = get_predecessors(graph.adjacency, node)`（注意邻接矩阵方向：`A[node, pred]=1` 表示 `pred -> node`）
   - peer 文本优先用同轮已生成的 `responses[pred]`，否则回退到 `prev_responses[pred]`
   - 若该节点没有前驱，则尝试注入上一轮最优 `best_prev` 作为唯一 peer（只要它非空且不同于该节点上一轮回复）
   - 调用 LLM 得到本轮 `responses[node]`
5) 本轮结束：重新 `embed_texts` → **SOO 贡献度** → `form_graph` → `select_final_response`

最终返回 `responses[final_idx]`。

---

## 5. 与当前实现一致的伪代码

### 5.1 SOO_Main：复用 SelfOrg 主循环，仅替换贡献度

```text
inputs:
  query
  agent_keys[0..N-1]
  T = max_rounds
  tau, top_k                    # graph params (SelfOrg)
  consensus_tau                 # SOO params
  embedding_model, embedding_device
  gamma = early_stop_gamma or None

function SOO_CONTRIBUTION(embeddings V[N,d], consensus_tau):
  S = V @ V^T
  A = exp(S / consensus_tau)
  (eigvals, eigvecs) = eigh(A)              # symmetric eigendecomp
  c = abs(eigvecs[:, -1])                   # principal eigenvector
  return c

function ROUND0(query):
  for i in 0..N-1:
    responses[i] = LLM(system=ROLE(agent_keys[i]),
                       user=BUILD_USER_PROMPT(query, peers=None))
  V = EMBED(responses, embedding_model, embedding_device)
  c = SOO_CONTRIBUTION(V, consensus_tau)
  graph = FORM_GRAPH(V, c, tau, top_k)      # same as SelfOrg (break cycles + topo sort)
  final_idx = SELECT_FINAL_RESPONSE(V, c)   # same as SelfOrg (weighted centroid nearest)
  return responses, V, c, graph, final_idx

function ITERATE(query, prev_responses, prev_graph, prev_final_idx):
  if gamma != None and EARLY_STOP(prev_graph.S, gamma):
    return STOP(prev_responses, prev_final_idx)

  responses = [""] * N
  best_prev = prev_responses[prev_final_idx]

  for node in prev_graph.topological_order:
    preds = PREDECESSORS(prev_graph.A, node)    # {p | A[node,p]=1 means p -> node}
    peers = []
    for p in preds:
      peers.append(responses[p] if responses[p] != "" else prev_responses[p])
    if preds empty and best_prev != "" and best_prev != prev_responses[node]:
      peers = [best_prev]

    responses[node] = LLM(system=ROLE(agent_keys[node]),
                          user=BUILD_USER_PROMPT(query, peers or None))

  V = EMBED(responses, embedding_model, embedding_device)
  c = SOO_CONTRIBUTION(V, consensus_tau)
  graph = FORM_GRAPH(V, c, tau, top_k)
  final_idx = SELECT_FINAL_RESPONSE(V, c)
  return responses, V, c, graph, final_idx

main(query):
  responses, V, c, graph, final_idx = ROUND0(query)
  for t in 1..T-1:
    out = ITERATE(query, responses, graph, final_idx)
    if out is STOP:
      break
    responses, V, c, graph, final_idx = out
  return responses[final_idx]
```

### 5.2 FORM_GRAPH（SelfOrg 公用）：相似度阈值 + “强者影响弱者” + 断环 + 拓扑序

SOO 没有改 `form_graph()`；其行为与 `methods/selforg/graph_formation.py::form_graph()` 一致：

```text
FORM_GRAPH(V, c, tau, k):
  S = COS_SIM_MATRIX(V); S[i,i]=0
  A = zeros(N,N)                       # A[n,m]=1 means m -> n
  for n in 0..N-1:
    candidates = { m!=n | S[n,m] >= tau }
    candidates = TOP_K_BY_SIM(candidates, k)
    for m in candidates:
      if c[m] > c[n]:
        A[n,m] = 1                     # keep only stronger->weaker edges
  A = BREAK_CYCLES(A, c)               # remove outgoing edge from weakest-in-cycle
  topo = TOPO_SORT(A, tie_break=-c)    # higher contribution first when multiple choices
  return (A, topo, c, S)
```

---

## 6. 配置项（实现里实际用到的）

文件：`methods/selforg/configs/config_main.yaml`

- `max_rounds`：最多轮数 \(T\)
- `embedding_model` / `embedding_device`：嵌入模型与设备
- `tau`：成图时的相似度阈值（注意：这是 **通信图阈值**，不是 SOO 的 `consensus_tau`）
- `top_k`：每个节点保留的最多候选前驱数
- `early_stop_gamma`：可选早停阈值（最小两两相似度 \(\ge\gamma\) 则停止）
- `consensus_tau`：SOO 贡献度里指数温度 \(\tau_{\text{consensus}}\)

---

## 7. 读代码容易踩的点（SOO 特有 + 与 SelfOrg 对齐）

- **SOO 只改贡献度，不改成图策略**：通信图仍由 `form_graph()` 决定，仍遵循“相似度阈值 + 强者影响弱者 + 断环 + 拓扑序”。
- **嵌入是否归一化很关键**：`estimate_contributions_soo()` 用的是 `embeddings @ embeddings.T`，如果嵌入未做 L2 normalize，则 \(S\) 不是余弦相似度，会改变 \(A\) 与主特征向量的含义。
- **`consensus_tau` 与 `tau` 是两套温度/阈值**：
  - `consensus_tau`：只影响 SOO 的可靠度（共识矩阵指数放大）
  - `tau`：只影响通信图边的筛选（相似度阈值）
- **邻接矩阵方向与直觉相反**：`adjacency[n,m]=1` 表示 **m → n**（m 影响 n），构造 peer 时取的是 node 的“前驱 m”。

