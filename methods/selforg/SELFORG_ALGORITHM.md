# SelfOrg（selforg）算法实现逻辑（仓库实现版）

本文面向本仓库 `methods/selforg/` 的当前实现，解释 **Self-Organized Multi-Agent Collaboration（SelfOrg）** 的实际执行流程、关键数据结构（通信图/贡献度/相似度矩阵）、每轮如何组织 agent 的“读他人→改写”，以及最终如何选择输出。  
并给出与代码结构一致的伪代码，写法参考 `methods/dylan/DYLAN_ALGORITHM.md`。

---

## 1. 这个仓库里的 SelfOrg 是什么形态？

入口类为 `methods/selforg/selforg_main.py::SelfOrg_Main`，特点是：

- **无异步/无复杂 agent wrapper**：直接用 `MAS.call_llm()` 调模型。
- **通信结构由“嵌入相似度 + 贡献度”自动形成**：每轮会重算嵌入、贡献度，然后构建一个 **有向无环图（DAG）** 作为下一轮的“信息流”。
- **每轮按拓扑序执行**：保证一个 agent 在本轮改写时，能优先看到其“前驱”在本轮（如果已生成）或上一轮的回复。
- **最终输出不是投票，而是“加权质心最近”**：用贡献度作权重算 centroid，再选与 centroid 余弦相似度最大的那条回复作为最终答案。

配置文件在 `methods/selforg/configs/config_main.yaml`：

- `max_rounds`：最多轮数（默认 3）
- `tau`：相似度阈值（默认 0.5）
- `top_k`：每个节点最多连接多少个“相似候选前驱”（默认 2；`null` 表示不截断）
- `early_stop_gamma`：可选的早停阈值（`null` 表示不开启）
- `embedding_model`：`all-MiniLM-L6-v2` / `tfidf` / `hybrid`（Dense+TF-IDF 0.5/0.5）
- `embedding_device`：如 `cpu` / `cuda:1`

---

## 2. 关键数据结构与模块

### 2.1 Agent 角色与提示词拼接

文件：`methods/selforg/prompt_main.py`

- **角色库**：`AGENT_ROLES`，key 如 `assistant/programmer/mathematician/...`，每个角色只有一段 `system_prompt`。
- **按数据集推断角色集合**：`infer_agent_keys(dataset_name, method_config)`
  - 若配置里显式给了 `agent_keys`，直接用
  - 否则根据 `dataset_name`（例如包含 `mmlu/gpqa/sci` 等）在两套默认 keys 间切换，并可用 `num_agents` 截断
- **user prompt**：`build_user_prompt(query, peer_responses)`
  - 无 peer：`Please answer... Question: ...`
  - 有 peer：拼 `Previous response i: ...`，并提示“可忽略错误/无用回复”

### 2.2 嵌入与相似度矩阵

文件：`methods/selforg/embedding.py`

- `embed_texts(texts, model_name, device)`：把每个 agent 的回复转成 \(d\) 维向量
  - 优先 `sentence-transformers`
  - 其次 `transformers + torch`
  - 最后 TF-IDF（`sklearn`）
  - `hybrid`：Dense 与 TF-IDF 拼接后再归一化（0.5/0.5）
- `pairwise_cosine_similarity(embeddings)`：得到 \(S\in\mathbb{R}^{N\times N}\) 的两两余弦相似度矩阵

### 2.3 贡献度估计（实现版）

文件：`methods/selforg/contribution.py`

实现采用“Shapley-inspired”的简化形式：

- 先求平均嵌入 \(r_{\text{avg}}=\frac{1}{N}\sum_n r_n\)
- 每个 agent 的贡献度（score）：

\[
\psi_n=\cos(r_n,\ r_{\text{avg}})
\]

这会倾向于给“更接近群体中心表达”的回复更高贡献度。

此外：

- `compute_weighted_centroid(embeddings, contributions)`：按贡献度加权求 centroid
- `select_final_response(embeddings, contributions)`：选与 centroid 最相似的回复作为当前轮 `final_idx`

### 2.4 通信图（Communication Graph）构建

文件：`methods/selforg/graph_formation.py`

通信图结构：

```text
CommunicationGraph:
  adjacency A[N,N]   # A[n,m]=1 表示 m -> n（m 影响 n）
  similarity_matrix S[N,N]
  contributions c[N]
  topological_order: list[int]
```

构建逻辑 `form_graph(embeddings, contributions, tau, k)`：

1) 计算两两相似度 \(S=\cos(r_n,r_m)\)，并把对角线置 0。
2) 对每个节点 \(n\)，找候选前驱集合：
   - `candidates = { m!=n | S[n,m] >= tau }`
   - 若 `k` 非空且候选过多，仅保留相似度最大的前 \(k\) 个
3) 只保留“强者影响弱者”的边：
   - 若 `contributions[m] > contributions[n]`，则置边 \(m\to n\)（编码为 `A[n,m]=1`）
4) **断环** `_break_cycles(A, contributions)`：
   - 找到一个有向环后，取环上贡献度最低的节点 `weakest`
   - 删除其在环中的一条出边 `weakest -> succ`（编码为 `A[succ, weakest]=0`）
   - 直到无环为止
5) **拓扑排序** `_topological_sort(A, contributions)`：
   - Kahn 算法：每次从入度为 0 的节点中，优先取贡献度更高者
   - 若仍有残留（异常情况），按贡献度降序追加

### 2.5 Early stop（可选）

文件：`methods/selforg/graph_formation.py::check_early_stop(S, gamma)`

实现并非“2/3 共识”，而是：

- 计算所有两两相似度的最小值 \(s_{\min}=\min_{n<m} S[n,m]\)
- 若 \(s_{\min}\ge \gamma\)，认为“所有回复都足够相似”，直接早停

是否启用由 `config_main.yaml` 的 `early_stop_gamma` 控制（默认 `null`，不开启）。

---

## 3. `SelfOrg_Main` 的逐步执行流程（与代码一致）

文件：`methods/selforg/selforg_main.py`

设：

- \(N\)：agent 数量（`agent_keys` 长度）
- \(T\)：最大轮数（`max_rounds`，最小为 1）

### 3.1 Round 0：独立作答 → 形成图 → 选一个“当前最优”

1) 对每个 agent \(i\in[0,N)\)：
   - 用该 agent 的 `system_prompt`
   - user prompt 只包含 `Question: query`
   - 得到 `responses[i]`
2) 将所有回复嵌入：`embeddings = embed_texts(responses, ...)`
3) 估计贡献度：`contributions = estimate_contributions(embeddings)`
4) 构建通信图：`graph = form_graph(embeddings, contributions, tau, top_k)`
5) 选当前最优下标：
   - `final_idx = select_final_response(embeddings, contributions)`
   - 本质：选与**贡献度加权质心**最接近的那条回复

### 3.2 Round 1..T-1：按拓扑序“读前驱→改写” → 重算图 → 更新最优

每一轮 `_t>=1`：

0) （可选）若设置了 `early_stop_gamma`：
   - 对上一轮图的相似度矩阵 `graph.similarity_matrix` 检查 `check_early_stop(S, gamma)`
   - 若满足，则停止迭代，直接输出当前 `final_idx`

1) 保存上一轮回复：`prev_responses = responses`，并初始化本轮 `responses = [""] * N`
2) 记录上一轮“最优回复文本”：
   - `best_prev = prev_responses[final_idx]`
3) 取本轮执行顺序：`topo = graph.topological_order`
4) 对 `topo` 中每个节点 `node`，构造其 peer_responses：
   - `predecessors = get_predecessors(graph.adjacency, node)` 取入边来源
   - 对每个 `pred`：
     - 如果本轮 `responses[pred]` 已生成，就用它（同轮最新信息）
     - 否则回退到 `prev_responses[pred]`
   - 特殊兜底：若 `node` **没有前驱**，则尝试把 `best_prev` 作为唯一 peer：
     - 条件：`best_prev` 非空且 `best_prev != prev_responses[node]`
     - 直觉：即使图上无前驱，也尽量把上一轮最“代表群体”的答案注入给该节点做改写
5) 调用 agent：
   - user prompt = `build_user_prompt(query, peer_texts)`
   - system prompt = 对应角色的 `system_prompt`
   - 得到本轮 `responses[node]`
6) 本轮结束后，重复 Round0 的嵌入/贡献/成图/选最优步骤，更新 `graph` 与 `final_idx`

最终返回 `responses[final_idx]`。

---

## 4. 与当前实现一致的伪代码

### 4.1 SelfOrg_Main：多轮自组织通信图协作

```text
inputs:
  query
  agent_keys[0..N-1]            # roles for agents
  T = max_rounds
  tau, k                        # graph parameters
  gamma = early_stop_gamma or None
  embedding_model, embedding_device

state per round:
  responses[0..N-1]             # texts
  embeddings[0..N-1]            # vectors
  contributions c[0..N-1]       # scalar scores
  similarity_matrix S[N,N]
  adjacency A[N,N]              # A[n,m]=1 means m -> n
  topo_order[0..N-1]
  final_idx                     # index of final response

function ROUND0(query):
  for i in 0..N-1:
    responses[i] = LLM(
      system = ROLE_SYSTEM(agent_keys[i]),
      user   = BUILD_USER_PROMPT(query, peer_responses=None)
    )
  embeddings = EMBED(responses, embedding_model, embedding_device)
  c = CONTRIBUTION(embeddings)                    # c[i] = cos(r_i, mean_r)
  graph = FORM_GRAPH(embeddings, c, tau, k)       # build A, break cycles, topo sort
  final_idx = SELECT_FINAL(embeddings, c)         # closest to weighted centroid
  return responses, graph, final_idx

function BUILD_PEERS(node, graph, responses, prev_responses, best_prev):
  preds = PREDECESSORS(graph.A, node)             # {m | A[node,m]=1}
  peer_texts = []
  for pred in preds:
    if responses[pred] != "":
      peer_texts.append(responses[pred])          # same-round updated
    else:
      peer_texts.append(prev_responses[pred])     # fallback to last round
  if preds is empty and best_prev != "" and best_prev != prev_responses[node]:
    peer_texts = [best_prev]                      # inject previous best
  return peer_texts or None

function ITERATE(query, prev_responses, prev_graph, prev_final_idx):
  if gamma is not None:
    if EARLY_STOP(prev_graph.S, gamma):           # min pairwise similarity >= gamma
      return STOP(prev_responses, prev_final_idx)

  responses = [""] * N
  best_prev = prev_responses[prev_final_idx]

  for node in prev_graph.topo_order:
    peers = BUILD_PEERS(node, prev_graph, responses, prev_responses, best_prev)
    responses[node] = LLM(
      system = ROLE_SYSTEM(agent_keys[node]),
      user   = BUILD_USER_PROMPT(query, peers)
    )

  embeddings = EMBED(responses, embedding_model, embedding_device)
  c = CONTRIBUTION(embeddings)
  graph = FORM_GRAPH(embeddings, c, tau, k)
  final_idx = SELECT_FINAL(embeddings, c)
  return responses, graph, final_idx

main(query):
  responses, graph, final_idx = ROUND0(query)
  for t in 1..T-1:
    (maybe_stop) = ITERATE(query, responses, graph, final_idx)
    if stop: break
    responses, graph, final_idx = result
  return responses[final_idx]
```

### 4.2 FORM_GRAPH：相似度阈值 + 贡献度方向 + 断环 + 拓扑序

```text
FORM_GRAPH(embeddings r[0..N-1], contributions c[0..N-1], tau, k):
  S = COS_SIM_MATRIX(r)                      # pairwise cosine similarity
  S[i,i] = 0
  A = zeros(N,N)                             # A[n,m]=1 means m -> n

  for n in 0..N-1:
    candidates = { m != n | S[n,m] >= tau }
    if k != None and |candidates| > k:
      candidates = TOP_K_BY_SIM(candidates, S[n,*], k)
    for m in candidates:
      if c[m] > c[n]:
        A[n,m] = 1                           # m influences n

  while HAS_CYCLE(A):
    cycle_nodes = FIND_ONE_CYCLE(A)
    weakest = argmin_{v in cycle_nodes} c[v]
    succ = NEXT_NODE_IN_CYCLE(weakest)
    A[succ, weakest] = 0                     # remove edge weakest -> succ

  topo = KAHN_TOPO_SORT(A, tie_break=-c)     # prefer higher contribution when multiple choices
  return (A, S, c, topo)
```

---

## 5. 读代码时容易踩的点（实现细节对齐）

- **边的编码方向**：`adjacency[n,m]=1` 表示 **m → n**（m 影响 n），与很多常见 `A[u,v]=1` 的约定相反。
- **同轮优先**：在 Round>=1 时，构造 peer_responses 会优先使用同轮已生成的 `responses[pred]`，否则回退到 `prev_responses[pred]`。
- **无前驱节点也可能收到信息**：若节点无前驱，会尝试注入上一轮的 `best_prev`（当前最优回复）作为 peer，以减少“孤立节点”无参考改写。
- **early stop 的含义**：实现是“最不相似的一对也足够相似”，不是多数投票或 2/3 共识。
- **最终选择策略**：每轮都用“贡献度加权质心最近”更新 `final_idx`，最后直接返回该下标的回复。

---

## 6. SelfOrg 家族变体（本仓库：`selforg` / `soo`）

框架入口映射在 `methods/__init__.py::method2class`：

- `selforg` → `SelfOrg_Main`
- `soo` → `SOO_Main`

二者共用 `SelfOrg_Main.inference()` 的**主循环骨架**，差异通过两个可覆盖 hook 注入：

- `_estimate_contributions(embeddings)`：贡献度估计
- `_form_graph(embeddings, contributions, responses)`：通信图构建

### 6.1 `SOO_Main`：只改贡献度（Consensus Matrix + Perron-Frobenius）

文件：`methods/selforg/soo_main.py`、`methods/selforg/soo_contribution.py`

它把 `SelfOrg` 的 \( \psi_n=\cos(r_n, r_{\text{avg}})\) 换成：

1) \(S = V V^\top\)（实现里直接用 `embeddings @ embeddings.T`，假设嵌入已 L2-normalize）
2) \(A_{ij} = \exp(S_{ij}/\tau_{\text{consensus}})\)
3) 取 \(A\) 的主特征向量（最大特征值对应特征向量）的绝对值作为贡献度 \(c\)

对应实现：

- `estimate_contributions_soo(embeddings, tau_consensus)` 用 `np.linalg.eigh(A)` 取最后一列特征向量
- 配置项：`consensus_tau`（见 `config_main.yaml`，默认 0.3）

### 6.2 `SOO_pu_Main`：谱聚类代表点 + Perron 贡献度

`SOO_pu` 沿用 `SelfOrg_Main.inference()` 的主循环与 `form_graph()` 的通信拓扑构建，只替换贡献度估计：

- 构建谱图的邻接矩阵 `A`：与 `addition` 中谱聚类一致（余弦相似度经 `shifted_cosine` 或 `exp_tau` 映射，且无自环）。
- 贡献度 `c`：取对称 `A` 的最大特征值对应特征向量的分量绝对值（Perron 向量）。
- 代表点：在对称归一化拉普拉斯的谱嵌入 `U` 空间中做 `kmeans`，对每个簇选择“离其质心欧氏距离最近”的样本点作为代表点（用于分析/调试，不改变通信图的构建逻辑）。

### 6.3 `SOO_pu_pro_Main`：judge 风格 prompt + 代表点候选注入

`SOO_pu_pro` 在 `SOO_pu` 的基础上做两处关键改动，其他流程保持不变：

- prompt：将多智能体的 user prompt 改成“judge 风格”，参考 Dylan 的组织方式，要求智能体从候选答案中选择最真实/最准确的最终答案（而不是简单“参考前文自行回答”）。
- 候选集合规则：
  - 若某个节点在通信拓扑中 **没有 predecessors**：使用 `SOO_pu` 的“每个簇代表点”（全部簇）作为候选答案注入到该节点的 prompt 中。
  - 若某个节点 **有 predecessors**：候选只来自其 predecessors 节点输出（保持与 `SelfOrg_Main` 一致），不额外注入代表点候选。

