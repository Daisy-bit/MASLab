# DyLAN（dylan）算法实现逻辑（仓库实现版）

本文面向本仓库 `methods/dylan/` 的当前实现，解释 **DyLAN（Dynamic LLM Agent Network）** 在不同任务（Main / MMLU / MATH / HumanEval）下的实际流程、关键数据结构、共识/筛选机制，并给出与代码结构一致的伪代码。  
同时对照 `methods/macnet/MACNET_ALGORITHM.md`，说明 DyLAN 与 MacNet 的核心差异。

---

## 1. DyLAN vs. MacNet（对照理解）

参考 `methods/macnet/MACNET_ALGORITHM.md`：

- **MacNet**：在一个 **DAG** 上做拓扑序执行；每个节点对前驱输出执行 *Instructor→Assistant* 两段交互；当多前驱满足阈值时执行 **聚合（aggregation）**，最终汇入 `node_out`。
- **DyLAN（本仓库）**：不是 DAG 拓扑序推理，而是“**按轮（round）展开的分层网络**”，核心是：
  - 多个 agent 先各自作答；
  - 后续轮次让 agent **读取其他 agent（或 judge）的输出后改写**；
  - 每轮尝试 **2/3 共识早停（consensus early stop）**；
  - 如未收敛，使用 **listwise ranker** 选择 top2 的“agent 槽位”进入下一轮（动态激活）。

一句话：**MacNet 是 DAG 上的前向传播 + 聚合；DyLAN 是多轮辩论/互看改写 + 共识早停 + 排名筛选激活。**

---

## 2. DyLAN 家族版本一览（本仓库有哪些“dylan”）

入口由 `methods/__init__.py` 暴露：

- `dylan` → `DyLAN_Main`（开放问答）
- `dylan_mmlu` → `DyLAN_MMLU`（单选题）
- `dylan_math` → `DyLAN_MATH`（数学题，最终抽取答案一致性为主）
- `dylan_humaneval` → `DyLAN_HumanEval`（代码生成，含 judge 与单元测试）

对应实现文件：

- `methods/dylan/dylan_main.py`
- `methods/dylan/dylan_mmlu.py`
- `methods/dylan/dylan_math.py`
- `methods/dylan/dylan_humaneval.py`

配置文件：

- `methods/dylan/configs/config_main.yaml`
- `methods/dylan/configs/config_mmlu.yaml`
- `methods/dylan/configs/config_math.yaml`
- `methods/dylan/configs/config_humaneval.yaml`

---

## 3. 共同的“DyLAN 核心套路”（跨版本共性）

尽管各任务版本细节不同，但有几条共同骨架：

- **分轮（Rounds）**：至少 3 轮（代码多处 `assert num_rounds > 2`）。
- **多 agent 并行候选**：每轮通常有 `num_agents` 个候选输出。
- **互看改写**：从第二轮开始，prompt 中会拼接“其他 agent 的答案/推理/实现”，要求当前 agent **批判性吸收并修订**。
- **共识早停**：当已有足够多 agent 输出后，检查是否存在一个候选能获得 **> floor(2/3 * group_size)** 的“多数一致”；若是，直接返回。
- **listwise ranker 选 Top2**：在后续轮（通常从第 3 轮开始），会让一个 LLM 扮演 ranker，从上一轮候选中选择 top2；下一轮只激活对应的“agent 槽位”。
- **（Main/MMLU/HumanEval）边权**：让 LLM 在输出末尾附带 `[[...]]` 的分数数组，对“前驱答案”打 1~5 分；实现会解析这个数组并写回 **入边权重**，再归一化。

> 注：**边权在当前实现中主要是“记录/可视化式”的信号**，并不直接参与“聚合公式”生成最终答案；真正决定下一轮激活的是 listwise ranker（top2），以及共识早停逻辑。

---

## 4. DyLAN_Main（开放问答）实现细节

文件：`methods/dylan/dylan_main.py`

### 4.1 网络结构（nodes/edges）

- 节点总数：`num_agents * num_rounds`
- 第 0 轮：节点 id `0..N-1`
- 第 r 轮：节点 id `r*N .. (r+1)*N - 1`
- 边：对每个 `r>=1`，本轮每个节点与上一轮所有节点全连接（prev round → current node）。

### 4.2 单节点激活 `_activate_node(idx, question)`

1) **System prompt**：按 `role` 从 `_get_role_map()` 取描述。
2) **拼 user message**：
   - 若无前驱：直接 user=question。
   - 若有前驱：拼接其他 agent 的 reply，并要求：
     - 基于他人答案进行批判性改写；
     - 同时对每个前驱答案打 1~5 分；
     - 分数以 `[[1,5,2,...]]` 形式输出在末尾。
3) **调用 LLM**：`reply = call_llm(messages=contexts)`。
4) **解析权重数组**：从 `reply` 中用正则提取最后一个 `[[...]]`。
5) **写回入边权重并归一化**：
   - 1~5 → weight=score/5
   - >5 → weight=1
   - 解析失败/长度不匹配 → 置 0；若全 0 则均分。
6) **答案**：`answer = reply`（Main 里 `ans_parser` 是恒等）。

### 4.3 共识 early stop `_check_consensus(idxs, idx_mask)`

- `candidates = [nodes[idx].answer]`
- 使用 `cmp_res(x,y)` 判断“近似相同”（Main 里是简化 BLEU/词重合阈值）
- 若某候选的等价计数 `> floor(2/3 * len(idx_mask))`，则返回该候选。

### 4.4 listwise ranker（top2 动态激活）

从第 3 轮开始（`rid>=2`，且 `num_agents>3`）：

- 收集上一轮 replies；
- 打乱后构造 ranking prompt；
- 让 LLM 输出 `[1,2]` 形式的 top2；
- 映射回“agent 槽位”（0..N-1），下一轮只激活这些槽位对应节点。

---

## 5. DyLAN_MMLU（单选题）实现细节

文件：`methods/dylan/dylan_mmlu.py`，工具：`methods/dylan/utils_mmlu.py`

与 `DyLAN_Main` 的结构几乎一致，但关键差异是：

- **输出解析**：用 `parse_single_choice()` 从回复中抽取 `(A)/(B)/(C)/(D)`（或 `A)` 形式）。
- **共识**：选项字符串完全一致即可（`lambda x,y: x==y`）。
- **system prompt**：role 描述 + `SYSTEM_PROMPT_MMLU`（强调 debate 且题型为 single choice）。
- **ranking prompt**：要求选择 top2 “solutions”。

---

## 6. DyLAN_MATH（数学题）实现细节

文件：`methods/dylan/dylan_math.py`

该版本更像“脚本化多阶段辩论”，并未显式维护 nodes/edges 的边权网络。

### 6.1 关键机制：答案抽取与等价判定

为了做共识判断，DyLAN_MATH 会：

- 从每个 agent 的解题文本中抽取最终答案（支持 `The answer is ...`、`boxed{...}`、数字等）。
- 对字符串进行大量清洗与归一化（去空格、处理 `\\frac`、`\\sqrt`、单位、百分号等）。
- 用 `_is_equiv` 判断等价。

### 6.2 实际流程（四阶段）

1) **Round 1：独立作答 + 早停共识**
   - 每个 agent 在相同 system prompt（debate）下作答；
   - 当输出数量达到 \( \lfloor 2/3 \cdot N \rfloor \) 后尝试共识早停。
2) **Round 2：全体互看改写 + 早停**
   - 把所有 agent 的解答拼进 prompt；
   - 要求每个 agent 给出更新解答；
   - 再次尝试共识早停。
3) **Round 3：rank 选 Top2**
   - 让 LLM 从候选中选择“最好两个解答”，解析 `[i,j]`。
4) **Final：Top2 再辩一次**
   - 只对 top2 再做一次互看改写；
   - 返回抽取后的最终答案，并额外包 `\\boxed{...}`（用于评测协议对齐）。

---

## 7. DyLAN_HumanEval（代码生成）实现细节

文件：`methods/dylan/dylan_humaneval.py`

该版本把“辩论”升级为 **agent→judge→agent** 的多轮循环，并用 **单元测试通过数**作为最终选择标准。

### 7.1 网络结构

每轮包含：

- `num_agents` 个 agent（写代码实现）
-（除最后一轮外）`num_judges` 个 judge（Tester/Reflector/Debugger/QualityManager 等）

边连接：

- `agents(round r) -> judges(round r)` 全连接
- `judges(round r) -> agents(round r+1)` 全连接

### 7.2 Judge 的作用（重点）

- **Tester**：产出断言单测（过滤语法不合法 assert、去重、最多 10 条），并在后续用于筛选候选代码。
- **Reflector/Debugger/QualityManager**：分别给出反思、调试建议、代码审查，并要求打分 `[[...]]`（用于更新边权）。
- **Ranker**：在候选足够多时帮助选 top2（或简化为 `[1,2]`）。

### 7.3 共识与最终选择

共识（early stop）不是直接比文本，而是：

1) 先用 `check_function_result()` 过滤掉语法/执行失败的代码；
2) 将代码裁剪到与 `entry_point` 对齐的函数实现（`cut_def_question`）；
3) 用 sentence BLEU（`sacrebleu`）判断实现相似度是否足够高；
4) 仍采用 2/3 多数阈值。

若共识早停，或轮次结束后：

- 汇总 Tester 生成的单测；
- 对所有活跃 agent 候选逐一跑这些单测；
- 选择通过数最多的实现作为最终输出（并列时在 top 中随机挑）。

---

## 8. 与当前实现一致的伪代码

### 8.1 DyLAN_Main / DyLAN_MMLU（分轮网络 + 共识 + listwise top2）

```text
inputs:
  query
  N = num_agents
  R = num_rounds (R > 2)
  roles[0..N-1]
  activation = listwise (top2)

state:
  nodes[r*N + i] for r in [0..R-1], i in [0..N-1]
  edges: fully connect prev-round nodes -> next-round nodes (store weight)

function ACTIVATE(node_id, query):
  node = nodes[node_id]
  predecessors = active predecessor nodes via node.from_edges
  sys = ROLE_PROMPT(node.role) + (task-specific system prompt)
  user = BUILD_MESSAGE(query, predecessors.replies)
        # user要求：更新答案 + 对每个 predecessor 给 1..5 分，输出[[...]]
  reply = LLM(sys, user)
  answer = PARSE_ANSWER(reply)  # Main: reply; MMLU: parse (A/B/C/D)
  scores = PARSE_LAST_ARRAY(reply)  # from [[...]]
  WRITE_SCORES_TO_INCOMING_EDGE_WEIGHTS(node, scores); NORMALIZE()
  node.reply = reply; node.answer = answer; node.active = True

function CONSENSUS(active_node_ids, mask_size, cmp):
  answers = [nodes[id].answer for id in active_node_ids]
  (best, cnt) = MOST_FREQUENT(answers, cmp)
  return (cnt > floor(2/3 * mask_size), best)

main(query):
  reset edge weights; deactivate all nodes

  # round 0
  order = shuffle(0..N-1)
  active = []
  for k, id in enumerate(order):
    ACTIVATE(id, query)
    active.append(id)
    if k >= floor(2/3*N) and CONSENSUS(active, N, CMP).reached:
      return best

  # round 1
  order = shuffle(N..2N-1)
  active = []
  for k, id in enumerate(order):
    ACTIVATE(id, query)
    active.append(id)
    if k >= floor(2/3*N) and CONSENSUS(active, N, CMP).reached:
      return best

  idx_mask = [0..N-1]
  idxs = [N..2N-1]  # last round activated node ids

  for r in 2..R-1:
    if N > 3 and activation == listwise:
      replies = [nodes[id].reply for id in idxs]
      tops = LLM_LISTWISE_TOP2(shuffle(replies), query)  # returns 2 indices
      idx_mask = MAP_TOPS_TO_AGENT_SLOTS(tops)

    order = shuffle(r*N .. (r+1)*N - 1)
    idxs = []
    for pos, id in enumerate(order):
      if pos in idx_mask:
        ACTIVATE(id, query)
        idxs.append(id)
        if len(idxs) > floor(2/3 * len(idx_mask)) and CONSENSUS(idxs, len(idx_mask), CMP).reached:
          return best

  return MOST_FREQUENT([nodes[id].answer for id in idxs], CMP).best
```

### 8.2 DyLAN_MATH（脚本化多阶段辩论 + rank）

```text
question = examples(mode) + "Problem: {query}"
agent_contexts[i] = [system(debate_prompt), user(question)]

# Round 1: independent solve + early consensus
for i in 0..N-1:
  agent_contexts[i].append(assistant(LLM(agent_contexts[i])))
  if i >= floor(2/3*N) and CONSENSUS(extract_math_answer(agent_contexts[0..i])).reached:
    return MOST_FREQUENT_EXTRACTED_ANSWER

# Round 2: debate with all solutions + early consensus
msg = BUILD_DEBATE_MESSAGE(all agent solutions)
for i in 0..N-1:
  RESET_CONTEXT(agent_contexts[i], system, user(msg))
  agent_contexts[i].append(assistant(LLM(...)))
  if i >= floor(2/3*N) and CONSENSUS(...).reached:
    return MOST_FREQUENT_EXTRACTED_ANSWER

# Round 3: rank top2
rank_msg = BUILD_RANK_MESSAGE(all solutions)
tops = PARSE_TOP2(LLM(user(rank_msg)))
keep agent_contexts[tops]

# Final: debate top2 once more
msg = BUILD_DEBATE_MESSAGE(top2 solutions)
for each i in tops:
  RESET_CONTEXT(...); call LLM
return "\\boxed{" + EXTRACT_FINAL_ANSWER(top2) + "}"
```

### 8.3 DyLAN_HumanEval（agent→judge→agent + 单测驱动最终选择）

```text
prompt, entry_point = PARSE_HUMANEVAL_QUERY(query)
INIT_NODES_AND_EDGES(num_rounds, num_agents, num_judges)
unit_tests = []

# round 0: agents produce initial implementations
ACTIVATE_ALL_AGENTS_RANDOM_ORDER()

# round 0: judges provide tests/reflections/debug/reviews/ranks
for each judge:
  ACTIVATE_JUDGE()
  if judge.role == Tester: unit_tests += judge.unit_tests

# round 1: agents improve; check 2/3 consensus on runnable+similar code
ACTIVATE_ALL_AGENTS_RANDOM_ORDER()
if CONSENSUS_REACHED: return PICK_BY_TESTS(prompt, unit_tests, entry_point)

# later rounds:
for rid in 2..R-1:
  if enough agents: idx_mask = LISTWISE_TOP2(previous_round_answers)
  ACTIVATE_JUDGES_AND_COLLECT_MORE_TESTS()
  ACTIVATE_SELECTED_AGENTS(idx_mask)
  if CONSENSUS_REACHED: return PICK_BY_TESTS(...)

return PICK_BY_TESTS(prompt, unit_tests, entry_point)
```

---

## 9. 各版本差异总结表（“不同版本 dylan 的区别”）

| 版本 | 任务 | 结构 | 共识判据 | 选择/收敛机制 | 最终输出 |
|---|---|---|---|---|---|
| `DyLAN_Main` | 开放问答 | round 分层 + prev→next 全连接（仅 agent） | 文本近似相同（简化 BLEU/词重合）达到 2/3 | early stop + listwise top2 激活 | 多数/最频答案 |
| `DyLAN_MMLU` | 单选题 | 同 Main（仅 agent） | 选项完全一致达到 2/3 | early stop + listwise top2 激活 | (A/B/C/D) |
| `DyLAN_MATH` | 数学题 | 脚本化 4 阶段（不显式 nodes/edges） | 抽取并清洗后的最终答案等价达到 2/3 | early stop + rank top2 + top2 再辩 | `\\boxed{ans}` |
| `DyLAN_HumanEval` | 代码补全 | agent→judge→agent（含 Tester/Reflector/Debugger/QA/Ranker） | 可运行代码 + BLEU 相似达到 2/3 | early stop + listwise/Ranker + 单测驱动 | 通过单测最多的代码 |

---

## 10. 实用定位（代码入口与关键函数）

- Main：
  - `DyLAN_Main.inference()`
  - `_activate_node()`, `_check_consensus()`, `_listwise_ranker()`
- MMLU：
  - `DyLAN_MMLU.inference()`
  - `activate_node()`, `check_consensus()`, `listwise_ranker_2()`
  - `utils_mmlu.parse_single_choice()`
- MATH：
  - `DyLAN_MATH.inference()`
  - `_check_reach_consensus()`, `_extract_math_answer()`, `_strip_string()`
- HumanEval：
  - `DyLAN_HumanEval.inference()`
  - `activate_llm_node()`, `activate_judge_node()`
  - `check_consensus()`, `all_tests_and_get_final_result()`

