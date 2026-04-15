# MacNet 算法说明与伪代码

基于论文 **《Scaling Large Language Model-based Multi-Agent Collaboration》(ICLR 2025, arXiv:2406.07155)** 以及本仓库中 `methods/macnet` 的实现整理。

---

## 一、论文中的 MacNet 核心思想

### 1.1 动机与问题

- 受**神经缩放律**启发：增加神经元能提升性能；论文探究**持续增加协作的 LLM 智能体**是否也能带来类似收益。
- 技术路线：用**有向无环图 (DAG)** 将智能体组织成**多智能体协作网络 (MacNet)**，按**拓扑顺序**进行交互推理，从而自主完成任务。

### 1.2 主要设计

| 概念 | 说明 |
|------|------|
| **拓扑 (Topology)** | 智能体之间的连接方式。支持链式、星型、树、全连接、MLP 式、随机等；论文发现**非常规/小世界拓扑**往往优于规则拓扑。 |
| **角色分工** | 可理解为「指导者 (Instructor/Reviewer)」与「执行者 (Assistant)」：前者对前序输出做审查与建议，后者根据建议生成/改进答案。 |
| **记忆控制** | 通过上下文管理（短时/长时记忆）控制信息量，避免过载。本实现中体现为「只把前驱节点的输出作为当前节点的输入」。 |
| **协作缩放律** | 性能随智能体数量呈**逻辑斯蒂增长**，且协作涌现早于传统神经涌现（约 24–25 个智能体即可趋于饱和）。 |

### 1.3 图上的计算方式

- 图是 **DAG**，保证无环、可分层执行。
- 存在**虚拟入口节点 (node_in)** 和**虚拟出口节点 (node_out)**，所有「无前驱」的节点接在 node_in 之后，所有「无后继」的节点接到 node_out 之前。
- 信息沿边流动：前驱 → 后继。每个节点在收到**所有前驱**的答案后，先做**审查/建议 (Instructor)**，再做**生成/改进 (Assistant)**；若前驱数 ≥ 某阈值，则对多路答案做**聚合 (Aggregation)**，否则直接采用单路结果。

---

## 二、本仓库 MacNet 实现要点（MacNet_Main）

### 2.1 配置与图结构

- **agent_num**：图中「实际智能体」数量（不含虚拟的 node_in/node_out）。
- **topology**：`chain` | `star` | `tree` | `net` | `mlp` | `random`，决定边集。
- **reverse**：是否将边反向（例如链 0→1→2 反向后变成 2→1→0）。
- **node_in_id / node_out_id**：默认 -1 与 -2，作为全局输入/输出的虚拟节点。
- **aggregate_unit_num**：前驱数量 ≥ 该值时才调用「多答案聚合」；否则直接用第一个前驱的交互结果。
- **aggregate_retry_limit**：聚合阶段失败时的重试次数（在 SRDD 版本中用到）。

图的边由 `generate_graph_topology()` 根据 `topology` 生成，例如：

- **chain**：0→1→2→…→(agent_num-1)
- **star**：0→1, 0→2, …, 0→(agent_num-1)
- **tree**：二叉树式 0→1,0→2, 1→3,1→4, …
- **net**：任意两点 u<v 连边
- **mlp**：按层划分节点，层间全连接
- **random**：在 (agent_num-1) 到 agent_num*(agent_num-1)/2 之间随机边数，再随机选边

生成边后，会加上「node_in → 所有原图入度为 0 的节点」和「所有原图出度为 0 的节点 → node_out」，并做**环检测**，保证最终为 DAG。

### 2.2 节点 (Node) 行为

每个节点维护：

- **predecessors / successors**：前驱/后继节点列表。
- **pre_answers**：来自各前驱的「交互后的答案」的字典 (前驱 id → 文本)。
- **generated_answer**：本节点对外输出（单前驱时即某条 pre_answer，多前驱且满足条件时为聚合结果）。
- **temperature**：按层深线性衰减，`1 - cur_depth / depth`，越深越低。
- **system_message**：系统提示。`type == 'default'` 时为固定句；否则从 SYSTEM_PROMPT 列表中随机取一个角色（如不同职业），实现「多视角」。

单次 **interact**（当前节点作为「后继」，对「某一前驱」的 previous_answer 做一次交互）：

1. **Instructor（审查）**  
   若有 previous_answer，则用 `INSTRUCTOR_PROMPT`（问题 + 待审查答案）调用 LLM，得到 **suggestions**（改进建议或 "<INFO>No revision needed"）。
2. **Assistant（执行）**  
   用 `ASSISTANT_PROMPT`（问题 + 原答案 + suggestions）再调 LLM，得到 **interacted_answer**，返回。

**aggregate_answers**（多前驱时）：

- 输入：`answer_dict` = 前驱 id → 该前驱与本节点交互后的答案。
- 使用 `CC_PROMPT` 加上「Node i 的 answer: …」列表，调用 LLM 做**信息综合**，返回一条聚合后的答案。

### 2.3 图执行顺序（分层执行）

1. **解析拓扑**  
   将配置里的拓扑字符串（如 `"0->1"`, `"1->2"`）转成节点与边，建立 `predecessors`/`successors`，并挂上 node_in、node_out，做环检测。

2. **分层与属性**  
   用「每次取当前入度为 0 的节点为一层、然后从图中删掉这一层」的方式给所有节点分层，得到 `depth`（图深度）。  
   对每层节点设置 `depth`、`temperature = 1 - cur_depth/depth`，以及 `system_message`（若 type 非 default 则随机角色）。

3. **按层执行**  
   - 若当前没有「入度为 0 的节点」则结束（此时只剩 node_out）。
   - 取当前**输入层** input_layer（入度为 0 的节点）。
   - 对每条边 (cur_node, next_node)：
     - 收集 cur_node 的所有前驱的 `generated_answer`，拼成 **pre_answer** 文本。
     - 调用 **next_node.interact(query, pre_answer)**，得到 interacted_answer。
     - 将结果记入 **next_node.pre_answers[cur_node.id]**。
   - 对每个 next_node：
     - 若 **len(pre_answers) == len(predecessors) 且 len(pre_answers) >= aggregate_unit_num**：  
       调用 **aggregate_answers(pre_answers)**，得到 **generated_answer**。
     - 否则：**generated_answer = 某一条 pre_answer**（实现里取第一条）。
   - 从图中删除本层：删掉已处理的边和 input_layer 中的节点。
   - 重复直到没有入度为 0 的节点。

4. **输出**  
   最终 **node_out.generated_answer** 即为整个 MacNet 的答案。

---

## 三、基于当前实现的伪代码

```text
===== 全局 =====
query: 用户问题
agent_num, topology, reverse, aggregate_unit_num, node_in_id=-1, node_out_id=-2

===== 1. 生成图拓扑 =====
function generate_graph_topology():
    edges = 根据 topology 生成边集  # chain/star/tree/net/mlp/random
    if reverse: edges = [(v,u) for (u,v) in edges]
    return [ "u->v" for (u,v) in edges ]

===== 2. 建图与分层 =====
function build_graph(topo):
    nodes = { node_in_id: Node(in), node_out_id: Node(out) }
    for 每条 "u->v" in topo:
        解析为 (from_ids, to_ids)
        for from_id in from_ids, to_id in to_ids:
            若未创建则 nodes[from_id] = Node(from_id), nodes[to_id] = Node(to_id)
            nodes[from_id].successors += nodes[to_id]
            nodes[to_id].predecessors += nodes[from_id]
    # 挂虚拟节点
    for n in 所有原图入度为0的节点 (且非 in/out):
        nodes[node_in_id].successors += n, n.predecessors += nodes[node_in_id]
    for n in 所有原图出度为0的节点 (且非 in/out):
        n.successors += nodes[node_out_id], nodes[node_out_id].predecessors += n
    assert 无环(circular_check)

    # 分层并设置 depth, temperature, system_message
    layers = []
    G_copy = 深拷贝(图)
    while G_copy 中仍有节点:
        input_nodes = 入度为0的节点
        layers.append(input_nodes)
        从 G_copy 中删除 input_nodes 及其出边
    depth = len(layers)
    for cur_depth, Layer in enumerate(layers):
        for node in Layer:
            node.depth = cur_depth
            node.temperature = 1 - cur_depth / depth
            node.system_message = 若 type=='default' 则 "You are helpful an assistant."
                            否则 从 SYSTEM_PROMPT 中随机选一条
    return nodes, depth

===== 3. 单节点：与一个前驱的交互 =====
function Node.interact(query, previous_answer):
    suggestions = "None."
    if previous_answer != '':
        suggestions = LLM(INSTRUCTOR_PROMPT(query, previous_answer), system=system_message, temp=temperature)
    interacted_answer = LLM(ASSISTANT_PROMPT(query, previous_answer, suggestions), system=system_message, temp=temperature)
    return interacted_answer

===== 4. 单节点：多前驱答案聚合 =====
function Node.aggregate_answers(answer_dict):  # answer_dict: 前驱id -> 答案文本
    if len(answer_dict) == 1: return answer_dict 中唯一的值
    prompt = CC_PROMPT + 拼接("[Node {id}'s answer:]\n{answer}\n\n" for id, answer in answer_dict)
    return LLM(prompt, system=system_message, temp=temperature)

===== 5. 图推理主循环 =====
function graph_inference(query, topo):
    nodes, depth = build_graph(topo)

    while true:
        input_layer = { n : n in nodes 且 len(n.predecessors)==0 }
        if input_layer 为空: break

        for cur_node in input_layer:
            for next_node in cur_node.successors:
                pre_answer = 拼接 "[Node {p.id}'s answer]\n{p.generated_answer}\n\n" for p in cur_node.predecessors
                interacted = next_node.interact(query, pre_answer)
                next_node.pre_answers[cur_node.id] = interacted

        for next_node in 本轮涉及的所有后继:
            if len(next_node.pre_answers) == len(next_node.predecessors) 且 len(next_node.pre_answers) >= aggregate_unit_num:
                next_node.generated_answer = next_node.aggregate_answers(next_node.pre_answers)
            else:
                next_node.generated_answer = next_node.pre_answers 中任一条（实现中取第一条）

        从图中删除 input_layer 中节点及其出边（更新 predecessors/successors，并从 nodes 中移除）

    return nodes[node_out_id].generated_answer

===== 6. 对外入口 =====
function inference(sample):
    query = sample['query']
    fix_random_seed(2025)
    topo = generate_graph_topology()
    response = graph_inference(query, topo)
    return { "response": response }
```

---

## 四、数据流小结

- **输入**：仅用户问题 `query`；node_in 在本实现中不直接参与 LLM 调用，其「输出」在逻辑上由「第一层节点的前驱」提供，即第一层节点用「空串或占位」作为 previous_answer 调用 interact。
- **第一层**：入度为 0 的节点只有 node_in，因此第一层节点的 pre_answer 来自 node_in；若 node_in 没有 generated_answer，则相当于用空字符串调用 interact，即「直接根据 query 生成」。
- **中间层**：每个节点先对每个前驱做一次 Instructor+Assistant 的 interact，再按前驱数是否 ≥ aggregate_unit_num 决定是聚合还是取单路。
- **输出**：最后一层节点把结果汇聚到 node_out，node_out.generated_answer 即为最终答案。

以上描述与伪代码与 `methods/macnet/macnet_main.py` 中的 **MacNet_Main** 行为一致；**MacNet_SRDD** 面向软件需求与设计（代码生成、编译与运行测试），交互与聚合接口不同，但「DAG + 分层执行 + 多前驱聚合」的整体框架一致。
