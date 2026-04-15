# h_swarm 重构合并至 MASLab 的设计说明

本文档说明将 `h_swarm/`（PSO 拓扑搜索）及其多目标扩展（`h_swarm/multi_objective_topology/`）重构合并至 MASLab 统一框架的整体思路、关键设计决策与实现细节。

---

## 1. 合并动机

MASLab 已集成 20+ 种 LLM 多智能体方法（CoT、SelfOrg、DyLAN、MAV 等），均通过统一的 `MAS` 基类、`inference.py` 入口和 `evaluate.py` 评估管线运行。h_swarm 原本作为独立项目存在于 `h_swarm/` 目录，拥有**独立的推理管线、评估逻辑和模型加载方式**，无法与 MASLab 其他方法进行公平对比。

合并的唯一目的：**让 h_swarm（单目标 PSO）和多目标 NSGA-II 拓扑优化方法能在同一管线下与 MASLab 已有方法直接对比**。

核心对比点：给定一个查询，各方法产出的回答质量如何？

---

## 2. 原始 h_swarm 与 MASLab 的关键差异

在设计合并方案前，需要理解两者在架构上的根本差异：

| 维度 | MASLab 方法 | 原始 h_swarm |
|------|------------|-------------|
| 基类 | 继承 `MAS`，实现 `inference(sample)` | 无基类，独立脚本 |
| LLM 调用 | `call_llm()`，OpenAI 兼容 API | 本地 `transformers` 加载 / vLLM API |
| 推理粒度 | 单样本（one sample per call） | 批量（batch of prompts） |
| 拓扑优化 | 无（固定拓扑） | PSO / NSGA-II 搜索最优拓扑 |
| 模型优化 | 无 | LoRA 合并（可选） |
| 评估管线 | `evaluate.py` + xverify | 自带多种评估（多选题/精确匹配/Gemini） |
| 数据格式 | `{"query": ..., "gt": ...}` | `{"question": ..., "answer": ..., "choices": ...}` |

**关键决策**：以 MASLab 的架构为基准进行适配，而非反向。

---

## 3. 整体设计思路

### 3.1 分离"搜索"与"推理"两个阶段

h_swarm 的工作流程天然分为两阶段：

```
阶段 1（搜索/优化）：在验证集上运行 PSO / NSGA-II → 得到最优拓扑
阶段 2（推理）：用最优拓扑对测试样本逐条推理 → 产出回答
```

MASLab 已有的 `optimizing(val_data)` 接口（用于 GPTSwarm、ADAS 等需要验证集的方法）天然对应阶段 1；`inference(sample)` 对应阶段 2。这使得 h_swarm 能无缝融入 MASLab 的 `inference.py` 主流程：

```python
# inference.py 中的已有逻辑（无需修改）
if args.require_val:
    mas = MAS_METHOD(general_config)
    mas.optimizing(val_dataset)      # ← 阶段 1：PSO / NSGA-II
# ...
mas.inference(sample)                # ← 阶段 2：图推理
```

### 3.2 统一 LLM 调用方式

原始 h_swarm 的 `evaluate.py::graph_generate()` 为每个图节点加载本地模型（或调用 vLLM），核心逻辑是按拓扑序在 DAG 上逐节点推理。合并后改用 `MAS.call_llm()` 替代本地模型加载，保证所有方法使用同一 LLM 后端，确保对比公平。

**改造前**（原始 h_swarm）：
```python
model = AutoModelForCausalLM.from_pretrained(model_path_list[assignment[node]], ...)
outputs = batch_generate(model, tokenizer, prompts, gpu_id, ...)
```

**改造后**（MASLab 集成版）：
```python
response = self.call_llm(prompt=prompt)
```

### 3.3 仅保留拓扑优化，去掉模型权重优化

原始 h_swarm 可同时优化**拓扑结构**（邻接矩阵）和**模型权重**（LoRA 合并）。但在 MASLab 框架中，LLM 通过 API 调用、无法修改权重，因此仅保留拓扑优化（对应原始代码的 `optimize_topology_only=1` 模式）。这也是更公平的对比方式——所有方法使用同一个冻结模型，差异仅在于多智能体的**协作拓扑**。

### 3.4 批量推理 → 单样本推理

原始 `graph_generate()` 一次处理整批 prompts，MASLab 的 `inference(sample)` 每次处理单个样本。改造时：

- 将批量循环提取到外层（由 `inference.py` 的 `ThreadPoolExecutor` 负责并行）
- 单次 `inference()` 内只对一个 query 在 DAG 上完成多节点推理

### 3.5 DAG 解码缓存

原始代码每次评估都重新执行 `graph_decode()`；在 MASLab 中 `inference()` 被逐样本调用，若不缓存则同一拓扑会被重复解码 \(|\text{dataset}|\) 次。因此引入 `_cached_dag` 机制：

- 邻接矩阵变更时（`_set_adjacency_matrix()`）自动清除缓存
- `inference()` 从缓存读取已解码的 DAG，避免冗余计算
- PSO 评估时（`_evaluate_topology()`）先解码一次、设入缓存，再遍历验证样本

---

## 4. 文件结构与模块映射

```
methods/h_swarm/
├── __init__.py                    # 导出 HSwarm_Main, HSwarm_MultiObj_Main
├── h_swarm_main.py                # 单目标 PSO 拓扑搜索 + 图推理（← search.py + evaluate.py）
├── h_swarm_multiobj_main.py       # 多目标 NSGA-II 拓扑搜索 + 图推理（← search_multiobj.py）
├── graph_utils.py                 # 连续邻接矩阵 → DAG 解码（← graph_decode.py）
├── graph_utils_multiobj.py        # (d, r) 编码 → DAG 解码 + 修复（← graph_decode_multiobj.py）
├── individual.py                  # 混合个体类（← individual_hybrid.py）
├── nsga2_utils.py                 # NSGA-II 算子：非支配排序/拥挤距离/PMX/Swap（← nsga2_hybrid.py）
├── pareto_archive.py              # Pareto 前沿归档（← pareto_archive.py）
├── dual_archive.py                # 收敛性/多样性双档案（← dual_archive.py）
├── persistent_homology.py         # 持久同调拓扑距离（← persistent_homology.py，可选依赖 ripser）
└── configs/
    ├── config_main.yaml           # 单目标 PSO 配置
    └── config_multiobj.yaml       # 多目标 NSGA-II 配置
```

### 原始文件 → 集成文件映射

| 原始文件 | 集成文件 | 主要改动 |
|---------|---------|---------|
| `h_swarm/search.py` | `h_swarm_main.py` | PSO 循环重写为 `optimizing()` 方法；去掉模型优化、wandb、多进程模型加载 |
| `h_swarm/evaluate.py` | `h_swarm_main.py` | `graph_generate()` 重写为 `inference()`；`call_llm()` 替代本地模型 |
| `h_swarm/graph_decode.py` | `graph_utils.py` | 去掉 `torch`/`random` 全局导入；增加 softmax 零值保护 |
| `multi_objective_topology/search_multiobj.py` | `h_swarm_multiobj_main.py` | NSGA-II 循环重写为 `optimizing()`；评估改用 `call_llm()` |
| `multi_objective_topology/graph_decode_multiobj.py` | `graph_utils_multiobj.py` | 内部辅助函数改为 `_` 前缀私有命名 |
| `multi_objective_topology/individual_hybrid.py` | `individual.py` | 无实质改动 |
| `multi_objective_topology/nsga2_hybrid.py` | `nsga2_utils.py` | 改为相对导入 `from .individual import ...` |
| `multi_objective_topology/pareto_archive.py` | `pareto_archive.py` | 无实质改动 |
| `multi_objective_topology/dual_archive.py` | `dual_archive.py` | 去掉对 `persistent_homology` 的硬依赖；DA 用目标空间距离替代 |
| `multi_objective_topology/persistent_homology.py` | `persistent_homology.py` | 去掉 `from search import log_with_flush`；ripser 缺失时优雅降级 |

---

## 5. 核心算法保持一致

### 5.1 图推理（Graph-based Inference）

推理逻辑与原始 `h_swarm/evaluate.py::graph_generate()` **完全一致**：

```text
function GRAPH_INFERENCE(query, dag):
    topo_order = topological_sort(dag)
    active = get_active_nodes(dag)          // 反向 DFS 找有效路径节点
    outputs[0..N-1] = [None, ...]

    for node in topo_order:
        if node not in active: continue

        if in_degree(node) == 0:            // 起始节点
            prompt = FIRST_INSTRUCTION + "\nQuestion: " + query
        else:
            preds = { j | dag[j, node] == 1 }
            prev_text = concat("Previous response k: " + outputs[pred])
            if out_degree(node) == 0:       // 终止节点
                prompt = LAST_INSTRUCTION + "\n" + prev_text + "Question: " + query
            else:                           // 中间节点
                prompt = NON_LAST_INSTRUCTION + "\n" + prev_text + "Question: " + query

        outputs[node] = call_llm(prompt)

    return outputs[topo_order[-1]]
```

提示词模板、前驱查找方式、拼接格式均与原始代码**逐字符一致**（已通过字符串比对验证）。

### 5.2 PSO 拓扑搜索

速度-位置更新公式与原始 `search.py::graph_update()` 一致：

\[
\mathbf{v}_{i}^{t+1} = \frac{r_w \cdot w}{W} \cdot \mathbf{v}_{i}^{t} + \frac{r_p \cdot c_1}{W} \cdot (\mathbf{p}_{i} - \mathbf{x}_{i}^{t}) + \frac{r_s \cdot c_2}{W} \cdot (\mathbf{g} - \mathbf{x}_{i}^{t}) + \frac{r_b \cdot c_3}{W} \cdot (\mathbf{x}_{i}^{t} - \mathbf{w})
\]

\[
\mathbf{x}_{i}^{t+1} = \mathbf{x}_{i}^{t} + \eta \cdot \mathbf{v}_{i}^{t+1}
\]

其中 \(W = r_w w + r_p c_1 + r_s c_2 + r_b c_3\)（归一化），\(r_w, r_p, r_s, r_b \sim \mathcal{U}(0,1)\)（当 `weight_randomness=true`），\(\eta\) 为步长（可衰减）。

集成版增加了 \(W=0\) 时的除零保护（原始代码无此保护）。

### 5.3 NSGA-II 多目标搜索

遗传算子与原始 `nsga2_hybrid.py` 完全一致：

- **非支配排序**：标准快速非支配排序（O(MN²)）
- **拥挤距离**：逐目标排序、边界解赋无穷大
- **锦标赛选择**：优先从 rank=0 前沿中选取
- **d 向量交叉**：均匀交叉（Uniform Crossover）
- **d 向量变异**：位翻转（Bit-flip, p=1/D）
- **π 排列交叉**：部分匹配交叉（PMX）
- **π 排列变异**：交换变异（Swap, p=1/N）
- **环境选择**：NSGA-II 精英保留 + 去重

双目标定义：
- \(f_1 = -\text{accuracy}\)（最小化 → 最大化准确率）
- \(f_2 = |\text{edges}| + |\text{connected\_nodes}|\)（最小化 → 更稀疏的拓扑）

### 5.4 图解码

**连续矩阵解码**（`graph_utils.py`，用于单目标 PSO）：

```text
1. 选择终点：取出度倒数经 softmax 后的 top-p 采样
2. 迭代添加节点：每次选出度最高的剩余节点，连接到一个已有节点
3. 结果：有向无环图，有且仅有一个汇点
4. top_p=0 时为确定性解码
```

**(d, r) 编码解码**（`graph_utils_multiobj.py`，用于多目标 NSGA-II）：

```text
1. 从 r（排列π）生成排列矩阵 P
2. 将 d（二进制连接向量）映射为严格上三角矩阵 U
3. 物理空间邻接矩阵：A = P^T U P
4. 修复策略：检测弱连通分量，保留最大非平凡分量，其余断开为孤立点
5. 动态识别汇点：在有效子图中找出度为 0 且 Rank 最大的节点
```

---

## 6. 注册与使用

### 6.1 方法注册

在 `methods/__init__.py` 中新增：

```python
from .h_swarm import HSwarm_Main, HSwarm_MultiObj_Main

method2class = {
    ...
    "h_swarm": HSwarm_Main,
    "h_swarm_multiobj": HSwarm_MultiObj_Main,
}
```

### 6.2 使用方式

**直接推理**（使用随机拓扑或预优化拓扑）：

```bash
python inference.py --method_name h_swarm --test_dataset_name MATH --model_name qwen25-3b-instruct
```

**带验证集优化**（PSO / NSGA-II 先搜索最优拓扑再推理）：

```bash
python inference.py --method_name h_swarm --test_dataset_name MATH --model_name qwen25-3b-instruct --require_val
python inference.py --method_name h_swarm_multiobj --test_dataset_name MATH --model_name qwen25-3b-instruct --require_val
```

**使用预优化拓扑**：修改 `configs/config_main.yaml`：

```yaml
adjacency_matrix_path: /path/to/optimized/adjacency.npy
```

**对比实验**：

```bash
# 各方法在同一数据集、同一模型下推理
python inference.py --method_name vanilla   --test_dataset_name MATH
python inference.py --method_name cot       --test_dataset_name MATH
python inference.py --method_name selforg   --test_dataset_name MATH
python inference.py --method_name h_swarm   --test_dataset_name MATH --require_val

# 统一评估
python evaluate.py --eval_protocol xverify --tested_dataset_name MATH --tested_method_name h_swarm --tested_mas_model_name qwen25-3b-instruct
```

### 6.3 配置参数

**config_main.yaml**（单目标 PSO）：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `num_agents` | 5 | 拓扑图中的节点/智能体数量 |
| `graph_num` | 10 | PSO 粒子（候选拓扑）数量 |
| `max_iteration` | 50 | 最大迭代次数 |
| `patience` | 10 | 早停：连续无改进的最大轮数 |
| `inertia` | 0.4 | 惯性权重 \(w\) |
| `cognitive_coeff` | 0.3 | 认知系数 \(c_1\) |
| `social_coeff` | 0.3 | 社会系数 \(c_2\) |
| `repel_coeff` | 0.0 | 排斥系数 \(c_3\)（远离全局最差） |
| `step_length` | 1.0 | 位置更新步长 \(\eta\) |
| `weight_randomness` | true | 是否随机化 PSO 系数 |

**config_multiobj.yaml**（多目标 NSGA-II）：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `num_agents` | 5 | 节点数量 |
| `num_of_individuals` | 50 | 种群大小 |
| `num_of_generations` | 100 | 进化代数 |
| `tournament_prob` | 0.9 | 锦标赛选择概率 |
| `crossover_prob` | 0.9 | 交叉概率 |
| `mutation_prob` | null | 变异概率（null 时 d 向量用 1/D，π 用 1/N） |
| `ca_max_size` | 20 | 收敛性档案最大容量 |
| `da_max_size` | 20 | 多样性档案最大容量 |
| `pareto_selection` | "utility" | 推理时的 Pareto 解选择策略 |

---

## 7. 与原始代码相比的改进

1. **除零保护**：PSO 权重归一化增加 \(W=0\) 检查（原始代码无此保护）
2. **Softmax 零值处理**：`graph_utils.py` 的 `softmax()` 在总和为 0 时返回均匀分布（原始代码会产生 NaN）
3. **DAG 缓存**：避免同一拓扑在多样本评估时被重复解码
4. **离散矩阵支持**：加载预优化拓扑时自动启用 `use_discrete_matrix`，跳过不必要的解码
5. **优雅降级**：`persistent_homology.py` 在 ripser 未安装时不报错，仅使用目标空间距离
6. **双档案简化**：`dual_archive.py` 的多样性档案（DA）在无持久同调时使用贪心最远点采样替代

---

## 8. 未移植的部分（及原因）

| 原始功能 | 原因 |
|---------|------|
| LoRA 模型合并（`merge.py`） | MASLab 通过 API 调用 LLM，无法修改权重 |
| 多 GPU 模型并行加载 | 改用 API 调用，无需管理 GPU |
| wandb 实验追踪 | MASLab 有自己的结果存储（JSONL + results/） |
| 独立的评估管线（多选题解析等） | 使用 MASLab 统一的 xverify 评估 |
| `evaluate_graph.py` | 功能已整合到 `_evaluate_topology()` |
| `save_dag_to_file` / `load_dag_from_file` | 原始代码中也未被调用，属冗余工具函数 |
