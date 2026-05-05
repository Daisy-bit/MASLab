# `mad_vote_scc` vs `soo_centered_v3` — 组件级对比 + 详细伪代码

> 目的：把两个号称「在不同 baseline（mad_vote / selforg）上加同一组 SCC 组件」的实现摊平对比，看看每个组件在两边的具体逻辑是否真的一致。
>
> **结论先行**：**不一致，差异极大**。两边只是名字相同，几乎每个组件的具体规则、触发位置、控制变量都不同。详见末尾「不一致性总览」。

源文件：
- `methods/mad_vote_scc/mad_vote_scc_main.py` (`MAD_Vote_SCC_Main`)
- `methods/soo_centered_v3/soo_centered_v3_main.py` (`SOO_Centered_v3_Main`)
  - 继承自 `methods/soo_centered_v2/soo_centered_v2_main.py` (`SOO_Centered_v2_Main`)
  - 后者继承自 `methods/selforg/selforg_main.py` (`SelfOrg_Main`)

---

## 0. 共享的核心数学

两边都用同一套 spectral primitives：

```
embed agents:  e_i = SBERT.encode(answer_i)            # L2 normalised
similarity:    S_ij = <e_i, e_j>                        # cosine
center:        H = I_N − (1/N) 1 1^T
double-center: S_c = H S H
trace signal:  trace(S_c)         (越小 → 共识越强)
PC1 vector:    v_1  = eigenvector of largest eigvec of S_c
raw contrib:   u_i  = |v_1[i]|
softmax(t=0.1):  c_i = exp(u_i / 0.1) / Σ exp(u_k / 0.1)
```

**两边都用 `c` 做 contribution 权重**。但底下利用 `c` 的方式完全不同。

---

## 1. mad_vote_scc 详细伪代码

按代码原文标号（`mad_vote_scc_main.py` 行号）逐步展开。

```python
# ───────────────────────────────────────────────────────────────────────
# class MAD_Vote_SCC_Main(MAD_Vote_Main)
#   父类: methods/mad_vote/mad_vote_main.py — 标准 fully-connected MAD baseline
# ───────────────────────────────────────────────────────────────────────

CONFIG_KNOBS:
    agents_num                     = 5
    rounds_num                     = 3                  # 固定上限
    enable_triggering              ∈ {False, True}
    enable_routing                 ∈ {False, True}
    enable_weighted_aggregation    ∈ {False, True}
    top_k                          = 2
    variance_consensus_thr (τ)     = 0.05
    answer_consensus_min (K)       = 3
    emb_model                      = model/all-MiniLM-L6-v2
    initial_pool_dir               = optional cache dir
    initial_pool_filename_pattern  = "mad_vote_{dataset}_infer.jsonl"


def inference(sample):
    query, gt = sample.query, sample.gt
    gold      = extract_gold(gt, task_type)
    judge_cache = {None: (False, "no-extraction")}

    def grade_canonical(ext):
        if ext in judge_cache: return judge_cache[ext]
        v = self._judge_or_default(query, canonical_response_text(ext), gt)  # xverify
        judge_cache[ext] = v
        return v

    # ── Build per-agent contexts ─────────────────────────────────────
    agent_contexts = []
    for role in self.roles:                         # roles = AGENT_ROLES[:5]
        ctx = [{"role":"system",  "content": role["system_prompt"]},
               {"role":"user",    "content": initial_user_prompt(query, task_type)}]
        agent_contexts.append(ctx)


    # ╔═══════════════════════ ROUND 0 ═══════════════════════╗
    cached_init = initial_pool.get(query) if initial_pool else None

    if cached_init and len(cached_init) == agents_num:
        # ───── Path A: replay cached responses ─────
        initial_records = []
        for i, ctx in enumerate(agent_contexts):
            rec  = cached_init[i]
            ctx.append({"role":"assistant","content": rec.raw_response})
            ext  = rec.extracted_answer
            judge_cache[ext] = (rec.is_correct, rec.judge_status)   # warm cache
            initial_records.append({...rec..., "cached_initial": True})
    else:
        # ───── Path B: live LLM ─────
        initial_records = []
        for i, ctx in enumerate(agent_contexts):
            resp, n_p, n_c = call_llm(ctx)                         # round-0 LLM
            ctx.append({"role":"assistant","content": resp})
            ext = extract_answer(resp, task_type)
            ic, js = grade_canonical(ext)                          # xverify
            initial_records.append({...})

    initial_extracted = [r.extracted_answer for r in initial_records]
    initial_correct   = [r.is_correct        for r in initial_records]
    initial_oracle_coverage = any(initial_correct)


    # ── Spectral analysis (only if any SCC module is on) ──
    need_spectral = enable_triggering or enable_routing or enable_weighted_aggregation
    if need_spectral:
        trace_Sc, contributions, _ = _spectral_analysis(
            [r.raw_response for r in initial_records]
        )
    else:
        trace_Sc, contributions = 0.0, [1/N] * N


    # ── Round-0 vote (always) ──
    initial_vote, _, _ = _vote(initial_extracted, contributions)
    initial_vote_correct, _ = grade_canonical(initial_vote)


    # ── Trigger AFTER round 0 ──
    triggered, reason = _trigger_hit(initial_extracted, trace_Sc)
    early_exit_round  = 0 if triggered else None


    # ╔═══════════════════ ROUNDS 1..rounds_num ══════════════╗
    for r = 1..rounds_num:
        if early_exit_round is not None:
            ── pad histories with the LAST values, NO LLM call ──
            continue

        last_round_responses = [ctx[-1].content for ctx in agent_contexts]

        for i in range(agents_num):                          # ALL agents speak every round
            peer_ids = _select_peers(i, agents_num, contributions)
            peer_responses = [last_round_responses[j] for j in peer_ids]
            user_prompt    = debate_user_prompt(query, peer_responses, task_type)

            agent_contexts[i].append({"role":"user","content": user_prompt})
            resp, n_p, n_c = call_llm(agent_contexts[i])     # round r LLM
            agent_contexts[i].append({"role":"assistant","content": resp})
            this_round_records[i] = {...peer_ids, ext, ic, js...}

        extracted_r  = [rec.extracted_answer for rec in this_round_records]

        if need_spectral:
            trace_r, contributions_r, _ = _spectral_analysis(
                [rec.raw_response for rec in this_round_records])
        else:
            trace_r, contributions_r = 0.0, [1/N] * N

        # ── Round r vote ──
        vote_r, _, weights_r = _vote(extracted_r, contributions_r)
        vote_r_correct, _    = grade_canonical(vote_r)
        history.append(vote_r, ...)
        contributions = contributions_r

        # ── Trigger AFTER round r ──
        triggered, reason = _trigger_hit(extracted_r, trace_r)
        if triggered:
            early_exit_round = r              # ← future rounds will be padded


    # ╔═════════════════════ FINAL VOTE ═══════════════════════╗
    final_vote          = round_vote_history[-1]            # 最后一轮的投票
    final_vote_correct  = round_vote_correct_history[-1]
    final_oracle_coverage = any(round_correct_history[-1])
    bucket = "already_solved" if initial_vote_correct
             else "recoverable" if initial_oracle_coverage
             else "unrecoverable"

    return {"response": final_vote or "", "diagnostic": {...}}


# ─── helpers ─────────────────────────────────────────────────────────

def _trigger_hit(extracted, trace_Sc):
    if not enable_triggering:           return False, None
    if max_count(extracted) >= K:       return True, "answer_plurality"
    if trace_Sc < τ:                    return True, "spectral_trace"
    return False, None


def _select_peers(i, N, contributions):
    if not enable_routing:                            # ─── full mesh ───
        return [j for j in range(N) if j != i]
    others = [j for j in range(N) if j != i]
    others.sort(key=lambda j: (-contributions[j], j))   # 高 c 在前；同 c 按 idx
    return others[: top_k]                              # 每个 i 都拿 top-k 同样的人


def _vote(extracted, contributions):
    if enable_weighted_aggregation:
        winner, weights, counts = _weighted_plurality(extracted, contributions, task_type)
        return winner, counts, weights
    # ── A0/A1/A2/A5 path: vanilla equal-weight plurality ──
    winner, counts = plurality_vote(extracted, task_type)   # methods/mad_vote/extractor.py
    weights = {k: v / N for k,v in counts.items()}          # 占位 only
    return winner, counts, weights


def _weighted_plurality(extracted, contributions, task_type):
    weights, counts, first_pos, first_orig, order = {}, {}, {}, {}, []
    for i, ext in enumerate(extracted):
        key = _canonicalize(ext, task_type)               # mcq → upper-strip;
                                                          # math → _normalize_number_str
        if key not in weights:
            weights[key], counts[key] = 0.0, 0
            first_pos[key], first_orig[key] = i, ext
            order.append(key)
        weights[key] += contributions[i]
        counts[key]  += 1
    real = [k for k in order if k is not None]
    if not real: return None, weights, counts
    winner_key = max(real, key=lambda k: (weights[k], counts[k], -first_pos[k]))
    #                                       ↑          ↑          ↑
    #                                    第一优先   第二优先    第三优先
    return first_orig[winner_key], weights, counts


def _spectral_analysis(answers):
    embs   = SBERT.encode(answers)        # 全新调用，不复用全局 emb model
    S      = cos_sim_matrix(embs)
    S_c    = double_center(S)
    trace  = trace(S_c)
    eigvecs,_ = eigh(S_c)                 # last column = largest eigvec
    pc1     = abs(eigvecs[:, -1])
    contributions = softmax(pc1, t=0.1)
    return trace, contributions, pc1
```

### 关键属性 — mad_vote_scc

| 性质 | 行为 |
|---|---|
| **每轮所有 agent 都发言** | ✓（无论 routing 是否开启，所有 agent 都被发起 LLM 调用） |
| **routing 影响什么** | **只影响 prompt 的 peer 上下文**：决定每个 agent 在新一轮看到哪些其他 agent 的上一轮回复 |
| **trigger 触发后** | 跳过剩余所有轮，**不再做 LLM 调用**，histories 用最后值填充 |
| **每轮独立投票** | ✓ Round 0 / 1 / 2 / 3 都各自投一次，`round_vote_history` 完整保留 |
| **final = 最后一轮投票结果** | ✓（不是跨轮累计） |
| **mid-debate early-stop** | ✗ 没有「投票后看 cluster 大小再决定停」的逻辑（trigger 检查的是答案 plurality 数 ≥ K，不是聚合后的 winner cluster size，但效果相近） |
| **DAG 强制非环** | ✗ 没有 DAG-ify 步骤 |
| **拓扑序** | ✗ 所有 agent 同一轮并发轮询，无 leader 概念 |
| **diversity 注入** | ✗ |
| **graph reform** | ✗ topology 在 round 1..rounds_num 间用同一份 contributions 排序，**不重建** |
| **聚合成败的回退** | 全部 ext 都 None 时 vote = None，response 为空字符串 |

---

## 2. soo_centered_v3 详细伪代码

```python
# ───────────────────────────────────────────────────────────────────────
# class SOO_Centered_v3_Main(SOO_Centered_v2_Main → SelfOrg_Main → MAS)
#   baseline: selforg (DAG-based propagation, embedding centroid agg)
#   v2 加: 双中心化 trace 共识检测
#   v3 加: 符号 plurality + 早停 + diversity 注入 + 任务类型适配
# ───────────────────────────────────────────────────────────────────────

CONFIG_KNOBS:                              # configs/config_main.yaml
    num_agents                       = 5
    top_k                            = 2
    max_rounds                       = 2                # 比 SCC 少一轮
    sim_threshold                    = 0.75             # ★ pruning 边的阈值
    enforce_dag                      = True
    reform                           = True             # 每轮后是否重建 graph
    aggregate_mode                   = "weighted"       # "single"|"weighted"
    consensus_min_sim                = 0.95
    consensus_range_eps              = 0.1
    variance_consensus_thr (τ)       = 0.05
    enable_spectral_consensus        = True
    answer_consensus_min_initial (Kᵢ)= 3
    answer_consensus_min_round   (Kᵣ)= 3
    enable_answer_consensus          = True
    diversity_p (p)                  = 0.2
    enable_contribution_routing      = True             # ★ ablation flag
    enable_contribution_aggregation  = True             # ★ ablation flag
    include_math_few_shot            = True
    include_mcq_format_hint          = True
    force_task_type                  = None             # 否则 auto-detect
    emit_diagnostic                  = False            # 默认 OFF；与 mad_vote_scc 不同


def inference(sample):
    query, reference = sample.query, sample.get("reference")
    task_type = _detect_task_type(sample)               # math / mcq / open

    # ╔═══════════════════ ROUND 0 ═════════════════════╗
    init_answers = []
    prompt0      = _init_prompt(query, task_type)       # task-typed
    for i in range(N):
        sysp = role_map[roles[i]]
        ans  = call_llm(prompt0, system_prompt=sysp, temperature=temperature)
        init_answers.append(ans)

    contributions = _approx_shapley(init_answers, None) # PC1 softmax
                                                        # 副作用: self._last_spectral
                                                        #         = {trace, gap_ratio, lam1, lam2}


    # ╔═════════════ EARLY-EXIT GATE A: answer plurality ════════════╗
    canonical, size = _plurality(init_answers, contributions, task_type)
    if enable_answer_consensus and canonical and size >= Kᵢ:    # 整数计数 ≥ 阈值
        return {"response": _format_final(canonical, task_type)}     # ← 直接返回，不 debate

    # ╔═════════════ EARLY-EXIT GATE B: spectral trace ══════════════╗
    if enable_spectral_consensus and self._last_spectral is not None
           and self._last_spectral["trace"] < τ:
        return {"response": _aggregate_from(init_answers, contributions, task_type)}
        #                                              ↑ 退化到 v2 的 centroid 聚合
        #                                              ext 提取在 _plurality 里失败时也会走这里


    # ── Build initial DAG ──
    sims          = _pairwise_sims(init_answers)
    edges, edge_w = _build_diverse_graph(sims, contributions, N)


    # ╔═══════════════════ DEBATE PROPAGATION ═════════════════╗
    final_answers = _propagate_with_typed_prompts(
        query, init_answers, edges, rounds=max_rounds,
        contributions=contributions, task_type=task_type,
    )


    # ── Final contribution refresh & aggregation ──
    contributions = _approx_shapley(final_answers, reference)
    canonical, _  = _plurality(final_answers, contributions, task_type)
    if canonical:
        return {"response": _format_final(canonical, task_type)}

    # 全部 ext 抽取失败 / open-ended:
    if not enable_contribution_aggregation:
        return {"response": init_answers[0]}            # deterministic fallback
    return {"response": _aggregate_from(final_answers, contributions, task_type)}


# ─── DAG construction ───────────────────────────────────────────────────

def _build_diverse_graph(sims, contributions, N):
    if not enable_contribution_routing:                 # ── full mesh fallback ──
        edges = {(j,i) for i in range(N) for j in range(N) if j != i}
        edge_w = {e: 1.0 for e in edges}
        if enforce_dag:
            edges, edge_w = _dagify(edges, edge_w)      # 即使 full mesh 也 DAG 化
        return edges, edge_w

    helpful = []
    for i in range(N):
        scored = []
        for j in range(N):
            if j == i: continue
            base = sims[i][j]
            adj  = base * (1 + N * (contributions[j] - contributions[i]))
            scored.append((j, adj, base))

        # ★ pruning by sim_threshold
        scored = [p for p in scored if p[2] >= sim_threshold]    # 0.75
        scored.sort(key=lambda x: (x[1], contributions[x[0]]), reverse=True)
        keep = [j for (j,_,_) in scored[:top_k]]

        # ★ diversity injection
        if diversity_p > 0:
            all_others = [j for j in range(N) if j != i]
            swapped = []
            for j in keep:
                if rng.random() < diversity_p:
                    pool = [k for k in all_others if k != j and k not in swapped]
                    if pool:
                        swapped.append(rng.choice(pool))
                        continue
                swapped.append(j)
            keep = swapped

        helpful.append(sorted(set(keep)))

    edges, edge_w = set(), {}
    for i in range(N):
        for j in helpful[i]:
            edges.add((j, i))
            edge_w[(j,i)] = max(0, sims[i][j] * (1 + N*(contributions[j]-contributions[i])))

    if enforce_dag:
        edges, edge_w = _dagify(edges, edge_w)          # 强制无环
    return edges, edge_w


def _dagify(edges, edge_w):
    """Repeatedly DFS for cycles; each found cycle removes its weakest edge.
       Loops until DFS sees no cycle. From SelfOrg_Main:246."""
    while cycle_found(edges):
        cycle = first_cycle_edges(edges)
        weakest = argmin(edge_w[e] for e in cycle)
        edges.remove(weakest); edge_w.pop(weakest)
    return edges, edge_w


def _topo_order_by_contributions(edges, contributions):
    """Kahn's algorithm with priority: -contribution as primary, idx as secondary."""
    indeg = compute_indegrees()
    heap  = [(-contributions[i], i) for i in indeg if indeg[i] == 0]
    heapify(heap)
    order = []
    while heap:
        _, u = heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heappush(heap, (-contributions[v], v))
    return order if len(order)==N else []


# ─── Debate propagation ─────────────────────────────────────────────────

def _propagate_with_typed_prompts(query, init_answers, edges, rounds, contributions, task_type):
    current = list(init_answers)                        # 槽位状态
    adj_in  = {v: [u for (u,v') in edges if v'==v] ...}
    order   = _topo_order_by_contributions(edges, contributions) or list(range(N))

    for r in range(max(1, rounds)):
        # ── 一轮内按拓扑序 sequential 更新 ──
        for i in order:
            preds     = adj_in.get(i, [])
            is_leader = (i == order[0])

            if not preds and not is_leader:             # ★ 没人指向 i 且不是 leader：跳过
                continue                                #   current[i] 保留原值（可能是上一轮 / round 0）

            incoming     = [(pid, current[pid]) for pid in preds]
            own_response = current[i]
            sysp         = role_map[roles[i]]

            if is_leader and not preds:
                prompt = _update_prompt_leader_agent(query, own_response, task_type)
            else:
                prompt = _update_prompt(query, own_response, incoming, task_type)

            current[i] = call_llm(prompt, sysp, temperature)
            #            ↑ 注意：current[i] 在本轮内立刻被更新，
            #              拓扑序后面的 j 看 preds 时 current[j_pred] 可能已是 r 轮新值
            #              ─ 即"瀑布式"更新，不是同步快照。

        sims          = _pairwise_sims(current)
        contributions = _approx_shapley(current, None)

        # ╔═════════════ MID-DEBATE GATE A: answer plurality ═════════════╗
        canonical, size = _plurality(current, contributions, task_type)
        if enable_answer_consensus and canonical and size >= Kᵣ:
            return current                              # ← 提前返回（unfinished rounds 不跑）

        # ╔═════════════ MID-DEBATE GATE B: spectral trace ═══════════════╗
        if enable_spectral_consensus and self._last_spectral["trace"] < τ:
            return current                              # ← 提前返回

        # ╔═════════════ GRAPH REFORM ════════════════════════════════════╗
        if reform and (r+1) < rounds:
            edges, _ = _build_diverse_graph(sims, contributions, N)   # ★ 重建 DAG
            adj_in   = ...
            order    = _topo_order_by_contributions(edges, contributions) or list(range(N))

    return current


# ─── Aggregation ────────────────────────────────────────────────────────

def _plurality(answers, contributions, task_type):
    """Symbolic plurality with task-typed equivalence and contribution tiebreak."""
    if task_type == "open" or N == 0: return "", 0
    extracted = [_extract_answer(a, task_type) for a in answers]
    pairs     = [(i, x) for i, x in enumerate(extracted) if x]
    if not pairs: return "", 0

    eq = math_is_equiv if task_type=="math" else _mcq_is_equiv

    # group equivalent answers (transitive cluster)
    groups = []
    for i, x in pairs:
        for g in groups:
            if eq(x, g[0][1]):
                g.append((i, x)); break
        else:
            groups.append([(i, x)])

    def score(g):
        size = len(g)
        if not enable_contribution_aggregation:
            return (size, -groups.index(g))                      # ★ count first, position tiebreak
        return (size, sum(contributions[i] for i, _ in g))       # ★ count first, contribution tiebreak

    best = max(groups, key=score)
    canonical = best[0][1]
    if task_type == "math":
        canonical = strip_string(canonical)
    return canonical, len(best)                                  # 返回 cluster size 给 caller


def _aggregate_from(answers, contributions, task_type):
    """Centroid fallback: only used when:
        - spectral early-stop fires at round 0, OR
        - final _plurality found nothing (full extraction failure)."""
    if aggregate_mode == "single":
        return answers[argmax(contributions)]
    embs    = _embed_many(answers)
    weights = contributions or uniform
    centroid = _weighted_centroid(embs, weights)
    nearest  = argmax([cos(centroid, embs[i]) for i in range(N)])
    return answers[nearest]
```

### 关键属性 — soo_centered_v3

| 性质 | 行为 |
|---|---|
| **每轮所有 agent 都发言** | ✗ **只有 leader 和有 incoming 边的 agent 说话**；其他保留原值 |
| **routing 影响什么** | **改变拓扑结构和谁说话**：不仅是 prompt 内容，还决定了哪些 agent 在该轮被调用 LLM |
| **早停时机** | **每个 round 后**做两次检查（plurality + spectral）；提前 `return current` |
| **每轮独立"投票"** | ✗ — 投票仅在两处发生：round 0 plurality early-exit、final aggregation；中间几轮 _plurality 只用作早停判定，结果不入"vote history" |
| **final = `_plurality(final_answers)`** | ✓ 在最后整体重新做一次（**不是**用某一轮历史投票） |
| **mid-debate early-stop** | ✓ 见 `_propagate_with_typed_prompts` 末尾两处 |
| **DAG 强制非环** | ✓ `_dagify` 保证图无环 |
| **拓扑序** | ✓ `_topo_order_by_contributions`（contribution 高的先发言） |
| **瀑布式更新** | ✓ 同一轮内 `current[i]` 即时生效，下游 j 的 `preds[j]` 看到的 `current[u]` 可能已是 r 轮新值 |
| **diversity 注入** | ✓ `diversity_p` 概率换成随机 peer |
| **graph reform** | ✓ 每轮结束（除最后一轮）按新 sims/contribs 重建 |
| **sim_threshold pruning** | ✓ base sim 必须 ≥ 0.75 才进入 candidate set |
| **聚合成败的回退** | embedding centroid 加权中心 + nearest agent（_aggregate_from） |
| **任务类型适配** | ✓ math/mcq/open 各有独立 prompt 和 extraction 规则 |
| **task-typed extraction** | ✓ math: `extract_math_answer`；mcq: 正则 `(X)` 抓字母 |
| **format final 后处理** | ✓ math 输出 `"The answer is X. \boxed{X}"`；mcq 输出 `"The answer is (X)"` |

---

## 3. 直接回答你提的两个具体问题

### Q1: mad_vote_scc 的 router top-k DAG 和 soo_centered_v3 的一样吗？

**完全不一样。** 把两个 routing 实现并排放：

| 维度 | mad_vote_scc `_select_peers` | soo_centered_v3 `_build_diverse_graph` |
|---|---|---|
| **入度选择规则** | `i` 选 contribution 最高的 `top_k` 个 peer | `i` 选 `sim[i][j] · (1 + N·(c_j - c_i))` 最高的 `top_k` 个 peer，**结合相似度和 contribution 差**，并按 `(adj, c_j)` 排序 |
| **是否过滤低相似度边** | ✗ | ✓ 必须 `sim[i][j] ≥ 0.75`（`sim_threshold`） |
| **是否注入 diversity** | ✗ | ✓ 每条选出的边以 `diversity_p=0.2` 概率换成随机 peer |
| **是否 DAG 化（去环）** | ✗ 没有 `_dagify` 步骤 | ✓ 必须无环（cycle 内删最弱边直到无环） |
| **是否计算拓扑序** | ✗ —— 同一轮内所有 agent 并发使用 *上一轮快照* | ✓ contribution-priority 拓扑序，控制本轮 agent 发言顺序 |
| **每轮是否重建** | ✗ —— 用 round-0 的 contributions 一直到结束 | ✓ `reform=True` 时每轮（除最后）重建 DAG |
| **谁会说话** | 所有 N 个 agent 每轮都说话；routing 只决定 *看谁的话* | 只有 `leader ∪ {有 incoming 边的 i}` 说话；其他人 `current[i]` 不变 |
| **更新模式** | 同一轮内并发：所有 agent 看上一轮所有人的回复 | 同一轮内瀑布：拓扑序内后面的 agent 可看到本轮已更新的 predecessor |

→ 名字都叫 "top-k DAG"，但 mad_vote_scc 的"DAG"实际上**根本不是 DAG**（没有 dagify、没有 topo order、没有 leader）。它只是「每个 agent 接收的上下文里只放 top-k 个 peer」，本质是 **prompt-level filtering**，并非图结构控制。soo_centered_v3 的版本才是**真正意义上的 DAG-on-debate**。

### Q2: mad_vote_scc 有 mid-debate early-stop 吗？

**有，但实现方式不同。**

| 维度 | mad_vote_scc `_trigger_hit` | soo_centered_v3 mid-debate gate |
|---|---|---|
| **检查时机** | round 0 之后 + 每轮 r 结束后（在投票之后） | round 0 之后（GATE A/B）+ 每轮内 propagation 结束后 |
| **第一类信号** | `max_count(extracted) ≥ 3` （`answer_consensus_min`） | `_plurality(current).cluster_size ≥ 3` （`answer_consensus_min_round`） — **看 cluster size，可能合并不同字面但 task-equivalent 的答案** |
| **第二类信号** | `trace(S_c) < 0.05` | `_last_spectral["trace"] < 0.05` |
| **触发后的行为** | 设置 `early_exit_round`，**剩余轮 padding 不调 LLM**；vote / contributions / trace 全部沿用最后一次值 | **直接 `return current`**，剩余轮不跑；对外接口只有最后一轮的 `current[]` |
| **早停后是否仍记录每轮** | ✓ 历史用最后值填充到 `rounds_num` 长度（保证 schema 对齐） | ✗ 历史/diagnostic 只在 `emit_diagnostic=True` 路径（`_inference_with_diagnostic`）记录；默认 path 不写历史 |
| **对 plurality 的「相等」判定** | `_canonicalize`：mcq 转大写 strip；math 转 `_normalize_number_str` —— 字符串相等 | `math_is_equiv` / `_mcq_is_equiv` —— **数学上等价**（如 `1/2` 和 `0.5`） |

→ 两边都有 mid-debate 的早停逻辑，但**判等口径不同**：
- mad_vote_scc 的 plurality 是 "字符串规范化后相等"
- soo_centered_v3 的 plurality 是 "数学/任务等价"（`is_equiv` 来自 `soo_math/math_answer_utils.py`，能把 `\frac{1}{2}`、`0.5`、`1/2` 都归为同一组）

后者更宽容，cluster size 通常更大 → 早停更激进。**这意味着同样的 K=3 在两个方法里含义不同**：mad_vote_scc 要求 3 个 agent 输出同字面的 ext，soo_centered_v3 要求 3 个 agent 输出数学等价的 ext。

---

## 4. 不一致性总览

把所有差异凝练成一个表，以便审稿人/合作者一眼看清：

| 组件 | mad_vote_scc | soo_centered_v3 | 是否同概念 |
|---|---|---|---|
| **Baseline** | mad_vote (fully-connected, 3 rounds, equal-weight plurality) | selforg (DAG propagation, embedding centroid, 2 rounds) | 不同 |
| **Round-0 prompts** | role-only system prompt + dataset-agnostic user | task-typed (math few-shot / mcq format hint / open) | **不同** |
| **Round-0 early-exit (plurality K)** | trigger 在投票**后**检查 (`max_count(extracted)≥K`) | gate 在 `_plurality` cluster size ≥ Kᵢ；**走快速 return 前对 canonical 做格式化** | 相同思想，时机不同 |
| **Round-0 early-exit (spectral)** | `trace<τ` → 设置 early_exit | `trace<τ` → 走 `_aggregate_from` centroid 聚合并返回 | 不同（一个是停止 debate 但保留 vote-0，一个是切到 centroid 聚合） |
| **Routing top-k** | contribution 排序，prompt-level filter | sim+contribution 加权，sim_threshold pruning，diversity 注入，DAG-ify，topo order | **完全不同** |
| **每轮谁说话** | 全员每轮都说 | 只有 leader + 有 incoming 的 agent | **完全不同** |
| **更新模式** | 并发（snapshot of last round） | 瀑布（同一轮内 in-order） | **不同** |
| **Graph reform** | ✗ 不重建 | ✓ `reform=True` 默认每轮重建 | **不同** |
| **Mid-debate early-stop** | trigger 检查 `max_count≥K` ∨ `trace<τ`；命中后 padding | gate 检查 `_plurality.size≥Kᵣ` ∨ `trace<τ`；命中后 return | 相同思想，**判等口径不同** |
| **每轮投票** | ✓ 每轮独立投票，记录 `round_vote_history` | ✗ 中间不投票，只用 plurality 做早停判定 | **不同** |
| **Final 决策** | `round_vote_history[-1]`（最后一轮的投票） | `_plurality(final_answers)`（在 `current` 末态上重新跑一次 plurality） | 类似但口径不同 |
| **Aggregation: 是否加权** | A0/A1/A2/A5 等权 plurality；A3/A4 用 contribution 加权（**weight-first lexicographic**） | 永远 count-first lexicographic（contribution 仅 tiebreak） | **不同** |
| **Aggregation: 等价判定** | `_canonicalize` 字符串规范化 | `math_is_equiv` / `_mcq_is_equiv` 数学等价 | **不同** |
| **Aggregation: 失败 fallback** | 返回空字符串 | `_aggregate_from` centroid + nearest | **不同** |
| **Output 后处理** | 直接返回 winner ext | `_format_final` 拼成 `"The answer is X. \boxed{X}"` | **不同** |
| **Judge / xverify** | 强制开启，每个 ext 都判（缓存） | 默认关闭；只有 `emit_diagnostic=True` 才走 _inference_with_diagnostic | **不同** |
| **Diagnostic schema** | 每个 sample 一个完整 diagnostic dict（initial_responses, round_responses, vote histories, scc_modules, ...） | 默认 path 无 diagnostic；emit 路径单独走 _inference_with_diagnostic（schema 对齐 mad_vote） | **不同** |

---

## 5. 含义：「同一组 SCC 组件」其实是个误导

如果论文/汇报里说「我们在 mad_vote 和 selforg 上加了相同的三个 SCC 组件 (Triggering / Routing / Aggregation)」，从代码层面看这是**不准确**的：

1. **Triggering** —— 名字相同，判等口径不同（字符串 vs 数学等价），早停后行为不同（padding vs return）
2. **Routing** —— 名字相同，但 mad_vote_scc 是 prompt-filter，soo_centered_v3 是真正的 DAG（含 dagify、topo order、reform、diversity、sim threshold）
3. **Aggregation** —— 名字相同，但加权策略相反（mad_vote_scc 是 weight-first，soo_centered_v3 是 count-first）

**这意味着 ablation 表的横向对比要小心**：
- A4 (mad_vote_scc 全开) 和 v3 (soo_centered_v3 默认) 即使数值差不多，背后机制完全不同
- 在论文里如果想用 v3 的实验数据「印证」mad_vote_scc 的设计，需要先承认两套实现的差异，否则审稿人很容易抓到漏洞

## 6. 下一步建议

如果目标是「mad_vote_scc 是 mad_vote 上加 SCC，soo_centered_v3 是 selforg 上加 SCC，**两者组件实现一致**」，需要做的最小一致化工程：

1. **Routing**：mad_vote_scc 加上 `_dagify` + `_topo_order_by_contributions` + 至少把 `sim_threshold` pruning 也加上；或把 v3 的 routing 改成不 DAG-ify 的 prompt-filter
2. **Aggregation**：把 `_weighted_plurality` 的 lexicographic 顺序改成 `(counts, weights, ...)`（**count-first**），与 v3 一致
3. **判等口径**：把 mad_vote_scc 的 `_canonicalize` 替换成 `math_is_equiv` / `_mcq_is_equiv`
4. **Mid-debate signal**：mad_vote_scc 的 `_trigger_hit` 改成看 `_plurality.cluster_size`（而非 `max_count`），与 v3 一致
5. **Final decision**：mad_vote_scc 在最后再跑一次 plurality 而非取 `round_vote_history[-1]`，与 v3 一致

每一项都不大，但合起来才能让两个方法**真正使用同一组组件**。否则 ablation 不可比。
