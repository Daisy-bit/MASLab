#!/usr/bin/env python3
"""
汇总 soo_math 在 trace_inference: true 下写入 jsonl 的 inference_trace，并绘制过程统计图。

用法:
  python scripts/visualize_soo_math_trace.py --jsonl results/MATH/qwen/soo_math_infer.jsonl --out-prefix /tmp/soo_trace

中文显示: 自动选用系统已安装的 Noto/Source Han/文泉驿 等字体；若无中文字体，可安装 fonts-noto-cjk，
或指定字体文件: --cjk-font /path/to/NotoSansSC-Regular.otf（环境变量 SOO_MATH_TRACE_FONT 同等）。

依赖 matplotlib（可选）:
  pip install matplotlib numpy
"""

import argparse
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

def load_traces(path: Path) -> Tuple[List[Dict[str, Any]], int]:
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "inference_trace" in obj:
                rows.append(obj)
            else:
                skipped += 1
    return rows, skipped


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """按 stage label 聚合正确率与 final_idx 频次。"""
    by_label: Dict[str, Dict[str, Any]] = {}
    exit_reasons: Dict[str, int] = defaultdict(int)
    event_kinds: Dict[str, int] = defaultdict(int)
    final_idx_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for row in rows:
        tr = row["inference_trace"]
        ex = tr.get("exit") or {}
        reason = str(ex.get("reason", "unknown"))
        exit_reasons[reason] += 1
        for ev in tr.get("events") or []:
            event_kinds[str(ev.get("kind", "?"))] += 1

        for st in tr.get("stages") or []:
            label = str(st.get("label", "?"))
            if label not in by_label:
                by_label[label] = {
                    "n": 0,
                    "any_agent_correct": 0,
                    "plurality_uniform_correct": 0,
                    "plurality_uniform_defined": 0,
                    "plurality_weighted_correct": 0,
                    "plurality_weighted_defined": 0,
                    "final_idx_correct": 0,
                    "final_idx_defined": 0,
                }
            b = by_label[label]
            b["n"] += 1
            if st.get("any_agent_correct") is True:
                b["any_agent_correct"] += 1
            if st.get("plurality_uniform_correct") is True:
                b["plurality_uniform_correct"] += 1
            if st.get("plurality_uniform_correct") is not None:
                b["plurality_uniform_defined"] += 1
            if st.get("plurality_weighted_correct") is True:
                b["plurality_weighted_correct"] += 1
            if st.get("plurality_weighted_correct") is not None:
                b["plurality_weighted_defined"] += 1
            if st.get("final_idx_correct") is True:
                b["final_idx_correct"] += 1
            if st.get("final_idx_correct") is not None:
                b["final_idx_defined"] += 1
            fi = st.get("final_idx")
            if isinstance(fi, int):
                final_idx_counts[label][fi] += 1

    repair_vs_regress = _repair_regression_stats(rows)

    by_xv = _aggregate_xverify_by_label(rows)

    out: Dict[str, Any] = {
        "by_label": by_label,
        "exit_reasons": dict(exit_reasons),
        "event_kinds": dict(event_kinds),
        "final_idx_counts": {k: dict(v) for k, v in final_idx_counts.items()},
        "num_traced_samples": len(rows),
        **repair_vs_regress,
    }
    if by_xv:
        out["by_label_xverify"] = by_xv
    return out


def _aggregate_xverify_by_label(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """由 trace_xverify（eval_trace_xverify.py 产出）汇总各阶段 xverify 正确率。"""
    by_label: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        tx = row.get("trace_xverify") or {}
        if tx.get("skipped") or tx.get("error") or not tx.get("stages"):
            continue
        for txs in tx["stages"]:
            label = str(txs.get("label") or "?")
            if label not in by_label:
                by_label[label] = {
                    "n": 0,
                    "any_agent_correct": 0,
                    "plurality_uniform_correct": 0,
                    "plurality_uniform_defined": 0,
                    "plurality_weighted_correct": 0,
                    "plurality_weighted_defined": 0,
                }
            b = by_label[label]
            b["n"] += 1
            pa = txs.get("per_agent") or []
            scores = [p.get("eval_score") for p in pa]
            if scores and all(s is not None for s in scores):
                if any(s == 1 for s in scores):
                    b["any_agent_correct"] += 1
            pu = txs.get("plurality_uniform")
            if pu is not None and pu.get("eval_score") is not None:
                b["plurality_uniform_defined"] += 1
                if pu.get("eval_score") == 1:
                    b["plurality_uniform_correct"] += 1
            pw = txs.get("plurality_weighted")
            if pw is not None and pw.get("eval_score") is not None:
                b["plurality_weighted_defined"] += 1
                if pw.get("eval_score") == 1:
                    b["plurality_weighted_correct"] += 1
    return by_label


def _final_ok_xverify(row: Dict[str, Any]) -> Optional[bool]:
    """
    仅依据评测结果判断最终答案是否正确（xverify 等写入的 eval 字段）。
    无有效评测结果时返回 None。
    """
    score = row.get("eval_score")
    if score == 1:
        return True
    if score == 0:
        return False
    content = row.get("eval_content")
    if content is None:
        return None
    if isinstance(content, str):
        if content.startswith("Eval Error") or content.startswith("Infer Error"):
            return None
        low = content.strip().lower()
        if low == "correct":
            return True
        if low == "incorrect":
            return False
    return None


def _initial_any_correct_for_repair(row: Dict[str, Any]) -> Optional[bool]:
    """首轮是否有任意智能体被判对：优先 trace_xverify，否则 legacy trace 的 any_agent_correct。"""
    tx = row.get("trace_xverify") or {}
    st0 = (tx.get("stages") or [None])[0]
    if isinstance(st0, dict) and st0.get("per_agent"):
        scores = [p.get("eval_score") for p in st0["per_agent"]]
        if scores and all(s is not None for s in scores):
            return any(s == 1 for s in scores)
    tr = row.get("inference_trace") or {}
    legacy = (tr.get("stages") or [None])[0]
    if isinstance(legacy, dict) and legacy.get("any_agent_correct") is not None:
        return bool(legacy["any_agent_correct"])
    return None


def _final_ok_for_repair(row: Dict[str, Any]) -> Optional[bool]:
    """最终输出是否对：优先 trace_xverify.final_output，否则 evaluate.py 写入的 eval_*。"""
    tx = row.get("trace_xverify") or {}
    fo = tx.get("final_output") or {}
    if fo.get("eval_score") == 1:
        return True
    if fo.get("eval_score") == 0:
        return False
    return _final_ok_xverify(row)


def _repair_regression_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    相对「初始并行」：纠错 / 带偏。
    - 首轮任意智能体是否对：优先 trace_xverify.stages[0].per_agent；否则旧 trace 的 any_agent_correct。
    - 最终是否对：优先 trace_xverify.final_output；否则行内 eval_score / eval_content。
    """
    n_ok = 0
    repair = 0
    regress = 0
    same_wrong = 0
    same_right = 0
    skipped_no_trace = 0
    skipped_no_initial = 0
    skipped_no_final = 0

    for row in rows:
        tr = row.get("inference_trace") or {}
        stages = tr.get("stages") or []
        if not stages:
            skipped_no_trace += 1
            continue
        init_any = _initial_any_correct_for_repair(row)
        if init_any is None:
            skipped_no_initial += 1
            continue
        fin = _final_ok_for_repair(row)
        if fin is None:
            skipped_no_final += 1
            continue
        n_ok += 1
        init_ok = bool(init_any)
        if init_ok and fin:
            same_right += 1
        elif init_ok and not fin:
            regress += 1
        elif not init_ok and fin:
            repair += 1
        else:
            same_wrong += 1

    return {
        "repair_regression": {
            "initial_signal": "trace_xverify.stages[0].per_agent if present else legacy any_agent_correct",
            "final_signal": "trace_xverify.final_output if present else row eval_score/eval_content",
            "defined_samples": n_ok,
            "skipped_no_trace": skipped_no_trace,
            "skipped_no_initial_judgement": skipped_no_initial,
            "skipped_no_final_judgement": skipped_no_final,
            "repair_initial_wrong_final_right": repair,
            "regress_initial_any_right_final_wrong": regress,
            "both_wrong": same_wrong,
            "both_right": same_right,
        }
    }


def _rate(num: int, den: int) -> float:
    return float(num) / den if den else 0.0


def print_report(summary: Dict[str, Any]) -> None:
    print(f"含 inference_trace 的样本数: {summary['num_traced_samples']}")
    rr = summary.get("repair_regression") or {}
    if rr.get("defined_samples"):
        print(
            "\n相对「初始并行」→ 最终输出（首轮/最终优先使用 trace_xverify；"
            "否则首轮用 legacy any_agent_correct，最终用 evaluate 的 eval_*）:\n"
            f"  可统计样本: {rr['defined_samples']}\n"
            f"  纠正(初无人对 → 终对): {rr['repair_initial_wrong_final_right']}\n"
            f"  退化(初有人对 → 终错): {rr['regress_initial_any_right_final_wrong']}\n"
            f"  始终终错: {rr['both_wrong']}\n"
            f"  初有人对且终对: {rr['both_right']}"
        )
    else:
        print(
            "\n提示: repair_regression 需要 (1) scripts/eval_trace_xverify.py 生成的 trace_xverify，"
            "或 (2) 旧 trace 含 any_agent_correct 且行内有 eval_*。"
            f" 未计入: skipped_no_trace={rr.get('skipped_no_trace', 0)}, "
            f"skipped_no_initial={rr.get('skipped_no_initial_judgement', 0)}, "
            f"skipped_no_final={rr.get('skipped_no_final_judgement', 0)}"
        )
    if summary.get("by_label_xverify"):
        print("\n各阶段（trace_xverify，xverify 评判）:")
        order_x = sorted(
            summary["by_label_xverify"].keys(),
            key=lambda x: (0 if x == "initial_parallel" else 1, x),
        )
        for label in order_x:
            b = summary["by_label_xverify"][label]
            n = b["n"]
            print(f"  [{label}] 样本数={n}")
            print(f"    任一智能体 xverify 正确: {_rate(b['any_agent_correct'], n):.3f}")
            ud = b["plurality_uniform_defined"]
            wd = b["plurality_weighted_defined"]
            if ud:
                print(f"    均匀 plurality xverify 正确: {_rate(b['plurality_uniform_correct'], ud):.3f} (分母={ud})")
            if wd:
                print(f"    加权 plurality xverify 正确: {_rate(b['plurality_weighted_correct'], wd):.3f} (分母={wd})")
    print("退出原因:", summary["exit_reasons"])
    if summary["event_kinds"]:
        print("过程事件:", summary["event_kinds"])
    print("\n各阶段（有标注 ground truth 时统计正确率）:")
    order = sorted(
        summary["by_label"].keys(),
        key=lambda x: (0 if x == "initial_parallel" else 1, x),
    )
    for label in order:
        b = summary["by_label"][label]
        n = b["n"]
        print(f"  [{label}] 样本数={n}")
        print(f"    任一智能体答案正确: {_rate(b['any_agent_correct'], n):.3f}")
        ud = b["plurality_uniform_defined"]
        wd = b["plurality_weighted_defined"]
        fd = b["final_idx_defined"]
        if ud:
            print(f"    均匀 plurality 正确: {_rate(b['plurality_uniform_correct'], ud):.3f} (分母={ud})")
        if wd:
            print(f"    加权 plurality 正确: {_rate(b['plurality_weighted_correct'], wd):.3f} (分母={wd})")
        if fd:
            print(f"    final_idx 对应答案正确: {_rate(b['final_idx_correct'], fd):.3f} (分母={fd})")
        fi = summary["final_idx_counts"].get(label)
        if fi:
            print(f"    final_idx 分布: {dict(sorted(fi.items()))}")


def _configure_matplotlib_cjk(font_file: Optional[str] = None) -> bool:
    """
    配置 matplotlib 使用中文字体，避免标题/图例显示为方框。
    优先顺序：--cjk-font 路径 → 环境变量 SOO_MATH_TRACE_FONT → 系统已安装常见 CJK 字体。
    返回 True 表示已启用中文字体；False 时绘图应使用英文标签以免方框。
    """
    import matplotlib
    from matplotlib import font_manager

    matplotlib.rcParams["axes.unicode_minus"] = False

    path = (font_file or os.environ.get("SOO_MATH_TRACE_FONT", "") or "").strip()
    if path and Path(path).is_file():
        try:
            fm = font_manager.fontManager
            if hasattr(fm, "addfont"):
                fm.addfont(path)
            prop = font_manager.FontProperties(fname=path)
            name = prop.get_name()
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            return True
        except Exception as ex:
            warnings.warn("无法从 SOO_MATH_TRACE_FONT / --cjk-font 加载字体: {}".format(ex))

    preferred = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "Noto Serif CJK SC",
        "Source Han Sans SC",
        "Source Han Sans CN",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "WenQuanYi Bitmap Song",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Hiragino Sans GB",
        "STHeiti",
        "Heiti SC",
        "Arial Unicode MS",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for fam in preferred:
        if fam in available:
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [fam, "DejaVu Sans", "sans-serif"]
            return True

    for f in font_manager.fontManager.ttflist:
        blob = " ".join(
            [f.name, getattr(f, "fname", "") or ""]
        ).lower()
        if any(
            k in blob
            for k in (
                "cjk",
                "noto sans sc",
                "noto serif c",
                "source han",
                "wqy",
                "wenquanyi",
                "simhei",
                "yahei",
                "pingfang",
            )
        ):
            matplotlib.rcParams["font.family"] = "sans-serif"
            matplotlib.rcParams["font.sans-serif"] = [f.name, "DejaVu Sans", "sans-serif"]
            return True

    warnings.warn(
        "未检测到可用的中文字体，图表将改用英文标签。"
        "若要显示中文: Linux 可 sudo apt install fonts-noto-cjk ；"
        "或下载 NotoSansSC-Regular.otf 后加参数 --cjk-font /path/to/NotoSansSC-Regular.otf"
    )
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "sans-serif"]
    return False


def plot_summary(summary: Dict[str, Any], out_prefix: Path, cjk_font_file: Optional[str] = None) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        raise SystemExit("需要 matplotlib 与 numpy 才能绘图: pip install matplotlib numpy") from e

    use_zh = _configure_matplotlib_cjk(font_file=cjk_font_file)

    def T(zh: str, en: str) -> str:
        return zh if use_zh else en

    labels = sorted(
        summary["by_label"].keys(),
        key=lambda x: (0 if x == "initial_parallel" else 1, x),
    )
    if not labels:
        print("无阶段数据，跳过作图")
        return

    x = np.arange(len(labels))
    w = 0.25
    any_corr = [_rate(summary["by_label"][lb]["any_agent_correct"], summary["by_label"][lb]["n"]) for lb in labels]
    pu = []
    pw = []
    pf = []
    for lb in labels:
        b = summary["by_label"][lb]
        pu.append(_rate(b["plurality_uniform_correct"], b["plurality_uniform_defined"] or b["n"]))
        pw.append(_rate(b["plurality_weighted_correct"], b["plurality_weighted_defined"] or b["n"]))
        pf.append(_rate(b["final_idx_correct"], b["final_idx_defined"] or b["n"]))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    ax.bar(x - w, any_corr, width=w, label=T("任一智能体正确", "Any agent correct"))
    ax.bar(x, pu, width=w, label=T("均匀 plurality 正确", "Plurality (uniform) correct"))
    ax.bar(x + w, pw, width=w, label=T("加权 plurality 正确", "Plurality (weighted) correct"))
    ax.bar(x + 2 * w, pf, width=w, label=T("final_idx 正确", "final_idx correct"))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(T("比率", "Rate"))
    ax.set_title(
        T(
            "soo_math 过程各阶段正确率（需 jsonl 中含 gt/answer）",
            "soo_math accuracy by stage (needs gt/answer in jsonl)",
        )
    )
    ax.legend()
    fig.tight_layout()
    p1 = out_prefix.parent / f"{out_prefix.name}_stage_accuracy.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"已保存: {p1}")

    # final_idx 堆叠条形图（每个 label 一行）
    fi_data = summary["final_idx_counts"]
    if any(fi_data.values()):
        all_idx = sorted({i for d in fi_data.values() for i in d})
        if all_idx:
            fig2, ax2 = plt.subplots(figsize=(max(8, len(labels) * 1.2), 4))
            bottom = np.zeros(len(labels))
            for j, idx in enumerate(all_idx):
                vals = np.array([fi_data.get(lb, {}).get(idx, 0) for lb in labels], dtype=float)
                ax2.bar(
                    labels,
                    vals,
                    bottom=bottom,
                    label=f"agent {idx}",
                    color=plt.cm.tab10(j % 10),
                )
                bottom += vals
            plt.setp(ax2.get_xticklabels(), rotation=25, ha="right")
            ax2.set_ylabel(T("计数", "Count"))
            ax2.set_title(T("各阶段 final_idx 分布", "final_idx distribution by stage"))
            ax2.legend(loc="upper right", fontsize=8)
            fig2.tight_layout()
            p2 = out_prefix.parent / f"{out_prefix.name}_final_idx_dist.png"
            fig2.savefig(p2, dpi=150)
            plt.close(fig2)
            print(f"已保存: {p2}")

    # 退出原因饼图
    if summary["exit_reasons"]:
        fig3, ax3 = plt.subplots(figsize=(6, 6))
        reasons = list(summary["exit_reasons"].keys())
        counts = [summary["exit_reasons"][r] for r in reasons]
        ax3.pie(counts, labels=reasons, autopct="%1.1f%%")
        ax3.set_title(T("最终答案来源（exit.reason）", "Exit reason (exit.reason)"))
        p3 = out_prefix.parent / f"{out_prefix.name}_exit_reasons.png"
        fig3.savefig(p3, dpi=150)
        plt.close(fig3)
        print(f"已保存: {p3}")


def main() -> None:
    ap = argparse.ArgumentParser(description="可视化 soo_math inference_trace 汇总")
    ap.add_argument("--jsonl", type=Path, required=True, help="推理结果 jsonl（含 inference_trace）")
    ap.add_argument(
        "--out-prefix",
        type=Path,
        default=Path("soo_math_trace_plot"),
        help="输出图片路径前缀（将追加 _*.png）",
    )
    ap.add_argument(
        "--cjk-font",
        type=str,
        default=None,
        metavar="PATH",
        help="含中文的 .ttf/.otf 字体文件路径（也可用环境变量 SOO_MATH_TRACE_FONT）",
    )
    ap.add_argument("--no-plot", action="store_true", help="只打印报告不绘图")
    args = ap.parse_args()

    rows, skipped = load_traces(args.jsonl)
    print(f"读取 {args.jsonl}: 含 trace 的条目 {len(rows)}, 跳过/无 trace {skipped}")
    if not rows:
        raise SystemExit("没有含 inference_trace 的记录；请在 config_soo_math.yaml 设置 trace_inference: true 后重跑。")

    summary = aggregate(rows)
    print_report(summary)
    summary_path = args.out_prefix.parent / f"{args.out_prefix.name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n已写入汇总 JSON: {summary_path}")

    if not args.no_plot:
        plot_summary(summary, args.out_prefix, cjk_font_file=args.cjk_font)


if __name__ == "__main__":
    main()
