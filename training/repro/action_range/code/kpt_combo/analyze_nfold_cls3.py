#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if float(den) > 0.0 else 0.0


def _to_int_matrix(x: Any) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"Confusion matrix must be 2D, got ndim={arr.ndim}")
    if arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Confusion matrix must be square, got shape={arr.shape}")
    if np.any(arr < 0):
        raise ValueError("Confusion matrix must be non-negative.")
    return arr


def _split_combo_name(name: str, strict: bool) -> Tuple[str, str]:
    s = str(name).strip()
    if "|" in s:
        parts = s.split("|")
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    if strict:
        raise ValueError(f"Invalid combo class name (expected 'tool|action'): {name!r}")
    return "unknown_tool", "unknown_action"


def _metrics_from_confusion(class_names: Sequence[str], cm: np.ndarray) -> Dict[str, Any]:
    if len(class_names) != int(cm.shape[0]):
        raise ValueError(
            f"class_names length mismatch: len(class_names)={len(class_names)} vs cm={cm.shape}"
        )
    n_classes = int(cm.shape[0])
    row_sum = cm.sum(axis=1)
    col_sum = cm.sum(axis=0)
    diag = np.diag(cm)
    total = int(cm.sum())
    correct = int(diag.sum())
    errors = int(total - correct)

    per_class: List[Dict[str, Any]] = []
    f1_vals: List[float] = []
    p_vals: List[float] = []
    r_vals: List[float] = []
    support_vals: List[int] = []

    for i in range(n_classes):
        tp = int(diag[i])
        fp = int(col_sum[i] - tp)
        fn = int(row_sum[i] - tp)
        tn = int(total - tp - fp - fn)
        support = int(row_sum[i])
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)

        fn_to = []
        if fn > 0:
            for j in np.argsort(-cm[i, :]):
                if int(j) == i:
                    continue
                c = int(cm[i, int(j)])
                if c <= 0:
                    continue
                fn_to.append(
                    {
                        "pred_class_id": int(j),
                        "pred_class_name": str(class_names[int(j)]),
                        "count": c,
                        "ratio_in_fn": _safe_div(c, fn),
                    }
                )
        fp_from = []
        if fp > 0:
            col = cm[:, i]
            for j in np.argsort(-col):
                if int(j) == i:
                    continue
                c = int(cm[int(j), i])
                if c <= 0:
                    continue
                fp_from.append(
                    {
                        "true_class_id": int(j),
                        "true_class_name": str(class_names[int(j)]),
                        "count": c,
                        "ratio_in_fp": _safe_div(c, fp),
                    }
                )

        per_class.append(
            {
                "class_id": int(i),
                "class_name": str(class_names[i]),
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_negative_to": fn_to,
                "false_positive_from": fp_from,
            }
        )
        f1_vals.append(float(f1))
        p_vals.append(float(precision))
        r_vals.append(float(recall))
        support_vals.append(int(support))

    macro_precision = float(np.mean(p_vals)) if p_vals else float("nan")
    macro_recall = float(np.mean(r_vals)) if r_vals else float("nan")
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
    weighted_f1 = (
        float(np.average(np.asarray(f1_vals, dtype=np.float64), weights=np.asarray(support_vals, dtype=np.float64)))
        if support_vals and sum(support_vals) > 0
        else float("nan")
    )

    mistakes: List[Dict[str, Any]] = []
    for i in range(n_classes):
        true_total = int(row_sum[i])
        if true_total <= 0:
            continue
        for j in range(n_classes):
            if i == j:
                continue
            c = int(cm[i, j])
            if c <= 0:
                continue
            mistakes.append(
                {
                    "true_class_id": int(i),
                    "true_class_name": str(class_names[i]),
                    "pred_class_id": int(j),
                    "pred_class_name": str(class_names[j]),
                    "count": c,
                    "ratio_in_true_class": _safe_div(c, true_total),
                }
            )
    mistakes.sort(
        key=lambda x: (-int(x["count"]), -float(x["ratio_in_true_class"]), x["true_class_name"], x["pred_class_name"])
    )

    return {
        "num_classes": int(n_classes),
        "class_names": [str(x) for x in class_names],
        "total_samples": total,
        "correct_samples": correct,
        "error_samples": errors,
        "accuracy": _safe_div(correct, total),
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix_true_rows_pred_cols": cm.astype(np.int64).tolist(),
        "per_class": per_class,
        "mistake_pairs": mistakes,
    }


def _validate_source_combo_fpfn(source: Dict[str, Any], computed: Dict[str, Any], strict: bool, debug: bool) -> None:
    checks = [
        ("total_samples", int(source.get("total_samples", -1)), int(computed["total_samples"])),
        ("correct_samples", int(source.get("correct_samples", -1)), int(computed["correct_samples"])),
        ("error_samples", int(source.get("error_samples", -1)), int(computed["error_samples"])),
    ]
    for key, src, cur in checks:
        if src >= 0 and src != cur:
            msg = f"source mismatch for {key}: src={src} vs computed={cur}"
            if strict:
                raise ValueError(msg)
            if debug:
                print(f"[warn] {msg}")

    float_checks = [
        ("accuracy", float(source.get("accuracy", float("nan"))), float(computed["accuracy"])),
        ("macro_f1", float(source.get("macro_f1", float("nan"))), float(computed["macro_f1"])),
        ("weighted_f1", float(source.get("weighted_f1", float("nan"))), float(computed["weighted_f1"])),
    ]
    for key, src, cur in float_checks:
        if math.isfinite(src) and abs(src - cur) > 1e-6:
            msg = f"source mismatch for {key}: src={src:.8f} vs computed={cur:.8f}"
            if strict:
                raise ValueError(msg)
            if debug:
                print(f"[warn] {msg}")


def _build_action_tool_from_combo(cm_combo: np.ndarray, combo_names: Sequence[str], strict: bool) -> Dict[str, Any]:
    tool_names: List[str] = []
    action_names: List[str] = []
    tool_to_idx: Dict[str, int] = {}
    action_to_idx: Dict[str, int] = {}
    combo_to_tool: List[int] = []
    combo_to_action: List[int] = []

    for name in combo_names:
        tool, action = _split_combo_name(name, strict=strict)
        if tool not in tool_to_idx:
            tool_to_idx[tool] = len(tool_names)
            tool_names.append(tool)
        if action not in action_to_idx:
            action_to_idx[action] = len(action_names)
            action_names.append(action)
        combo_to_tool.append(int(tool_to_idx[tool]))
        combo_to_action.append(int(action_to_idx[action]))

    n_tool = len(tool_names)
    n_action = len(action_names)
    cm_tool = np.zeros((n_tool, n_tool), dtype=np.int64)
    cm_action = np.zeros((n_action, n_action), dtype=np.int64)

    c = int(cm_combo.shape[0])
    for i in range(c):
        ti = int(combo_to_tool[i])
        ai = int(combo_to_action[i])
        row = cm_combo[i, :]
        for j in range(c):
            v = int(row[j])
            if v <= 0:
                continue
            tj = int(combo_to_tool[j])
            aj = int(combo_to_action[j])
            cm_tool[ti, tj] += v
            cm_action[ai, aj] += v

    return {
        "tool_names": tool_names,
        "action_names": action_names,
        "combo_to_tool": combo_to_tool,
        "combo_to_action": combo_to_action,
        "cm_tool": cm_tool,
        "cm_action": cm_action,
    }


def _extract_epoch(path: Path) -> int:
    m = re.search(r"epoch_(\d+)\.json$", path.name)
    if not m:
        return -1
    return int(m.group(1))


def _pick_fold_combo_fpfn_file(
    fold_dir: Path,
    *,
    file_pattern: str,
    target_epoch: int,
    strict: bool,
) -> Optional[Path]:
    files = sorted((fold_dir / "checkpoints").glob(file_pattern))
    if not files:
        if strict:
            raise FileNotFoundError(f"No combo fpfn file under: {fold_dir / 'checkpoints'} pattern={file_pattern}")
        return None
    if target_epoch > 0:
        candidates = [p for p in files if _extract_epoch(p) == int(target_epoch)]
        if candidates:
            return candidates[0]
        if strict:
            raise FileNotFoundError(
                f"Target epoch file not found: fold={fold_dir.name} epoch={target_epoch} pattern={file_pattern}"
            )
    files.sort(key=lambda p: (_extract_epoch(p), p.name))
    return files[-1]


def _is_numeric_fold_dir_name(name: str) -> bool:
    return re.fullmatch(r"fold_\d+", str(name).strip()) is not None


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    clean = [float(v) for v in vals if math.isfinite(float(v))]
    if not clean:
        return float("nan"), float("nan")
    if len(clean) == 1:
        return float(clean[0]), 0.0
    return float(statistics.mean(clean)), float(statistics.pstdev(clean))


def _write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join([str(h) for h in header]) + "\n")
        for row in rows:
            f.write("\t".join([str(x) for x in row]) + "\n")


def _fmt(x: float, nd: int = 6) -> str:
    if not math.isfinite(float(x)):
        return "nan"
    return f"{float(x):.{int(nd)}f}"


@dataclass
class FoldTaskEval:
    task: str
    class_names: List[str]
    confusion: np.ndarray
    metrics: Dict[str, Any]


@dataclass
class FoldEval:
    fold: str
    fold_index: int
    epoch: int
    source_file: str
    combo: FoldTaskEval
    action: FoldTaskEval
    tool: FoldTaskEval


def _compute_fold_eval(combo_fpfn_path: Path, strict: bool, debug: bool) -> FoldEval:
    src = json.loads(combo_fpfn_path.read_text(encoding="utf-8"))
    class_names = [str(x) for x in src.get("class_names", [])]
    cm_combo = _to_int_matrix(src.get("confusion_matrix_true_rows_pred_cols", []))
    if not class_names:
        raise ValueError(f"Missing class_names in {combo_fpfn_path}")
    if len(class_names) != int(cm_combo.shape[0]):
        raise ValueError(
            f"class_names / confusion size mismatch in {combo_fpfn_path}: "
            f"{len(class_names)} vs {cm_combo.shape}"
        )

    combo_metrics = _metrics_from_confusion(class_names=class_names, cm=cm_combo)
    _validate_source_combo_fpfn(source=src, computed=combo_metrics, strict=strict, debug=debug)

    derived = _build_action_tool_from_combo(cm_combo=cm_combo, combo_names=class_names, strict=strict)
    action_metrics = _metrics_from_confusion(class_names=derived["action_names"], cm=derived["cm_action"])
    tool_metrics = _metrics_from_confusion(class_names=derived["tool_names"], cm=derived["cm_tool"])

    fold_name = combo_fpfn_path.parent.parent.name
    m = re.search(r"(\d+)$", fold_name)
    fold_idx = int(m.group(1)) if m else -1
    epoch = _extract_epoch(combo_fpfn_path)

    return FoldEval(
        fold=fold_name,
        fold_index=fold_idx,
        epoch=epoch,
        source_file=str(combo_fpfn_path),
        combo=FoldTaskEval(
            task="combo",
            class_names=[str(x) for x in combo_metrics["class_names"]],
            confusion=np.asarray(combo_metrics["confusion_matrix_true_rows_pred_cols"], dtype=np.int64),
            metrics=combo_metrics,
        ),
        action=FoldTaskEval(
            task="action",
            class_names=[str(x) for x in action_metrics["class_names"]],
            confusion=np.asarray(action_metrics["confusion_matrix_true_rows_pred_cols"], dtype=np.int64),
            metrics=action_metrics,
        ),
        tool=FoldTaskEval(
            task="tool",
            class_names=[str(x) for x in tool_metrics["class_names"]],
            confusion=np.asarray(tool_metrics["confusion_matrix_true_rows_pred_cols"], dtype=np.int64),
            metrics=tool_metrics,
        ),
    )


def _accumulate_global_confusion(
    task_fold_evals: Sequence[FoldTaskEval],
) -> Tuple[List[str], np.ndarray]:
    global_names: List[str] = []
    name_to_idx: Dict[str, int] = {}
    global_cm = np.zeros((0, 0), dtype=np.int64)

    for fe in task_fold_evals:
        local_names = [str(x) for x in fe.class_names]
        local_cm = fe.confusion
        if len(local_names) != int(local_cm.shape[0]):
            raise ValueError(f"Local confusion/name mismatch for task={fe.task}")

        for name in local_names:
            if name in name_to_idx:
                continue
            idx_new = len(global_names)
            global_names.append(name)
            name_to_idx[name] = idx_new
            new_cm = np.zeros((idx_new + 1, idx_new + 1), dtype=np.int64)
            if global_cm.size > 0:
                new_cm[: idx_new, : idx_new] = global_cm
            global_cm = new_cm

        idx_map = [int(name_to_idx[n]) for n in local_names]
        for i_local, i_global in enumerate(idx_map):
            row = local_cm[i_local, :]
            for j_local, j_global in enumerate(idx_map):
                v = int(row[j_local])
                if v > 0:
                    global_cm[i_global, j_global] += v

    return global_names, global_cm


def _summarize_per_class_across_folds(
    task: str,
    fold_evals: Sequence[FoldTaskEval],
) -> Dict[str, Any]:
    by_class: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    support_sum = Counter()
    folds_present = Counter()

    for fe in fold_evals:
        per_class = list(fe.metrics.get("per_class", []))
        seen_in_fold = set()
        for row in per_class:
            cname = str(row.get("class_name", "unknown"))
            support = int(row.get("support", 0))
            support_sum[cname] += support
            by_class[cname]["precision"].append(float(row.get("precision", 0.0)))
            by_class[cname]["recall"].append(float(row.get("recall", 0.0)))
            by_class[cname]["f1"].append(float(row.get("f1", 0.0)))
            by_class[cname]["support"].append(float(support))
            seen_in_fold.add(cname)
        for cname in seen_in_fold:
            folds_present[cname] += 1

    rows = []
    for cname in sorted(by_class.keys()):
        p_mean, p_std = _mean_std(by_class[cname]["precision"])
        r_mean, r_std = _mean_std(by_class[cname]["recall"])
        f_mean, f_std = _mean_std(by_class[cname]["f1"])
        s_mean, _ = _mean_std(by_class[cname]["support"])
        rows.append(
            {
                "task": task,
                "class_name": cname,
                "folds_present": int(folds_present[cname]),
                "support_sum": int(support_sum[cname]),
                "support_mean": float(s_mean),
                "precision_mean": float(p_mean),
                "precision_std": float(p_std),
                "recall_mean": float(r_mean),
                "recall_std": float(r_std),
                "f1_mean": float(f_mean),
                "f1_std": float(f_std),
                "f1_min": float(min(by_class[cname]["f1"])) if by_class[cname]["f1"] else float("nan"),
                "f1_max": float(max(by_class[cname]["f1"])) if by_class[cname]["f1"] else float("nan"),
            }
        )
    rows.sort(key=lambda x: (-x["f1_mean"], x["class_name"]))
    return {"task": task, "rows": rows}


def _write_markdown_report(
    path: Path,
    *,
    fold_metrics_rows: List[Dict[str, Any]],
    task_summary: Dict[str, Any],
    global_top_confusions: Dict[str, List[Dict[str, Any]]],
    top_k: int,
) -> None:
    lines: List[str] = []
    lines.append("# 10-Fold Classification Report (combo/action/tool)")
    lines.append("")
    lines.append(f"- fold_count: `{len(fold_metrics_rows)}`")
    lines.append("")

    for task in ("combo", "action", "tool"):
        summ = task_summary[task]
        lines.append(f"## {task.upper()} Summary")
        lines.append("")
        lines.append("| metric | mean | std | global |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(
            f"| accuracy | {_fmt(summ['accuracy_mean'], 4)} | {_fmt(summ['accuracy_std'], 4)} | {_fmt(summ['global_accuracy'], 4)} |"
        )
        lines.append(
            f"| macro_f1 | {_fmt(summ['macro_f1_mean'], 4)} | {_fmt(summ['macro_f1_std'], 4)} | {_fmt(summ['global_macro_f1'], 4)} |"
        )
        lines.append(
            f"| weighted_f1 | {_fmt(summ['weighted_f1_mean'], 4)} | {_fmt(summ['weighted_f1_std'], 4)} | {_fmt(summ['global_weighted_f1'], 4)} |"
        )
        lines.append("")
        lines.append(f"### Top confusion pairs (global top {int(top_k)})")
        lines.append("")
        lines.append("| mistake_pair | count | ratio_in_true |")
        lines.append("| --- | ---: | ---: |")
        for row in global_top_confusions[task][: int(top_k)]:
            lines.append(
                f"| {row['true_class_name']} -> {row['pred_class_name']} | {int(row['count'])} | {_fmt(row['ratio_in_true_class'], 4)} |"
            )
        lines.append("")

    lines.append("## Per-fold F1")
    lines.append("")
    lines.append("| fold | combo_macro_f1 | action_macro_f1 | tool_macro_f1 |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in sorted(fold_metrics_rows, key=lambda x: int(x["fold_index"])):
        lines.append(
            f"| {row['fold']} | {_fmt(row['combo_macro_f1'], 4)} | {_fmt(row['action_macro_f1'], 4)} | {_fmt(row['tool_macro_f1'], 4)} |"
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_analysis(args: argparse.Namespace) -> Dict[str, Any]:
    nfold_root = Path(args.nfold_root).resolve()
    if not nfold_root.is_dir():
        raise FileNotFoundError(f"--nfold-root not found: {nfold_root}")
    out_dir = Path(args.out_dir).resolve() if str(args.out_dir).strip() else nfold_root
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_dirs = [p for p in sorted(nfold_root.glob(str(args.fold_glob))) if p.is_dir()]
    if bool(args.only_numeric_folds):
        fold_dirs = [p for p in fold_dirs if _is_numeric_fold_dir_name(p.name)]
    if args.max_folds > 0:
        fold_dirs = fold_dirs[: int(args.max_folds)]
    if not fold_dirs:
        raise RuntimeError(f"No fold dirs matched: root={nfold_root} glob={args.fold_glob}")

    fold_results: List[FoldEval] = []
    for fd in fold_dirs:
        src_path = _pick_fold_combo_fpfn_file(
            fold_dir=fd,
            file_pattern=str(args.combo_fpfn_pattern),
            target_epoch=int(args.target_epoch),
            strict=bool(args.strict),
        )
        if src_path is None:
            continue
        fold_eval = _compute_fold_eval(combo_fpfn_path=src_path, strict=bool(args.strict), debug=bool(args.debug))
        fold_results.append(fold_eval)
        if args.debug:
            print(
                f"[debug] fold={fold_eval.fold} epoch={fold_eval.epoch} "
                f"combo_f1={fold_eval.combo.metrics['macro_f1']:.4f} "
                f"action_f1={fold_eval.action.metrics['macro_f1']:.4f} "
                f"tool_f1={fold_eval.tool.metrics['macro_f1']:.4f}"
            )

    if not fold_results:
        raise RuntimeError("No usable fold results loaded.")

    fold_results.sort(key=lambda x: (x.fold_index, x.fold))

    fold_metrics_rows: List[Dict[str, Any]] = []
    for fr in fold_results:
        fold_metrics_rows.append(
            {
                "fold": fr.fold,
                "fold_index": int(fr.fold_index),
                "epoch": int(fr.epoch),
                "total_samples": int(fr.combo.metrics["total_samples"]),
                "combo_accuracy": float(fr.combo.metrics["accuracy"]),
                "combo_macro_f1": float(fr.combo.metrics["macro_f1"]),
                "combo_weighted_f1": float(fr.combo.metrics["weighted_f1"]),
                "action_accuracy": float(fr.action.metrics["accuracy"]),
                "action_macro_f1": float(fr.action.metrics["macro_f1"]),
                "action_weighted_f1": float(fr.action.metrics["weighted_f1"]),
                "tool_accuracy": float(fr.tool.metrics["accuracy"]),
                "tool_macro_f1": float(fr.tool.metrics["macro_f1"]),
                "tool_weighted_f1": float(fr.tool.metrics["weighted_f1"]),
                "source_file": fr.source_file,
            }
        )

    task_to_fold_task_evals = {
        "combo": [fr.combo for fr in fold_results],
        "action": [fr.action for fr in fold_results],
        "tool": [fr.tool for fr in fold_results],
    }

    task_summary: Dict[str, Any] = {}
    global_top_confusions: Dict[str, List[Dict[str, Any]]] = {}
    task_per_class_summary: Dict[str, Dict[str, Any]] = {}

    for task, task_evals in task_to_fold_task_evals.items():
        acc_mean, acc_std = _mean_std([fe.metrics["accuracy"] for fe in task_evals])
        f1_mean, f1_std = _mean_std([fe.metrics["macro_f1"] for fe in task_evals])
        wf1_mean, wf1_std = _mean_std([fe.metrics["weighted_f1"] for fe in task_evals])

        global_names, global_cm = _accumulate_global_confusion(task_evals)
        global_metrics = _metrics_from_confusion(class_names=global_names, cm=global_cm)
        global_mistakes = list(global_metrics["mistake_pairs"])[: int(args.top_k)]
        global_top_confusions[task] = global_mistakes
        task_per_class_summary[task] = _summarize_per_class_across_folds(task=task, fold_evals=task_evals)

        task_summary[task] = {
            "fold_count": int(len(task_evals)),
            "accuracy_mean": float(acc_mean),
            "accuracy_std": float(acc_std),
            "macro_f1_mean": float(f1_mean),
            "macro_f1_std": float(f1_std),
            "weighted_f1_mean": float(wf1_mean),
            "weighted_f1_std": float(wf1_std),
            "global_accuracy": float(global_metrics["accuracy"]),
            "global_macro_f1": float(global_metrics["macro_f1"]),
            "global_weighted_f1": float(global_metrics["weighted_f1"]),
            "global_total_samples": int(global_metrics["total_samples"]),
            "global_class_names": list(global_metrics["class_names"]),
            "global_confusion_matrix_true_rows_pred_cols": list(global_metrics["confusion_matrix_true_rows_pred_cols"]),
            "global_top_confusions": global_mistakes,
        }

    # Fold metrics table
    fold_metrics_tsv = out_dir / f"{args.output_prefix}_fold_metrics.tsv"
    _write_tsv(
        fold_metrics_tsv,
        header=[
            "fold",
            "epoch",
            "total_samples",
            "combo_accuracy",
            "combo_macro_f1",
            "combo_weighted_f1",
            "action_accuracy",
            "action_macro_f1",
            "action_weighted_f1",
            "tool_accuracy",
            "tool_macro_f1",
            "tool_weighted_f1",
            "source_file",
        ],
        rows=[
            [
                r["fold"],
                int(r["epoch"]),
                int(r["total_samples"]),
                _fmt(r["combo_accuracy"], 6),
                _fmt(r["combo_macro_f1"], 6),
                _fmt(r["combo_weighted_f1"], 6),
                _fmt(r["action_accuracy"], 6),
                _fmt(r["action_macro_f1"], 6),
                _fmt(r["action_weighted_f1"], 6),
                _fmt(r["tool_accuracy"], 6),
                _fmt(r["tool_macro_f1"], 6),
                _fmt(r["tool_weighted_f1"], 6),
                r["source_file"],
            ]
            for r in fold_metrics_rows
        ],
    )

    # Top confusions by fold
    per_fold_conf_tsv = out_dir / f"{args.output_prefix}_top_confusions_by_fold.tsv"
    conf_rows = []
    for fr in fold_results:
        for task, fe in (("combo", fr.combo), ("action", fr.action), ("tool", fr.tool)):
            for row in list(fe.metrics["mistake_pairs"])[: int(args.top_k)]:
                conf_rows.append(
                    [
                        fr.fold,
                        int(fr.fold_index),
                        task,
                        row["true_class_name"],
                        row["pred_class_name"],
                        int(row["count"]),
                        _fmt(float(row["ratio_in_true_class"]), 6),
                    ]
                )
    _write_tsv(
        per_fold_conf_tsv,
        header=["fold", "fold_index", "task", "true_class", "pred_class", "count", "ratio_in_true_class"],
        rows=conf_rows,
    )

    # Global top confusions
    global_conf_tsv = out_dir / f"{args.output_prefix}_top_confusions_global.tsv"
    _write_tsv(
        global_conf_tsv,
        header=["task", "true_class", "pred_class", "count", "ratio_in_true_class"],
        rows=[
            [
                task,
                row["true_class_name"],
                row["pred_class_name"],
                int(row["count"]),
                _fmt(float(row["ratio_in_true_class"]), 6),
            ]
            for task in ("combo", "action", "tool")
            for row in global_top_confusions[task]
        ],
    )

    # Per-class summary files for each task
    per_class_paths: Dict[str, str] = {}
    for task in ("combo", "action", "tool"):
        rows = task_per_class_summary[task]["rows"]
        json_path = out_dir / f"{args.output_prefix}_{task}_f1_summary.json"
        tsv_path = out_dir / f"{args.output_prefix}_{task}_f1_summary.tsv"
        json_path.write_text(json.dumps(task_per_class_summary[task], indent=2), encoding="utf-8")
        _write_tsv(
            tsv_path,
            header=[
                "task",
                "class_name",
                "folds_present",
                "support_sum",
                "support_mean",
                "precision_mean",
                "precision_std",
                "recall_mean",
                "recall_std",
                "f1_mean",
                "f1_std",
                "f1_min",
                "f1_max",
            ],
            rows=[
                [
                    r["task"],
                    r["class_name"],
                    int(r["folds_present"]),
                    int(r["support_sum"]),
                    _fmt(r["support_mean"], 3),
                    _fmt(r["precision_mean"], 6),
                    _fmt(r["precision_std"], 6),
                    _fmt(r["recall_mean"], 6),
                    _fmt(r["recall_std"], 6),
                    _fmt(r["f1_mean"], 6),
                    _fmt(r["f1_std"], 6),
                    _fmt(r["f1_min"], 6),
                    _fmt(r["f1_max"], 6),
                ]
                for r in rows
            ],
        )
        per_class_paths[task] = str(tsv_path)

    report_md = out_dir / f"{args.output_prefix}_report.md"
    if args.write_markdown:
        _write_markdown_report(
            path=report_md,
            fold_metrics_rows=fold_metrics_rows,
            task_summary=task_summary,
            global_top_confusions=global_top_confusions,
            top_k=int(args.top_k),
        )

    summary = {
        "config": {
            "nfold_root": str(nfold_root),
            "out_dir": str(out_dir),
            "fold_glob": str(args.fold_glob),
            "combo_fpfn_pattern": str(args.combo_fpfn_pattern),
            "target_epoch": int(args.target_epoch),
            "top_k": int(args.top_k),
            "strict": bool(args.strict),
            "debug": bool(args.debug),
            "max_folds": int(args.max_folds),
        },
        "fold_count": int(len(fold_results)),
        "folds": fold_metrics_rows,
        "task_summary": task_summary,
        "output_files": {
            "fold_metrics_tsv": str(fold_metrics_tsv),
            "top_confusions_by_fold_tsv": str(per_fold_conf_tsv),
            "top_confusions_global_tsv": str(global_conf_tsv),
            "per_class_summary_tsv": per_class_paths,
            "report_md": str(report_md) if args.write_markdown else "",
        },
    }
    summary_json = out_dir / f"{args.output_prefix}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["output_files"]["summary_json"] = str(summary_json)
    return summary


def _write_fake_combo_fpfn(path: Path, cm: np.ndarray, class_names: List[str]) -> None:
    m = _metrics_from_confusion(class_names=class_names, cm=cm)
    payload = {
        "num_classes": int(len(class_names)),
        "class_ids": [int(i) for i in range(len(class_names))],
        "class_names": class_names,
        "combo_id_to_name": {str(i): class_names[i] for i in range(len(class_names))},
        "total_samples": int(m["total_samples"]),
        "correct_samples": int(m["correct_samples"]),
        "error_samples": int(m["error_samples"]),
        "accuracy": float(m["accuracy"]),
        "macro_precision": float(m["macro_precision"]),
        "macro_recall": float(m["macro_recall"]),
        "macro_f1": float(m["macro_f1"]),
        "weighted_f1": float(m["weighted_f1"]),
        "confusion_matrix_true_rows_pred_cols": m["confusion_matrix_true_rows_pred_cols"],
        "per_class": m["per_class"],
        "mistake_pairs": m["mistake_pairs"],
        "misclassified_samples": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_smoke_synthetic(top_k: int, debug: bool) -> None:
    with tempfile.TemporaryDirectory(prefix="nfold_cls3_smoke_") as td:
        root = Path(td) / "fake_nfold"
        class_names = [
            "hook|dissect",
            "grasper|grasp",
            "clipper|clip",
            "scissors|cut",
            "scissors|dissect",
            "grasper|dissect",
        ]
        cm1 = np.asarray(
            [
                [90, 0, 0, 1, 3, 6],
                [0, 60, 0, 0, 0, 10],
                [0, 1, 30, 5, 0, 4],
                [0, 0, 3, 18, 1, 8],
                [1, 1, 0, 6, 12, 10],
                [0, 8, 2, 3, 2, 65],
            ],
            dtype=np.int64,
        )
        cm2 = np.asarray(
            [
                [95, 0, 0, 0, 2, 3],
                [1, 58, 0, 0, 0, 9],
                [0, 1, 29, 6, 0, 4],
                [0, 0, 4, 16, 2, 9],
                [1, 1, 0, 5, 14, 9],
                [0, 7, 3, 4, 1, 66],
            ],
            dtype=np.int64,
        )
        _write_fake_combo_fpfn(root / "fold_01" / "checkpoints" / "val_combo_fpfn_epoch_100.json", cm1, class_names)
        _write_fake_combo_fpfn(root / "fold_02" / "checkpoints" / "val_combo_fpfn_epoch_100.json", cm2, class_names)

        ns = argparse.Namespace(
            nfold_root=str(root),
            out_dir=str(root),
            fold_glob="fold_*",
            combo_fpfn_pattern="val_combo_fpfn_epoch_*.json",
            target_epoch=100,
            top_k=int(top_k),
            strict=True,
            debug=bool(debug),
            max_folds=0,
            only_numeric_folds=True,
            output_prefix="smoke_cls3",
            write_markdown=True,
        )
        summary = _run_analysis(ns)
        out_json = Path(summary["output_files"]["summary_json"])
        if not out_json.is_file():
            raise RuntimeError("Smoke failed: summary json not found.")
        loaded = json.loads(out_json.read_text(encoding="utf-8"))
        if int(loaded.get("fold_count", 0)) != 2:
            raise RuntimeError("Smoke failed: fold_count mismatch.")
        for task in ("combo", "action", "tool"):
            if task not in loaded.get("task_summary", {}):
                raise RuntimeError(f"Smoke failed: missing task summary: {task}")
        print(f"[smoke] synthetic PASS: {out_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Analyze 10-fold classification from combo confusion json and derive "
            "combo/action/tool metrics (F1, confusion pairs)."
        )
    )
    p.add_argument("--nfold-root", type=str, default="", help="Root dir containing fold_XX subdirs.")
    p.add_argument("--out-dir", type=str, default="", help="Output dir. Default: --nfold-root")
    p.add_argument("--fold-glob", type=str, default="fold_*", help="Fold dir glob under --nfold-root.")
    p.add_argument(
        "--only-numeric-folds",
        dest="only_numeric_folds",
        action="store_true",
        help="Only keep dirs whose names match fold_\\d+ (default on).",
    )
    p.add_argument(
        "--no-only-numeric-folds",
        dest="only_numeric_folds",
        action="store_false",
        help="Disable numeric fold-name filtering.",
    )
    p.set_defaults(only_numeric_folds=True)
    p.add_argument(
        "--combo-fpfn-pattern",
        type=str,
        default="val_combo_fpfn_epoch_*.json",
        help="Glob under fold/checkpoints to locate combo fpfn files.",
    )
    p.add_argument("--target-epoch", type=int, default=0, help="If >0, require this epoch file when present.")
    p.add_argument("--top-k", type=int, default=20, help="Top-K confusion pairs to keep.")
    p.add_argument("--max-folds", type=int, default=0, help="For smoke/debug: only analyze first N folds if >0.")
    p.add_argument("--output-prefix", type=str, default="nfold_cls3", help="Prefix for output files.")
    p.add_argument("--write-markdown", action="store_true", help="Write markdown report file.")
    p.add_argument("--debug", action="store_true", help="Verbose debug logs.")
    p.add_argument("--strict", dest="strict", action="store_true", help="Fail on schema/metric mismatch.")
    p.add_argument("--no-strict", dest="strict", action="store_false", help="Warn on mismatch and continue.")
    p.set_defaults(strict=True)
    p.add_argument(
        "--smoke-synthetic",
        action="store_true",
        help="Run built-in synthetic smoke test (no project data required).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_synthetic:
        _run_smoke_synthetic(top_k=int(args.top_k), debug=bool(args.debug))
        return
    if not str(args.nfold_root).strip():
        raise ValueError("--nfold-root is required unless --smoke-synthetic is used.")
    if not str(args.out_dir).strip():
        args.out_dir = str(Path(args.nfold_root).resolve())
    summary = _run_analysis(args)
    print(f"[done] fold_count={summary['fold_count']}")
    print(f"[done] summary_json={summary['output_files']['summary_json']}")
    print(f"[done] fold_metrics_tsv={summary['output_files']['fold_metrics_tsv']}")
    print(f"[done] top_confusions_global_tsv={summary['output_files']['top_confusions_global_tsv']}")
    if args.write_markdown:
        print(f"[done] report_md={summary['output_files']['report_md']}")


if __name__ == "__main__":
    main()
