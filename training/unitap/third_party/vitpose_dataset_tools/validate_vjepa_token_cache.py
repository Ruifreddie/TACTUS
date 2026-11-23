#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _read_split_ids(split_file: Path) -> List[str]:
    if not split_file.is_file():
        return []
    lines = [x.strip() for x in split_file.read_text(encoding="utf-8").splitlines()]
    return [x for x in lines if x]


def _resolve_split_file(data_root: Path, split: str, override: str) -> Path:
    if override and str(override).strip():
        p = Path(str(override))
        if not p.is_absolute():
            p = data_root / p
        return p
    return data_root / f"{split}.txt"


def _validate_token_file(path: Path) -> Tuple[bool, Optional[Tuple[int, ...]], Optional[str]]:
    if not path.is_file():
        return False, None, "missing"
    try:
        arr = np.load(path, allow_pickle=False)
        if arr.ndim not in (4, 5):
            return False, tuple(arr.shape), f"unexpected_ndim={arr.ndim}"
        return True, tuple(arr.shape), None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Validate a V-JEPA token cache against split files, optionally deleting invalid cache files "
            "so conversion can safely resume from the remaining valid subset."
        )
    )
    ap.add_argument("--data-root", type=str, required=True, help="Dataset root containing split txt files.")
    ap.add_argument("--cache-root", type=str, required=True, help="Token cache root with split subdirectories.")
    ap.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to validate.")
    ap.add_argument("--train-split-file", type=str, default="")
    ap.add_argument("--val-split-file", type=str, default="")
    ap.add_argument("--test-split-file", type=str, default="")
    ap.add_argument(
        "--delete-invalid",
        action="store_true",
        help="Delete invalid cache files after recording them in the manifest/report.",
    )
    ap.add_argument(
        "--fail-on-invalid",
        action="store_true",
        help="Exit non-zero if any invalid cache files are found.",
    )
    ap.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Exit non-zero if any expected cache files are missing.",
    )
    ap.add_argument(
        "--report-path",
        type=str,
        default="",
        help="Optional report json path. Default: <cache-root>/cache_validation_report.json",
    )
    ap.add_argument(
        "--manifest-path",
        type=str,
        default="",
        help="Optional TSV manifest for invalid or missing entries.",
    )
    ap.add_argument("--log-interval", type=int, default=500, help="Progress interval per split.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root not found: {data_root}")
    if not cache_root.is_dir():
        raise FileNotFoundError(f"cache root not found: {cache_root}")

    report_path = (
        Path(args.report_path).expanduser().resolve()
        if str(args.report_path).strip()
        else (cache_root / "cache_validation_report.json")
    )
    manifest_path = (
        Path(args.manifest_path).expanduser().resolve()
        if str(args.manifest_path).strip()
        else None
    )

    split_override = {
        "train": str(args.train_split_file),
        "val": str(args.val_split_file),
        "test": str(args.test_split_file),
    }
    split_names = [x.strip() for x in str(args.splits).split(",") if x.strip()]

    summary: Dict[str, Dict[str, int]] = {}
    invalid_entries: List[Dict[str, object]] = []
    missing_entries: List[Dict[str, object]] = []
    extra_entries: List[Dict[str, object]] = []

    print("[validate-setup]")
    print(f"  data_root={data_root}")
    print(f"  cache_root={cache_root}")
    print(f"  splits={split_names}")
    print(f"  delete_invalid={bool(args.delete_invalid)}")

    for split in split_names:
        split_file = _resolve_split_file(data_root=data_root, split=split, override=split_override.get(split, ""))
        sids = _read_split_ids(split_file)
        sid_set = set(sids)
        split_dir = cache_root / split
        if not split_dir.is_dir():
            split_dir.mkdir(parents=True, exist_ok=True)

        total = len(sids)
        valid = 0
        invalid = 0
        missing = 0
        deleted_invalid = 0

        print(f"[split={split}] start total={total} split_file={split_file}")
        for idx, sid in enumerate(sids, start=1):
            token_path = split_dir / f"{sid}.npy"
            ok, shape, err = _validate_token_file(token_path)
            if ok:
                valid += 1
            elif token_path.is_file():
                invalid += 1
                invalid_entries.append(
                    {
                        "split": split,
                        "sid": sid,
                        "path": str(token_path),
                        "shape": list(shape) if shape is not None else None,
                        "error": err,
                    }
                )
                if args.delete_invalid:
                    token_path.unlink()
                    deleted_invalid += 1
            else:
                missing += 1
                missing_entries.append({"split": split, "sid": sid, "path": str(token_path)})

            if int(args.log_interval) > 0 and ((idx % int(args.log_interval) == 0) or idx == total):
                print(
                    f"[split={split}] {idx}/{total} valid={valid} invalid={invalid} "
                    f"missing={missing} deleted_invalid={deleted_invalid}"
                )

        extra = 0
        for token_path in sorted(split_dir.glob("*.npy")):
            sid = token_path.stem
            if sid in sid_set:
                continue
            extra += 1
            extra_entries.append({"split": split, "sid": sid, "path": str(token_path)})

        summary[split] = {
            "total": int(total),
            "valid": int(valid),
            "invalid": int(invalid),
            "missing": int(missing),
            "deleted_invalid": int(deleted_invalid),
            "extra": int(extra),
        }
        print(f"[split={split}] done -> {summary[split]}")

    report = {
        "data_root": str(data_root),
        "cache_root": str(cache_root),
        "splits": split_names,
        "delete_invalid": bool(args.delete_invalid),
        "summary": summary,
        "invalid_entries": invalid_entries,
        "missing_entries": missing_entries,
        "extra_entries": extra_entries,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        for row in invalid_entries:
            lines.append(f"{row['path']}\tinvalid\t{row.get('error', '')}")
        for row in missing_entries:
            lines.append(f"{row['path']}\tmissing\t")
        manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    total_invalid = sum(int(v["invalid"]) for v in summary.values())
    total_missing = sum(int(v["missing"]) for v in summary.values())
    total_deleted = sum(int(v["deleted_invalid"]) for v in summary.values())

    print("[validate-done]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  report={report_path}")
    if manifest_path is not None:
        print(f"  manifest={manifest_path}")
    print(f"  invalid_total={total_invalid}")
    print(f"  missing_total={total_missing}")
    print(f"  deleted_invalid_total={total_deleted}")

    errors: List[str] = []
    if args.fail_on_invalid and total_invalid > 0:
        errors.append(f"invalid={total_invalid}")
    if args.fail_on_missing and total_missing > 0:
        errors.append(f"missing={total_missing}")
    if errors:
        raise RuntimeError("cache validation failed: " + ", ".join(errors))


if __name__ == "__main__":
    main()
