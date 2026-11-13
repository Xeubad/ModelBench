"""Optional ModelTrace bank trainer, adapted from MIT upstream 6cd4a8f.

NumPy is used only by this explicitly invoked training command. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError as exc:
    raise RuntimeError("训练参考库需要可选 NumPy；请在独立虚拟环境中安装 numpy>=1.26,<3") from exc

from . import banks, modeltrace, modeltrace_suite

DIMENSION = modeltrace.DIM
ALPHA = modeltrace.ALPHA
ORDERED_BLOCK_WEIGHT = 0.25


def hellinger_feature(counts):
    values = np.asarray(counts, dtype=np.float64) + ALPHA
    return np.sqrt(values / values.sum())


def ordered_block_feature(numbers):
    return np.asarray(modeltrace._ordered_features(numbers), dtype=np.float64)


def robust_score_numbers(numbers, bank):
    return {"fused": modeltrace._scores(numbers, bank)[1]}


def fit_robust_artifacts(rows: list[dict], model_ids: list[str]) -> dict:
    environments = sorted({row["condition_id"] for row in rows})
    complete = [
        environment
        for environment in environments
        if {row["source"] for row in rows if row["condition_id"] == environment}
        == set(model_ids)
    ]
    robust_ready = bool(complete)
    if complete:
        robust_rows = [row for row in rows if row["condition_id"] in complete]
    else:
        robust_rows = rows

    features = np.stack([hellinger_feature(row["counts"]) for row in robust_rows])
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    standardized = (features - feature_mean) / feature_scale

    nuisance_environments = sorted(
        {
            row.get("nuisance_condition_id", row["condition_id"])
            for row in robust_rows
        }
    )
    environment_means = []
    for environment in nuisance_environments:
        indices = np.asarray(
            [
                row.get("nuisance_condition_id", row["condition_id"]) == environment
                for row in robust_rows
            ],
            dtype=bool,
        )
        environment_means.append(standardized[indices].mean(axis=0))
    offsets = np.stack(environment_means)
    offsets -= offsets.mean(axis=0, keepdims=True)
    basis = np.empty((0, DIMENSION), dtype=np.float64)
    if len(offsets) > 1:
        _, singular, right = np.linalg.svd(offsets, full_matrices=False)
        rank = min(2, int(np.sum(singular > singular[0] * 1e-8))) if singular[0] else 0
        basis = right[:rank]
        # NumPy 2.0/macOS ARM 的 matmul 可能继承 SVD 的浮点标志并误报；einsum 等价且不依赖该 BLAS 路径。
        projection = np.einsum("ij,kj->ik", standardized, basis)
        standardized -= np.einsum("ik,kj->ij", projection, basis)

    labels = np.asarray([row["source"] for row in robust_rows])
    centroids = np.stack(
        [standardized[labels == model_id].mean(axis=0) for model_id in model_ids]
    )
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids /= np.maximum(centroid_norms, 1e-12)

    ordered_features = np.stack([ordered_block_feature(row["numbers"]) for row in robust_rows])
    ordered_mean = ordered_features.mean(axis=0)
    ordered_scale = ordered_features.std(axis=0)
    ordered_scale[ordered_scale < 1e-12] = 1.0
    ordered_standardized = (ordered_features - ordered_mean) / ordered_scale
    ordered_environment_means = []
    for environment in nuisance_environments:
        indices = np.asarray(
            [
                row.get("nuisance_condition_id", row["condition_id"]) == environment
                for row in robust_rows
            ],
            dtype=bool,
        )
        ordered_environment_means.append(ordered_standardized[indices].mean(axis=0))
    ordered_offsets = np.stack(ordered_environment_means)
    ordered_offsets -= ordered_offsets.mean(axis=0, keepdims=True)
    ordered_basis = np.empty((0, ordered_standardized.shape[1]), dtype=np.float64)
    if len(ordered_offsets) > 1:
        _, singular, right = np.linalg.svd(ordered_offsets, full_matrices=False)
        rank = min(2, int(np.sum(singular > singular[0] * 1e-8))) if singular[0] else 0
        ordered_basis = right[:rank]
        ordered_projection = np.einsum("ij,kj->ik", ordered_standardized, ordered_basis)
        ordered_standardized -= np.einsum("ik,kj->ij", ordered_projection, ordered_basis)
    ordered_centroids = np.stack(
        [ordered_standardized[labels == model_id].mean(axis=0) for model_id in model_ids]
    )
    ordered_centroids /= np.maximum(np.linalg.norm(ordered_centroids, axis=1, keepdims=True), 1e-12)
    ordered_environment_centroids = []
    unprojected_ordered = (ordered_features - ordered_mean) / ordered_scale
    template_environments = complete or [None]
    for environment in template_environments:
        environment_centroids = []
        for model_id in model_ids:
            indices = np.asarray(
                [
                    (environment is None or row["condition_id"] == environment)
                    and row["source"] == model_id
                    for row in robust_rows
                ],
                dtype=bool,
            )
            centroid = unprojected_ordered[indices].mean(axis=0)
            centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
            environment_centroids.append(centroid)
        ordered_environment_centroids.append(np.stack(environment_centroids))
    ordered_weight = ORDERED_BLOCK_WEIGHT

    return {
        "model_order": model_ids,
        "robust_ready": robust_ready,
        "training_rows": len(robust_rows),
        "complete_environments": complete,
        "hellinger": {
            "feature_mean": feature_mean.tolist(),
            "feature_scale": feature_scale.tolist(),
            "nuisance_rank": len(basis),
            "nuisance_environments": nuisance_environments,
            "nuisance_basis": basis.tolist(),
            "centroids": centroids.tolist(),
        },
        "ordered_blocks": {
            "weight": ordered_weight,
            "feature": "four position blocks x 16 value bins plus final-digit distribution",
            "feature_mean": ordered_mean.tolist(),
            "feature_scale": ordered_scale.tolist(),
            "nuisance_rank": len(ordered_basis),
            "nuisance_basis": ordered_basis.tolist(),
            "centroids": ordered_centroids.tolist(),
            "environment_centroids": [centroids.tolist() for centroids in ordered_environment_centroids],
        },
    }


def calibration_records(rows: list[dict], model_ids: list[str], query_count: int) -> list[tuple[list[float], int]]:
    records = []
    conditions = sorted({row["condition_id"] for row in rows})
    for condition in conditions:
        challenge_ids = sorted({row["challenge_id"] for row in rows if row["condition_id"] == condition})
        challenge_ids = [
            challenge_id
            for challenge_id in challenge_ids
            if all(
                any(
                    row["source"] == model_id and row["challenge_id"] == challenge_id
                    for row in rows
                )
                for model_id in model_ids
            )
        ]
        if len(challenge_ids) < query_count:
            continue
        for held_ids in itertools.combinations(challenge_ids, query_count):
            held_set = set(held_ids)
            train = [row for row in rows if row["challenge_id"] not in held_set]
            robust = fit_robust_artifacts(train, model_ids)
            score_bank = {"robust": robust}
            for truth_index, model_id in enumerate(model_ids):
                held = [
                    row
                    for row in rows
                    if row["source"] == model_id and row["challenge_id"] in held_set
                ]
                if len(held) != query_count:
                    continue
                response_scores = [
                    robust_score_numbers(row["numbers"], score_bank)["fused"] for row in held
                ]
                combined = [
                    sum(scores[index] for scores in response_scores) / query_count
                    for index in range(len(model_ids))
                ]
                records.append((combined, truth_index))
    return records


def fit_beta(records: list[tuple[list[float], int]]) -> tuple[float, float]:
    best_beta = 1.0
    best_loss = float("inf")
    for step in range(801):
        beta = math.exp(math.log(0.05) + step * (math.log(12.0) - math.log(0.05)) / 800)
        loss = 0.0
        for scores, truth in records:
            scaled = [beta * value for value in scores]
            maximum = max(scaled)
            log_sum = maximum + math.log(sum(math.exp(value - maximum) for value in scaled))
            loss += log_sum - scaled[truth]
        loss /= len(records)
        if loss < best_loss:
            best_beta = beta
            best_loss = loss
    return best_beta, best_loss


def build_bank(rows: list[dict], fallback_calibration: dict | None = None) -> dict:
    model_ids = list(dict.fromkeys(row["source"] for row in rows))
    model_entries = []
    for model_id in model_ids:
        selected = [row for row in rows if row["source"] == model_id]
        pooled = [sum(row["counts"][index] for row in selected) for index in range(DIMENSION)]
        family_id = selected[0].get("family_id")
        family_name = selected[0].get("family_name")
        model_entries.append(
            {
                "id": model_id,
                "display_name": model_id,
                "family": family_id,
                "family_name": family_name,
                "response_count": len(selected),
                "valid_number_count": sum(len(row["numbers"]) for row in selected),
                "conditions": dict(Counter(row["condition_id"] for row in selected)),
                "counts": pooled,
            }
        )

    robust = fit_robust_artifacts(rows, model_ids)
    calibration = {}
    for query_count in (1, 2, 3):
        records = calibration_records(rows, model_ids, query_count)
        key = str(query_count)
        if records:
            beta, nll = fit_beta(records)
            correct = sum(max(range(len(scores)), key=lambda index: scores[index]) == truth for scores, truth in records)
            calibration[key] = {
                "beta": beta,
                "cv_accuracy": correct / len(records),
                "cv_correct": correct,
                "cv_samples": len(records),
                "cv_nll": nll,
                "fallback": False,
            }
        else:
            previous = fallback_calibration.get(key) if fallback_calibration else None
            calibration[key] = {
                **(
                    previous
                    or {
                        "beta": 1.0,
                        "cv_accuracy": None,
                        "cv_correct": 0,
                        "cv_samples": 0,
                        "cv_nll": None,
                    }
                ),
                "fallback": True,
            }

    providers = sorted({row.get("provider") for row in rows if row.get("provider")})
    source_scope = "Reference outputs enrolled for the listed model labels."
    method_name = "Ordered-block + nuisance-Hellinger"
    return {
        "schema": "robust-number-fingerprint-bank",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "method": {
            "name": method_name,
            "range": [1, 355],
            "alpha": ALPHA,
            "ordered_block_weight": robust["ordered_blocks"]["weight"],
            "parser": "longest digit run; alphabetic separators split runs; retain values 1..355",
            "response_score": "shared nuisance-Hellinger and ordered-block environment/nuisance fusion",
            "aggregation": "mean of response-wise fused class z-scores",
            "probability": "uniform-prior softmax with fusion-specific grouped cross-validation temperature scaling",
        },
        "recommended_queries": 3,
        "minimum_valid_numbers": 80,
        "source_scope": source_scope,
        "providers": providers,
        "models": model_entries,
        "robust": robust,
        "calibration": calibration,
    }



def _prepare_rows(rows):
    """只接受可核对的 ModelBench 36 条采集格式；重新检查每条输出。"""
    if not isinstance(rows, list):
        raise ValueError("采集记录必须是列表")
    suite = {item["challenge_id"]: item for item in modeltrace_suite.fingerprint_suite()}
    prepared, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("采集记录格式无效")
        source, run_id, challenge_id = row.get("source"), row.get("run_id"), row.get("challenge_id")
        if not isinstance(source, str) or not source.strip() or len(source) > banks.MAX_MODEL_LABEL or not all(
                isinstance(v, str) and v.strip() and len(v) <= 160 for v in (run_id, challenge_id)):
            raise ValueError("采集记录缺少来源、运行 ID 或挑战 ID")
        key = (source, challenge_id)
        if key in seen:
            raise ValueError("同一模型的挑战重复；每个模型请选择一次完整采集")
        seen.add(key)
        task = suite.get(challenge_id)
        if not task or row.get("suite_commit") != modeltrace.UPSTREAM_COMMIT:
            raise ValueError("采集挑战版本与当前训练器不一致")
        prompt = ((task["user_prefix"] + "\n\nFinal task:\n") if task["user_prefix"] else "") + task["prompt"]
        system_sha = hashlib.sha256(task["system"].encode()).hexdigest()
        if (row.get("condition_id") != task["condition"] or row.get("nuisance_condition_id") != task["condition"]
                or row.get("requested_count") != task["expected_count"] or row.get("prompt") != prompt
                or row.get("system_sha256") != system_sha):
            raise ValueError("采集的提示词、环境或数量与固定挑战不一致")
        text = row.get("text")
        if not isinstance(text, str) or len(text) > 20000 or type(row.get("strict_valid")) is not bool:
            raise ValueError("采集回答或有效性标记无效")
        numbers = modeltrace.parse_numbers(text)
        threshold = max(80, math.ceil(task["expected_count"] * .55))
        valid = row["strict_valid"] and len(numbers) >= threshold and not modeltrace._obvious_pattern(numbers)
        prepared.append({**row, "strict_valid": valid, "numbers": numbers, "counts": modeltrace._counts(numbers)})
    return prepared


def _rows_hash(rows):
    raw = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _evaluate(bank, held_rows):
    candidates = {item["id"] for item in bank["models"]}
    groups = {}
    for row in held_rows:
        groups.setdefault((row["source"], row["condition_id"]), []).append(row)
    result = {"known_groups": 0, "known_correct": 0, "known_ranked": 0, "known_abstained": 0,
              "unknown_groups": 0, "unknown_ranked": 0, "unknown_abstained": 0,
              "independent_run_ids": True,
              "limitation": "按独立运行的三条挑战分组；闭集排名不识别未知模型，留出结果不是身份真实性概率。"}
    artifact = {"bank": bank, "sha256": "training", "source": "local", "upstream_commit": None}
    sources = sorted({row["source"] for row in held_rows})
    conditions = sorted({item["condition"] for item in modeltrace_suite.fingerprint_suite()})
    # 已停止或整组失败的环境也计入拒判，避免只统计有结果的环境。
    for source, condition in itertools.product(sources, conditions):
        group = groups.get((source, condition), [])
        outputs = [{"text": row["text"], "expected_count": row["requested_count"], "ok": row["strict_valid"]} for row in group]
        ranked = modeltrace.analyze_outputs(outputs, artifact=artifact)
        kind = "known" if source in candidates else "unknown"
        result[kind + "_groups"] += 1
        if ranked["status"] != "ranked":
            result[kind + "_abstained"] += 1
        else:
            result[kind + "_ranked"] += 1
            if kind == "known" and ranked["prediction"] == source:
                result["known_correct"] += 1
    return result


def train_bank(training_rows, holdout_rows=None):
    """拟合参考库；留出集须来自独立运行，未知模型只统计是否仍被闭集排序。"""
    holdout_rows = holdout_rows or []
    training = _prepare_rows(training_rows)
    held = _prepare_rows(holdout_rows)
    train_runs = {row["run_id"] for row in training}
    held_runs = {row["run_id"] for row in held}
    if train_runs & held_runs:
        raise ValueError("训练集和留出集必须来自独立运行")
    usable = [row for row in training if row["strict_valid"]]
    ids = list(dict.fromkeys(row["source"] for row in usable))
    if not 2 <= len(ids) <= banks.MAX_MODELS:
        raise ValueError("训练需要至少两个、最多 40 个有有效记录的已知模型")
    for source in ids:
        if sum(row["source"] == source for row in usable) < 3:
            raise ValueError("每个已知模型至少需要 3 条有效采集记录")
    bank = build_bank(usable)
    bank["source_scope"] = "ModelBench 用户采集的已知来源；来源真实性由用户核实"
    bank["provenance"] = {"training_rows": len(usable), "training_total_rows": len(training),
                          "training_run_ids": sorted(train_runs), "training_sha256": _rows_hash(training_rows),
                          "suite_commit": modeltrace.UPSTREAM_COMMIT, "numpy_version": np.__version__,
                          "python_version": platform.python_version(),
                          "holdout_sha256": _rows_hash(holdout_rows) if held else None,
                          "holdout_run_ids": sorted(held_runs)}
    bank["evaluation"] = _evaluate(bank, held) if held else None
    banks.validate_bank(bank)
    return bank


def _read_jsonl(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path} 第 {line_no} 行不是有效 JSON") from exc
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python3 -m model_test train-bank", description="使用本地采集数据训练 ModelTrace 参考库")
    parser.add_argument("--train", action="append", required=True, help="训练 JSONL；可重复指定")
    parser.add_argument("--holdout", action="append", default=[], help="独立运行的留出 JSONL；可重复指定")
    parser.add_argument("--output", required=True, help="输出新 JSON 文件；不会覆盖已有文件")
    args = parser.parse_args(argv)
    bank = train_bank(_read_jsonl(args.train), _read_jsonl(args.holdout))
    with open(args.output, "x", encoding="utf-8") as handle:
        json.dump(bank, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": args.output, "training_rows": bank["provenance"]["training_rows"],
                      "models": len(bank["models"]), "evaluation": bank["evaluation"]}, ensure_ascii=False, indent=2))
