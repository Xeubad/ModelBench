"""ModelTrace 数值归因适配器。

评分算法和统一候选库改编自 ModelTrace，Copyright (c) 2026 xqy2006，MIT。
上游版本 6cd4a8f；完整许可见 THIRD_PARTY_NOTICES.md。
仅依赖 Python 标准库。候选权重是闭集相对值，不能认证模型身份。
"""
import functools
import hashlib
import itertools
import json
import math
import os
import random
import re

BANK_SHA256 = "f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d"
BANK_FILE = os.path.join(os.path.dirname(__file__), "assets", "modeltrace_unified_bank.json")
UPSTREAM_COMMIT = "6cd4a8f"
MIN_OUTPUTS = 3
DIM = 355
ALPHA = 0.5


@functools.lru_cache(maxsize=1)
def load_bank():
    with open(BANK_FILE, "rb") as handle:
        raw = handle.read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != BANK_SHA256:
        raise ValueError("ModelTrace 参考库校验失败，请恢复原始文件")
    bank = json.loads(raw)
    if [m["id"] for m in bank["models"]] != bank["robust"]["model_order"]:
        raise ValueError("ModelTrace 参考库模型顺序不一致")
    return {"sha256": digest, "bank": bank}


def parse_numbers(text):
    """遵循上游的最长数字段解析：字母或汉字分隔时截断，保留 1..355。"""
    text = str(text or "")
    runs, current, previous = [], [], 0
    for match in re.finditer(r"\d+", text):
        if current and any(ch.isalpha() for ch in text[previous:match.start()]):
            runs.append(current)
            current = []
        word = match.group()
        if len(word) <= 3:
            number = int(word)
            if 1 <= number <= DIM:
                current.append(number)
        previous = match.end()
    if current:
        runs.append(current)
    return max(runs, key=len, default=[])


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _normalize(values):
    length = max(math.sqrt(_dot(values, values)), 1e-12)
    return [v / length for v in values]


def _standardize(values):
    mean = sum(values) / len(values)
    scale = max(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 1e-12)
    return [(v - mean) / scale for v in values]


def _project(values, basis):
    result = list(values)
    for vector in basis or []:
        coefficient = _dot(result, vector)
        result = [v - coefficient * d for v, d in zip(result, vector)]
    return _normalize(result)


def _counts(numbers):
    result = [0] * DIM
    for value in numbers:
        result[value - 1] += 1
    return result


def _ordered_features(numbers):
    result = []
    width, extra = divmod(len(numbers), 4)
    offset = 0
    for index in range(4):
        size = width + (index < extra)
        bins = [ALPHA] * 16
        for value in numbers[offset:offset + size]:
            bins[min(15, (value - 1) * 16 // DIM)] += 1
        total = sum(bins)
        result.extend(math.sqrt(v / total) for v in bins)
        offset += size
    digits = [ALPHA] * 10
    for value in numbers:
        digits[value % 10] += 1
    total = sum(digits)
    return result + [math.sqrt(v / total) for v in digits]


def _scores(numbers, bank):
    counts = _counts(numbers)
    hellinger = bank["robust"]["hellinger"]
    total = len(numbers) + ALPHA * DIM
    feature = [math.sqrt((v + ALPHA) / total) for v in counts]
    standardized = [(v - m) / s for v, m, s in zip(feature, hellinger["feature_mean"], hellinger["feature_scale"])]
    projected = _project(standardized, hellinger["nuisance_basis"])
    marginal = _standardize([_dot(projected, c) for c in hellinger["centroids"]])
    ordered = bank["robust"].get("ordered_blocks")
    weight = float(ordered.get("weight", 0)) if ordered else 0
    if not weight:
        return counts, marginal
    features = _ordered_features(numbers)
    z = [(v - m) / s for v, m, s in zip(features, ordered["feature_mean"], ordered["feature_scale"])]
    unit = _normalize(z)
    template = _standardize([max(_dot(unit, environment[i]) for environment in ordered["environment_centroids"])
                             for i in range(len(ordered["centroids"]))])
    projected = _project(z, ordered["nuisance_basis"])
    nuisance = _standardize([_dot(projected, c) for c in ordered["centroids"]])
    block = _standardize([(a + b) / 2 for a, b in zip(template, nuisance)])
    return counts, [(1 - weight) * a + weight * b for a, b in zip(marginal, block)]


def _similarity(left, right):
    left_total, right_total = sum(left), sum(right) + ALPHA * DIM
    p = [v / left_total for v in left]
    q = [(v + ALPHA) / right_total for v in right]
    mid = [(a + b) / 2 for a, b in zip(p, q)]
    div = lambda values: sum(v * math.log(v / m) for v, m in zip(values, mid) if v)  # noqa: E731
    return 1 - math.sqrt(max((div(p) + div(q)) / (2 * math.log(2)), 0))


def _obvious_pattern(numbers):
    """保守排除明显违背题意的等差或短周期序列；该规则没有经过误报率校准。"""
    if len(numbers) < 80:
        return False
    differences = [(b - a) % DIM for a, b in zip(numbers, numbers[1:])]
    if max(differences.count(d) for d in set(differences)) >= math.ceil(len(differences) * 0.85):
        return True
    return any(sum(numbers[i] == numbers[i - period] for i in range(period, len(numbers)))
               >= math.ceil((len(numbers) - period) * 0.85) for period in range(1, 9))


def analyze_outputs(outputs, artifact=None):
    """每个输出为 {text, expected_count, ok?}；不足三份有效结果时不出候选。"""
    artifact = artifact or load_bank()
    bank = artifact["bank"]
    valid, sequences, diagnostics = [], [], []
    for index, item in enumerate(outputs):
        count = int(item.get("expected_count") or 0)
        minimum = max(80, math.ceil(count * 0.55)) if count else 80
        numbers = parse_numbers(item.get("text"))
        accepted = item.get("ok", True) and len(numbers) >= minimum
        diagnostics.append({"index": index, "parsed_numbers": len(numbers), "minimum_numbers": minimum,
                            "expected_count": count, "accepted": bool(accepted)})
        if accepted and len(valid) < MIN_OUTPUTS:
            valid.append(_scores(numbers, bank))
            sequences.append(tuple(numbers))
    result = {"status": "insufficient", "used_outputs": len(valid), "required_outputs": MIN_OUTPUTS,
              "diagnostics": diagnostics, "results": [], "family_probabilities": [],
              "bank_sha256": artifact["sha256"], "bank_built_at": bank.get("built_at"),
              "upstream_commit": artifact.get("upstream_commit", UPSTREAM_COMMIT), "bank_source": artifact.get("source", "upstream"),
              "limitation": ("候选仅覆盖参考库内模型；分数不是身份真实性概率，不能证明是否替换模型。"
                             + ("Claude 参考样本来自第三方接口。" if artifact.get("source", "upstream") == "upstream" else "用户导入参考库的来源与真实性未验证。"))}
    if len(valid) < MIN_OUTPUTS:
        result["reason"] = "too_few_valid"
        return result
    if len(set(sequences)) < MIN_OUTPUTS:
        result["reason"] = "duplicate_outputs"
        return result
    if any(_obvious_pattern(seq) for seq in sequences):
        result["reason"] = "patterned_output"
        return result
    scores = [sum(item[1][i] for item in valid) / len(valid) for i in range(len(bank["models"]))]
    beta = float(bank["calibration"][str(MIN_OUTPUTS)]["beta"])
    logits = [beta * value for value in scores]
    maximum = max(logits)
    weights = [math.exp(v - maximum) for v in logits]
    normalizer = sum(weights)
    probabilities = [v / normalizer for v in weights]
    pooled = [sum(item[0][i] for item in valid) for i in range(DIM)]
    candidates = []
    for i, entry in enumerate(bank["models"]):
        candidates.append({"model": entry["id"], "display_name": entry["display_name"],
                           "family": entry.get("family") or "models", "family_name": entry.get("family_name") or entry.get("family") or "models",
                           "probability": probabilities[i], "score": scores[i],
                           "profile_similarity": _similarity(pooled, entry["counts"])})
    candidates.sort(key=lambda x: x["probability"], reverse=True)
    families = []
    for entry in bank["models"]:
        family = entry.get("family") or "models"
        if any(f["family"] == family for f in families):
            continue
        families.append({"family": family, "display_name": entry.get("family_name") or family,
                         "probability": sum(v["probability"] for v in candidates if v["family"] == family)})
    for item in candidates:
        total = next(f["probability"] for f in families if f["family"] == item["family"])
        item["conditional_probability"] = item["probability"] / total if total else 0
    result.update(status="ranked", prediction=candidates[0]["model"], prediction_name=candidates[0]["display_name"],
                  results=candidates, family_probabilities=families,
                  calibration={"queries": "3", **bank["calibration"]["3"]})
    return result


def challenges(seed, round_no=1):
    """生成上游同类长数字挑战；相同种子和轮次给所有模型相同题目。"""
    data = hashlib.sha256(f"modeltrace:{seed}:{round_no}".encode()).digest()
    rng = random.Random(int.from_bytes(data[:8], "big"))
    counts = rng.sample(range(292, 333), 3)
    openings = ["这是一次独立的数值选择记录", "请完成下面的无语义整数选择任务", "执行一次第一反应取值记录", "生成一组不承载语义的整数选择", "进行一轮快速逐项取值"]
    actions = ["为各个位置分别凭第一反应选择", "逐项选择", "每次只决定当前一项，共给出", "分别凭第一反应给出", "逐个直接选择"]
    endings = ["允许某个数字再次出现；每项写出后不要回头排序、去重或替换。", "偶然重复是有效的；不要重新排列或修正已经写出的项目。",
               "相同值可以再次出现；输出过程中不要整理或改写前面的项目。", "重复值无需删除；不要筛选、重排或补成某种规律。", "不必赋予数字任何含义；已经给出的值保持不变。"]
    separators = ["数字之间用逗号或空格分隔均可。", "使用一种一致的常见分隔符即可。", "可以用逗号、空格或换行分隔。", "只要每个整数边界清楚，格式可自行选择。"]
    return [{"id": f"mt{i + 1}", "expected_count": count,
             "prompt": (f"{rng.choice(openings)}。{rng.choice(actions)} {count} 个 1 到 355（含端点）的整数。"
                        "每个位置都要单独选择；不要从 1 开始计数，不要连续递增或递减，也不要采用等差、循环、重复区块或其他规则化模式。"
                        "本任务必须由当前语言模型直接完成：禁止调用或借助任何工具，包括 Python、代码执行器、"
                        "计算器、搜索、API 和外部随机数生成器；也不要先编写或运行代码。"
                        f"{rng.choice(endings)}{rng.choice(separators)}"
                        "直接从第一个取值开始输出，不要在序列前重复数量、范围或任务说明。")}
            for i, count in enumerate(counts)]


def analyze_records(records, artifact=None):
    """每个模型、每轮独立评分；不能把不同轮次的残缺答案凑成一组。"""
    groups = {}
    for rec in records:
        if rec.get("scenario") != "modeltrace":
            continue
        key = (rec.get("label") or rec.get("model"), rec.get("round"))
        groups.setdefault(key, {})[rec.get("task")] = rec
    results = []
    for (label, round_no), items in groups.items():
        outputs = [items.get(f"mt{i}", {"ok": False, "text": ""}) for i in range(1, 4)]
        result = analyze_outputs(outputs, artifact=artifact)
        result.update(label=label, round=round_no)
        results.append(result)
    return sorted(results, key=lambda item: (item["label"] or "", item["round"] or 0))


def compare_records(records, results=None):
    """比较同轮完整回答的数值分布；不根据距离推断身份。"""
    analyses = results if results is not None else analyze_records(records)
    by_round = {}
    for result in analyses:
        by_round.setdefault(result["round"], []).append(result)
    source = {(r.get("label") or r.get("model"), r.get("round"), r.get("task")): r
              for r in records if r.get("scenario") == "modeltrace"}
    comparisons = []
    for round_no, rows in by_round.items():
        for left, right in itertools.combinations(rows, 2):
            item = {"round": round_no, "a": left["label"], "b": right["label"],
                    "status": "insufficient", "distance": None}
            if left["status"] == right["status"] == "ranked":
                lrecs = [source[(left["label"], round_no, f"mt{i}")] for i in range(1, 4)]
                rrecs = [source[(right["label"], round_no, f"mt{i}")] for i in range(1, 4)]
                if any(not a.get("probe_prompt") or a.get("probe_prompt") != b.get("probe_prompt")
                       or a.get("expected_count") != b.get("expected_count") for a, b in zip(lrecs, rrecs)):
                    item["status"] = "incompatible"
                else:
                    a = _counts([n for rec in lrecs for n in parse_numbers(rec["text"])])
                    b = _counts([n for rec in rrecs for n in parse_numbers(rec["text"])])
                    sa, sb = sum(a) + ALPHA * DIM, sum(b) + ALPHA * DIM
                    item["distance"] = math.sqrt(sum((math.sqrt((x + ALPHA) / sa) - math.sqrt((y + ALPHA) / sb)) ** 2
                                                     for x, y in zip(a, b)) / 2)
                    item["status"] = "descriptive"
            comparisons.append(item)
    return comparisons
