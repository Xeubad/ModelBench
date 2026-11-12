"""短问题回答分布；仅作描述，不包含模型身份基准或真实性判定。"""
import collections
import itertools
import math

# 自行设计的短问题，不使用 meow 的题目、参考分布或判定线。
PROBES = (
    ("b01", "两位数", "写出一个你想到的两位数，只输出数字。"),
    ("b02", "交通工具", "请只写出一种常见交通工具的名称。"),
    ("b03", "英文颜色", "Name one color in English. Reply with one word only."),
    ("b04", "方向", "从北、南、东、西中选择一个方向，只输出一个字。"),
)
MIN_VALID = 5  # 仅为避免极少样本的展示门槛，并非经过准确率校准的判定线。


def tasks():
    return [{"id": key, "name": name, "prompt": prompt, "scenario": "behavior", "kind": "behavior"}
            for key, name, prompt in PROBES]


def js_divergence(left, right):
    """两组频数的 Jensen–Shannon 散度（以 2 为底，0 到 1）。"""
    total_a, total_b = sum(left.values()), sum(right.values())
    if not total_a or not total_b:
        return None
    result = 0.0
    for key in set(left) | set(right):
        a, b = left.get(key, 0) / total_a, right.get(key, 0) / total_b
        mid = (a + b) / 2
        if a:
            result += a * math.log2(a / mid) / 2
        if b:
            result += b * math.log2(b / mid) / 2
    return max(0.0, min(1.0, result))


def analyze(records, planned_rounds):
    names = {key: name for key, name, _ in PROBES}
    selected = [r for r in records if r.get("scenario") == "behavior"]
    profiles = {}
    for label in dict.fromkeys(r.get("label") or r.get("model") for r in selected):
        profiles[label] = {}
        for key, name in names.items():
            matching = [r for r in selected if (r.get("label") or r.get("model")) == label and r.get("task") == key]
            answers = [str(r.get("text") or "").strip().casefold() for r in matching if r.get("ok") and str(r.get("text") or "").strip()]
            counts = dict(collections.Counter(answers))
            required = max(MIN_VALID, math.ceil(planned_rounds * 0.6))
            profiles[label][key] = {"name": name, "counts": counts, "valid": len(answers), "attempted": len(matching),
                                    "planned": planned_rounds, "required": required}
    pairs = []
    for a, b in itertools.combinations(profiles, 2):
        probes = []
        for key, name in names.items():
            x, y = profiles[a][key], profiles[b][key]
            eligible = x["valid"] >= x["required"] and y["valid"] >= y["required"]
            probes.append({"id": key, "name": name, "a_valid": x["valid"], "b_valid": y["valid"],
                           "jsd": js_divergence(x["counts"], y["counts"]) if eligible else None})
        available = [p["jsd"] for p in probes if p["jsd"] is not None]
        complete = len(available) == len(names)
        pairs.append({"a": a, "b": b, "status": "descriptive" if complete else "insufficient", "probes": probes,
                      "mean_jsd": sum(available) / len(available) if complete else None})
    return {"profiles": profiles, "pairs": pairs, "planned_rounds": planned_rounds,
            "limitation": "无官方基准或经验证判定线；分布差异不证明模型身份或替换。"}
