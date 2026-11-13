"""本地 ModelTrace 参考库导入：内容寻址、结构校验和不可变快照。"""
import hashlib
import json
import math
import os
import re
import tempfile
import threading

from . import modeltrace

MAX_BANK_BYTES = 4_000_000
MAX_MODELS = 40
MAX_MODEL_LABEL = 2200  # 接口名称 + 2048 字符的 Bedrock 推理配置 ARN
_LOCK = threading.RLock()
_SENSITIVE_FIELDS = {"api_key", "apikey", "access_token", "authorization", "password", "secret", "cookie", "private_key"}


def _number(value, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1e12:
        raise ValueError("参考库包含无效数值")
    if positive and value <= 0:
        raise ValueError("参考库比例或校准参数必须大于零")


def _vector(values, length, *, positive=False):
    if not isinstance(values, list) or len(values) != length:
        raise ValueError("参考库特征维度不正确")
    for value in values:
        _number(value, positive=positive)


def _contains_sensitive_field(value):
    if isinstance(value, dict):
        return any(str(k).lower() in _SENSITIVE_FIELDS or _contains_sensitive_field(v) for k, v in value.items())
    if isinstance(value, list):
        return any(_contains_sensitive_field(v) for v in value)
    return False


def validate_bank(bank):
    """检查上游 ModelTrace 统一库所需结构，不验证样本来源或统计有效性。"""
    try:
        if not isinstance(bank, dict) or bank.get("schema") != "robust-number-fingerprint-bank":
            raise ValueError("只支持 ModelTrace robust-number-fingerprint-bank 格式")
        if _contains_sensitive_field(bank):
            raise ValueError("参考库含敏感字段，拒绝导入")
        models = bank["models"]
        if not isinstance(models, list) or not 2 <= len(models) <= MAX_MODELS:
            raise ValueError(f"参考库需要 2–{MAX_MODELS} 个候选模型")
        ids = []
        for item in models:
            if not isinstance(item, dict):
                raise ValueError("模型条目格式不正确")
            mid, display = item.get("id"), item.get("display_name")
            if not isinstance(mid, str) or not mid.strip() or len(mid) > MAX_MODEL_LABEL or not isinstance(display, str) or not display or len(display) > MAX_MODEL_LABEL:
                raise ValueError("模型名称无效")
            for field in ("family", "family_name"):
                if item.get(field) is not None and (not isinstance(item[field], str) or len(item[field]) > 160):
                    raise ValueError("模型家族名称无效")
            counts = item.get("counts")
            _vector(counts, modeltrace.DIM)
            if any(v < 0 for v in counts) or sum(counts) <= 0:
                raise ValueError("模型参考频数无效")
            ids.append(mid)
        if len(set(ids)) != len(ids):
            raise ValueError("参考库模型 ID 重复")
        robust = bank["robust"]
        if robust["model_order"] != ids:
            raise ValueError("参考库模型顺序不一致")
        hellinger, ordered = robust["hellinger"], robust["ordered_blocks"]
        for artifact, dim in ((hellinger, modeltrace.DIM), (ordered, 74)):
            _vector(artifact["feature_mean"], dim)
            _vector(artifact["feature_scale"], dim, positive=True)
            basis = artifact["nuisance_basis"]
            if not isinstance(basis, list) or len(basis) > 16:
                raise ValueError("参考库干扰方向无效")
            for vector in basis:
                _vector(vector, dim)
            centroids = artifact["centroids"]
            if not isinstance(centroids, list) or len(centroids) != len(models):
                raise ValueError("参考库模型中心数量不正确")
            for vector in centroids:
                _vector(vector, dim)
        weight = ordered.get("weight", 0)
        _number(weight)
        if not 0 <= weight <= 1:
            raise ValueError("参考库特征权重无效")
        environments = ordered["environment_centroids"]
        if not isinstance(environments, list) or not 1 <= len(environments) <= 32:
            raise ValueError("参考库环境模板数量无效")
        for environment in environments:
            if not isinstance(environment, list) or len(environment) != len(models):
                raise ValueError("参考库环境模板数量不正确")
            for vector in environment:
                _vector(vector, 74)
        _number(bank["calibration"]["3"]["beta"], positive=True)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("参考库缺少必需字段") from exc
    return bank


def bank_path(data_dir, bank_id):
    if not isinstance(bank_id, str) or not re.fullmatch(r"[0-9a-f]{64}", bank_id):
        raise ValueError("参考库 ID 无效")
    return os.path.join(data_dir, "modeltrace-banks", bank_id + ".json")


def _info(bank_id, artifact):
    bank = artifact["bank"]
    return {"id": bank_id, "sha256": artifact["sha256"], "source": artifact.get("source", "upstream"),
            "name": "ModelTrace 内置参考库" if bank_id == "builtin" else str(bank.get("source_scope") or "用户导入参考库")[:160],
            "built_at": bank.get("built_at"), "model_count": len(bank["models"]),
            "models": [m["id"] for m in bank["models"]]}


def load_bank(data_dir, bank_id="builtin"):
    if bank_id == "builtin":
        return dict(modeltrace.load_bank(), source="upstream", upstream_commit=modeltrace.UPSTREAM_COMMIT)
    path = bank_path(data_dir, bank_id)
    with open(path, "rb") as handle:
        raw = handle.read(MAX_BANK_BYTES + 1)
    if len(raw) > MAX_BANK_BYTES or hashlib.sha256(raw).hexdigest() != bank_id:
        raise ValueError("本地参考库哈希不匹配或文件过大")
    bank = json.loads(raw, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("非有限数值")))
    return {"bank": validate_bank(bank), "sha256": bank_id, "source": "local", "upstream_commit": None}


def import_bank(data_dir, raw, known_secrets):
    if not isinstance(raw, bytes) or len(raw) > MAX_BANK_BYTES:
        raise ValueError("参考库文件超过 4 MB 或格式无效")
    try:
        text = raw.decode("utf-8-sig")
        from .core import _redact_value, redact_text
        if any(secret and redact_text(text, secret) != text for secret in known_secrets):
            raise ValueError("参考库包含已保存的接口密钥，拒绝导入")
        bank = json.loads(text, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("非有限数值")))
        if any(secret and _redact_value(bank, secret, mask_keys=True) != bank for secret in known_secrets):
            raise ValueError("参考库包含已保存的接口密钥，拒绝导入")
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("参考库不是有效 UTF-8 JSON") from exc
    validate_bank(bank)
    bank_id = hashlib.sha256(raw).hexdigest()
    path = bank_path(data_dir, bank_id)
    with _LOCK:
        os.makedirs(os.path.dirname(path), mode=0o700, exist_ok=True)
        if os.path.exists(path):
            return _info(bank_id, load_bank(data_dir, bank_id))
        fd, tmp = tempfile.mkstemp(prefix=".bank-", dir=os.path.dirname(path))
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return _info(bank_id, {"bank": bank, "sha256": bank_id, "source": "local"})


def list_banks(data_dir):
    output = [_info("builtin", load_bank(data_dir))]
    folder = os.path.join(data_dir, "modeltrace-banks")
    if not os.path.isdir(folder):
        return output
    for filename in sorted(os.listdir(folder)):
        if not re.fullmatch(r"[0-9a-f]{64}\.json", filename):
            continue
        bank_id = filename[:-5]
        try:
            output.append(_info(bank_id, load_bank(data_dir, bank_id)))
        except (OSError, ValueError):
            continue
    return output
