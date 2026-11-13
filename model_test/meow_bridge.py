"""独立 meow 进程的最小本机 API 适配器；不包含上游代码或基准。"""
import hashlib
import http.client
import json
import re
import urllib.parse

MAX_RESPONSE = 8_000_000
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[a-zA-Z0-9.-]+)?\Z")
_SESSION = re.compile(r"[0-9a-f]{32}\Z")


def _port(value):
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError("meow 端口必须是 1–65535 的整数")
    return value


def _call(port, path, token=None, body=None, uncertain=None):
    conn = http.client.HTTPConnection("127.0.0.1", _port(port), timeout=15)
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Meow-Token"] = token
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
    try:
        conn.request("POST" if body is not None else "GET", path, body=data, headers=headers)
        response = conn.getresponse()
        raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise ValueError(f"meow {uncertain}请求可能已送达，请在 meow 中检查状态" if uncertain else "meow 响应超过大小限制")
        try:
            obj = json.loads(raw)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"meow {uncertain}请求可能已送达，请在 meow 中检查状态" if uncertain else "meow 返回了无效 JSON") from exc
        if response.status != 200:
            if uncertain and response.status >= 500:
                raise ValueError(f"meow {uncertain}请求可能已送达，请在 meow 中检查状态")
            error = obj.get("error") if isinstance(obj, dict) else None
            code = error.get("code") if isinstance(error, dict) else None
            code = code if isinstance(code, str) and re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", code) else "unknown"
            raise ValueError(f"meow 返回 HTTP {response.status}（{code}）")
        if not isinstance(obj, dict):
            raise ValueError(f"meow {uncertain}请求可能已送达，请在 meow 中检查状态" if uncertain else "meow 响应格式无效")
        return obj
    except (OSError, http.client.HTTPException) as exc:
        if uncertain:
            raise ValueError(f"meow {uncertain}请求的送达状态未知，请先在 meow 中检查状态") from exc
        raise ValueError("无法连接本机 meow，请确认它已在所选端口启动") from exc
    finally:
        conn.close()


def _snapshot(port):
    boot = _call(port, "/api/bootstrap")
    token = boot.get("token")
    if not isinstance(token, str) or not 8 <= len(token) <= 256:
        raise ValueError("该端口不是受支持的 meow 服务")
    snapshot = _call(port, "/api/snapshot", token)
    if snapshot.get("brand") != "meow LLM Detector" or not isinstance(snapshot.get("packages"), list) or not isinstance(snapshot.get("endpoints"), list):
        raise ValueError("该端口不是受支持的 meow 服务")
    return token, snapshot


def _fields(obj, names):
    return {key: obj[key] for key in names if key in obj}


def _endpoint_hash(endpoint):
    fields = _fields(endpoint, ("id", "mode", "base_url", "model", "credential_saved", "allow_insecure"))
    return hashlib.sha256(json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_snapshot(snapshot):
    packages = []
    for item in snapshot["packages"][:100]:
        if not isinstance(item, dict):
            continue
        package = _fields(item, ("id", "version", "name", "mode", "publisher", "content_sha256", "withdrawn"))
        models = item.get("models")
        package["models"] = [_fields(model, ("id", "name")) for model in models[:100] if isinstance(model, dict)] if isinstance(models, list) else []
        packages.append(package)
    endpoints = [dict(_fields(item, ("id", "name", "mode", "model", "credential_saved")), config_sha256=_endpoint_hash(item))
                 for item in snapshot["endpoints"][:100] if isinstance(item, dict)]
    return {"brand": "meow LLM Detector", "version": snapshot.get("version"), "packages": packages, "endpoints": endpoints}


def status(port):
    _, snapshot = _snapshot(port)
    return _safe_snapshot(snapshot)


def _config(body):
    if not isinstance(body, dict):
        raise ValueError("meow 配置无效")
    port = _port(body.get("port"))
    package, version, endpoint = body.get("package_id"), body.get("package_version"), body.get("endpoint_id")
    claimed, request_model, bank_sha = body.get("claimed_model"), body.get("request_model"), body.get("benchmark_sha256")
    endpoint_sha = body.get("endpoint_sha256")
    if not all(isinstance(s, str) and _ID.fullmatch(s) for s in (package, endpoint, claimed)):
        raise ValueError("基准包、接口或候选模型 ID 无效")
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError("基准包版本无效")
    if not isinstance(bank_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", bank_sha):
        raise ValueError("基准包哈希无效，请重新连接 meow")
    if not isinstance(endpoint_sha, str) or not re.fullmatch(r"[a-f0-9]{64}", endpoint_sha):
        raise ValueError("接口配置哈希无效，请重新连接 meow")
    if not isinstance(request_model, str) or not request_model.strip() or len(request_model) > 256:
        raise ValueError("请填写实际请求模型")
    tier, workers, retries = body.get("tier", "low"), body.get("workers", 1), body.get("retry_budget", 0)
    if tier not in ("low", "medium", "high") or type(workers) is not int or not 1 <= workers <= 4 or type(retries) is not int or not 0 <= retries <= 1000:
        raise ValueError("检测档位、并发或重试预算无效")
    return {"port": port, "package_id": package, "package_version": version, "benchmark_sha256": bank_sha, "endpoint_id": endpoint, "endpoint_sha256": endpoint_sha,
            "claimed_model": claimed, "request_model": request_model.strip(), "tier": tier,
            "runtime": {"workers": workers, "retry_budget": retries, "retain_raw": False}}


def _verified(config):
    token, snapshot = _snapshot(config["port"])
    packages = [p for p in snapshot["packages"] if isinstance(p, dict) and p.get("id") == config["package_id"] and p.get("version") == config["package_version"]]
    endpoints = [e for e in snapshot["endpoints"] if isinstance(e, dict) and e.get("id") == config["endpoint_id"]]
    if len(packages) != 1 or len(endpoints) != 1:
        raise ValueError("meow 中找不到所选基准包或接口")
    package, endpoint = packages[0], endpoints[0]
    if package.get("content_sha256") != config["benchmark_sha256"]:
        raise ValueError("meow 基准包已变化，请重新连接并估算")
    if _endpoint_hash(endpoint) != config["endpoint_sha256"]:
        raise ValueError("meow 接口配置已变化，请重新连接并估算")
    if package.get("withdrawn") == "security" or endpoint.get("mode") != package.get("mode") or endpoint.get("credential_saved") is not True:
        raise ValueError("meow 接口未保存凭据、协议不匹配或基准已撤回")
    if config["claimed_model"] not in [m.get("id") for m in package.get("models", []) if isinstance(m, dict)]:
        raise ValueError("候选模型不在所选基准包中")
    return token


def _estimate(config, token):
    fields = ("package_id", "package_version", "tier", "runtime")
    result = _call(config["port"], "/api/run/estimate", token, _fields(config, fields))
    logical, maximum = result.get("logical_requests"), result.get("maximum_http_attempts")
    if type(logical) is not int or type(maximum) is not int or not 0 < logical <= maximum <= 100000:
        raise ValueError("meow 返回了无效的请求数估算")
    return _fields(result, ("logical_requests", "maximum_http_attempts", "retry_budget", "known_price", "estimated_usd"))


def estimate(body):
    config = _config(body)
    return _estimate(config, _verified(config))


def start(body):
    config = _config(body)
    token = _verified(config)
    estimate_result = _estimate(config, token)
    if type(body.get("expected_max_http_attempts")) is not int or body["expected_max_http_attempts"] != estimate_result["maximum_http_attempts"]:
        raise ValueError("请求数估算已变化，请重新估算后启动")
    fields = ("package_id", "package_version", "endpoint_id", "claimed_model", "request_model", "tier", "runtime")
    result = _call(config["port"], "/api/run/start", token, _fields(config, fields), uncertain="启动")
    session_id = result.get("session_id")
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise ValueError("meow 返回了无效会话 ID，请在 meow 中检查是否已启动")
    return {"session_id": session_id, "estimate": estimate_result}


def report(port, session_id):
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise ValueError("meow 会话 ID 无效")
    token, _ = _snapshot(port)
    result = _call(port, "/api/report/" + urllib.parse.quote(session_id), token)
    if result.get("product") != "meow LLM Detector" or result.get("session_id") != session_id:
        raise ValueError("meow 报告格式无效")
    safe = _fields(result, ("product", "version", "session_id", "updated_at", "operational_status", "mode", "tier", "claimed_model", "request_model"))
    for key, fields in (("benchmark", ("id", "version", "publisher", "content_sha256")),
                        ("fingerprint", ("verdict", "model", "reasons", "matches", "quality_status", "valid_samples", "planned_samples")),
                        ("progress", ("status", "planned", "logical_completed", "valid_samples", "retries", "retry_budget"))):
        safe[key] = _fields(result.get(key) or {}, fields) if isinstance(result.get(key), dict) else {}
    return safe


def stop(port, session_id):
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise ValueError("meow 会话 ID 无效")
    token, _ = _snapshot(port)
    _call(port, "/api/run/stop", token, {"session_id": session_id}, uncertain="停止")
    return {"ok": True}
