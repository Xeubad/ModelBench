#!/usr/bin/env python3
"""模型测试平台的存储层：接口（地址 + 密钥 + 模型列表）、测试结果目录、历史、一次完整测试的执行。

- 目录和路径由 config.json 的 paths 决定（默认 data/、results/）。接口存在 data/endpoints.json（权限 600，明文保存密钥；页面只拿到“末四位提示”，改接口时密钥留空表示不变）。
- 每次测试存成独立文件夹 results/<日期>-<当天第几次>/（如 2026-10-12-1）：result.json（汇总+环境快照+每次请求）、
  raw.json（含全部流事件）、report.md。写完设为只读，不再改动。密钥不会进入任何结果文件。
- 旧版结果（~/.openclaw/logs/model-bench/<时间>.json）仍可在历史里查看。
"""
import datetime as dt
import json
import os
import random
import re
import secrets
import shutil
import threading

from . import config, core

DATA_DIR = config.path("data_dir")
RESULTS_DIR = config.path("results_dir")
LEGACY_DIR = config.path("legacy_results_dir")
OPENCLAW_CONFIG = config.path("openclaw_config")
RUN_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-\d+(?:_[\w-]{1,30})?\Z")  # 日期-序号[_测试名称]
NAME_RE = re.compile(r"^[\w-]{1,30}\Z")  # 测试名称：中文、字母、数字、- 和 _
LEGACY_ID_RE = re.compile(r"^\d{8}-\d{6}\Z")
API_CHOICES = ("auto", "responses", "chat")
LIMITS = {"name": 40, "models": 300, "model_len": 120}
_lock = threading.RLock()
_hist_cache = {}


def now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 接口

def _ep_file():
    return os.path.join(DATA_DIR, "endpoints.json")


def _load():
    try:
        with open(_ep_file(), encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"schema_version": 1, "endpoints": []}


def _save(data):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = _ep_file() + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, _ep_file())


def to_public(ep):
    out = {k: v for k, v in ep.items() if k != "api_key"}
    k = ep.get("api_key") or ""
    out["key_hint"] = ("…" + k[-4:]) if len(k) > 8 else ("已设置" if k else "未设置")
    return out


def list_endpoints(public=True):  # noqa: A002  public=False 时带密钥，仅服务端内部使用
    with _lock:
        eps = _load()["endpoints"]
    return [to_public(e) for e in eps] if public else eps


def get_endpoint(eid):
    return next((e for e in list_endpoints(public=False) if e["id"] == eid), None)


def _clean_models(models):
    if not isinstance(models, list):
        raise ValueError("模型列表格式不对")
    out = []
    for m in models:
        if not isinstance(m, str) or not m.strip() or len(m.strip()) > LIMITS["model_len"]:
            raise ValueError("模型名不能为空，最长 %d 字" % LIMITS["model_len"])
        out.append(m.strip())
    out = list(dict.fromkeys(out))
    if len(out) > LIMITS["models"]:
        raise ValueError("模型最多 %d 个" % LIMITS["models"])
    return out


def upsert_endpoint(body):
    name = (body.get("name") or "").strip() if isinstance(body.get("name"), str) else ""
    if not name or len(name) > LIMITS["name"] or "/" in name:
        raise ValueError("名称必填，最长 %d 字，不能含 /" % LIMITS["name"])
    base = body.get("base_url")
    if not isinstance(base, str):
        raise ValueError("请填写接口地址")
    core.endpoint_parts(base)  # 校验
    api = body.get("api", "auto")
    if api not in API_CHOICES:
        raise ValueError("接口类型只能是 auto / responses / chat")
    key = body.get("api_key")
    if key is not None and (not isinstance(key, str) or len(key) > 400 or any(c.isspace() for c in key)):
        raise ValueError("密钥格式不对（不能含空白）")
    with _lock:
        data = _load()
        eps = data["endpoints"]
        cur = next((e for e in eps if e["id"] == body.get("id")), None) if body.get("id") else None
        if body.get("id") and not cur:
            raise ValueError("接口不存在")
        if any(e["name"] == name and e is not cur for e in eps):
            raise ValueError("已有同名接口")
        if not cur and not key:
            raise ValueError("新增接口必须填写密钥")
        if cur is None:
            cur = {"id": "ep-" + secrets.token_hex(4), "models": [], "api_detected": None, "models_checked_at": None}
            eps.append(cur)
        cur.update(name=name, base_url=base.strip().rstrip("/"), api=api, note=str(body.get("note") or "")[:200])
        if key:
            cur["api_key"] = key
        if "models" in body:
            cur["models"] = _clean_models(body["models"])
        if api != "auto":
            cur["api_detected"] = None
        _save(data)
        return to_public(cur)


def delete_endpoint(eid):
    with _lock:
        data = _load()
        n = len(data["endpoints"])
        data["endpoints"] = [e for e in data["endpoints"] if e["id"] != eid]
        if len(data["endpoints"]) == n:
            raise ValueError("接口不存在")
        _save(data)


def effective_api(ep):
    api = ep.get("api")
    if api in ("responses", "chat"):
        return api
    if ep.get("api_detected") in ("responses", "chat"):
        return ep["api_detected"]
    raise ValueError(f"接口“{ep['name']}”的类型还没确定，请先点“探测”，或手动指定 responses / chat")


def detect_api(ep, model, timeout=40):
    """用一个模型发最小请求，先试 responses，再试 chat。返回 (api 或 None, 错误说明列表)。"""
    pool, errs = core.Pool(), []
    try:
        for api in ("responses", "chat"):
            r = core.stream_request(pool, dict(ep, api=api), core.make_body(api, model, core.DEFAULT_PROMPT), timeout)
            if r["ok"]:
                return api, errs
            errs.append(f"{api}: HTTP {r['status']} {r['error'][:80]}")
    finally:
        pool.close()
    return None, errs


def discover(eid):
    """读取模型列表并保存；接口类型为 auto 时再探测一次。"""
    ep = get_endpoint(eid)
    if not ep:
        raise ValueError("接口不存在")
    result = {"models": None, "api": ep.get("api_detected") or ep["api"], "errors": []}
    ids = None
    try:
        ids = core.fetch_models(ep)
    except Exception as e:  # noqa: BLE001
        result["errors"].append(f"读取模型列表失败：{type(e).__name__}: {e}")
    models = ids if ids is not None else ep["models"]
    detected = ep.get("api_detected")
    if ep["api"] == "auto" and models:
        detected, errs = detect_api(ep, models[0])
        result["errors"] += ["探测接口类型失败：" + "；".join(errs)] if not detected else []
    with _lock:
        data = _load()
        cur = next((e for e in data["endpoints"] if e["id"] == eid), None)
        if cur:
            if ids is not None:
                cur["models"], cur["models_checked_at"] = ids, now_iso()
            if ep["api"] == "auto":
                cur["api_detected"] = detected
            _save(data)
    result.update(models=len(models or []), api=detected if ep["api"] == "auto" else ep["api"])
    return result


def import_openclaw():
    """从 OpenClaw 配置导入供应商（只读配置）：名称相同的更新，其余新增。返回导入数量。"""
    with open(OPENCLAW_CONFIG, encoding="utf-8") as f:
        providers = json.load(f).get("models", {}).get("providers", {})
    n = 0
    for pid, p in providers.items():
        if not isinstance(p.get("apiKey"), str) or not p.get("apiKey") or not p.get("baseUrl"):
            continue
        models = [(m.get("id") if isinstance(m, dict) else m) for m in p.get("models", [])]
        cur = next((e for e in list_endpoints(public=False) if e["name"] == pid), None)
        api = {"openai-responses": "responses", "openai-completions": "chat"}.get(p.get("api"), "auto")
        upsert_endpoint({"id": cur["id"] if cur else None, "name": pid, "base_url": p["baseUrl"], "api_key": p["apiKey"], "api": api,
                         "models": [m for m in models if isinstance(m, str)]})
        n += 1
    return n


# ---------------------------------------------------------------- 结果

def run_dir(run_id):
    return os.path.join(RESULTS_DIR, run_id)


def new_run_dir(name=None):
    """创建 results/<日期>-<当天第几次>[_测试名称]/，返回 run_id。序号按当天已有文件夹取最大值 +1，与名称无关。"""
    if name and not NAME_RE.match(name):
        raise ValueError("测试名称只能含中文、字母、数字、- 和 _，最长 30 字")
    day = dt.date.today().isoformat()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with _lock:
        nums = [int(m.group(1)) for n in os.listdir(RESULTS_DIR) for m in [re.fullmatch(re.escape(day) + r"-(\d+)(?:_.*)?", n)] if m]
        run_id = f"{day}-{max(nums, default=0) + 1}" + (f"_{name}" if name else "")
        os.makedirs(run_dir(run_id))
    return run_id


def save_result(result, records):
    d = run_dir(result["id"])
    result = dict(result, records=[core.public_rec(r, 20000) for r in records])
    full = {"schema_version": core.SCHEMA_VERSION, "id": result["id"],
            "records": [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]}
    paths = {"result.json": result, "raw.json": full}
    for name, obj in paths.items():
        p = os.path.join(d, name)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)
    core.write_report(os.path.join(d, "report.md"), result)
    for name in ("result.json", "raw.json", "report.md"):
        os.chmod(os.path.join(d, name), 0o444)  # 保存后不再改动


def _cond(env, sc):
    """同一条趋势线必须“条件相同”：推理强度 + （固定内容的场景用提示词指纹；自定义提示词连同背景长度一起算进指纹）。"""
    eff = env.get("effort") or ""
    fp = (env.get("prompt_fingerprints") or {}).get(sc, "")
    label = "强度 " + (eff or "未指定")
    if sc == "speed":
        label = f"提示词「{(env.get('prompt') or '')[:14]}」 · 背景 {env.get('context_tokens') or '无'} · " + label
    return f"{eff}|{fp}", label


def _meta_new(run_id, d):
    env = d.get("env", {})
    return {"id": run_id, "created": env.get("created", ""), "scenarios": env.get("scenarios", []), "models": env.get("models", []),
            "name": env.get("name", ""), "n": len(d.get("records", [])), "state": d.get("state", "done"), "rounds": env.get("rounds"), "legacy": False,
            "brief": [dict(zip(("cond", "cond_label"), _cond(env, r["scenario"])), label=r["label"], scenario=r["scenario"],
                           total_median=(r["total"] or {}).get("median"), pass_rate=r["pass_rate"]) for r in d.get("summary", [])]}


def _seq(run_id):
    m = RUN_ID_RE.match(run_id)
    return int(run_id.split("-")[3].split("_")[0]) if m else 0


def history(limit=60):
    out = []
    if os.path.isdir(RESULTS_DIR):
        for n in os.listdir(RESULTS_DIR):
            p = os.path.join(RESULTS_DIR, n, "result.json")
            if RUN_ID_RE.match(n) and os.path.isfile(p):
                m = os.path.getmtime(p)
                if _hist_cache.get(n, (0,))[0] != m:
                    try:
                        with open(p, encoding="utf-8") as f:
                            _hist_cache[n] = (m, _meta_new(n, json.load(f)))
                    except Exception:  # noqa: BLE001
                        continue
                out.append(_hist_cache[n][1])
    if os.path.isdir(LEGACY_DIR):
        for n in os.listdir(LEGACY_DIR):
            if n.endswith(".json") and LEGACY_ID_RE.match(n[:-5]):
                s = n[:-5]
                out.append({"id": s, "created": f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:]}", "scenarios": ["speed"], "models": [], "name": "",
                            "n": None, "state": "done", "rounds": None, "legacy": True})
    out.sort(key=lambda r: (r["created"], _seq(r["id"])), reverse=True)  # 同一秒创建的两次测试，用当天序号决定先后
    return out[:limit]


def load_run(run_id):
    if RUN_ID_RE.match(run_id):
        with open(os.path.join(run_dir(run_id), "result.json"), encoding="utf-8") as f:
            return json.load(f)
    if LEGACY_ID_RE.match(run_id):
        with open(os.path.join(LEGACY_DIR, run_id + ".json"), encoding="utf-8") as f:
            d = json.load(f)
        recs = [core.public_rec(dict(r, label=r.get("label") or r["model"], scenario="speed")) for r in d.get("records", [])]
        rows = core.summarize(recs)
        return {"id": run_id, "legacy": True, "env": {"created": history_created(run_id), "scenarios": ["speed"], "prompt": d.get("prompt", ""),
                                                       "context_tokens": d.get("context_tokens", 0), "tool_version": "1.x", "endpoints": [], "models": []},
                "summary": rows, "records": recs, "conclusion": core.conclusion(rows)}
    raise FileNotFoundError(run_id)


def merge_runs(run_ids, limit=12):
    """读取多次测试并合并统计。run_ids 已由调用方校验格式；缺文件的结果跳过并记入 missing。"""
    ids = list(dict.fromkeys(i for i in run_ids if isinstance(i, str)))[:limit]
    if len(ids) < 2:
        raise ValueError("合并至少需要 2 次测试")
    runs, missing = [], []
    for rid in ids:
        try:
            d = load_run(rid)
        except FileNotFoundError:
            missing.append(rid)
            continue
        except Exception:  # noqa: BLE001
            missing.append(rid)
            continue
        if d.get("legacy"):
            missing.append(rid)  # 旧版结果缺少采样参数/指纹等条件信息，合并会不可比
            continue
        runs.append(d)
    if len(runs) < 2:
        raise ValueError("可合并的测试不足 2 次（" + "；".join(f"{m} 缺条件信息或已删除" for m in missing) + "）" if missing else "可合并的测试不足 2 次")
    res = core.merge_analysis(runs)
    if not res:
        raise ValueError("这些测试里没有可合并的请求记录")
    res["missing"] = missing
    res["total_requests"] = sum(len(r.get("records") or []) for r in runs)
    res["runs_meta"] = [{"id": r.get("id"), "created": (r.get("env") or {}).get("created"),
                         "scenarios": (r.get("env") or {}).get("scenarios") or []} for r in runs]
    return res


def history_created(s):
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:]}"


def raw_path(run_id):
    if RUN_ID_RE.match(run_id):
        return os.path.join(run_dir(run_id), "raw.json")
    if LEGACY_ID_RE.match(run_id):
        return os.path.join(LEGACY_DIR, run_id + ".json")
    raise FileNotFoundError(run_id)


def delete_run(run_id):
    """删除一次测试结果（整个文件夹移到 results/.trash/，不直接删除）。"""
    if not RUN_ID_RE.match(run_id) or not os.path.isdir(run_dir(run_id)):
        raise ValueError("结果不存在")
    trash = os.path.join(RESULTS_DIR, ".trash")
    os.makedirs(trash, exist_ok=True)
    dst = os.path.join(trash, run_id + "-" + secrets.token_hex(2))
    for root, _, files in os.walk(run_dir(run_id)):
        for fn in files:
            os.chmod(os.path.join(root, fn), 0o644)
    shutil.move(run_dir(run_id), dst)
    _hist_cache.pop(run_id, None)


# ---------------------------------------------------------------- 执行

def execute(params, emit, stop):
    """params 已校验：selection[{endpoint, models}]、scenarios、prompt、rounds、timeout、context、gap、warmup。保存结果并返回 result。"""
    all_eps = {e["id"]: e for e in list_endpoints(public=False)}
    models = [(s["endpoint"], m) for s in params["selection"] for m in s["models"]]
    plan_eps = {}
    for eid in dict.fromkeys(e for e, _ in models):
        ep = dict(all_eps[eid])
        ep["api"] = effective_api(ep)
        plan_eps[eid] = ep
    plan = {"endpoints": plan_eps, "models": models, "tasks": core.build_tasks(params["scenarios"], params["prompt"], params["context"]),
            "rounds": params["rounds"], "timeout": params["timeout"], "gap": params["gap"], "warmup": params["warmup"],
            "effort": params.get("effort", ""), "seed": random.randrange(1 << 31)}
    run_id = new_run_dir(params.get("name"))
    created = now_iso()
    emit({"run_id": run_id, "total": plan["rounds"] * len(models) * len(plan["tasks"]), "done": 0})
    try:
        records, extra = core.run_plan(plan, emit, stop)
    except BaseException:
        shutil.rmtree(run_dir(run_id), ignore_errors=True)
        raise
    env = core.make_env(plan, extra, {"created": created, "prompt": params["prompt"], "context": params["context"], "name": params.get("name", "")})
    rows = core.summarize(records)
    fp = core.fp_analysis(records)
    result = {"id": run_id, "state": "stopped" if stop.is_set() else "done", "env": env, "summary": rows, "fingerprint": fp,
              "conclusion": core.conclusion(rows, fp) if rows else ""}
    if records:
        save_result(result, records)
    else:
        shutil.rmtree(run_dir(run_id), ignore_errors=True)
        result["id"] = None
    result["records"] = [core.public_rec(r) for r in records]
    return result
