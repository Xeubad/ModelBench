#!/usr/bin/env python3
"""模型测试平台（本机网页）：添加任意 OpenAI 兼容接口（地址 + 密钥），探测模型列表，选模型和场景，测完在右侧看数据。

用法：python3 -m model_test serve [--port 18888] [--open]    或    ./start.sh（在 ModelBench 目录下）
然后打开 http://127.0.0.1:18888

设计要点（详见同目录 README.md）：
- 监听地址和端口见 config.json（默认只绑 127.0.0.1；需要远程组网访问时自行在 bind_hosts 追加）；检查 Host 和 Origin，POST 必须是 JSON，防止别的网站借浏览器触发测试。
- 接口信息（含密钥）存在 data/endpoints.json（权限 600），不发给页面；每次测试结果存在 results/<日期>-<次数>/，不含密钥。
- 页面只能触发固定操作；参数全部白名单校验；一次只跑一个任务；测试期间不允许探测/自检（避免干扰测量）。
- 不改 OpenClaw 配置（“导入”只读取它），不经过网关，不影响微信用户。
"""
import argparse
import base64
import json
import os
import re
import secrets
import socket
import sys
import subprocess
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config, core, selftest, store

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dashboard.html")
_L = config.CFG["limits"]
MAX_PROMPT, MAX_ROUNDS, MAX_TIMEOUT, MAX_REQUESTS, MAX_BODY = (_L["max_prompt"], _L["max_rounds"], _L["max_timeout"], _L["max_requests"], _L["max_body"])
MAX_CMP_MERGE = 20  # 一次最多合并多少次测试结果（只读，代价很低，但太多也没意义）
PORT = config.CFG["server"]["port"]
BIND_HOSTS = list(config.CFG["server"]["bind_hosts"])  # 第一个是本机；其余（如远程组网地址）绑定失败就跳过
LOCK = threading.Lock()
SELFTEST_LOCK = threading.Lock()
JOB = {"state": "idle"}
STOP = threading.Event()


LAN = {"srv": None, "host": None, "port": None, "token": None}  # 局域网访问：会话级，重启即失效


def running():
    return JOB.get("state") in ("starting", "running")


def lan_ips():
    """本机可用的局域网 IPv4 地址（排除回环）。UDP connect 只为问内核要默认出口网卡的地址，不实际发包。"""
    found = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.settimeout(0.2)
            s.connect(("8.8.8.8", 80))
            found.append(s.getsockname()[0])
        finally:
            s.close()
    except OSError:
        pass
    try:
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            found.append(ip)
    except OSError:
        pass
    out = []
    for ip in found:
        if ip == "0.0.0.0" or ip.startswith("127.") or ip in out:
            continue
        try:
            socket.inet_aton(ip)
        except OSError:
            continue
        out.append(ip)
    return out


def lan_denied_html():
    """局域网访问没带令牌时给的提示页（比 JSON 报错友好，也说明怎么拿令牌）。"""
    return ("<!doctype html><meta charset=\"utf-8\"><title>需要访问令牌</title>"
            "<body style=\"font:14px/1.8 -apple-system,PingFang SC,sans-serif;background:#0a0d12;color:#dce3ed;"
            "max-width:520px;margin:14vh auto;padding:0 24px\">"
            "<h2 style=\"color:#ffb454;font-weight:600\">需要访问令牌</h2>"
            "<p>这个 ModelBench 实例允许局域网访问，但每个设备都要用<b>带令牌的地址</b>打开。</p>"
            "<p>请在运行 ModelBench 的那台电脑上，打开它的「局域网访问」设置，复制完整地址"
            "（形如 <code style=\"background:#151b28;padding:2px 6px;border-radius:3px\">http://192.168.x.x:18889/?k=…</code>）"
            "再打开。</p>"
            "<p style=\"color:#6b7789;font-size:12.5px\">令牌只在本机启动后有效，关闭局域网访问或重启服务即失效。</p>"
            "</body>").encode("utf-8")


def lan_start(port):
    """在检测到的局域网地址上开额外监听。返回可访问地址；失败抛 ValueError。"""
    if LAN["srv"] is not None:
        return lan_info()
    ips = lan_ips()
    if not ips:
        raise ValueError("没有检测到局域网地址（本机可能没连网络）")
    errs = []
    for host in ips:  # 逐个试，跳过不可绑定的（如已被占用的虚拟网卡）
        try:
            srv = ThreadingHTTPServer((host, port), Handler)
        except OSError as e:
            errs.append(f"{host}（{e}）")
            continue
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        LAN.update(srv=srv, host=host, port=port, token=base64.urlsafe_b64encode(os.urandom(18)).decode().rstrip("="))
        return lan_info()
    raise ValueError("局域网地址都绑定不上：" + "；".join(errs))


def lan_stop():
    srv = LAN.get("srv")
    if srv is not None:
        try:
            srv.shutdown()
            srv.server_close()
        except Exception:  # noqa: BLE001
            pass
    LAN.update(srv=None, host=None, port=None, token=None)


def lan_info():
    """给页面看的局域网状态。令牌只回传给本机请求，不回传给已经连上的局域网设备。"""
    on = LAN["srv"] is not None
    return {"enabled": on, "host": LAN["host"], "port": LAN["port"], "token": LAN["token"],
            "url": (f"http://{LAN['host']}:{LAN['port']}/?k={LAN['token']}" if on else None),
            "candidates": lan_ips()}


def is_lan_addr(addr):
    """这个监听地址是不是对外（非回环）。"""
    return bool(addr) and not str(addr).startswith("127.")


def run_job(params):
    def emit(u):
        with LOCK:
            for k, v in u.items():
                if k == "records":
                    snap = list(v)
                    JOB["records"], JOB["rows"] = [core.public_rec(r) for r in snap], core.summarize(snap)
                else:
                    JOB[k] = v
            if u.get("phase") == "warmup" or "total" in u:
                JOB["state"] = "running"

    try:
        res = store.execute(params, emit, STOP)
        with LOCK:
            JOB.update(state=res["state"], rows=res["summary"], conclusion=res["conclusion"], run_id=res["id"], records=res["records"],
                       fingerprint=res.get("fingerprint"), env=res.get("env"), current="")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            JOB.update(state="error", error=f"{type(e).__name__}: {e}")


def validate(body):
    scen = body.get("scenarios")
    if not isinstance(scen, list) or not scen or any(s not in core.SCENARIOS for s in scen):
        raise ValueError("请至少选择一个测试场景")
    scen = list(dict.fromkeys(scen))
    prompt = body.get("prompt", "")
    if "speed" in scen and (not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT):
        raise ValueError(f"自定义提示词不能为空，最多 {MAX_PROMPT} 字")
    sel_in, selection = body.get("selection"), []
    if not isinstance(sel_in, list):
        raise ValueError("请选择要测试的模型")
    for s in sel_in:
        ep = store.get_endpoint(s.get("endpoint")) if isinstance(s, dict) and isinstance(s.get("endpoint"), str) else None
        if not ep:
            raise ValueError("所选接口不存在")
        ms = s.get("models")
        if not isinstance(ms, list) or any(not isinstance(m, str) or m not in ep["models"] for m in ms):
            raise ValueError(f"接口“{ep['name']}”的模型必须在它已保存的模型列表里")
        if ms:
            store.effective_api(ep)
            selection.append({"endpoint": ep["id"], "models": list(dict.fromkeys(ms))})
    if not selection:
        raise ValueError("请选择要测试的模型")
    rounds, timeout, context, gap = body.get("rounds"), body.get("timeout"), body.get("context", 0), body.get("gap", 1.5)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or not 1 <= rounds <= MAX_ROUNDS:
        raise ValueError(f"轮数范围 1–{MAX_ROUNDS}")
    if not isinstance(timeout, int) or isinstance(timeout, bool) or not 10 <= timeout <= MAX_TIMEOUT:
        raise ValueError(f"超时范围 10–{MAX_TIMEOUT} 秒")
    if context not in core.CONTEXT_CHOICES:
        raise ValueError("背景文字长度只能是 " + "、".join(map(str, core.CONTEXT_CHOICES)))
    if not isinstance(gap, (int, float)) or isinstance(gap, bool) or not 0 <= gap <= 10:
        raise ValueError("请求间隔范围 0–10 秒")
    name = body.get("name") or ""
    if name and (not isinstance(name, str) or not store.NAME_RE.match(name.strip())):
        raise ValueError("测试名称只能含中文、字母、数字、- 和 _，最长 30 字")
    effort = body.get("effort", "")
    if effort not in core.EFFORTS:
        raise ValueError("推理强度只能是 不指定 / low / medium / high")
    concurrency = body.get("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or not 1 <= concurrency <= 4:
        raise ValueError("并发范围 1–4")
    connection_mode = body.get("connection_mode", "reuse")
    if connection_mode not in ("reuse", "cold"):
        raise ValueError("连接模式只能为 reuse / cold")
    seed = body.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 31):
        raise ValueError("种子必须是 0 到 2147483647 的整数或留空")
    tasks = core.build_tasks(scen, "x", 0)
    n = rounds * sum(len(s["models"]) for s in selection) * sum(2 if t["kind"] == "tool" else 1 for t in tasks)
    n += sum(len(s["models"]) for s in selection) if body.get("warmup", True) is not False else 0
    if n > MAX_REQUESTS:
        raise ValueError(f"一次最多 {MAX_REQUESTS} 个预定请求（含预热与工具两步，不含连接重试），当前 {n}")
    return {"selection": selection, "scenarios": scen, "prompt": (prompt or "").strip(), "rounds": rounds, "timeout": timeout,
            "context": context, "gap": float(gap), "name": name.strip(), "effort": effort, "warmup": body.get("warmup", True) is not False,
            "concurrency": concurrency, "connection_mode": connection_mode, "seed": seed}


def public_config():
    per = {sc: len(core.build_tasks([sc], "x", 0)) for sc in core.SCENARIOS}
    return {"version": core.VERSION, "scenarios": core.SCENARIOS, "tasks_per_scenario": per, "context_choices": list(core.CONTEXT_CHOICES),
            "efforts": [e for e in core.EFFORTS if e], "p95_min_n": core.P95_MIN_N, "defaults": config.CFG["defaults"], "max_rounds": MAX_ROUNDS, "max_timeout": MAX_TIMEOUT, "max_requests": MAX_REQUESTS, "low_sample": core.LOW_SAMPLE,
            "quiz": [{"id": q["id"], "name": q["name"]} for q in core.QUIZ]}


class Handler(BaseHTTPRequestHandler):
    server_version = "ModelTest/2.0"

    def log_message(self, fmt, *a):  # 不把请求内容写到终端
        pass

    def _allowed_hosts(self):
        hosts = list(BIND_HOSTS) + ["localhost"]
        if LAN["srv"] is not None:
            hosts.append(LAN["host"])
        return [f"{h}:{PORT}" for h in hosts if h]

    def _is_lan_request(self):
        """这个请求走的是对外监听（局域网），还是只监听本机的那个。"""
        addr = (getattr(self.server, "server_address", None) or (None,))[0]
        return is_lan_addr(addr)

    def _token_ok(self):
        """局域网请求的令牌校验：先看 cookie，再看 ?k= 查询参数。"""
        want = LAN.get("token")
        if not want:
            return False
        for part in (self.headers.get("Cookie") or "").split(";"):
            name, _, val = part.strip().partition("=")
            if name == "mbtoken" and val == want:
                return True
        qs = urllib.parse.urlsplit(self.path).query
        return urllib.parse.parse_qs(qs).get("k", [""])[0] == want

    def _ok_origin(self):
        host = (self.headers.get("Host") or "").lower()
        allowed = self._allowed_hosts()
        if host not in allowed:
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in [f"http://{a}" for a in allowed]:
            return False
        if self._is_lan_request():
            if not self._token_ok():
                return False
            # 用 ?k= 打开过之后种 cookie，后续同源请求就不用一直带参数了
            if "mbtoken=" not in (self.headers.get("Cookie") or ""):
                self._set_token_cookie = True
        return True

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        data = body if isinstance(body, bytes) else (body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)).encode("utf-8")
        extra = dict(extra or {})
        if getattr(self, "_set_token_cookie", False) and LAN.get("token"):
            extra["Set-Cookie"] = f"mbtoken={LAN['token']}; Path=/; SameSite=Lax; Max-Age=86400"
            self._set_token_cookie = False
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._ok_origin():
            if self._is_lan_request():
                return self._send(403, lan_denied_html(), "text/html; charset=utf-8")
            return self._send(403, {"error": "forbidden"})
        path = self.path.split("?")[0]
        try:
            if path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if path == "/api/config":
                return self._send(200, public_config())
            if path == "/api/endpoints":
                return self._send(200, {"endpoints": store.list_endpoints()})
            if path == "/api/job":
                with LOCK:
                    return self._send(200, dict(JOB))
            if path == "/api/history":
                return self._send(200, {"runs": store.history()})
            if path == "/api/network":
                info = lan_info()
                if self._is_lan_request():
                    info["token"] = info["url"] = None  # 令牌只回给本机，避免已连上的设备互相串联
                return self._send(200, info)
            upath = urllib.parse.unquote(path)  # 测试名称可含中文，URL 里是百分号编码
            m = re.match(r"^/api/run/([^/]+)$", upath)
            if m:
                return self._send(200, store.load_run(m.group(1)))
            m = re.match(r"^/raw/([^/]+)\.json$", upath)
            if m:
                with open(store.raw_path(m.group(1)), "rb") as f:
                    fn = urllib.parse.quote(f"model-test-{m.group(1)}.json")
                    return self._send(200, f.read(), extra={"Content-Disposition": f"attachment; filename=\"model-test-raw.json\"; filename*=UTF-8''{fn}"})
            return self._send(404, {"error": "not found"})
        except FileNotFoundError:
            return self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        if not self._ok_origin():
            return self._send(403, {"error": "forbidden"})
        if "application/json" not in (self.headers.get("Content-Type") or ""):
            return self._send(415, {"error": "需要 application/json"})
        path = self.path.split("?")[0]
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._send(413, {"error": "请求太大"})
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("请求格式不对")
            if path == "/api/run":
                if SELFTEST_LOCK.locked():
                    return self._send(409, {"error": "自检正在运行，请稍后"})
                params = validate(body)
                with LOCK:
                    if running():
                        return self._send(409, {"error": "已有测试在运行"})
                    JOB.clear()
                    JOB.update(state="starting", total=0, done=0, current="", records=[], rows=[], conclusion="", error="", run_id=None, fingerprint=None,
                               params={k: params[k] for k in ("scenarios", "prompt", "rounds", "timeout", "context", "gap", "warmup", "name", "effort")})
                STOP.clear()
                threading.Thread(target=run_job, args=(params,), daemon=True).start()
                return self._send(200, {"ok": True})
            if path == "/api/stop":
                STOP.set()
                return self._send(200, {"ok": True})
            if path in ("/api/endpoints/probe", "/api/selftest") and running():
                return self._send(409, {"error": "测试运行期间不能探测或自检（会干扰测量）"})
            if path == "/api/endpoints":
                return self._send(200, {"endpoint": store.upsert_endpoint(body)})
            if path == "/api/endpoints/delete":
                store.delete_endpoint(body.get("id"))
                return self._send(200, {"ok": True})
            if path == "/api/endpoints/probe":
                return self._send(200, store.discover(body.get("id")))
            if path == "/api/endpoints/import":
                return self._send(200, {"imported": store.import_openclaw()})
            if path == "/api/run/delete":
                store.delete_run(str(body.get("id")))
                return self._send(200, {"ok": True})
            if path == "/api/network":
                want = body.get("enable")
                if not isinstance(want, bool):
                    raise ValueError("参数 enable 必须是 true 或 false")
                if want:
                    info = lan_start(PORT)
                    info["candidates"] = lan_ips()
                else:
                    lan_stop()
                    info = lan_info()
                info["warning"] = ("页面没有登录，同网段任何人拿到这个地址都能使用你保存的接口密钥。" if want else None)
                return self._send(200, info)
            if path == "/api/merge":
                ids = body.get("ids")
                if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                    raise ValueError("请选择要合并的测试结果")
                return self._send(200, store.merge_runs(ids, limit=MAX_CMP_MERGE))
            if path == "/api/selftest":
                if not SELFTEST_LOCK.acquire(blocking=False):
                    return self._send(409, {"error": "自检正在运行"})
                try:
                    return self._send(200, {"checks": selftest.run_checks()})
                finally:
                    SELFTEST_LOCK.release()
            return self._send(404, {"error": "not found"})
        except (ValueError, FileNotFoundError) as e:
            return self._send(400, {"error": str(e)})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})


def make_servers(port):
    """本机必须成功；其余地址（远程组网）绑定失败就跳过。返回 (主服务, 附加服务列表, 提示列表)。"""
    main_srv = ThreadingHTTPServer((BIND_HOSTS[0], port), Handler)
    extra, notes = [], []
    for h in BIND_HOSTS[1:]:
        try:
            extra.append(ThreadingHTTPServer((h, port), Handler))
        except OSError as e:
            notes.append(f"未监听 {h}:{port}（{e}）")
    return main_srv, extra, notes


def main():
    global PORT
    try:
        sys.stdout.reconfigure(line_buffering=True)  # 重定向到日志文件时也能立刻看到启动提示
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(prog="python3 -m model_test serve", description="模型测试平台（网页）")
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()
    PORT = args.port
    srv, extra, notes = make_servers(PORT)
    for e in extra:
        threading.Thread(target=e.serve_forever, daemon=True).start()
        print(f"同时监听 http://{e.server_address[0]}:{PORT}")
    for n in notes:
        print(n)
    url = f"http://127.0.0.1:{PORT}"
    ips = lan_ips()
    if ips:
        print("局域网访问：在页面“局域网访问”里打开开关（默认关闭，开启后生成带令牌的地址）")
        print("  本机可用地址：" + "、".join(f"http://{ip}:{PORT}" for ip in ips))
    print(f"模型测试平台已启动：{url}（Ctrl+C 退出）")
    if args.open:  # 用 macOS 的 open，不走 webbrowser（在 VS Code 终端里它会连 VS Code 的已失效通道而报错）
        subprocess.run(["open", url], check=False)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已退出")
