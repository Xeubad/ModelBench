"""Amazon Bedrock ConverseStream、AWS SigV4 与事件流；只使用 Python 标准库。"""
import base64
import configparser
import datetime as dt
import hashlib
import hmac
import ipaddress
import json
import os
import re
import struct
import urllib.parse
import zlib

MAX_EVENT_BYTES = 16 * 1024 * 1024
DEFAULT_REGION = "us-east-1"


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_settings(region, profile, base_url=None):
    if not isinstance(region, str) or not re.fullmatch(r"[a-z]{2}(?:-[a-z0-9]+){1,4}-[0-9]+", region):
        raise ValueError("AWS 区域无效，例如 us-east-1")
    if not isinstance(profile, str) or (profile and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", profile)):
        raise ValueError("AWS profile 名称无效")
    if base_url:
        parsed = urllib.parse.urlsplit(base_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme == "http" and not _loopback(host):
            raise ValueError("Bedrock 凭据请求必须使用 HTTPS（本机回环测试除外）")
        match = re.fullmatch(r"bedrock(?:-runtime)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?", host)
        if match and match.group(1) != region:
            raise ValueError("AWS 区域与 Bedrock 接口地址不一致")


def resolve_credentials(profile=""):
    """显式 profile 使用共享凭据文件；否则优先环境变量，再使用默认 profile。"""
    validate_settings(DEFAULT_REGION, profile)
    if not profile:
        access = os.environ.get("AWS_ACCESS_KEY_ID", "")
        secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        token = os.environ.get("AWS_SESSION_TOKEN", "")
        if access or secret or token:
            if not access or not secret:
                raise ValueError("AWS 环境变量必须同时提供 Access Key ID 和 Secret Access Key")
            return dict(_validate_credentials(access, secret, token), source="environment")
        profile = os.environ.get("AWS_PROFILE") or "default"
        validate_settings(DEFAULT_REGION, profile)
    path = os.path.expanduser(os.environ.get("AWS_SHARED_CREDENTIALS_FILE") or "~/.aws/credentials")
    parser = configparser.RawConfigParser(interpolation=None)
    try:
        loaded = parser.read(path, encoding="utf-8")
        if not loaded or not parser.has_section(profile):
            raise ValueError("找不到 AWS profile；请配置环境变量或共享凭据文件")
        access = parser.get(profile, "aws_access_key_id", fallback="")
        secret = parser.get(profile, "aws_secret_access_key", fallback="")
        token = parser.get(profile, "aws_session_token", fallback="")
    except (OSError, configparser.Error) as exc:
        raise ValueError("无法读取 AWS 共享凭据文件") from exc
    return dict(_validate_credentials(access, secret, token), source="profile:" + profile)


def _validate_credentials(access, secret, token):
    for value, maximum in ((access, 256), (secret, 512), (token, 8192)):
        if not isinstance(value, str) or len(value) > maximum or any(c.isspace() or ord(c) < 32 for c in value):
            raise ValueError("AWS 凭据格式无效")
    if not access or not secret:
        raise ValueError("AWS profile 必须包含 Access Key ID 和 Secret Access Key；SSO/角色请先导出临时凭据")
    return {"access_key": access, "secret_key": secret, "session_token": token}


def secrets_of(credentials):
    return [credentials[k] for k in ("access_key", "secret_key", "session_token") if credentials.get(k)]


def request_path(base_url, model):
    if not isinstance(model, str) or not model or len(model) > 2048 or any(ord(c) < 33 or ord(c) == 127 for c in model):
        raise ValueError("Bedrock 模型 ID 或推理配置 ARN 无效")
    if model in (".", ".."):
        raise ValueError("Bedrock 模型 ID 无效")
    path = urllib.parse.urlsplit(base_url).path.rstrip("/")
    return path + "/model/" + urllib.parse.quote(model, safe="") + "/converse-stream"


def _canonical_path(path):
    parts = []
    for piece in (path or "/").split("/"):
        if not piece or piece == ".":
            continue
        if piece == "..":
            if parts:
                parts.pop()
        else:
            parts.append(piece)
    result = "/" + "/".join(parts)
    if path.endswith("/") and result != "/":
        result += "/"
    return urllib.parse.quote(result, safe="/~")


def sign_headers(method, url, body, credentials, region, now=None):
    """AWS Signature Version 4；凭据和签名仅用于构造请求头。"""
    validate_settings(region, "", url)
    credentials = _validate_credentials(credentials.get("access_key"), credentials.get("secret_key"), credentials.get("session_token") or "")
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise ValueError("AWS 接口地址无效")
    host = parsed.hostname.lower()
    if ":" in host:
        host = "[" + host + "]"
    if parsed.port and parsed.port != (443 if parsed.scheme == "https" else 80):
        host += ":" + str(parsed.port)
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    now = now.astimezone(dt.timezone.utc)
    timestamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    hdrs = {"content-type": "application/json", "host": host, "x-amz-date": timestamp}
    if credentials["session_token"]:
        hdrs["x-amz-security-token"] = credentials["session_token"]
    signed = ";".join(sorted(hdrs))
    canonical_headers = "".join(k + ":" + " ".join(hdrs[k].strip().split()) + "\n" for k in sorted(hdrs))
    pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    encoded = sorted((urllib.parse.quote(k, safe="-_.~"), urllib.parse.quote(v, safe="-_.~")) for k, v in pairs)
    query = "&".join(k + "=" + v for k, v in encoded)
    payload_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join((method.upper(), _canonical_path(parsed.path), query, canonical_headers, signed, payload_hash))
    scope = "/".join((day, region, "bedrock", "aws4_request"))
    to_sign = "\n".join(("AWS4-HMAC-SHA256", timestamp, scope, hashlib.sha256(canonical.encode()).hexdigest()))
    def digest(key, data):
        return hmac.new(key, data.encode(), hashlib.sha256).digest()
    derived = digest(digest(digest(digest(("AWS4" + credentials["secret_key"]).encode(), day), region), "bedrock"), "aws4_request")
    signature = hmac.new(derived, to_sign.encode(), hashlib.sha256).hexdigest()
    result = {"Host": host, "Content-Type": "application/json", "X-Amz-Date": timestamp,
              "Authorization": f"AWS4-HMAC-SHA256 Credential={credentials['access_key']}/{scope}, SignedHeaders={signed}, Signature={signature}"}
    if credentials["session_token"]:
        result["X-Amz-Security-Token"] = credentials["session_token"]
    return result


def control_url(base_url):
    parsed = urllib.parse.urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    if host.startswith("bedrock-runtime."):
        replacement = "bedrock." + host[len("bedrock-runtime."):]
        if parsed.port:
            replacement += ":" + str(parsed.port)
        return urllib.parse.urlunsplit((parsed.scheme, replacement, "", "", ""))
    if _loopback(host):
        return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))
    raise ValueError("自定义 Bedrock 地址无法自动推导模型目录，请手工填写模型 ID")


def fetch_models(ep, timeout, new_conn, redact):
    region = ep.get("aws_region", DEFAULT_REGION)
    validate_settings(region, ep.get("aws_profile", ""), ep["base_url"])
    credentials = ep.get("_aws_credentials") or resolve_credentials(ep.get("aws_profile", ""))
    base = control_url(ep["base_url"])
    parsed = urllib.parse.urlsplit(base)
    path = parsed.path.rstrip("/") + "/foundation-models"
    url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
    headers = sign_headers("GET", url, b"", credentials, region)
    protected = secrets_of(credentials) + [headers["Authorization"], headers["Authorization"].split("Signature=")[-1]]
    conn, _ = new_conn(dict(ep, base_url=base), timeout)
    try:
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        raw = response.read(2_000_001)
        if len(raw) > 2_000_000:
            raise ValueError("Bedrock 模型列表过大")
        if response.status != 200:
            text = raw.decode("utf-8", "replace")
            for secret in protected:
                text = redact(text, secret)
            raise RuntimeError("读取 Bedrock 模型列表失败：HTTP " + str(response.status) + " " + text[:160])
        data = json.loads(raw)
    finally:
        conn.close()
    if not isinstance(data, dict) or not isinstance(data.get("modelSummaries"), list):
        raise ValueError("Bedrock 模型列表格式不正确")
    found = set()
    for item in data["modelSummaries"]:
        if not isinstance(item, dict) or item.get("responseStreamingSupported") is False:
            continue
        modalities = item.get("outputModalities")
        if modalities is not None and (not isinstance(modalities, list) or "TEXT" not in modalities):
            continue
        model = item.get("modelId")
        if not isinstance(model, str):
            continue
        try:
            request_path(ep["base_url"], model)
        except ValueError:
            continue
        if all(redact(model, secret) == model for secret in protected):
            found.add(model)
    return sorted(found)


def make_body(model, prompt, system, max_tokens):
    body = {"model": model, "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens}}
    if system:
        body["system"] = [{"text": system}]
    return body


def _read_exact(stream, n):
    data = stream.read(n)
    if len(data) != n:
        raise ValueError("Bedrock 事件流被截断")
    return data


def read_events(stream):
    """读取 AWS event-stream 帧，并验证 prelude 和整帧 CRC32。"""
    while True:
        prelude = stream.read(12)
        if not prelude:
            return
        if len(prelude) != 12:
            raise ValueError("Bedrock 事件流前导数据被截断")
        total, header_len, prelude_crc = struct.unpack(">III", prelude)
        if total < 16 or total > MAX_EVENT_BYTES or header_len > total - 16:
            raise ValueError("Bedrock 事件流长度无效")
        if zlib.crc32(prelude[:8]) & 0xffffffff != prelude_crc:
            raise ValueError("Bedrock 事件流前导校验失败")
        remainder = _read_exact(stream, total - 12)
        raw = prelude + remainder
        if zlib.crc32(raw[:-4]) & 0xffffffff != struct.unpack(">I", raw[-4:])[0]:
            raise ValueError("Bedrock 事件流消息校验失败")
        data, headers, offset = raw[12:12 + header_len], {}, 0
        while offset < len(data):
            name_len = data[offset]
            offset += 1
            if not name_len or offset + name_len + 1 > len(data):
                raise ValueError("Bedrock 事件流头格式无效")
            name = data[offset:offset + name_len].decode("utf-8")
            offset += name_len
            typ = data[offset]
            offset += 1
            if typ in (0, 1):
                value = typ == 0
            elif typ in (6, 7):
                if offset + 2 > len(data):
                    raise ValueError("Bedrock 事件流头被截断")
                length = struct.unpack(">H", data[offset:offset + 2])[0]
                offset += 2
                if offset + length > len(data):
                    raise ValueError("Bedrock 事件流头被截断")
                value = data[offset:offset + length]
                offset += length
                if typ == 7:
                    value = value.decode("utf-8")
            else:
                sizes = {2: 1, 3: 2, 4: 4, 5: 8, 8: 8, 9: 16}
                size = sizes.get(typ)
                if size is None or offset + size > len(data):
                    raise ValueError("Bedrock 事件流头类型无效")
                value = data[offset:offset + size]
                offset += size
            if name in headers:
                raise ValueError("Bedrock 事件流头重复")
            headers[name] = value
        payload = raw[12 + header_len:-4]
        try:
            value = json.loads(payload) if payload else {}
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Bedrock 事件内容不是有效 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("Bedrock 事件内容格式无效")
        message_type = headers.get(":message-type")
        name = headers.get(":event-type") if message_type == "event" else headers.get(":exception-type") or headers.get(":error-code")
        if not isinstance(name, str) or message_type not in ("event", "exception", "error"):
            raise ValueError("Bedrock 事件类型无效")
        yield name, value, message_type


def _usage(data):
    return {"input_tokens": data.get("inputTokens"), "output_tokens": data.get("outputTokens"),
            "cached_tokens": data.get("cacheReadInputTokens"), "cache_creation_input_tokens": data.get("cacheWriteInputTokens"),
            "reasoning_tokens": None}


def consume(rec, kind, value, message_type, acc, now):
    if rec["first_event_s"] is None:
        rec["first_event_s"] = now
    if message_type != "event":
        rec["error"] = "Bedrock 流错误：" + str(kind)[:100] + " " + str(value.get("message", ""))[:150]
        return
    blocks = acc.setdefault("bedrock_blocks", {})
    opened = acc.setdefault("bedrock_open", set())
    if kind == "messageStart":
        if acc.get("bedrock_started"):
            rec["error"] = "Bedrock 消息重复开始"
        acc["bedrock_started"] = True
    elif kind == "contentBlockStart":
        index = value.get("contentBlockIndex")
        tool = (value.get("start") or {}).get("toolUse")
        if not acc.get("bedrock_started") or not isinstance(index, int) or not isinstance(tool, dict) or index in blocks:
            rec["error"] = "Bedrock 内容块开始事件无效"
            return
        blocks[index] = {"toolUse": {"toolUseId": tool.get("toolUseId", ""), "name": tool.get("name", ""), "input": {}}}
        acc["calls"][index] = {"id": tool.get("toolUseId", ""), "name": tool.get("name", ""), "arguments": ""}
        opened.add(index)
    elif kind == "contentBlockDelta":
        index, delta = value.get("contentBlockIndex"), value.get("delta") or {}
        if not acc.get("bedrock_started") or not isinstance(index, int) or not isinstance(delta, dict):
            rec["error"] = "Bedrock 内容增量无效"
            return
        if isinstance(delta.get("text"), str):
            text = delta["text"]
            block = blocks.setdefault(index, {"text": ""})
            if "text" not in block:
                rec["error"] = "Bedrock 内容块类型冲突"
                return
            block["text"] += text
            if text:
                if rec["first_token_s"] is None:
                    rec["first_token_s"] = now
                acc["text"].append(text)
        elif isinstance(delta.get("toolUse"), dict):
            if index not in opened or index not in acc["calls"]:
                rec["error"] = "Bedrock 工具增量缺少开始事件"
                return
            acc["calls"][index]["arguments"] += delta["toolUse"].get("input") or ""
        elif isinstance(delta.get("reasoningContent"), dict):
            reasoning = delta["reasoningContent"]
            block = blocks.setdefault(index, {"reasoningContent": {"reasoningText": {}}})
            if "reasoningContent" not in block:
                rec["error"] = "Bedrock 思考块类型冲突"
                return
            rec["reasoning_tokens_unknown"] = True
            if rec["first_think_s"] is None:
                rec["first_think_s"] = now
            if "redactedContent" in reasoning:
                try:
                    binary = base64.b64decode(reasoning["redactedContent"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise ValueError("Bedrock 加密思考块无效") from exc
                data = acc.setdefault("bedrock_redacted", {})
                data[index] = data.get(index, b"") + binary
                block["reasoningContent"] = {"redactedContent": base64.b64encode(data[index]).decode("ascii")}
            else:
                target = block["reasoningContent"].get("reasoningText")
                if target is None:
                    rec["error"] = "Bedrock 思考块类型冲突"
                    return
                for field in ("text", "signature"):
                    if field in reasoning:
                        target[field] = target.get(field, "") + (reasoning[field] or "")
    elif kind == "contentBlockStop":
        index = value.get("contentBlockIndex")
        if index in opened:
            opened.remove(index)
            try:
                args = json.loads(acc["calls"][index]["arguments"] or "{}")
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须是对象")
                blocks[index]["toolUse"]["input"] = args
            except (ValueError, TypeError):
                rec["error"] = "Bedrock 工具参数不是合法 JSON 对象"
    elif kind == "messageStop":
        reason = value.get("stopReason")
        rec["stop_reason"] = reason
        if not acc.get("bedrock_started") or opened:
            rec["error"] = "Bedrock 消息结束时内容块未完成"
        rec["response_status"] = "completed" if reason in ("end_turn", "tool_use", "stop_sequence") else "incomplete"
        if rec["response_status"] != "completed" and not rec["error"]:
            rec["error"] = "Bedrock 结束原因：" + str(reason)[:100]
    elif kind == "metadata" and isinstance(value.get("usage"), dict):
        rec["usage"] = _usage(value["usage"])


def consume_json(rec, payload, acc, now):
    message = (payload.get("output") or {}).get("message") or {}
    parts = message.get("content") or []
    acc["items"] = parts
    for index, part in enumerate(parts):
        if isinstance(part.get("text"), str):
            acc["text"].append(part["text"])
            if part["text"] and rec["first_token_s"] is None:
                rec["first_token_s"] = now
        if isinstance(part.get("reasoningContent"), dict):
            rec["reasoning_tokens_unknown"] = True
            if rec["first_think_s"] is None:
                rec["first_think_s"] = now
        tool = part.get("toolUse")
        if isinstance(tool, dict):
            acc["calls"][index] = {"id": tool.get("toolUseId") or "", "name": tool.get("name") or "",
                                   "arguments": json.dumps(tool.get("input") or {}, ensure_ascii=False)}
    rec["text"] = "".join(acc["text"])
    reason = payload.get("stopReason")
    rec["stop_reason"] = reason
    rec["response_status"] = "completed" if reason in ("end_turn", "tool_use", "stop_sequence") else "incomplete"
    if rec["response_status"] != "completed":
        rec["error"] = "Bedrock 结束原因：" + str(reason)[:100]
    if isinstance(payload.get("usage"), dict):
        rec["usage"] = _usage(payload["usage"])
