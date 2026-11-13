"""Gemini GenerateContent 的请求形状、SSE 响应和分页，零外部依赖。"""
import json
import re
import urllib.parse


def prefix(base_url):
    path = urllib.parse.urlsplit(base_url).path.rstrip("/")
    if path.endswith("/models"):
        path = path[:-7]
    if not re.search(r"/v\d+(?:beta\d*)?$", path):
        path += "/v1beta"
    return path


def request_path(base_url, model):
    if not isinstance(model, str):
        raise ValueError("Gemini 模型 ID 无效")
    parts = model.split("/")
    if len(parts) == 1:
        parts.insert(0, "models")
    if len(parts) != 2 or parts[0] not in ("models", "tunedModels", "dynamic") or not re.fullmatch(r"[A-Za-z0-9._-]{1,120}", parts[1]):
        raise ValueError("Gemini 模型 ID 无效")
    return prefix(base_url) + "/" + parts[0] + "/" + urllib.parse.quote(parts[1], safe="") + ":streamGenerateContent?alt=sse"


def fetch_models(ep, request, redact):
    """读取 Gemini nextPageToken；失败时不返回部分结果。"""
    seen, found, cursor = set(), set(), None
    for _ in range(50):
        query = "?" + urllib.parse.urlencode({"pageToken": cursor}) if cursor else ""
        status, body = request("gemini", prefix(ep["base_url"]) + "/models" + query)
        if status != 200:
            raise RuntimeError("读取 Gemini 模型列表失败：HTTP " + str(status) + " " + redact(body, ep.get("api_key"))[:160])
        data = json.loads(body)
        if not isinstance(data, dict) or not isinstance(data.get("models"), list):
            raise ValueError("Gemini 模型列表格式不正确")
        for item in data["models"]:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                continue
            methods = item.get("supportedGenerationMethods")
            if methods is not None and (not isinstance(methods, list) or "generateContent" not in methods):
                continue
            name = item["name"]
            if name.startswith("models/"):
                name = name[7:]
            try:
                request_path(ep["base_url"], name)
            except ValueError:
                continue
            if redact(name, ep.get("api_key")) == name:
                found.add(name)
        if len(found) > 5000:
            raise ValueError("模型列表超过 5000 个，请手动配置")
        cursor = data.get("nextPageToken")
        if not cursor:
            return sorted(found)
        if not isinstance(cursor, str) or len(cursor) > 512 or cursor in seen or redact(cursor, ep.get("api_key")) != cursor:
            raise ValueError("Gemini 分页游标无效或重复")
        seen.add(cursor)
    raise ValueError("Gemini 模型列表页数超过上限")


def make_body(model, prompt, system, max_tokens):
    body = {"model": model, "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": max_tokens}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    return body


def consume(rec, event, acc, now):
    """处理一个 GenerateContentResponse，保留函数调用的 thoughtSignature。"""
    if rec["first_event_s"] is None:
        rec["first_event_s"] = now
    if event.get("error"):
        rec["error"] = json.dumps(event["error"], ensure_ascii=False)[:300]
    feedback = event.get("promptFeedback") or {}
    if feedback.get("blockReason"):
        rec["error"] = "Gemini 拒绝请求：" + str(feedback["blockReason"])[:100]
    rec["returned_model"] = event.get("modelVersion") or rec.get("returned_model")
    usage = event.get("usageMetadata")
    if isinstance(usage, dict):
        visible, thoughts = usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount")
        total_out = (visible or 0) + (thoughts or 0) if visible is not None or thoughts is not None else None
        rec["usage"] = {"input_tokens": usage.get("promptTokenCount"), "output_tokens": total_out,
                        "reasoning_tokens": thoughts, "cached_tokens": usage.get("cachedContentTokenCount")}
    candidates = event.get("candidates") or []
    if not candidates:
        return
    candidate = candidates[0]
    for part in (candidate.get("content") or {}).get("parts") or []:
        if not isinstance(part, dict):
            continue
        acc["items"].append(part)
        text = part.get("text")
        if isinstance(text, str) and text:
            if part.get("thought") is True:
                if rec["first_think_s"] is None:
                    rec["first_think_s"] = now
            else:
                if rec["first_token_s"] is None:
                    rec["first_token_s"] = now
                acc["text"].append(text)
        call = part.get("functionCall")
        if call:
            args = call.get("args", {}) if isinstance(call, dict) else None
            if not isinstance(args, dict):
                rec["error"] = "Gemini 工具参数不是 JSON 对象"
                continue
            index = len(acc["calls"])
            acc["calls"][index] = {"id": call.get("id") or "", "name": call.get("name") or "",
                                   "arguments": json.dumps(args, ensure_ascii=False)}
    reason = candidate.get("finishReason")
    if reason:
        rec["stop_reason"] = reason
        rec["response_status"] = "completed" if reason == "STOP" else "incomplete"
        if reason != "STOP" and not rec["error"]:
            rec["error"] = "Gemini 结束原因：" + str(reason)[:100]
