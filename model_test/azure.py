"""Azure OpenAI 部署式 Chat Completions 路径与鉴权参数。"""
import datetime
import re
import urllib.parse

DEFAULT_API_VERSION = "2024-10-21"
AUTH_MODES = ("api_key", "bearer")


def validate_settings(version, auth):
    if not isinstance(version, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:-preview)?", version):
        raise ValueError("Azure API 版本必须为 YYYY-MM-DD 或 YYYY-MM-DD-preview")
    try:
        datetime.date.fromisoformat(version[:10])
    except ValueError as exc:
        raise ValueError("Azure API 版本日期无效") from exc
    if auth not in AUTH_MODES:
        raise ValueError("Azure 鉴权方式只能是 api_key 或 bearer")


def request_path(base_url, deployment, version):
    validate_settings(version, "api_key")
    if not isinstance(deployment, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", deployment):
        raise ValueError("Azure 模型需填写有效的部署名称")
    path = urllib.parse.urlsplit(base_url).path.rstrip("/")
    if not path.endswith("/openai"):
        path += "/openai"
    query = urllib.parse.urlencode({"api-version": version})
    return path + "/deployments/" + urllib.parse.quote(deployment, safe="") + "/chat/completions?" + query
