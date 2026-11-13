"""兼容接口的非密钥查询参数和自定义鉴权头。"""
import re
import urllib.parse

MAX_QUERY_PARAMS = 16
RESERVED_QUERY = {"after", "after_id", "pagetoken", "pagesize", "alt"}
FORBIDDEN_HEADERS = {"host", "content-length", "transfer-encoding", "connection", "proxy-authorization", "te", "trailer", "upgrade", "content-type", "accept", "user-agent"}


def validate_auth(mode, header, prefix):
    if mode not in ("bearer", "api-key", "x-api-key", "custom"):
        raise ValueError("兼容接口鉴权方式无效")
    if mode != "custom":
        return
    if not isinstance(header, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", header) or header.lower() in FORBIDDEN_HEADERS:
        raise ValueError("自定义鉴权头名称无效或为保留头")
    if not isinstance(prefix, str) or len(prefix) > 32 or any(not 32 <= ord(c) <= 126 for c in prefix):
        raise ValueError("自定义鉴权前缀无效")


def validate_query(value):
    if not isinstance(value, dict) or len(value) > MAX_QUERY_PARAMS:
        raise ValueError(f"非密钥查询参数必须为对象，最多 {MAX_QUERY_PARAMS} 项")
    seen = set()
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name):
            raise ValueError("查询参数名称无效")
        normalized = re.sub(r"[^a-z0-9]", "", name.lower())
        sensitive = ("token", "secret", "signature", "credential", "authorization", "password", "cookie", "session")
        if (name.lower() in RESERVED_QUERY or normalized in RESERVED_QUERY or normalized in {"auth", "authcode", "code", "sig", "sas", "jwt", "bearer"}
                or normalized.startswith("key") or normalized.endswith("key") or any(part in normalized for part in sensitive)):
            raise ValueError("查询参数不能包含密钥字段或分页保留字段")
        if name.lower() in seen:
            raise ValueError("查询参数名称不能重复")
        seen.add(name.lower())
        if not isinstance(item, str) or len(item) > 256 or any(ord(c) < 32 or ord(c) == 127 for c in item):
            raise ValueError("查询参数值无效")
    return dict(value)


def append_query(path, params):
    params = validate_query(params or {})
    if not params:
        return path
    parsed = urllib.parse.urlsplit(path)
    existing = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if {k.lower() for k, _ in existing} & {k.lower() for k in params}:
        raise ValueError("自定义查询参数与协议参数冲突")
    query = urllib.parse.urlencode(existing + list(params.items()))
    return urllib.parse.urlunsplit(("", "", parsed.path, query, ""))
