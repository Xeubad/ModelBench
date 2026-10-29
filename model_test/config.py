"""集中配置：端口、监听地址、限额、页面默认值、数据与结果目录。

- 配置文件是 ModelBench 目录下的 config.json（可选，没有就用下面的内置默认）；环境变量 MODEL_TEST_CONFIG 可指定别的文件。
- 只需写想改的项，其余沿用默认；写了不认识的项会报错（防止拼错了却没生效）；以 _ 开头的键当注释忽略。
- paths 里的相对路径相对于 ModelBench 目录，~ 会展开为用户目录。
- 算法常量（样本不足阈值、p95 最少样本数、非流式判定等）不在这里，它们是测量方法的一部分，改了会让历史结果不可比，见 core.py。
"""
import copy
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ModelBench 目录
DEFAULTS = {
    "server": {"port": 18888, "bind_hosts": ["127.0.0.1"]},  # 第一个必须是本机；需要远程访问时自行追加（绑定失败会自动跳过）
    "limits": {"max_prompt": 2000, "max_rounds": 20, "max_timeout": 300, "max_requests": 1200, "max_body": 400000},
    "defaults": {"rounds": 5, "timeout": 90, "gap": 1.5},  # 页面第一次打开时的参数默认值
    "paths": {"data_dir": "data", "results_dir": "results", "legacy_results_dir": "~/.openclaw/logs/model-bench",
              "openclaw_config": "~/.openclaw/openclaw.json"},
}


def config_file():
    return os.environ.get("MODEL_TEST_CONFIG") or os.path.join(ROOT, "config.json")


def _merge(base, over, where=""):
    if not isinstance(over, dict):
        raise ValueError(f"配置项 {where or '根'} 必须是对象")
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if k not in base:
            raise ValueError(f"配置里有不认识的项：{where}{k}（拼错了？可用项见 config.example.json）")
        if isinstance(base[k], dict):
            _merge(base[k], v, where + k + ".")
        else:
            base[k] = v
    return base


def _int(v, lo, hi, name):
    if not isinstance(v, int) or isinstance(v, bool) or not lo <= v <= hi:
        raise ValueError(f"配置项 {name} 必须是 {lo}–{hi} 的整数，当前是 {v!r}")


def validate(c):
    _int(c["server"]["port"], 1, 65535, "server.port")
    hosts = c["server"]["bind_hosts"]
    if not isinstance(hosts, list) or not hosts or any(not isinstance(h, str) or not h for h in hosts):
        raise ValueError("配置项 server.bind_hosts 必须是非空的地址列表，第一个是本机 127.0.0.1")
    for k, v in c["limits"].items():
        _int(v, 1, 10 ** 9, "limits." + k)
    d, lim = c["defaults"], c["limits"]
    _int(d["rounds"], 1, lim["max_rounds"], "defaults.rounds（不能超过 limits.max_rounds）")
    _int(d["timeout"], 10, lim["max_timeout"], "defaults.timeout（10 到 limits.max_timeout）")
    if not isinstance(d["gap"], (int, float)) or isinstance(d["gap"], bool) or not 0 <= d["gap"] <= 10:
        raise ValueError(f"配置项 defaults.gap 必须是 0–10 的数字，当前是 {d['gap']!r}")
    for k, v in c["paths"].items():
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f"配置项 paths.{k} 必须是非空路径")
    return c


def load(path=None):
    cfg, p = copy.deepcopy(DEFAULTS), path or config_file()
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except ValueError as e:
            raise ValueError(f"配置文件 {p} 不是合法的 JSON：{e}") from e
        _merge(cfg, data)
    return validate(cfg)


CFG = load()


def resolve(p):
    p = os.path.expanduser(p)
    return os.path.normpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


def path(key):
    return resolve(CFG["paths"][key])
