"""
my-agent 全局配置加载器
=======================
职责：
  1. 读取 YAML 配置（默认 my-agent/config.yaml，可用环境变量 AGENT_CONFIG_PATH 覆盖）；
  2. 处理 env 覆盖（OPENAI_API_KEY / OPENAI_BASE_URL / AGENT_CODE_PATH）；
  3. 对外暴露 `get_config()` 单例 + 若干便捷访问器（模型配置、意图表等）。

设计原则：
  - 本模块**只做读/合并**，不缓存业务副作用，便于单测重置。
  - 所有调用方拿到的都是普通 dict，不强依赖 pydantic/attrs，避免引入额外依赖。
  - 字段缺失走代码里的 _DEFAULTS 兜底，确保 yaml 删掉某一节也不会炸。
"""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "需要 PyYAML 才能加载配置文件，请先执行：pip install -r requirements.txt"
    ) from e


# ---------------------------------------------------------------------------
# 兜底默认值：yaml 缺字段时用这里的值，保证代码永远能跑
# ---------------------------------------------------------------------------
_DEFAULTS: Dict[str, Any] = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "default_base_url": "",
    },
    "models": {
        "plan":       {"name": "qwen-turbo", "temperature": 0.5, "enable_thinking": False},
        "exe":        {"name": "qwen-plus",  "temperature": 0.0, "enable_thinking": False},
        "answer":     {"name": "qwen-plus",  "temperature": 0.5, "enable_thinking": False},
        "structured": {"name": "qwen3-8b",   "temperature": 0.0, "enable_thinking": False},
    },
    "runtime": {
        "thread_id": "10086",
        "user_id": "1",
        "default_code_path": "",
        "max_step": 20,
        "summary_threshold": 5,
    },
    "intents": [
        {"key": "sql",  "label": "慢Sql分析", "keywords": ["慢sql", "sql分析", "sql 分析"], "route": "sql_parse"},
        {"key": "mail", "label": "发送邮件", "keywords": ["发送邮件", "发邮件", "send mail", "send email"], "route": "send_email"},
        {"key": "exit", "label": "结束对话", "keywords": ["结束对话", "退出", "exit", "quit"], "route": "__end__"},
        {"key": "chat", "label": "日常闲聊", "keywords": ["日常闲聊", "闲聊", "聊天"], "route": "chat"},
    ],
    "intent_fallback_key": "chat",
    "sql_parse": {
        "slow_sqls": [],
        "default_index_list": [],
        "output_markdown_name": "optimization_plan.md",
    },
    "send_mail": {
        "cli": {
            "binary": "agently-cli",
            "min_version": "1.0.18",
            "install_cmd": ["npm", "install", "-g", "@tencent-qqmail/agently-cli"],
        },
        "token_expiry_buffer_seconds": 300,
        "timeouts": {"version": 30, "status": 60, "login": 600, "send": 120, "install": 300},
    },
}

# 延迟加载的全局单例
_CONFIG: Optional[Dict[str, Any]] = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """把 override 深合并进 base 的拷贝；list 直接整体替换，dict 递归合并。"""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_config_path() -> str:
    """定位 config.yaml：env > my-agent/config.yaml（本文件父目录）。"""
    env_path = os.environ.get("AGENT_CONFIG_PATH", "").strip()
    if env_path:
        return env_path
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), "config.yaml")


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    加载并返回配置单例。
    - 第一次调用：读 yaml -> 与 _DEFAULTS 合并 -> 应用 env 覆盖。
    - 之后调用：返回同一个 dict（除非 force_reload=True）。
    """
    global _CONFIG
    if _CONFIG is not None and not force_reload:
        return _CONFIG

    cfg = deepcopy(_DEFAULTS)
    path = _resolve_config_path()
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"配置文件根节点必须是 dict：{path}")
        cfg = _deep_merge(cfg, data)
    # else: yaml 不存在也允许，走 _DEFAULTS，便于最小化跑通

    # env 覆盖
    code_path_env = os.environ.get("AGENT_CODE_PATH", "").strip()
    if code_path_env:
        cfg["runtime"]["default_code_path"] = code_path_env

    _CONFIG = cfg
    return _CONFIG


def get_config() -> Dict[str, Any]:
    """便捷别名，等价于 load_config()。"""
    return load_config()


# ---------------------------------------------------------------------------
# 便捷访问器（避免散落在业务代码里到处写 cfg["xxx"]["yyy"]）
# ---------------------------------------------------------------------------
def get_openai_credentials() -> Dict[str, str]:
    """返回 {api_key, base_url}，均从 env 读；env 缺失则回落 yaml.default_base_url / 空串。"""
    cfg = get_config()
    oc = cfg.get("openai", {})
    api_key = os.environ.get(oc.get("api_key_env", "OPENAI_API_KEY"), "")
    base_url = os.environ.get(oc.get("base_url_env", "OPENAI_BASE_URL"), "") or oc.get("default_base_url", "")
    return {"api_key": api_key, "base_url": base_url}


def get_model_cfg(kind: str) -> Dict[str, Any]:
    """kind in {plan, exe, answer, structured}；缺失时抛错，提醒用户补 yaml。"""
    models = get_config().get("models", {})
    if kind not in models:
        raise KeyError(f"config.yaml 未配置 models.{kind}；请参考默认 config.yaml 补齐。")
    return models[kind]


def get_runtime_cfg() -> Dict[str, Any]:
    return get_config().get("runtime", {})


def get_intents() -> List[Dict[str, Any]]:
    """按 yaml 顺序返回意图列表（normalize_intent 的匹配顺序即此）。"""
    return list(get_config().get("intents", []) or [])


def get_intent_label(key: str) -> str:
    """根据 key 拿到 canonical label；找不到时返回 key 本身。"""
    for it in get_intents():
        if it.get("key") == key:
            return it.get("label", key)
    return key


def get_intent_fallback_key() -> str:
    return get_config().get("intent_fallback_key", "chat")


def get_sql_parse_cfg() -> Dict[str, Any]:
    return get_config().get("sql_parse", {})


def get_send_mail_cfg() -> Dict[str, Any]:
    return get_config().get("send_mail", {})
