from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.callbacks import BaseCallbackHandler
from threading import Lock

from .config import (
    get_openai_credentials,
    get_model_cfg,
    get_sql_parse_cfg,
)

class SqlParseState(TypedDict):
    sql: str
    table_name: str
    action: str
    select_columns: list
    condition: str
    offset: int
    limit: int
    position: str
    index: list
    code_path: str
    # optimize 子图产出的最终方案文本，需要逐层透传到最外层 sql_parse 主子图
    response: str

class SendMailState(TypedDict):
    user_message: str
    code_path: str
    to: list
    cc: list
    bcc: list
    subject: str
    body: str
    attachment: list
    auth_ok: bool
    auth_url: str
    auth_reason: str
    send_summary: str
    response: str

FindPosGraphName = "find_pos"


# ---------------------------------------------------------------------------
# 根据配置构造 ChatOpenAI 实例
# ---------------------------------------------------------------------------
def _build_chat_model(kind: str, **extra_bind) -> ChatOpenAI:
    """
    从配置里拿 models.<kind> 的 name/temperature/enable_thinking，并把 openai
    base_url / api_key 一并注入；若需要关闭 thinking（Qwen3 系列），则通过
    extra_body={"enable_thinking": False} 的方式 bind 到实例上。
    """
    mcfg = get_model_cfg(kind)
    creds = get_openai_credentials()
    kwargs = {
        "model": mcfg.get("name"),
        "temperature": mcfg.get("temperature", 0),
    }
    if creds.get("api_key"):
        kwargs["api_key"] = creds["api_key"]
    if creds.get("base_url"):
        kwargs["base_url"] = creds["base_url"]
    llm = ChatOpenAI(**kwargs)
    # Qwen3 等默认开启 reasoning，JSON 抽取/意图分析场景必须关掉
    if not mcfg.get("enable_thinking", True):
        llm = llm.bind(extra_body={"enable_thinking": False}, **extra_bind)
    elif extra_bind:
        llm = llm.bind(**extra_bind)
    return llm


# 对外沿用原来的变量名，内部实现换成「按配置构造」
PlanModel = _build_chat_model("plan")
ExeModel = _build_chat_model("exe")
AnswerModel = _build_chat_model("answer")
# 用于严格 JSON 抽取（SQL 结构 / 邮件草稿），独立实例，温度 0 + 关 thinking
StructuredModel = _build_chat_model("structured")


# 保留旧式三个模型名字符串，仅供需要原始 model 名的代码（如 with_structured_output
# 的调试日志）使用；注意这些值不再是「可修改的开关」。
plan_model = get_model_cfg("plan").get("name")
exe_model = get_model_cfg("exe").get("name")
answer_model = get_model_cfg("answer").get("name")


def get_index_by_tablename(tablename: str) -> list:
    """
    返回指定表的索引列表。
    当前实现从配置 sql_parse.default_index_list 读取；
    真实上线建议把配置改成 `{"table_name": [idx1, idx2]}` 的字典结构，
    这里就可以按表名精确命中。
    """
    cfg = get_sql_parse_cfg()
    idx = cfg.get("default_index_list", [])
    if isinstance(idx, dict):
        return list(idx.get(tablename, []))
    return list(idx or [])


def get_slow_sql() -> list[str]:
    """从配置返回待分析的慢 SQL 列表。"""
    cfg = get_sql_parse_cfg()
    return list(cfg.get("slow_sqls", []) or [])


class TokenUsageCallback(BaseCallbackHandler):
    """
    全局 token 用量统计回调。
    通过 config={"callbacks": [handler]} 传入 graph.stream / graph.invoke，
    LangChain 会自动沿执行链传递给所有 LLM 调用（包括 create_react_agent 内部循环）。
    """

    def __init__(self):
        self._lock = Lock()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_count = 0
        # 按模型名细分统计
        self.per_model = {}  # {model_name: {"prompt": x, "completion": y, "total": z, "calls": n}}

    def _extract_usage(self, response):
        """兼容多种字段位置提取 token usage。"""
        usage = None
        model_name = "unknown"

        # 1) llm_output（OpenAI 格式最常见）
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")
            model_name = llm_output.get("model_name") or llm_output.get("model") or model_name

        # 2) 从 generations[*].message.usage_metadata 兜底（新版 langchain-openai）
        if not usage:
            try:
                gens = response.generations
                for gen_list in gens:
                    for gen in gen_list:
                        msg = getattr(gen, "message", None)
                        um = getattr(msg, "usage_metadata", None) if msg else None
                        if um:
                            usage = {
                                "prompt_tokens": um.get("input_tokens", 0),
                                "completion_tokens": um.get("output_tokens", 0),
                                "total_tokens": um.get("total_tokens", 0),
                            }
                            # model 名兜底
                            resp_meta = getattr(msg, "response_metadata", {}) or {}
                            model_name = resp_meta.get("model_name") or resp_meta.get("model") or model_name
                            break
                    if usage:
                        break
            except Exception:
                pass

        return usage, model_name

    def on_llm_end(self, response, **kwargs):
        usage, model_name = self._extract_usage(response)
        if not usage:
            return
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        total = int(usage.get("total_tokens", 0) or (prompt + completion))

        with self._lock:
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.total_tokens += total
            self.llm_call_count += 1

            slot = self.per_model.setdefault(
                model_name, {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
            )
            slot["prompt"] += prompt
            slot["completion"] += completion
            slot["total"] += total
            slot["calls"] += 1

    def report(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("Token 用量汇总")
        lines.append("-" * 60)
        lines.append(f"LLM 调用次数：{self.llm_call_count}")
        lines.append(f"总 prompt tokens    ：{self.total_prompt_tokens}")
        lines.append(f"总 completion tokens：{self.total_completion_tokens}")
        lines.append(f"总 tokens           ：{self.total_tokens}")
        if self.per_model:
            lines.append("-" * 60)
            lines.append("按模型细分：")
            for name, s in self.per_model.items():
                lines.append(
                    f"  [{name}] calls={s['calls']}, "
                    f"prompt={s['prompt']}, completion={s['completion']}, total={s['total']}"
                )
        lines.append("=" * 60)
        return "\n".join(lines)

    def print_report(self):
        print("\n" + self.report())