from .sql_parse import build_sql_parse_graph
from common.basic import get_slow_sql, SqlParseState, PlanModel
from .tools import build_project_graph
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
import os
import operator
from typing import Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class GraphState(MessagesState):
    sqls: list
    code_path: str
    # 多个 sql_parse 子图并行 fan-in，这里必须用 list + add reducer，否则 LangGraph
    # 会因“同一步 N 个分支写同一个字段”抛 InvalidUpdateError。
    response: Annotated[list, operator.add]

GET_CODE_PATH_PROMPT = """你是一个路径抽取助手。

用户的消息里可能会提到一个“待分析的代码目录/项目路径”。
请从中抽取出用户想要分析的代码目录路径（尽量给绝对路径）。

规则：
1. 如果用户明确给出了一个目录路径（如 /Users/xxx/project、/data/app/server，或带盘符的 Windows 路径），请原样返回该路径。
2. 如果用户只给了一个项目名（如 my-agent、my-app），而没有给出完整路径，请返回该项目名（程序会结合当前工作目录去定位）。
3. 如果用户完全没有提到任何目录或项目，请只返回字符串 "__DEFAULT__"。
4. 只返回路径或 "__DEFAULT__" 本身，不要加任何解释、引号或多余文字。

例如：
- 用户输入：分析一下 /Users/src/code 下的慢SQL
  返回：/Users/src/code
- 用户输入：帮我看看 my-agent 项目里的SQL
  返回：my-agent
- 用户输入：分析一下慢SQL怎么优化
  返回：__DEFAULT__
"""

def _resolve_code_path(raw: str) -> str:
    """把 LLM 抽取出的路径解析为一个真实存在的绝对路径；找不到则返回当前工作目录。"""
    raw = (raw or "").strip()
    default = os.getcwd()

    if not raw or raw == "__DEFAULT__":
        return default

    # 已经是存在的绝对/相对路径
    candidate = os.path.abspath(raw)
    if os.path.isdir(candidate):
        return candidate

    # LLM 可能只给了项目名，尝试在当前工作目录下查找同名目录
    name_only = os.path.basename(raw)
    guess = os.path.join(default, name_only)
    if os.path.isdir(guess):
        return guess

    # 退化为：相对路径（虽然 isdir 失败，但 abspath 仍可用），作为兜底保留
    # 这里仍返回 abspath，交给下游 get_sql 做存在性判断与提示
    return candidate

def get_code_path(state: MessagesState):
    """根据用户消息分析出要分析的代码目录；用户未指定则用程序当前路径。"""
    messages = state.get("messages", [])
    # 倒序取最近一条有内容的用户消息（HumanMessage）作为分析依据
    user_text = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and isinstance(getattr(m, "content", ""), str) and m.content.strip():
            user_text = m.content
            break

    if not user_text:
        code_path = os.getcwd()
    else:
        system_msg = SystemMessage(content=GET_CODE_PATH_PROMPT)
        human_msg = HumanMessage(content=f"用户消息：{user_text}")
        resp = PlanModel.invoke([system_msg, human_msg])
        code_path = _resolve_code_path(resp.content)

    print("=" * 60)
    print(f"[get_code_path] 分析得到代码目录: {code_path}")
    print("=" * 60)
    return {"code_path": code_path}



def get_sql(state: GraphState):
    # 项目根路径确定后，立即为项目建立目录图谱（只扫一次，之后所有 list_project_directory 走缓存）
    code_path = state.get("code_path", "")
    if code_path:
        summary = build_project_graph(code_path)
        print("=" * 60)
        print(f"[get_sql] 已为项目建立目录图谱: {summary['root']}")
        print(f"  总有效代码文件数: {summary['total_files']}")
        print(f"  顶层目录: {summary['top_level_dirs']}")
        print(f"  扩展名分布: {summary['ext_stats']}")
        print("=" * 60)
    return {"sqls": get_slow_sql()}

def parse_all_sql(state: GraphState):
    return [Send("run_one_sql_parse", {"sql": sql, "code_path": state["code_path"]}) for sql in state["sqls"]]


# 预编译一份子图，供下面的 wrapper 节点重复调用（每个并行分支独立 invoke）
_sql_parse_subgraph = build_sql_parse_graph()


def run_one_sql_parse(state: SqlParseState):
    """
    包一层子图调用：
    - 子图 state schema 是 SqlParseState.response: str
    - 父图 state schema 是 GraphState.response: list[str]（带 add reducer）
    - 两者类型不一致，不能让子图直接当节点挂进父图，否则 fan-in 时父图会试图
      做 list + str，报类型错误。这里显式 invoke 子图，再把 str 包成单元素 list
      交给父图 reducer 做 list.extend，保证 N 个并行分支的结果都能收集齐。
    """
    sub_result = _sql_parse_subgraph.invoke(state)
    one_response = sub_result.get("response", "") or ""
    return {"response": [one_response] if one_response else []}


def report_result(state: GraphState):
    """
    把 sql_parse 子图产出的所有方案（state["response"] 是 list）汇总为一条
    AIMessage 追加到主图 messages，让主图回到 wait_people_message 时用户能
    明确看到「本次慢SQL分析已完成，这是方案」——而不是只看到一个光秃秃的
    interrupt 让用户误以为还需要再确认、再输入一句话。
    """
    responses = state.get("response", []) or []
    if not responses:
        body = "（本次慢SQL分析未产出方案，可能是执行提前结束。）"
    elif len(responses) == 1:
        body = responses[0]
    else:
        parts = []
        for i, r in enumerate(responses, 1):
            parts.append(f"────── 慢SQL #{i} 的优化方案 ──────\n{r}")
        body = "\n\n".join(parts)

    ai_text = (
        "✅ 慢SQL分析任务已完成，以下是生成的优化方案：\n\n"
        + body
        + "\n\n如需发起新的任务，请继续输入；输入 退出 可结束对话。"
    )
    print("\n" + "=" * 60)
    print(f"[report_result] 子图产出已汇总（{len(responses)} 条），返回主图")
    print("=" * 60)
    return {"messages": [AIMessage(content=ai_text)]}


def build_graph(store: InMemoryStore):
    builder = StateGraph(GraphState)
    builder.add_node("get_code_path", get_code_path)
    builder.add_node("get_sql", get_sql)
    # 不再把子图直接当节点，而是用 wrapper 包一层以适配 fan-in 的 list reducer
    builder.add_node("run_one_sql_parse", run_one_sql_parse)
    builder.add_node("report_result", report_result)

    builder.add_edge(START, "get_code_path")
    builder.add_edge("get_code_path", "get_sql")
    builder.add_conditional_edges("get_sql", parse_all_sql)
    # 多个并行 run_one_sql_parse 完成后，统一汇总一次给用户，避免主图回到
    # wait_people_message 时只看到一个光秃秃的 interrupt，让用户误以为还要确认。
    builder.add_edge("run_one_sql_parse", "report_result")
    builder.add_edge("report_result", END)
    return builder.compile(checkpointer=MemorySaver(), store=store)
