import os
import re

from sql_parse.build_graph import build_graph as build_sql_graph
from send_mail.build_graph import build_graph as build_send_mail_graph
from common.basic import TokenUsageCallback, AnswerModel
from common.config import (
    get_config,
    get_intents,
    get_intent_fallback_key,
    get_intent_label,
    get_runtime_cfg,
)
from langgraph.store.memory import InMemoryStore
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, AIMessage


# ---------------------------------------------------------------------------
# 从配置派生出来的意图标签（canonical label）
# ---------------------------------------------------------------------------
# 这里保留四个常量名是为了让旧代码/调用方不需要改引用；真正的值来自 config.yaml，
# 新增/修改一个意图只需要改 yaml，不用动代码。
INTENT_SQL = get_intent_label("sql")
INTENT_MAIL = get_intent_label("mail")
INTENT_CHAT = get_intent_label("chat")
INTENT_EXIT = get_intent_label("exit")


def normalize_intent(raw: str) -> str:
    """
    把 LLM 回复里可能带的噪声（序号、markdown、标点、前后缀、think 片段）剥掉，
    然后按 config.yaml 中的意图顺序 + keywords 逐条匹配；全部不中则回落到
    intent_fallback_key（默认 chat），与 INTENT_PROMPT 中"无法归类返回日常闲聊"的约定保持一致。
    """
    if not isinstance(raw, str):
        raw = str(raw or "")
    text = raw.strip()

    # 1) 去掉 <think>...</think> 等推理泄漏
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # 2) 去掉 markdown 代码块包裹
    fence = re.search(r"```(?:\w+)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 3) 去掉 markdown 加粗/斜体标记
    text = text.replace("**", "").replace("*", "").strip()
    # 4) 去掉常见"意图：""返回："前缀 + 列表序号 "1. " / "1、"
    text = re.sub(r"^(意图|返回|结果|answer|intent)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\d+\s*[\.\、\)]\s*", "", text)
    # 5) 剥前后空白与常见标点
    text = text.strip().strip("。.!！\"'：:，, \t\n")
    lowered = text.lower()

    # 6) 按配置顺序匹配：第一条命中为准
    for intent in get_intents():
        label = intent.get("label", "")
        for kw in intent.get("keywords", []) or []:
            kw_l = (kw or "").lower()
            if not kw_l:
                continue
            if kw_l in lowered or kw in text:
                return label
        # label 自身也作为关键字兜底一次（兼容 LLM 原样返回 label 的情况）
        if label and (label in text or label.lower() in lowered):
            return label

    # 7) 兜底：无法归类 → 配置的 fallback（默认 chat）
    return get_intent_label(get_intent_fallback_key())

class AgentState(MessagesState):
    intent: str
    summary: str
    code_path: str

SUMMARY_PROMPT = """
你是一个消息总结助手

你能根据历史的消息，历史的总结，对用户的历史行为、历史的回复消息进行总结

约束：
1. 用户的关键信息不能忽略
2. 输出的总结字数不能多于历史消息总和
3. 历史最近的3条答复消息关键信息不能忽略
"""

CHAT_PROMPT = """
你是一个聊天助手

你能根据用户输入的内容和前面消息的总结，进行聊天

约束：回复内容必须为中文
"""


def _build_intent_prompt() -> str:
    """
    根据 config.yaml 里的意图列表动态生成 INTENT_PROMPT：
      - 可选意图：取 intents[*].label
      - fallback：intent_fallback_key 对应的 label
    避免「yaml 加了一个意图，prompt 却没同步更新」的脱节问题。
    """
    labels = [it.get("label", "") for it in get_intents() if it.get("label")]
    fallback_label = get_intent_label(get_intent_fallback_key())
    labels_block = "\n".join(labels)
    allowed = " / ".join(f'"{l}"' for l in labels)
    return f"""
你是一个意图分析助手

你能请根据用户输入的内容和前面消息的总结，判断用户的意图，返回用户的意图。

你只能返回如下{len(labels)}种意图之一（原样返回，不要加任何修饰）：
{labels_block}
如果无法归类，那么返回{fallback_label}

【严格输出约束】
1. 只能输出 {allowed} 其中之一，一个词，不能是其他内容。
2. 禁止输出序号（例如"1."、"1、"）、禁止 markdown 加粗、禁止代码块、禁止引号。
3. 禁止输出"意图：""返回："等前缀，禁止输出任何解释、推理过程。
4. 禁止输出任何标点符号（句号、感叹号、换行等）。
"""


INTENT_PROMPT = _build_intent_prompt()

def wait_people_message(state: AgentState):
    """
    等待用户输入节点（类似 claude-code 的交互式输入框）。

    使用 langgraph 的 interrupt 机制挂起图执行，把"读终端输入"这件事
    交给外部驱动循环（见 __main__）。当用户通过 Command(resume=...) 提交
    输入后，图会从这个节点自动恢复，并把返回值作为用户输入继续往后走。

    返回: {"messages": [HumanMessage(用户输入)]}
    """
    # 打印一个漂亮的输入框提示，交互体验更接近 claude-code
    print("\n" + "=" * 60)
    print("请输入你的消息（输入 exit / quit / 退出 结束对话）：")
    print("=" * 60)

    # interrupt 会在此挂起，直到外部用 Command(resume=text) 唤醒
    user_input = interrupt({"prompt": "等待用户输入"})

    # 兼容用户误输入非字符串的情况
    if not isinstance(user_input, str):
        user_input = str(user_input)

    return {"messages": [HumanMessage(content=user_input)]}


def summary_message(state: AgentState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    system_msg = SystemMessage(content=SUMMARY_PROMPT)
    response = AnswerModel.invoke([system_msg, HumanMessage(content="""历史消息如下：
                                   {messages}
                                   历史总结如下，是在历史消息之前的其他消息的总结，内容有可能为空：
    {summary}""".format(messages=messages, summary=summary))])
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-1]]
    return {"summary": response.content, "messages": delete_messages}

def should_summary(state: AgentState):
    messages = state.get("messages", [])
    threshold = get_runtime_cfg().get("summary_threshold", 5)
    if len(messages) > threshold:
        return "summary_message"
    return "wait_people_message"

def identify_intent(state: AgentState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    system_msg = SystemMessage(content=INTENT_PROMPT)
    response = AnswerModel.invoke([system_msg, HumanMessage(content="""消息如下，以最后一条为意图消息，其余消息作为背景判断：
                                   {messages}
                                   历史总结如下，是在历史消息之前的其他消息的总结，内容有可能为空：
    {summary}""".format(messages=messages, summary=summary))])
    # 规整模型输出，避免 "1. 慢Sql分析" / "**慢Sql分析**" 等噪声让下游分流崩溃
    intent = normalize_intent(response.content)
    return {"intent": intent}

def next_step(state: AgentState):
    """
    把当前意图映射到主图节点名：
      - 从 config.yaml 的 intents[*].route 读取；
      - route == "__end__" 的意图会直接走到 END（例如"结束对话"）；
      - 未知/未配置的意图统一回落到 chat，彻底杜绝 ValueError 让主图崩溃。
    """
    intent = state.get("intent", "")
    for it in get_intents():
        if it.get("label") == intent:
            route = it.get("route", "chat")
            return END if route == "__end__" else route
    return "chat"

def chat(state: AgentState):
    summary = state.get("summary", "")
    messages = state.get("messages", [])
    system_msg = SystemMessage(content=CHAT_PROMPT)
    response = AnswerModel.invoke([system_msg, HumanMessage(content="""消息如下，以最后一条为聊天消息，其余消息作为背景判断：
                                   {messages}
                                   历史总结如下，是在历史消息之前的其他消息的总结，内容有可能为空：
    {summary}""".format(messages=messages, summary=summary))])
    return {"summary": response.content, "messages": [response]}

def build_graph(store: InMemoryStore):
    """
    主图只负责「等输入 → 识别意图 → 分流到对应子图/节点 → 回到等输入」的循环，
    具体的业务（如慢 SQL 分析、发送邮件）都在各自目录的子图里闭环，
    agent.py 不再感知子图内部状态与 interrupt 细节。

    各子图作为节点加入主图（而非 .invoke() 调用）：这样子图内部的 interrupt
    会和主图共享同一个 checkpointer/thread_id/config，
    主程序的 Command(resume=...) 能直接唤醒被挂起的子图节点。
    """
    sql_parse = build_sql_graph(store)
    # 发邮件子图：节点形式挂入，内部的草稿抽取/授权/OAuth 等待/确认/发送全部闭环
    send_email = build_send_mail_graph()

    graph = StateGraph(AgentState)

    graph.add_node("wait_people_message", wait_people_message)
    graph.add_node("identify_intent", identify_intent)
    graph.add_node("summary_message", summary_message)
    graph.add_node("sql_parse", sql_parse)
    graph.add_node("chat", chat)
    graph.add_node("send_email", send_email)

    # 入口：先等用户输入
    graph.add_edge(START, "wait_people_message")
    # 用户输入后 → 识别意图
    graph.add_edge("wait_people_message", "identify_intent")
    # 根据意图分流到对应处理节点
    graph.add_conditional_edges("identify_intent", next_step)
    # 各处理节点执行完后，判断是否需要压缩历史：
    #   - 消息过多 -> summary_message 压缩后回到等待用户输入
    #   - 否则     -> 直接回到等待用户输入
    # 从而形成 claude-code 式的人机交互循环
    graph.add_conditional_edges("sql_parse", should_summary)
    graph.add_conditional_edges("chat", should_summary)
    graph.add_conditional_edges("send_email", should_summary)
    graph.add_edge("summary_message", "wait_people_message")

    # interrupt 必须配合 checkpointer 才能挂起/恢复
    return graph.compile(checkpointer=MemorySaver(), store=store)

if __name__ == "__main__":
    # 一次性加载配置并打印关键信息，便于启动时自检
    cfg = get_config()
    rt = get_runtime_cfg()
    print("=" * 60)
    print("my-agent 启动中（配置摘要）：")
    print(f"  thread_id          = {rt.get('thread_id')}")
    print(f"  user_id            = {rt.get('user_id')}")
    print(f"  default_code_path  = {rt.get('default_code_path') or '(=当前工作目录)'}")
    print(f"  max_step           = {rt.get('max_step')}")
    print(f"  summary_threshold  = {rt.get('summary_threshold')}")
    print(f"  意图：{[it.get('label') for it in get_intents()]}")
    print("=" * 60)

    store = InMemoryStore()
    graph = build_graph(store)
    token_cb = TokenUsageCallback()
    config = {
        "configurable": {
            "thread_id": str(rt.get("thread_id", "10086")),
            "user_id": str(rt.get("user_id", "1")),
        },
        "callbacks": [token_cb],
    }

    # 初始状态：把项目路径带进去，供子图 sql_parse 使用
    # 配置没给就用当前工作目录，避免写死某个开发机绝对路径
    init_code_path = rt.get("default_code_path") or os.getcwd()
    init_state = {"code_path": init_code_path}

    try:
        # 第一轮：从 START 进入，走到第一个 interrupt（等待用户输入）
        for event in graph.stream(init_state, config, stream_mode="values"):
            print(event)

        # 交互循环：每轮读取终端输入，用 Command(resume=...) 唤醒被 interrupt 挂起的图
        while True:
            # 读取被挂起节点的提示词（若有），并判断图是否已结束
            state_snapshot = graph.get_state(config)
            if not state_snapshot.next:
                # 没有任何待执行节点，说明图已走到 END（如"结束对话"意图）
                print("对话已结束。")
                break

            try:
                user_text = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见！")
                break

            if user_text.lower() in ("exit", "quit", "q", "退出"):
                print("结束对话，再见！")
                break

            # 把用户输入交给被挂起的 wait_people_message 节点，图自动恢复执行
            for event in graph.stream(
                Command(resume=user_text), config, stream_mode="values"
            ):
                print(event)
    finally:
        # 无论正常结束还是异常，都打印本次任务的 token 汇总
        token_cb.print_report()