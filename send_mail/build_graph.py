from .send_mail import (
    parse_mail,
    check_auth,
    ensure_auth,
    do_send,
    build_preview_text,
    parse_auth_status,
    token_is_usable,
    TOKEN_EXPIRY_BUFFER,
)
from common.basic import SendMailState
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import AIMessage, HumanMessage


class GraphState(MessagesState, SendMailState):
    pass


def get_user_message(state: GraphState):
    """从主图传入的 messages 中取最近一条 HumanMessage，作为发邮件的原始诉求。"""
    user_message = ""
    for m in reversed(state.get("messages", [])):
        if isinstance(m, HumanMessage) and isinstance(getattr(m, "content", ""), str) and m.content.strip():
            user_message = m.content
            break
    print("=" * 60)
    print(f"[get_user_message] 用户诉求：{user_message!r}")
    print("=" * 60)
    return {"user_message": user_message}


def should_after_parse(state: GraphState):
    """未抽到收件人 → 直接结束并提示；否则进入授权检查。"""
    if not (state.get("to") or []):
        return "no_recipient"
    return "check_auth"


def no_recipient(state: GraphState):
    resp = (
        "未能从你的描述中识别出收件人邮箱，请重新发起并明确提供收件人地址，"
        "例如：\n给 xxx@example.com 发邮件，主题是…，内容是…"
    )
    return {"response": resp, "summary": resp, "messages": [AIMessage(content=resp)]}


def should_after_auth(state: GraphState):
    """CLI 硬错误 → finalize_fail；凭据不可用 → ensure_oauth；否则 → preview。"""
    if state.get("response"):
        return "finalize_fail"
    if not state.get("auth_ok", False):
        return "ensure_oauth"
    return "preview"


def finalize_fail(state: GraphState):
    resp = state.get("response", "") or "发邮件流程失败"
    return {"summary": resp, "messages": [AIMessage(content=resp)]}


def ensure_oauth(state: GraphState):
    """触发 OAuth 登录拿到 URL，interrupt 等用户在浏览器完成授权后再验证凭据。"""
    ensure_auth(state)
    auth_url = state.get("auth_url", "")
    reason = state.get("auth_reason", "")
    hint = (
        "请点击或复制以下链接在浏览器中完成授权"
        + ("（凭据已过期，需重新授权）" if reason else "")
        + "，完成后回复任意内容（如“已授权”）继续。"
    )
    interrupt({"type": "oauth", "auth_url": auth_url, "hint": hint})

    st = parse_auth_status()
    if not token_is_usable(st, buffer_seconds=TOKEN_EXPIRY_BUFFER):
        resp = f"授权验证失败：{(st.get('raw') or '').strip() or '无法读取授权状态'}"
        return {
            "auth_ok": False,
            "response": resp,
            "summary": resp,
            "messages": [AIMessage(content=f"等待 OAuth 授权：{auth_url}\n{hint}"), AIMessage(content=resp)],
        }
    return {
        "auth_ok": True,
        "messages": [AIMessage(content=f"等待 OAuth 授权：{auth_url}\n{hint}")],
    }


def should_after_oauth(state: GraphState):
    return "preview" if state.get("auth_ok", False) else END


def preview(state: GraphState):
    """展示草稿 → interrupt 等用户确认 → 同轮判定/发送（用户只需确认一次）。"""
    preview_text = build_preview_text(state)
    hint = "请确认以上邮件内容，回复“确认/发/ok”以真正发送；回复其他内容可取消。"
    user_reply = interrupt({"type": "send_confirm", "summary": preview_text, "hint": hint})
    user_reply = (user_reply if isinstance(user_reply, str) else str(user_reply or "")).strip()

    positive = ("确认", "确定", "发", "发送", "ok", "yes", "y", "好", "可以", "同意")
    if not any(kw in user_reply.lower() for kw in positive):
        resp = f"已取消发送（用户回复：{user_reply or '(空)'}）。如需重新发送，请再次发起请求。"
        return {
            "send_summary": preview_text,
            "response": resp,
            "summary": resp,
            "messages": [AIMessage(content=preview_text), AIMessage(content=resp)],
        }

    do_send(state)
    resp = state.get("response", "") or ""
    return {
        "send_summary": preview_text,
        "response": resp,
        "summary": resp,
        "messages": [AIMessage(content=preview_text), AIMessage(content=resp)],
    }


def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("get_user_message", get_user_message)
    builder.add_node("parse_mail", parse_mail)
    builder.add_node("no_recipient", no_recipient)
    builder.add_node("check_auth", check_auth)
    builder.add_node("finalize_fail", finalize_fail)
    builder.add_node("ensure_oauth", ensure_oauth)
    builder.add_node("preview", preview)

    builder.add_edge(START, "get_user_message")
    builder.add_edge("get_user_message", "parse_mail")
    builder.add_conditional_edges("parse_mail", should_after_parse)
    builder.add_edge("no_recipient", END)
    builder.add_conditional_edges("check_auth", should_after_auth)
    builder.add_edge("finalize_fail", END)
    builder.add_conditional_edges("ensure_oauth", should_after_oauth)
    builder.add_edge("preview", END)
    return builder.compile()