from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from langgraph.graph import END, StateGraph, START

from common.basic import AnswerModel, SendMailState, StructuredModel
from common.config import get_send_mail_cfg
from .tools import (
    ensure_cli_installed,
    auth_login,
    send_mail,
    parse_auth_status,
    token_is_usable,
)

# 本地 OAuth token 剩余不足此缓冲秒数即视为需重授权（来自 config.yaml）
TOKEN_EXPIRY_BUFFER = int(get_send_mail_cfg().get("token_expiry_buffer_seconds", 300))


class MailDraft(BaseModel):
    model_config = {"populate_by_name": True}

    to: list[str] = Field(default_factory=list, description="收件人邮箱列表，如 ['a@x.com','b@x.com']")
    cc: list[str] = Field(default_factory=list, description="抄送邮箱列表")
    bcc: list[str] = Field(default_factory=list, description="密送邮箱列表")
    subject: str = Field(default="", description="邮件主题")
    body: str = Field(default="", description="邮件正文（纯文本）")
    attachment: list[str] = Field(default_factory=list, description="附件相对路径列表（最多 3 个）")


ParseMailPrompt = """
你是一个邮件起草助手。用户会用自然语言描述“要发一封邮件”，可能包含收件人、主题、正文、附件。
请从中抽取出结构化字段，注意：
1. 收件人/抄送/密送必须是合法邮箱地址（含 @）；如果用户只给了姓名（如“发给张三”），请留空并在 body 备注，不要编造邮箱。
2. 主题应简洁概括邮件目的；正文保留用户要求的全部关键信息，使用清晰中文。
3. 附件：如果用户提到某个文件路径（如 /data/report.pdf 或相对路径），填入 attachment；否则留空列表。
4. 必须且只能输出一个纯 JSON 对象（不是数组，不要包裹在 [] 里），字段严格匹配 MailDraft 的 schema，不要输出任何解释文字、不要使用代码块包裹、不要输出思考过程。
"""


def _build_mail_model():
    """
    用于邮件草稿抽取的严格 JSON 模型，直接复用 common.basic 里按配置构造好的
    StructuredModel（已经做了关 thinking + 温度 0 等稳定输出设置）。
    """
    return StructuredModel


def _parse_draft(text: str) -> MailDraft:
    """从模型原始输出里鲁棒还原 MailDraft（兼容数组包裹、代码块、字段类型错配）。"""
    import json
    import re

    raw = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start != -1 and end > start else {}

    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        data = {}

    for key in ("to", "cc", "bcc", "attachment"):
        val = data.get(key, [])
        if isinstance(val, str):
            val = [val] if val.strip() else []
        elif not isinstance(val, list):
            val = []
        data[key] = [str(v) for v in val if str(v).strip()]

    data["subject"] = str(data.get("subject", "") or "")
    data["body"] = str(data.get("body", "") or "")
    return MailDraft.model_validate(data)


def parse_mail(state: SendMailState):
    system_message = SystemMessage(content=ParseMailPrompt)
    human_message = HumanMessage(content=f"用户诉求：{state.get('user_message', '')}")
    response = _build_mail_model().invoke([system_message, human_message])
    content = response.content if isinstance(response.content, str) else str(response.content)
    draft = _parse_draft(content)

    print("=" * 60)
    print(f"[parse_mail] 抽取到邮件草稿：to={draft.to} subject={draft.subject!r} attachment={draft.attachment}")
    print("=" * 60)

    state["to"] = draft.to
    state["cc"] = draft.cc
    state["bcc"] = draft.bcc
    state["subject"] = draft.subject
    state["body"] = draft.body
    state["attachment"] = draft.attachment
    return state


def check_auth(state: SendMailState):
    install = ensure_cli_installed()
    if not install["ok"]:
        state["auth_ok"] = False
        state["response"] = f"CLI 未就绪：{install['detail']}"
        return state

    st = parse_auth_status()
    if token_is_usable(st, buffer_seconds=TOKEN_EXPIRY_BUFFER):
        print(f"[check_auth] 凭据有效，剩余约 {st.get('seconds_left', 0):.0f}s")
        state["auth_ok"] = True
        return state

    if not st.get("logged_in"):
        reason = "本地未发现 OAuth 凭据（未登录）"
    elif st.get("token_status") and st["token_status"] != "valid":
        reason = f"凭据状态为 {st['token_status']}"
    elif st.get("seconds_left") is not None:
        reason = f"凭据即将过期（剩余 {st['seconds_left']:.0f}s < 缓冲 {TOKEN_EXPIRY_BUFFER}s）"
    else:
        reason = "凭据状态未知（无法解析过期时间）"
    print(f"[check_auth] 凭据不可用：{reason}")
    state["auth_ok"] = False
    state["auth_reason"] = reason
    return state


def _extract_auth_url(res: dict) -> str:
    combined = (res.get("stdout", "") + "\n" + res.get("stderr", "")).strip()
    for line in combined.splitlines():
        line = line.strip()
        if line.startswith("http") and any(k in line for k in ("auth", "token", "oauth", "login")):
            return line
    if combined.startswith("http"):
        return combined.splitlines()[0].strip()
    return ""


def ensure_auth(state: SendMailState):
    reason = state.get("auth_reason", "")
    print(f"[ensure_auth] {'（重新）授权：' + reason if reason else '未检测到有效授权，触发 OAuth 登录'}")
    res = auth_login()
    state["auth_url"] = _extract_auth_url(res)
    return state


def should_send(state: SendMailState):
    """授权检查后分流：CLI 硬错误直接结束；凭据不可用走 ensure_auth；可用则直接 do_send。"""
    if state.get("response", ""):
        return END
    if not state.get("auth_ok", False):
        return "ensure_auth"
    return "do_send"


def _extract_confirmation(text: str) -> str:
    """
    从 CLI 输出里提取 confirmation-token（ctk_xxx）。
    注意：新版 agently-cli 的 ctk 是 UUID 形式含横杠，必须把 '-' 视为合法字符，
    否则会截断为残缺 token，导致 phase2 返回 exit=6（token invalid）。
    """
    for line in text.splitlines():
        if "ctk_" in line:
            start = line.find("ctk_")
            end = start
            while end < len(line) and (line[end].isalnum() or line[end] in ("_", "-")):
                end += 1
            return line[start:end]
    return ""


def build_preview_text(state: SendMailState) -> str:
    """生成展示给用户确认的邮件草稿预览文本。"""
    to = state.get("to", []) or []
    cc = state.get("cc", []) or []
    bcc = state.get("bcc", []) or []
    subject = state.get("subject", "") or ""
    body = state.get("body", "") or ""
    att = state.get("attachment", []) or []
    lines = [
        "📧 即将发送邮件，请确认：",
        f"  收件人(to)  ：{', '.join(to) if to else '（空）'}",
        f"  抄送(cc)    ：{', '.join(cc) if cc else '（空）'}",
        f"  密送(bcc)   ：{', '.join(bcc) if bcc else '（空）'}",
        f"  主题        ：{subject or '（空）'}",
        f"  附件        ：{', '.join(att) if att else '（无）'}",
        "  ── 正文 ──",
        body or "（空）",
    ]
    return "\n".join(lines)


def _send_fail(res: dict, stage: str) -> str:
    """按 agently-cli skill 的 exit code 语义给出针对性提示（不自动重试）。"""
    rc = res.get("returncode", -1)
    raw = (res.get("stderr") or "").strip() or (res.get("stdout") or "").strip()
    msg = f"{stage}失败（exit={rc}）：{raw}"
    if rc == 2:
        msg += "\n\n⚠️ 参数不合规。请检查收件人地址、附件路径（相对路径且 ≤3 个）、主题/正文是否齐全后重试。"
    elif rc == 3:
        msg += "\n\n⚠️ 授权已失效，请重新发送邮件以触发重新授权。"
    elif rc == 6:
        msg += (
            "\n\n⚠️ 该邮件被服务端永久拒绝（可能命中以下情形之一）：\n"
            "  1) 内容触发反垃圾/反滥用风控（含敏感词、营销/诱导性词汇等）\n"
            "  2) 收件人地址已退订/被加入黑名单/不存在/已被删除\n"
            "  3) 其他业务规则限制\n"
            "建议：修改邮件内容或更换收件人后，重新发起发送请求。"
        )
    elif rc == 7:
        msg += "\n\n⚠️ 触发发送限频。请稍候再试。"
    return msg


def do_send(state: SendMailState):
    """
    同一轮无 interrupt 连续「拿 ctk → 用 ctk」：
      1. 不带 token 调 CLI 拿 ctk_xxx；
      2. 立刻携带该 ctk 真正发送。
    ctk 有效期较短，不能把 interrupt 夹在中间；用户确认已由上游 preview 的 interrupt 完成。
    """
    print("=" * 60)
    print("[do_send] 同轮连续执行「拿 token → 用 token」……")
    print("=" * 60)

    kwargs = dict(
        to=state.get("to", []),
        subject=state.get("subject", ""),
        body=state.get("body", ""),
        cc=state.get("cc", []),
        bcc=state.get("bcc", []),
        attachment=state.get("attachment", []),
    )

    res1 = send_mail(**kwargs, confirmation_token=None)
    if res1["returncode"] != 0:
        state["response"] = _send_fail(res1, "预发送")
        return state

    token = _extract_confirmation(res1["stdout"] + "\n" + res1["stderr"])
    if not token:
        state["response"] = f"未从预发送结果中提取到 confirmation-token：{res1['stdout'].strip()}"
        return state

    res2 = send_mail(**kwargs, confirmation_token=token)
    if res2["returncode"] == 0:
        state["response"] = f"邮件已发送成功 ✅\n{res2['stdout'].strip()}"
    else:
        state["response"] = _send_fail(res2, "发送")
    return state


def build_send_mail_graph():
    builder = StateGraph(SendMailState)
    builder.add_node("parse_mail", parse_mail)
    builder.add_node("check_auth", check_auth)
    builder.add_node("ensure_auth", ensure_auth)
    builder.add_node("do_send", do_send)

    builder.add_edge(START, "parse_mail")
    builder.add_edge("parse_mail", "check_auth")
    builder.add_conditional_edges("check_auth", should_send)
    builder.add_edge("ensure_auth", "do_send")
    builder.add_edge("do_send", END)
    return builder.compile()