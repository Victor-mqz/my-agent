"""
send_mail 工具集：把 agently-cli 命令行封装成可供 react agent 调用的函数。

主要步骤来自 https://agent.qq.com/doc/cli-setup.md 的邮件功能说明：
  1. 安装 CLI:   npm install -g @tencent-qqmail/agently-cli
  2. 授权登录:   agently-cli auth login        （OAuth，需人工点击授权 URL）
  3. 验证授权:   agently-cli +me
  4. 发送邮件:   agently-cli message +send      （两阶段确认，先拿 ctk 再带 --confirmation-token）

设计说明：
  - 所有函数都是“薄封装”：执行 shell 命令，返回 stdout/stderr 文本，由调用方（LLM）自行解读。
  - 不在这里做复杂的状态机；两阶段确认、OAuth 等待等流程交给 `send_mail.py` 里的 plan-executor 编排。
  - 命令失败时返回非 0 exit code + 错误信息，交由上层按 skill 的“错误处理”规则处理。
  - CLI 可执行文件名、超时时间、最低版本等参数全部走 config.yaml，不再硬编码。
"""
import os
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Optional

from common.config import get_send_mail_cfg


# ---------------------------------------------------------------------------
# 从配置派生出来的常量（第一次 import 时固化；修改 yaml 后请重启进程）
# ---------------------------------------------------------------------------
_SM_CFG = get_send_mail_cfg()
_CLI_CFG = _SM_CFG.get("cli", {}) or {}
_TIMEOUTS = _SM_CFG.get("timeouts", {}) or {}

CLI_BIN: str = _CLI_CFG.get("binary", "agently-cli")
CLI_MIN_VERSION: str = _CLI_CFG.get("min_version", "1.0.18")
CLI_INSTALL_CMD: list = list(_CLI_CFG.get("install_cmd", ["npm", "install", "-g", "@tencent-qqmail/agently-cli"]))

TIMEOUT_VERSION: int = int(_TIMEOUTS.get("version", 30))
TIMEOUT_STATUS: int = int(_TIMEOUTS.get("status", 60))
TIMEOUT_LOGIN: int = int(_TIMEOUTS.get("login", 600))
TIMEOUT_SEND: int = int(_TIMEOUTS.get("send", 120))
TIMEOUT_INSTALL: int = int(_TIMEOUTS.get("install", 300))


def _run(cmd: list[str], timeout: int = 120, cwd: Optional[str] = None) -> dict:
    """
    执行一条 shell 命令，返回结构化结果。

    :param cmd: 命令参数列表（不使用 shell=True，避免注入）
    :param timeout: 超时秒数
    :param cwd: 工作目录
    :return: {"returncode": int, "stdout": str, "stderr": str}
    """
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }
    except FileNotFoundError:
        return {
            "returncode": 127,
            "stdout": "",
            "stderr": (
            f"command not found: {cmd[0]!r}. "
                f"请先安装 CLI：`{' '.join(CLI_INSTALL_CMD)}`"
            ),
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": 124,
            "stdout": "",
            "stderr": f"command timed out after {timeout}s: {' '.join(cmd)}",
        }
    except Exception as e:  # noqa: BLE001 - 兜底，避免工具异常击穿 agent
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": f"unexpected error running {' '.join(cmd)}: {e}",
        }


def _cli_version() -> Optional[str]:
    """探测已安装的 agently-cli 版本，失败返回 None。"""
    res = _run([CLI_BIN, "--version"], timeout=TIMEOUT_VERSION)
    if res["returncode"] != 0:
        return None
    # 典型输出： "1.0.18" 或 "agently-cli/1.0.18 ..."
    for token in (res["stdout"] + " " + res["stderr"]).split():
        if token[0].isdigit() and "." in token:
            return token.strip().lstrip("v")
    return None


def ensure_cli_installed() -> dict:
    """
    检查 agently-cli 是否已安装；未安装则尝试全局安装。
    若版本过旧（< 1.0.18，旧版存在 confirmation-token 有效期过短等问题），
    返回 ok=True 但附 upgrade_hint 提示用户手动升级（不自动升级以免破坏环境）。

    :return: {"ok": bool, "installed": bool, "detail": str, "upgrade_hint": str}
    """
    if not _which_on_path(CLI_BIN):
        res = _run(CLI_INSTALL_CMD, timeout=TIMEOUT_INSTALL)
        if res["returncode"] == 0:
            return {"ok": True, "installed": True, "detail": f"{CLI_BIN} 安装成功", "upgrade_hint": ""}
        return {
            "ok": False,
            "installed": False,
            "detail": f"安装失败（exit={res['returncode']}）：{res['stderr'].strip() or res['stdout'].strip()}",
            "upgrade_hint": "",
        }

    # 已安装：检查版本
    ver = _cli_version()
    upgrade_hint = ""
    if ver:
        try:
            min_parts = [int(x) for x in CLI_MIN_VERSION.split(".")[:3]]
            parts = [int(x) for x in ver.split(".")[:3]]
            if parts < min_parts:
                upgrade_hint = (
                    f"当前 {CLI_BIN} 版本 {ver} 偏旧（建议 ≥ {CLI_MIN_VERSION}），"
                    f"旧版存在 confirmation-token 有效期过短等问题，"
                    f"建议手动升级：`{' '.join(CLI_INSTALL_CMD)}`"
                )
        except Exception:  # noqa: BLE001
            pass

    return {"ok": True, "installed": True, "detail": f"{CLI_BIN} 已安装（{ver or '未知版本'}）", "upgrade_hint": upgrade_hint}


def _which_on_path(name: str) -> bool:
    """shutil.which 的兜底（某些环境 PATH 解析异常时）。"""
    try:
        return shutil.which(name) is not None
    except Exception:  # noqa: BLE001
        return False


def auth_login() -> dict:
    """
    触发 OAuth 登录。注意：这是一个交互式长命令，会向 stdout/stderr 打印授权 URL。

    调用方（编排节点）应当：
      1. 以后台 pty 方式运行，从输出中提取原始授权 URL；
      2. 把 URL 原样展示给用户，提示「请点击或复制以下链接在浏览器中完成授权：」；
      3. 用户在浏览器完成授权后，命令会自动退出。

    本封装在非 pty 环境下直接运行，便于在 agent 流程中先探测是否已有有效凭据。
    若已有凭据，命令会快速返回；若没有，通常会阻塞等待/报错，上层应改用 interrupt 等待用户。

    :return: {"returncode": int, "stdout": str, "stderr": str}
    """
    return _run([CLI_BIN, "auth", "login"], timeout=TIMEOUT_LOGIN)


def auth_status() -> dict:
    """
    查看当前 OAuth 凭据和授权状态（无需交互）。

    :return: {"returncode": int, "stdout": str, "stderr": str}
    """
    return _run([CLI_BIN, "auth", "status"], timeout=TIMEOUT_STATUS)


def whoami() -> dict:
    """
    获取当前用户信息（邮箱地址 + alias 列表），用于验证授权是否生效。

    :return: {"returncode": int, "stdout": str, "stderr": str}
    """
    return _run([CLI_BIN, "+me"], timeout=TIMEOUT_STATUS)


def send_mail(
    to: list[str],
    subject: str,
    body: str,
    cc: Optional[list[str]] = None,
    bcc: Optional[list[str]] = None,
    attachment: Optional[list[str]] = None,
    body_file: Optional[str] = None,
    confirmation_token: Optional[str] = None,
) -> dict:
    """
    发送邮件（两阶段确认）。

    第一阶段（不带 confirmation_token）：CLI 返回 ctk_xxx 和一个 summary，
    调用方应展示 summary 给用户并停下等待确认，切勿在同一轮自确认。

    第二阶段（带 confirmation_token）：用户明确许可后，用相同参数 + token 再次调用完成发送。

    :param to: 收件人邮箱列表（可多个）
    :param subject: 邮件主题
    :param body: 邮件正文（纯文本；若提供 body_file 则优先用文件）
    :param cc: 抄送列表
    :param bcc: 密送列表
    :param attachment: 附件路径列表（相对路径，最多 3 个）
    :param body_file: 正文文件路径（与 body 二选一）
    :param confirmation_token: 第一阶段拿到的 ctk_xxx；为空表示第一阶段
    :return: {"returncode": int, "stdout": str, "stderr": str}
    """
    cmd = [CLI_BIN, "message", "+send"]
    for addr in (to or []):
        cmd += ["--to", addr]
    for addr in (cc or []):
        cmd += ["--cc", addr]
    for addr in (bcc or []):
        cmd += ["--bcc", addr]
    for att in (attachment or []):
        cmd += ["--attachment", att]

    if body_file:
        cmd += ["--body-file", body_file]
    else:
        cmd += ["--subject", subject, "--body", body]

    if confirmation_token:
        cmd += ["--confirmation-token", confirmation_token]

    return _run(cmd, timeout=TIMEOUT_SEND)


# ---------------------------------------------------------------------------
# OAuth 凭据时效处理：本地 token 仅 1 小时有效，需在过期前主动重授权
# ---------------------------------------------------------------------------
def parse_auth_status() -> dict:
    """
    解析 `agently-cli auth status` 输出，判断本地 OAuth 凭据是否可用，
    并算出距离过期的剩余秒数（用于提前重授权，避免发送时才发现过期）。

    :return: {
        "ok": bool,            # 命令本身是否执行成功
        "logged_in": bool,     # 是否存在凭据
        "token_status": str,   # valid / expired / ...（以 CLI 为准）
        "seconds_left": float|None,  # 距 expires_at 的剩余秒数；无法解析时为 None
        "expires_at": str,     # 原始过期时间字符串
        "raw": str,            # 原始 stdout
    }
    """
    res = auth_status()
    out = (res.get("stdout") or "").strip()
    result = {
        "ok": res["returncode"] == 0,
        "logged_in": False,
        "token_status": "",
        "seconds_left": None,
        "expires_at": "",
        "raw": out,
    }
    if not out:
        return result

    # 轻量 JSON 提取（不强制整体解析， tolerate 前后多余文本）
    try:
        # 找到第一个 { 到最后一个 } 之间的内容
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            import json
            data = json.loads(out[start:end + 1])
            d = (data.get("data") or {}) if isinstance(data, dict) else {}
            result["logged_in"] = bool(d.get("logged_in", False))
            result["token_status"] = str(d.get("token_status", "") or "")
            result["expires_at"] = str(d.get("expires_at", "") or "")
    except Exception:  # noqa: BLE001 - 解析失败则回退到保守判断
        pass

    # 计算剩余秒数（expires_at 形如 2026-10-09T03:42:10Z，UTC）
    if result["expires_at"]:
        try:
            exp = result["expires_at"].replace("Z", "+00:00")
            exp_dt = datetime.fromisoformat(exp)
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            result["seconds_left"] = max(0.0, (exp_dt - now).total_seconds())
        except Exception:  # noqa: BLE001
            result["seconds_left"] = None

    return result


def token_is_usable(status: dict, buffer_seconds: float = 300.0) -> bool:
    """
    判断凭据是否可立即使用：已登录、状态 valid、且剩余有效期 > 缓冲期。

    :param status: parse_auth_status 的返回值
    :param buffer_seconds: 缓冲秒数（默认 300=5 分钟），小于此值即视为“需要重授权”，
                           防止“检查通过但真正发送时已过期”的竞态。
    """
    if not status.get("ok") or not status.get("logged_in"):
        return False
    if status.get("token_status") and status["token_status"] != "valid":
        return False
    secs = status.get("seconds_left")
    if secs is None:
        # 无法解析过期时间：保守起见，仅当 token_status 明确为 valid 时才放行
        return status.get("token_status") == "valid"
    return secs > buffer_seconds
