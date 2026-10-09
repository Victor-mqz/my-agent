from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from typing import Annotated, Union
import operator
from langchain_core.prompts import ChatPromptTemplate
from common.basic import PlanModel, ExeModel, AnswerModel
from common.config import get_sql_parse_cfg
from trustcall import create_extractor
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from .tools import read_file, write_file, create_file, list_project_directory, grep_file
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

class PlanExecute(TypedDict):
    input: str
    plan: list[str]
    past_steps: Annotated[list, operator.add]
    response: str
    max_step: int
    cur_step: int
    code_path: str

class Plan(BaseModel):
    steps: list[str] = Field(description="The steps to execute the plan.")

class Act(BaseModel):
    action: Union[Plan, str] = Field(description="Next step plan or final response.")

planner_prompt = ChatPromptTemplate.from_template("""
项目绝对路径 code_path = {code_path}（后续所有涉及文件/目录参数的工具调用，都必须使用这个绝对路径，不要用 '.' 或相对路径）。
请为实现目标制定一个简单的分布计划，以Json格式输出步骤列表。每一步必须写明需要操作的绝对路径。
目标为：{input}
""")

replanner_prompt = ChatPromptTemplate.from_template("""
项目绝对路径 code_path = {code_path}（所有工具调用均需使用该绝对路径）。
原始目标：{input}
原始计划：{plan}
已完成步骤：{past_steps}

请根据已完成的结果更新计划。判断规则：
1. 已完成步骤中的每条记录形如 (step_desc, result_repr)。result_repr 中若包含 "tool_trace=" 字样，其后即为本步骤实际发生的工具调用列表；不含则表示模型没有调用任何工具。
2. 如果原始计划或目标中要求创建/修改文件，则必须在 past_steps 的某一条 tool_trace 里看到对应的 create_file 或 write_file 工具调用且返回 "[create_file ok]" / 没有 error 前缀，才算真正完成；否则视为**未完成**，必须继续规划让执行代理去实际调用工具。
3. 如果 past_steps 中出现 "⚠️ WARNING: model CLAIMED to have written a file but NO create_file / write_file was actually called" 字样，说明模型只是口头声称完成，实际未落盘，必须继续规划一步"使用 create_file 工具创建 <绝对路径> 文件"来补做。
4. 只有当所有目标要求的动作都在 tool_trace 中被真实执行过，才可以直接输出最终回复；否则输出剩余步骤列表。
""")

plan_extractor = create_extractor(PlanModel, tools=[Plan], tool_choice="Plan")

def make_plan(state: PlanExecute):
    code_path = state.get("code_path", "")
    system_msg = planner_prompt.format(input=state["input"], code_path=code_path)
    planner = plan_extractor.invoke([SystemMessage(content=system_msg)])
    plan = planner["responses"][0].steps
    print("\n" + "=" * 60)
    print("[make_plan] 生成计划，共 {} 步（code_path={}）：".format(len(plan), code_path))
    for i, step in enumerate(plan, 1):
        print(f"  {i}. {step}")
    print("=" * 60)
    return {"plan": plan}

executor_system_prompt = (
    "你是一个执行代理。项目根目录（绝对路径）为：{code_path}。\n"
    "调用任何需要路径参数的工具时，必须使用该绝对路径或其子路径，\n"
    "严禁传入 '.'、'./'、相对路径，也不要自己编造其他路径。\n"
    "\n"
    "工具使用铁律（务必遵守以节省 token）：\n"
    "1. list_project_directory(path)：只用于目录，返回该目录下的一层内容。\n"
    "   - 返回值里 dirs[*].path 是子目录，若想深入请再次调用 list_project_directory。\n"
    "   - 返回值里 files[*].path 是文件，若想看文件内容请优先用 grep_file 定位，再用 read_file 精读。\n"
    "2. grep_file(pattern, path, context=3)：在文件或目录中做正则搜索，返回命中行及其上下文。\n"
    "   - 只要目标是‘找某个函数/符号/字符串在哪儿’、‘看某处代码怎么写’，一律先用 grep_file，不要一上来就 read_file 整个文件。\n"
    "   - path 可以是文件（只搜该文件）或目录（按代码白名单递归搜）。\n"
    "3. read_file(path, start_line=..., end_line=...)：读取单个文件；path 必须是文件的绝对路径。\n"
    "   - 严禁一上来就 read_file 整个大文件——太浪费 token。\n"
    "   - 推荐流程：先 grep_file 定位到关键行号 L，然后 read_file(path, start_line=L-20, end_line=L+40) 精读上下文窗口。\n"
    "   - 返回内容每行都带绝对行号前缀（形如 '  123: xxx'），后续 write_file 生成的 diff 请使用这些真实行号。\n"
    "   - 严禁把目录路径传给 read_file。\n"
    "4. write_file(path, diff)：只用于修改**已存在**的文件，diff 必须是 unified-diff（git diff）字符串，行号需与文件当前内容匹配。\n"
    "   - 若目标文件尚不存在，禁止调用 write_file，请改用 create_file。\n"
    "5. create_file(path, content)：只用于创建**新文件**，content 是文件的完整文本内容（不是 diff）。\n"
    "   - 若目标文件已存在，禁止用 create_file 覆盖它，请改用 write_file 生成 diff 进行修改。\n"
    "\n"
    "【严禁 shell 口嗨】你没有执行 shell 命令的能力。禁止在最终回复里说\n"
    "   “已通过 touch 创建…”、“已 echo 写入…”、“已用 cat 生成…” 等类似表达。\n"
    "   唯一合法的创建新文件方式是真实调用 create_file 工具；只要你没调过 create_file，\n"
    "   就不许声称文件已创建——否则该步骤会被判定为失败并强制回炉重做。\n"
    "\n"
    "完成当前步骤后，请用简短的自然语言总结你做了什么以及关键结果。"
)

# 使用 create_react_agent，模型请求工具调用后会由 agent 自动执行工具并把结果喂回模型，
# 直到模型不再请求工具、给出最终回复为止。
executor = create_react_agent(
    ExeModel,
    tools=[read_file, write_file, create_file, list_project_directory, grep_file],
)

def _format_tool_args(args):
    try:
        s = str(args)
    except Exception:
        s = repr(args)
    if len(s) > 500:
        s = s[:500] + "...(truncated)"
    return s

def _format_tool_result(content):
    if not isinstance(content, str):
        content = str(content)
    if len(content) > 800:
        content = content[:800] + "...(truncated)"
    return content

def execute_plan(state: PlanExecute):
    task = state["plan"][0]
    cur_step = state.get("cur_step", 0)
    code_path = state.get("code_path", "")
    system_msg = SystemMessage(content=executor_system_prompt.format(code_path=code_path))
    task_formatted = (
        f"项目绝对路径（code_path）：{code_path}\n"
        f"请执行以下步骤（涉及路径时必须使用上面的绝对路径）：{task}"
    )
    print("\n" + "-" * 60)
    print(f"[execute_plan] 第 {cur_step + 1} 步任务（code_path={code_path}）：{task}")

    # react agent 的输入是 {"messages": [...]}，返回值也是 {"messages": [...]}
    agent_response = executor.invoke(
        {"messages": [system_msg, HumanMessage(content=task_formatted)]}
    )
    messages = agent_response.get("messages", [])

    # 打印本次 react 循环里所有 AIMessage 的 tool_calls 和 ToolMessage 的结果
    tool_trace = []
    for m in messages:
        if isinstance(m, AIMessage):
            tcs = getattr(m, "tool_calls", None) or []
            if tcs:
                print(f"[execute_plan] <AI> 请求调用 {len(tcs)} 个工具：")
                for tc in tcs:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                    args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                    print(f"    - {name}({_format_tool_args(args)})")
                    tool_trace.append({"tool": name, "args": args})
            elif m.content:
                print(f"[execute_plan] <AI> 文本：{m.content!r}")
        elif isinstance(m, ToolMessage):
            result = _format_tool_result(m.content)
            print(f"[execute_plan] <Tool:{m.name}> 结果：{result}")
            tool_trace.append({"tool": m.name, "result": result})

    # 最后一条通常是 AIMessage 的最终回复
    final_text = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content and not (getattr(m, "tool_calls", None) or []):
            final_text = m.content
            break
    print(f"[execute_plan] 最终回复：{final_text!r}")
    print("-" * 60)

    # 把最终回复 + 工具轨迹一并塞进 past_steps，让 replan_step 能看到实际发生了什么
    result_repr = final_text
    if tool_trace:
        result_repr = (final_text + "\n" if final_text else "") + "tool_trace=" + str(tool_trace)

    # === 汇总日志：这一步实际调用了哪些工具（防止模型口嗨骗过 replan）===
    called_tools = [t.get("tool") for t in tool_trace if "tool" in t and "args" in t]
    write_tools = [t for t in called_tools if t in ("create_file", "write_file")]
    print(f"[execute_plan] 本轮工具调用列表：{called_tools}")
    print(f"[execute_plan] 本轮写盘工具调用：{write_tools if write_tools else '(无)'}")

    # 口嗨检测关键词：覆盖中文自述 + shell/命令式动词 + 典型方案文件名
    # （以前只认 "已创建/已写入"，模型用 touch/echo 一绕就过）
    CLAIM_KEYWORDS = (
        "已创建", "已写入", "已记录", "已保存", "已生成", "已输出",
        "创建文件", "写入文件", "创建文档", "生成文档", "生成方案", "创建方案",
        "touch ", "echo ", "cat >", "cat <<", ">>", "`cat", "`echo",
        ".md", "optimization_plan",
    )
    if not write_tools and any(kw in final_text for kw in CLAIM_KEYWORDS):
        print("[execute_plan] ⚠️  警告：模型声称已创建/写入文件（含 shell 口嗨），但本轮 tool_trace 中并没有 create_file/write_file 调用！")
        result_repr = (
            (final_text + "\n" if final_text else "")
            + "⚠️ WARNING: model CLAIMED to have created/written a file (possibly via shell cmds like touch/echo) "
            + "but NO create_file / write_file was actually called. "
            + "The file was NOT created on disk. This step must be redone with an actual create_file tool call.\n"
            + "tool_trace=" + str(tool_trace)
        )
    return {"past_steps": [(task, result_repr)]}

replan_extractor = create_extractor(PlanModel, tools=[Act], tool_choice="Act")

# ---------------------------------------------------------------------------
# 最终方案生成节点（answer）
# ---------------------------------------------------------------------------
# 设计原则：executor / replanner 只关心「怎么探索代码、执行到哪了」，
# 永远不承载「最终方案要长什么样」的格式约束，避免 prompt 无限膨胀。
# 所有「输出要详细 / 要带 diff / 要分步骤」这类诉求，都集中到 OUTPUT_CONSTRAINTS，
# 以后遇到新的 case 只需 append 一行，无需改动 prompt 模板本身。
#
# 注意：这份 prompt 产出的文本会被整体写入 `<code_path>/optimization_plan.md`，
# 读者是**工程师**，想拿到就能改代码；绝不是"Agent 的执行日志/计划汇报"。
# 因此下面所有约束都在强调「以慢 SQL 为主体」而不是「以 Agent 执行流程为主体」。
OUTPUT_CONSTRAINTS: list[str] = [
    "最终输出的是一份**面向工程师的《慢 SQL 优化方案》Markdown 文档**，直接复制进 .md 文件即可阅读。",
    "严禁出现任何 Agent 过程相关的词汇与结构，包括但不限于："
    "`原始目标`、`原始计划`、`已完成步骤`、`past_steps`、`tool_trace`、`执行代理`、`规划`、`replan`、`Agent`、`本次任务`、"
    "`第 N 步`（指代 Agent 自己的步骤时）；也不要以“我/Agent 做了什么”为叙事主语。",
    "文档结构固定：\n"
    "   # 慢 SQL 优化方案\n"
    "   ## 1. 概述（这次扫描到了几条慢 SQL，分别位于哪些文件/函数，整体风险等级）\n"
    "   ## 2. 慢 SQL 清单与逐条优化\n"
    "   （每条慢 SQL 一个 `### 2.x <简短命名>` 子节，内部必须包含：\n"
    "     - **定位**：目标文件绝对路径 + 关键函数/符号 + 真实行号区间\n"
    "     - **当前 SQL**：原样贴出 SQL 片段（用 ```sql 代码块）\n"
    "     - **慢的原因**：从索引/扫描量/N+1/锁/排序/子查询等角度解释，必须引用真实行号或代码片段\n"
    "     - **优化方案**：代码改动给出 unified diff；若需新增文件给出路径与骨架；若需加索引给出 DDL\n"
    "     - **预期收益与风险**：QPS/耗时/锁影响；可能的副作用\n"
    "     - **验证方式**：如何自测或回归，例如 EXPLAIN 对比、单测入口、灰度步骤）\n"
    "   ## 3. 建议的落地顺序（按依赖/风险排序列出 2.x 的执行顺序）\n"
    "   ## 4. 需要人工确认的事项（如拿不准的字段/业务含义）",
    "一切结论必须有真实代码事实支撑；若扫描过程没有查到足够信息，宁可在对应子节写"
    "“当前信息不足，需补充 XXX” 也**不要编造**文件名、行号、SQL 文本或表结构。",
    "语言：清晰的中文；SQL/代码/路径用反引号或代码块包裹。整篇文档自成一体，读者无需了解 Agent 内部运行过程。",
]

answer_prompt = ChatPromptTemplate.from_template(
    """\
你现在要产出一份**写入 `optimization_plan.md` 的《慢 SQL 优化方案》文档**。

读者是工程师，他们**不关心也不应该看到** Agent 的执行流程、计划步骤、工具调用记录。
请把下面的信息当作**事实材料库**查阅使用，但**最终文档中不要复述它们的形式**：

[事实材料库 - 仅供你内部查阅]
- 用户原始诉求（用于理解扫描目标，不要出现在文档正文）：{input}
- 过程中真实观察到的代码/SQL 片段与行号（这是你唯一可信的事实来源，禁止编造；但不要把它当作文档章节罗列出来）：{past_steps}

请严格按下面的约束，产出一份独立成文的 Markdown 文档：
{constraints}
"""
)
answer_chain = answer_prompt | AnswerModel

def _mentions_markdown_output(input_text: str) -> bool:
    """判断原始目标是否明确要求落一个 markdown 方案文档。"""
    if not isinstance(input_text, str):
        return False
    t = input_text.lower()
    return ("markdown" in t) or ("方案" in input_text and "文档" in input_text) or (".md" in t)


def _already_created_markdown(past_steps: list, code_path: str) -> bool:
    """
    扫 past_steps 的 tool_trace 字符串，判断是否已有一次成功的 create_file 调用
    用于创建 .md 文档。str(dict) 默认是单引号，而 json.dumps 会用双引号，
    这里两种格式都认，防止因序列化方式不同而漏判。
    """
    needles = ("'tool': 'create_file'", '"tool": "create_file"')
    for item in past_steps or []:
        try:
            _, result_repr = item
        except Exception:
            continue
        if not isinstance(result_repr, str):
            continue
        if any(n in result_repr for n in needles) and ".md" in result_repr:
            return True
    return False


def replan_step(state: PlanExecute):
    code_path = state.get("code_path", "")
    system_msg = replanner_prompt.format(
        input=state["input"], plan=state["plan"], past_steps=state["past_steps"], code_path=code_path,
    )
    cur_step = state.get("cur_step", 0) + 1
    print("\n" + "*" * 60)
    print(f"[replan_step] 已完成步数：{cur_step}")

    # —— trustcall 结果兜底 ——
    # Qwen3-8b 偶发不触发 Act 工具调用（thinking 泄漏 / 纯文本返回），
    # 这种情况下 replanner["responses"] 会是空 list，直接 [0] 就是
    # IndexError: list index out of range。这里统一吃掉异常并退化为
    # “保留原计划继续执行”，不让整个子图崩溃。
    action = None
    try:
        replanner = replan_extractor.invoke([SystemMessage(content=system_msg)])
        responses = replanner.get("responses", []) if isinstance(replanner, dict) else []
        if responses:
            action = getattr(responses[0], "action", None)
        else:
            print("[replan_step] ⚠️  replan_extractor 未返回任何 Act 结构化结果，"
                  "将退化为“保留原计划继续执行”")
    except Exception as e:
        print(f"[replan_step] ⚠️  replan_extractor 调用异常：{e!r}，"
              "将退化为“保留原计划继续执行”")

    # 如果连 action 都没拿到，就既不改 plan 也不出 response，
    # 让主循环 should_execute 检查到 response 为空且 cur_step 未爆 → 继续执行下一步。
    if action is None:
        remaining_plan = state.get("plan", []) or []
        # 若原计划已空（上一轮刚执行完最后一步），避免死循环，直接生成最终方案。
        if not remaining_plan:
            answer = answer_chain.invoke({
                "input": state["input"],
                "past_steps": state["past_steps"],
                "constraints": "\n".join(f"{i}. {c}" for i, c in enumerate(OUTPUT_CONSTRAINTS, 1)),
            }).content
            print(f"[replan_step] replanner 失败且原计划已空，直接生成最终方案（长度={len(answer)}）")
            print("*" * 60)
            return {"response": answer, "cur_step": cur_step}
        # 否则保留剩余计划，等待下一轮执行
        print(f"[replan_step] 保留剩余 {len(remaining_plan)} 步计划，等待下一轮执行")
        print("*" * 60)
        return {"cur_step": cur_step}

    # —— 硬兜底闸门 ——
    # 当原始目标明确要求产出 markdown 方案文档，但 past_steps 中确实
    # 没有一次成功的 create_file(.md) 调用时，无论 LLM 现在想"直接完结"
    # 还是继续规划什么，都强制覆写为「用 create_file 真正落盘文档」这一步。
    # 这样可以彻底堵住 "shell 口嗨 / 自述完成" 骗过 replanner 的漏洞。
    needs_md = _mentions_markdown_output(state.get("input", ""))
    md_done = _already_created_markdown(state.get("past_steps", []), code_path)
    if needs_md and not md_done:
        # 先生成一份详尽方案文本（和最终答复用的是同一个 answer_chain），
        # 再把"用 create_file 把这份文本落到绝对路径"的指令塞回执行层。
        plan_text = answer_chain.invoke({
            "input": state["input"],
            "past_steps": state["past_steps"],
            "constraints": "\n".join(f"{i}. {c}" for i, c in enumerate(OUTPUT_CONSTRAINTS, 1)),
        }).content
        md_name = get_sql_parse_cfg().get("output_markdown_name", "optimization_plan.md")
        target_md = f"{code_path.rstrip('/')}/{md_name}"
        forced_step = (
            f"调用 create_file 工具（禁止用 touch/echo/shell），path={target_md}。\n"
            f"content 参数 = 下面三引号中的完整 Markdown 文档全文，必须 **1:1 原样**写入目标文件；\n"
            f"禁止在 content 前后添加任何自然语言说明、解释、总结、代码块围栏或提示语；\n"
            f"禁止修改、省略、改写、重排其中任何一行；禁止把它当成‘方案摘要’再二次加工。\n"
            f"这段内容本身就是一份面向工程师的《慢 SQL 优化方案》Markdown 文档，直接读、直接 commit。\n"
            f"==== 文档全文 BEGIN ====\n"
            f"\"\"\"\n{plan_text}\n\"\"\"\n"
            f"==== 文档全文 END ===="
        )
        print("[replan_step] ⚠️  检测到目标要求 markdown 文档但 past_steps 无真实 create_file，"
              "强制插入补做步骤：create_file -> " + target_md)
        print("*" * 60)
        return {"plan": [forced_step], "cur_step": cur_step}

    if isinstance(action, str):
        # 不再把 str 直接当最终方案，而是交给 answer_chain 生成详尽、可落地的方案。
        # 这样「输出详细度/格式」的约束全部收敛到 answer_prompt，executor/replanner 永不膨胀。
        answer = answer_chain.invoke({
            "input": state["input"],
            "past_steps": state["past_steps"],
            "constraints": "\n".join(f"{i}. {c}" for i, c in enumerate(OUTPUT_CONSTRAINTS, 1)),
        }).content
        print(f"[replan_step] 生成最终详细方案（长度={len(answer)}）：")
        print("*" * 60)
        return {"response": answer, "cur_step": cur_step}
    else:
        steps = getattr(action, "steps", None) or []
        if not steps:
            # LLM 返回了 Plan 但 steps 空（或字段缺失）：保留原计划继续执行，别清空。
            print("[replan_step] ⚠️  Plan.steps 为空，保留原计划继续执行")
            print("*" * 60)
            return {"cur_step": cur_step}
        print(f"[replan_step] 更新计划，剩余 {len(steps)} 步：")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
        print("*" * 60)
        return {"plan": steps, "cur_step": cur_step}

def should_execute(state: PlanExecute):
    # 还没得到最终 response，且没超过最大步数 → 继续执行下一步
    # 否则（已经拿到 response 或步数用尽）→ 结束
    if not state.get("response", "") and state.get("cur_step", 0) < state.get("max_step", 0):
        return "execute_plan"
    return END