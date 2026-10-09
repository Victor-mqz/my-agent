from typing import Annotated
from operator import add
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from .plan_executor import make_plan, execute_plan, replan_step, should_execute
from common.basic import SqlParseState
from common.config import get_runtime_cfg, get_sql_parse_cfg

class OptimizeState(SqlParseState):
    input: str
    plan: list
    past_steps: Annotated[list, add]
    response: str

def start_node(state: OptimizeState):
    position = state["position"]
    index = state["index"]
    table_name = state["table_name"]
    code_path = state["code_path"]
    md_name = get_sql_parse_cfg().get("output_markdown_name", "optimization_plan.md")
    max_step = int(get_runtime_cfg().get("max_step", 20))
    input = (f"请优化{code_path}项目里{position}位置的SQL：\n"
             f"已知该调用会产生慢查询，表{table_name}的索引是{index}\n"
             f"要求：1.如果索引已经超过15个，不要新增索引；"
             f"2.可以尝试从 SQL 调用的用途上来进行分析，找到精简查询的办法；"
             f"3.把优化方案落到当前目录中，新建一个 markdown 文档 {md_name}；"
             f"4.优化方案需要有详细的执行步骤，但是整体方案不要超过500行，用中文，方案包含："
             f" 1）背景说明；2）方案概述；3）方案详解，包含如何做，为什么这样做可行；4）具体实施方法，包含代码修改方法/索引修改方法等\n"
             f"注意：所有涉及路径的工具调用都必须使用绝对路径 {code_path}，不要使用 '.'。"
             )
    return {"input": input, "max_step": max_step, "cur_step": 0, "code_path": code_path}

def build_optimize_graph():
    builder = StateGraph(OptimizeState)
    builder.add_node("start", start_node)
    builder.add_node("make_plan", make_plan)
    builder.add_node("execute_plan", execute_plan)
    builder.add_node("replan_step", replan_step)

    builder.add_edge(START, "start")
    builder.add_edge("start", "make_plan")
    builder.add_edge("make_plan", "execute_plan")
    builder.add_edge("execute_plan", "replan_step")
    builder.add_conditional_edges("replan_step", should_execute)
    return builder.compile()