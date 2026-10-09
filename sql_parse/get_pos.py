from typing import Annotated
from operator import add
from common.basic import SqlParseState
from common.config import get_runtime_cfg
from langgraph.graph import StateGraph, START, END
from .plan_executor import make_plan, execute_plan, replan_step, should_execute

class GetPosState(SqlParseState):
    input: str
    plan: list
    past_steps: Annotated[list, add]
    response: str

def start_node(state: GetPosState):
    table_name = state["table_name"]
    action = state["action"]
    select_columns = state["select_columns"]
    condition = state["condition"]
    offset = state["offset"]
    limit = state["limit"]
    code_path = state["code_path"]
    input = (f"请在项目目录 {code_path} 下（这是一个绝对路径，所有涉及路径的工具调用都必须使用这个绝对路径，不要使用 '.'），"
             f"找到调用符合如下条件的SQL的代码位置："
                      f"表名：{table_name}"
                      f"动作：{action}"
                      f"选择列：{select_columns}"
                      f"条件：{condition}"
                      f"offset（0也可能表示未设置）：{offset}"
                      f"limit（0也可能表示未设置）：{limit}\n 输出格式形如/Users/home/user/sandhub/internal/impl.go:50")
    return {"input": input, "max_step": int(get_runtime_cfg().get("max_step", 20)), "cur_step": 0, "code_path": code_path}

def get_pos_end(state: GetPosState):
    return {"position": state["response"]}

def build_get_pos_graph():
    builder = StateGraph(GetPosState)
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