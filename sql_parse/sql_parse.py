import json
import re

from pydantic import BaseModel, Field, ValidationError
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from common.basic import AnswerModel, FindPosGraphName, get_index_by_tablename, StructuredModel
from langgraph.graph import END, StateGraph, START
from .get_pos import build_get_pos_graph
from common.basic import SqlParseState
from .optimize import build_optimize_graph

class SqlParseResult(BaseModel):
    model_config = {"populate_by_name": True}

    table_name: str = Field(default="", alias="table", description="The name of the table to query")
    action: str = Field(default="", description="The action to perform on the table. e.g. select/insert/update/delete")
    select_columns: list = Field(default_factory=list, alias="select", description="The columns to select from the table")
    condition: str = Field(default="", description="The condition to filter the rows")
    offset: int = Field(default=0, description="The offset to start the query")
    limit: int = Field(default=0, description="The limit to the number of rows to return")
    position: str = Field(default="", description="The position of the code where the query is located, the path is relative path which start with my-app. e.g. my-app/pkg/sql_parse.go:10")

SqlParsePrompt = """
You are a helpful assistant that can parse sql.
You are given a SQL query and you need to parse it and return a JSON object with the following fields (use exactly these field names and types):

- table_name (string): the target table name.
- action (string): one of "select" / "insert" / "update" / "delete".
- select_columns (array of strings): the columns to select. Empty array if not applicable.
- condition (string): the WHERE clause without the "WHERE" keyword. Empty string if none.
- offset (integer): OFFSET value, 0 if not specified.
- limit (integer): LIMIT value, 0 if not specified.
- position (string): the code location in the form "<relative_path>:<line_number>", e.g. "my-app/pkg/sql_parse.go:10". Empty string if unknown. Do NOT return only a number.

Return ONLY the JSON object, no extra commentary.
"""

def _build_parse_model():
    """
    SQL 结构抽取用的严格 JSON 模型，直接复用 common.basic.StructuredModel
    （按 config.yaml 中 models.structured 构造，默认温度 0 + 关 thinking）。
    """
    return StructuredModel


def _parse_result(text: str) -> SqlParseResult:
    """从模型原始输出里鲁棒还原 SqlParseResult（兼容数组包裹、代码块、think 泄漏）。"""
    raw = (text or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start:end + 1]) if start != -1 and end > start else {}

    if isinstance(data, list):
        data = data[0] if data else {}
    if not isinstance(data, dict):
        data = {}
    return SqlParseResult.model_validate(data)


def parse_sql(state: SqlParseState):
    system_message = SystemMessage(content=SqlParsePrompt)
    human_message = HumanMessage(
        content=(
            "Parse the sql below and return a JSON object matching the schema of SqlParseResult:\n"
            + state["sql"]
        )
    )
    model = _build_parse_model()
    try:
        structured_model = model.with_structured_output(SqlParseResult, method="json_mode")
        result: SqlParseResult = structured_model.invoke([system_message, human_message])
    except (ValidationError, ValueError, json.JSONDecodeError):
        # json_mode 偶发被 thinking 碎片污染时，走文本回退解析
        response = model.invoke([system_message, human_message])
        content = response.content if isinstance(response.content, str) else str(response.content)
        result = _parse_result(content)

    state["table_name"] = result.table_name
    state["action"] = result.action
    state["select_columns"] = result.select_columns
    state["condition"] = result.condition
    state["offset"] = result.offset
    state["limit"] = result.limit
    state["position"] = result.position
    return state

def should_get_position(state: SqlParseState):
    if state.get("position", ""):
        return "get_index"
    return FindPosGraphName

def get_index(state: SqlParseState):
    return {"index": get_index_by_tablename(state["table_name"])}

def build_sql_parse_graph():
    builder = StateGraph(SqlParseState)
    builder.add_node("parse_sql", parse_sql)
    builder.add_node(FindPosGraphName, build_get_pos_graph())
    builder.add_node("get_index", get_index)
    builder.add_node("optimize", build_optimize_graph())
    builder.add_edge(START, "parse_sql")
    builder.add_conditional_edges("parse_sql", should_get_position)
    builder.add_edge(FindPosGraphName, "get_index")
    builder.add_edge("get_index", "optimize")
    return builder.compile()