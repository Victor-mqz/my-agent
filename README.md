# my-agent

> 基于 LangGraph 的多意图 Agent：**自然语言输入 → 自动识别意图 → 分流到慢 SQL 分析 / 发邮件 / 闲聊子图 → 回到交互循环**。

## 项目简介

一句话：**用一个常驻对话式 Agent 替代"手工挑慢SQL + 手工翻代码 + 手工写方案 + 手工发邮件"的全链路工作**。

目前内置三类技能，可按 `config.yaml` 继续扩展：

| 技能 | 说明 |
| --- | --- |
| 慢 SQL 分析 | 给定代码仓绝对路径 + 一组慢 SQL，自动在仓库中定位调用点、结合索引信息产出一份面向工程师的《慢 SQL 优化方案》Markdown 文档。 |
| 发送邮件 | 自然语言描述收件人/主题/正文/附件 → LLM 抽取草稿 → 用户确认 → 走 `agently-cli` 的两阶段确认机制真实落信。 |
| 日常闲聊 | 兜底节点，顺带承载"用户说了句不属于上述两类技能"的场景。 |

## 技术栈

- **语言**：Python 3.10+
- **编排框架**：[LangGraph](https://github.com/langchain-ai/langgraph)（主图 + 子图 + `interrupt / Command.resume` 实现"等用户输入"）
- **LLM 协议**：OpenAI 兼容协议；默认接阿里云百炼（Qwen 系列），也支持本地 vLLM / Azure OpenAI / 公司内网网关
- **Agent 模式**：
  - 主图：意图识别 → 条件分流
  - sql_parse 子图：Plan-Execute-Replan（`trustcall` 做结构化规划）+ `create_react_agent` 跑工具调用
  - send_mail 子图：抽取草稿 → OAuth 授权 → `interrupt` 等用户确认 → 两阶段发送
- **配置**：YAML + 环境变量（见下文）
- **外部依赖**：
  - `agently-cli`（腾讯 QQ 邮箱 CLI，用于真实收发邮件）
  - 若不启用邮件技能，可只装 Python 依赖

## 目录结构

```
my-agent/
├── agent.py                 # 主入口：主图、意图识别、交互循环
├── config.yaml              # 全部可配置项（模型、CLI、意图、慢SQL、路径...）
├── requirements.txt         # Python 依赖
├── README.md
│
├── common/                  # 公共基础设施
│   ├── config.py            # YAML 配置加载器 + 便捷访问器
│   └── basic.py             # 按配置构造的 ChatOpenAI 实例、TokenUsage 回调、State 定义
│
├── sql_parse/               # 慢 SQL 分析子图
│   ├── build_graph.py       # 子图顶层：取 code_path → 读 SQL 列表 → 并行扇出子分析
│   ├── sql_parse.py         # 把一条 SQL 结构化（table/action/condition/...）
│   ├── get_pos.py           # 在仓库中定位该 SQL 的调用代码行
│   ├── optimize.py          # 产出面向工程师的优化方案 Markdown
│   ├── plan_executor.py     # Plan-Execute-Replan 核心（含防口嗨硬兜底）
│   └── tools.py             # read_file / write_file / grep / list_dir 等工具
│
└── send_mail/               # 发送邮件子图
    ├── build_graph.py       # 子图顶层：抽草稿 → 查授权 → 确认 → 发送
    ├── send_mail.py         # 节点实现（parse_mail / check_auth / do_send ...）
    └── tools.py             # agently-cli 命令行薄封装
```

## 快速开始

### 1. 克隆 & 创建虚拟环境

```bash
cd my-agent
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 3. 可选：安装 `agently-cli`（仅当你要用"发送邮件"技能）

```bash
npm install -g @tencent-qqmail/agently-cli
agently-cli --version   # 建议 ≥ 1.0.18
```

### 4. 配置模型凭据

默认接阿里云百炼（Qwen）。只需导出两个环境变量：

```bash
export OPENAI_API_KEY="你的-dashscope-key"
export OPENAI_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

> 想换其他模型？改 [config.yaml](./config.yaml) 的 `models.*` 节即可，`agent.py` 自己不感知任何模型名。

### 5. （可选）编辑 `config.yaml`

常改的几项：

| 字段 | 作用 |
| --- | --- |
| `runtime.default_code_path` | 慢 SQL 分析默认扫描哪个代码目录；留空则用启动时的当前工作目录。也可用 env `AGENT_CODE_PATH` 临时覆盖。 |
| `sql_parse.slow_sqls` | 待分析的慢 SQL 列表（上线前请改成从监控系统/CSV 读） |
| `sql_parse.default_index_list` | 表索引信息，用于优化建议 |
| `models.*.name` | 各任务使用的 LLM 名字 |
| `send_mail.cli.binary` | 若 `agently-cli` 不在 `PATH`，这里填绝对路径 |
| `intents` | 新增/修改意图（label + keywords + 分流到的主图节点） |

### 6. 启动

```bash
python agent.py
```

启动时会打印一次配置摘要，然后进入 claude-code 风格的交互循环：

```
============================================================
my-agent 启动中（配置摘要）：
  thread_id          = 10086
  user_id            = 1
  default_code_path  = /Users/you/src
  max_step           = 20
  summary_threshold  = 5
  意图：['慢Sql分析', '发送邮件', '结束对话', '日常闲聊']
============================================================

请输入你的消息（输入 exit / quit / 退出 结束对话）：
>
```

## 使用示例

### 示例 1：慢 SQL 分析

```
> 分析一下 /Users/you/src下的慢 SQL
```

Agent 会：
1. 识别意图为"慢Sql分析"；
2. 从消息里抽出 `code_path`；
3. 读取 `config.yaml` 里配置的慢 SQL 列表，并行跑子分析；
4. 每条 SQL 产出 → 结构化 → 定位代码位置 → 结合索引生成优化方案；
5. 在 `code_path/<output_markdown_name>`（默认 `optimization_plan.md`）写出一份面向工程师的 Markdown。

产出文档结构（固定）：
```
# 慢 SQL 优化方案
## 1. 概述
## 2. 慢 SQL 清单与逐条优化
   ### 2.1 ...
     - 定位（文件绝对路径 + 行号）
     - 当前 SQL
     - 慢的原因
     - 优化方案（含 diff / DDL）
     - 预期收益与风险
     - 验证方式
## 3. 建议的落地顺序
## 4. 需要人工确认的事项
```

### 示例 2：发送邮件

```
> 给 someone@example.com 发封邮件，主题是周报，内容是本周完成了 A/B/C 三项工作
```

Agent 会：
1. 识别意图为"发送邮件"；
2. 通过 LLM 从自然语言中抽取 `to / cc / bcc / subject / body / attachment` 结构化字段；
3. 检查本地 `agently-cli` 的 OAuth token，必要时触发浏览器登录（给出授权 URL 并 `interrupt` 等用户确认）；
4. 展示草稿预览 + `interrupt` 等用户回复"确认 / ok / 发"；
5. 用户确认后，同一轮内连续调用两阶段 `agently-cli message +send`：先拿 `ctk_xxx`，再带 token 真正发出；
6. 回到主图，等待下一次输入。

### 示例 3：闲聊 & 退出

```
> 你好，讲个冷笑话
> 退出
```

闲聊走 `AnswerModel` 直接回复；输入 `退出 / exit / quit` 终止对话，程序结束时打印 token 用量汇总。

## 环境变量速查

| 变量 | 作用 | 默认值 |
| --- | --- | --- |
| `OPENAI_API_KEY` | LLM 鉴权 | 无（必填） |
| `OPENAI_BASE_URL` | LLM 服务地址 | `config.yaml` 里的 `openai.default_base_url` |
| `AGENT_CONFIG_PATH` | 指定其他 yaml 配置文件 | `my-agent/config.yaml` |
| `AGENT_CODE_PATH` | 覆盖 `runtime.default_code_path` | 无 |

## 常见问题

**Q：发邮件时报 `Confirmation token expired or invalid`？**
A：`agently-cli` 的 ctk 有效期极短，必须在同一轮里"拿 ctk → 用 ctk"，中间不能插 `interrupt`。本项目已经通过 `send_mail/send_mail.py :: do_send` 的"同轮连续发送"策略处理。如果仍遇到，先 `agently-cli --version` 看是否 ≥ 1.0.18，必要时升级。

**Q：慢 SQL 分析一直循环、不收尾？**
A：调大 `runtime.max_step`，或在 `config.yaml` 里把 `sql_parse.slow_sqls` 数量减少后再跑。

**Q：切到自建 vLLM / Azure OpenAI？**
A：改 `config.yaml` 的 `openai.default_base_url` 和 `models.*.name` 即可，无需改任何代码。

## License

Internal use.
