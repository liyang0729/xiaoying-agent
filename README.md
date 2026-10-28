# 企业知识问答 Agent

帮助员工就公司制度问题获得**可溯源**的回答：答案必须引用文档 ID 与原文片段，
资料冲突时列出冲突并请求确认，资料不足或工具失败时如实报告，不编造。

实现方式为原生 Python 的 ReAct 决策循环（不依赖 Agent 框架）：
模型每轮根据最新工具观察结果自主选择下一步动作
（检索 / 读原文 / 追问 / 结束），循环受步数上限约束。

## 环境要求

- Python 3.10+（开发验证使用 3.12.11）
- 一个 OpenAI 兼容的模型 API（DeepSeek / 通义 / 智谱等均可）

## 安装

```powershell
pip install -r requirements.txt
```

## 配置

复制 `.env.example` 为 `.env`，填入你的 API 信息：

```
LLM_API_KEY=你的密钥
LLM_BASE_URL=https://api.deepseek.com/v1   # 以实际服务商为准，通常以 /v1 结尾
LLM_MODEL=deepseek-chat
```

- `.env` 含真实密钥，已列入 `.gitignore`，不会被提交；
- 离线 Mock 模式：不配置 `LLM_API_KEY` 或设为 `mock` 即启用
  （仅测试流程用，见 docs/03 的边界声明）。

## 运行

```powershell
# 单次提问
python src/cli.py -q "报销期限多久"

# 多轮对话模式（支持上下文追问），输入 exit 退出
python src/cli.py

# 调整步数上限（演示 unfinished 状态）
python src/cli.py -q "报销期限多久" --max-steps 1
```

终端会输出任务状态、回答、引用与资料缺口，以及每一步的运行记录。

## 测试

```powershell
pytest tests -v
```

14 条测试覆盖工具行为与 Agent 循环机制（Mock 驱动，不需要 API key）。

## 评审入口

| 材料 | 位置 |
|---|---|
| 需求与验收点 | docs/01-requirements.md |
| 方案与动态决策设计 | docs/02-solution.md |
| 测试与真实运行结果 | docs/03-evaluation.md、examples/（测试存档 + 8 组场景记录） |
| AI 辅助记录 | docs/04-ai-usage.md |
| 复盘 | docs/05-retrospective.md |
| 合成知识库（含故意冲突的 D01/D02） | data/knowledge_base.json |

## 目录结构

```
src/        tools.py(工具) llm.py(模型层) agent.py(ReAct循环) cli.py(入口) context_demo.py(多轮场景复现)
tests/      pytest 测试（conftest.py 处理 src 路径）
examples/   测试结果存档 + 8 组输入、实际输出与运行记录
data/       合成知识库
docs/       01~05 文档
```
