"""工具层：Agent 可以调用的"业务工具"。

本题只需要两个工具：
1. search(query)    —— 按关键词检索知识库，返回命中文档的摘要信息
2. read_doc(doc_id) —— 按文档 ID 读取全文

设计取舍（将写入 docs/02-solution.md）：
- 语料只有 7 条合成文档，采用"同义词扩展 + 关键词计数打分"即可满足
  "改写后的相似问题"这一验收点，不引入向量数据库 / embedding；
- read_doc 对不存在的 doc_id 抛出 ToolError，作为"工具失败"
  验收场景的触发源，用于观察模型在工具报错后如何调整动作。
"""

import json
from pathlib import Path

# __file__ 是本文件（tools.py）的绝对路径；
# Path(...).parent 取所在目录（src/），再 .parent 退到项目根目录，
# 最后拼上 data/knowledge_base.json —— 这样无论从哪个目录启动脚本都能找到数据文件。
KB_PATH = Path(__file__).resolve().parent.parent / "data" / "knowledge_base.json"

# 同义词表：把口语说法归入同一组关键词，查询命中任意一个词就扩展出整组词。
# 这是应对"改写后的相似问题"验收点的主要手段。
SYNONYMS = {
    "报销": ["报销", "发票", "费用"],
    "期限": ["期限", "时限", "天内", "多久", "拖"],
    "请假": ["请假", "休假", "事假", "病假"],
    "采购": ["采购", "购买", "设备"],
    "工牌": ["工牌", "工卡", "门禁"],
    "会议": ["会议", "会议室", "预订"],
}


class ToolError(Exception):
    """业务工具抛出的可控异常。

    单独定义一个异常类（而不是复用内置 Exception），
    是为了让 Agent 循环能用 `except ToolError` 精确捕获"工具失败"，
    再把失败信息交给模型决策下一步，而不是让程序直接崩溃。
    """


def load_documents():
    """读取知识库，返回文档列表。

    json.load(f) 把 JSON 文件解析成 Python 对象（这里是 dict，
    其中 documents 键对应一个 list[dict]）。
    encoding="utf-8" 必须显式写：Windows 默认编码是 GBK，
    不写的话读中文文件会乱码或直接报错。
    """
    with open(KB_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


def expand_query(query):
    """把用户查询扩展成一组检索关键词。

    例如 "报销能拖多久" 会命中 "报销" 组和 "期限" 组，
    扩展出 {报销, 发票, 费用, 期限, 时限, 天内, 多久, 拖} 等关键词。
    """
    keywords = set()
    for group in SYNONYMS.values():
        # any(...) 判断"组里是否有至少一个词出现在查询里"
        if any(word in query for word in group):
            keywords.update(group)  # set.update：把整组词并入集合（自动去重）
    # 查询完全没命中任何同义词组时，退回把原始查询当关键词
    if not keywords:
        keywords.add(query)
    return keywords


def search(query, top_k=3):
    """检索工具：返回与 query 最相关的至多 top_k 条文档摘要。

    打分规则：标题命中一个关键词记 3 分（标题更代表文档主题），
    正文命中一个记 1 分。得分 0 的文档不返回。
    """
    if not query or not query.strip():
        # 参数本身非法也算工具失败，同样交给上层（模型）处理
        raise ToolError("search 工具的 query 参数不能为空")

    keywords = expand_query(query)
    results = []
    for doc in load_documents():
        score = 0
        for kw in keywords:
            # str.count(kw)：统计子串出现次数
            score += doc["title"].count(kw) * 3
            score += doc["content"].count(kw)
        if score > 0:
            # snippet：截取正文开头 80 字作为"原文片段"，
            # 题目要求交付结果里包含引用片段，search 就先把片段给出来
            results.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "version_date": doc["version_date"],
                "score": score,
                "snippet": doc["content"][:80],
            })
    # 按得分从高到低排序；key 指定按哪个字段排，reverse=True 是降序
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


def read_doc(doc_id):
    """读原文工具：按 doc_id 返回文档全文；ID 不存在则抛 ToolError。"""
    for doc in load_documents():
        if doc["doc_id"] == doc_id:
            return {
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "version_date": doc["version_date"],
                "content": doc["content"],
            }
    # 走到这里说明循环结束都没匹配上 —— 工具失败场景的触发点
    raise ToolError(f"文档 {doc_id} 不存在，请检查 doc_id 是否正确")


def run_tool(name, args):
    """工具分发器：按名字调用对应工具。

    把"工具名 -> 函数"的映射集中在这里，Agent 循环只需要
    调 run_tool(action["action"], action["args"])，不用写一长串 if-else。
    """
    if name == "search":
        return search(args["query"])
    if name == "read_doc":
        return read_doc(args["doc_id"])
    # 未知工具名也按工具失败处理（模型可能幻觉出不存在的工具名）
    raise ToolError(f"未知工具：{name}")