"""工具层测试：不依赖模型，直接测两个工具的行为。"""

import pytest

from tools import ToolError, expand_query, load_documents, read_doc, search


def test_load_documents_returns_list():
    docs = load_documents()
    assert isinstance(docs, list) and len(docs) >= 5
    # 每条文档都应具备题目要求的"文档 ID + 版本日期"
    for doc in docs:
        assert {"doc_id", "version_date", "content"} <= set(doc.keys())


def test_search_hits_conflicting_docs():
    """报销期限问题应同时命中 D01 和 D02（冲突场景的前提）。"""
    results = search("报销期限多久")
    ids = [r["doc_id"] for r in results]
    assert "D01" in ids and "D02" in ids
    # 返回结构应包含引用所需的字段
    assert {"doc_id", "version_date", "snippet"} <= set(results[0].keys())


def test_search_handles_rephrased_question():
    """改写后的相似问题（验收点）：'拖' 等口语词也要能命中。"""
    ids = [r["doc_id"] for r in search("报销最迟能拖多久再交")]
    assert "D01" in ids and "D02" in ids


def test_search_out_of_scope_returns_empty():
    """资料外问题：检索为空而不是硬凑结果。"""
    assert search("公司年会什么时候开") == []


def test_search_empty_query_raises():
    with pytest.raises(ToolError):
        search("   ")


def test_read_doc_returns_full_text():
    doc = read_doc("D03")
    assert doc["doc_id"] == "D03"
    assert "部门负责人" in doc["content"]


def test_read_doc_unknown_id_raises():
    """工具失败场景的触发源：不存在的 doc_id 抛 ToolError。"""
    with pytest.raises(ToolError, match="D99"):
        read_doc("D99")


def test_expand_query_groups_synonyms():
    kws = expand_query("报销能拖多久")
    assert "报销" in kws and "多久" in kws and "发票" in kws  # 整组词被扩展进来
    assert expand_query("完全无关词xyz") == {"完全无关词xyz"}  # 未命中时退回原查询