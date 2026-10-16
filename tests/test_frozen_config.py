"""冻结的方法：检索配置的默认值、方法表里各方法的参数与控制器的模型文件都钉住。

发表的数字（N2–N12）要能从 data/cache 原样重放。请求快照（test_eval_leak.py）只在迷你数据上跑：
整段历史都装得进预算，L1 / L2 的默认值（预算、候选数、权重……）改了也看不出来，但在真实数据上
检索结果与请求都会变。这里把它们与基线提交 e7064ef 上的值逐项对齐。有意修改冻结的方法时（在
decisions 里写明），连同这里的值一起改。"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any

import pytest

from cuemem.evals.methods import N4_L1, build_methods
from cuemem.read.controller import DEFAULT_MODEL
from cuemem.read.l1 import L1Config
from cuemem.read.l2 import L2Config

READ = DEFAULT_MODEL.parent  # src/cuemem/read

L1_DEFAULT: dict[str, Any] = {
    "k": 10,
    "budget_tokens": 4000,
    "max_rounds": 40,
    "fact_k": 10,
    "pool": 30,
    "sparse": "none",
    "fusion": "modality",
    "rrf_k": 60,
    "sparse_weight": 0.5,
    "round_weight": 1.0,
    "fact_weight": 1.0,
    "summary_weight": 0.0,
    "time_weight": 1.0,
    "order_weight": 0.0,
    "chain": True,
    "image_kind": "image",
    "image_weight": 1.0,
    "visual_weight": 0.5,
    "attach_images": 0,
}
N4_DEFAULT = {**L1_DEFAULT, "sparse": "bm25"}  # B4（N4 定稿时带 BM25）
L2_DEFAULT: dict[str, Any] = {
    "l1": {**L1_DEFAULT, "budget_tokens": 6000, "max_rounds": 60},
    "plan": True,
    "fusion": "max",
    "sub_query_weight": 1.0,
    "ppr": False,
    "ppr_weight": 0.5,
    "graph": {
        "damping": 0.5,
        "hub_ratio": 0.2,
        "iterations": 50,
        "mention_edges": True,
        "min_alias_chars": 3,
        "similar_edges": True,
        "similar_k": 5,
        "similar_threshold": 0.8,
        "tolerance": 1e-08,
    },
    "seed_rounds": 5,
    "seed_facts": 10,
    "rerank": False,
    "rerank_pool": 30,
    "iterations": 2,
    "followup_budget": 1000,
    "followup_facts": 5,
    "fact_k": 10,
    "aggregate_fact_k": 25,
}
# 冻结的方法（L1 = B4-nobm25、L2 = B5、控制器 B6）及 B1、B2、B4、B3、B6a：类与构造参数
FROZEN_METHODS: dict[str, dict[str, Any]] = {
    "B1": {"class": "FullContext"},
    "B2": {"class": "NaiveRAG", "name": "B2", "k": 10},
    "B4": {
        "class": "L1RAG",
        "name": "B4",
        "config": N4_DEFAULT,
        "relative_time": True,
        "notes": False,
        "show_facts": True,
    },
    "B4-nobm25": {
        "class": "L1RAG",
        "name": "B4-nobm25",
        "config": L1_DEFAULT,
        "relative_time": True,
        "notes": False,
        "show_facts": True,
    },
    "B5": {"class": "L2RAG", "name": "B5", "config": L2_DEFAULT, "notes": True},
    "B6": {
        "class": "RoutedRAG",
        "name": "B6",
        "l1": L1_DEFAULT,
        "l2": L2_DEFAULT,
        "router": "learned",
    },
    "B3": {"class": "AgentRAG", "name": "B3", "hatch": False},
    "B6a": {"class": "AgentRAG", "name": "B6a", "hatch": True},
}
# 控制器的模型文件（retrieval.controller）：v1 是 N7 / N9、v2 是 N11 的冻结评测用的，默认的
# controller_model.json 是 v3（D-058）。重新训练、导出会改写它们（scripts/n7_controller.py 等）
CONTROLLERS = {
    "controller_model.json": "204f970981279fca0425077902d237818f3087535c3f7845bc8157da46827673",
    "controller_model_v1.json": "abd1ba1293fe383f0706a6aeac7cde5a2ebc6e58e3221952c2c054ec0374c3ca",
    "controller_model_v2.json": "3c2a4dd8d2c09bf80c4041c12a4ca6a425e372c80393da4c286c03f9fcdc380b",
}


def test_retrieval_defaults_are_frozen() -> None:
    assert asdict(L1Config()) == L1_DEFAULT
    assert asdict(N4_L1) == N4_DEFAULT
    assert asdict(L2Config()) == L2_DEFAULT


def _params(method: Any) -> dict[str, Any]:
    """方法的类与构造参数（显示用的中文名除外），配置展开成字典。"""
    params: dict[str, Any] = {"class": type(method).__name__}
    for key, value in vars(method).items():
        if key != "label":
            params[key] = asdict(value) if is_dataclass(value) else value
    return params


@pytest.mark.parametrize("name", list(FROZEN_METHODS))
def test_frozen_methods_keep_their_parameters(name: str) -> None:
    [method] = build_methods([name])
    assert _params(method) == FROZEN_METHODS[name]


def test_controller_model_files_are_frozen() -> None:
    found = {p.name for p in READ.glob("controller_model*.json")}
    assert found == set(CONTROLLERS)
    for name, digest in CONTROLLERS.items():
        assert hashlib.sha256((READ / name).read_bytes()).hexdigest() == digest, name
