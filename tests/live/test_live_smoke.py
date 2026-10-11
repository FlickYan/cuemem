"""联网冒烟测试：需要 .env 中的真实 API Key，会产生极少量费用。

运行：uv run pytest -m live
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from cuemem.config import Settings, load_settings
from cuemem.embed import build_embedder
from cuemem.llm import Ledger, LLMClient

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    loaded = load_settings()
    missing = [p.api_key_env for p in loaded.providers.values() if p.api_key() is None]
    if missing:
        pytest.skip(f"缺少 API Key：{missing}")
    return loaded


@pytest.mark.parametrize("role", ["reader", "extractor", "judge"])
def test_role_models_reply(live_settings: Settings, role: str) -> None:
    client = LLMClient(live_settings, ledger=Ledger(live_settings.paths.ledger))
    result = client.chat(
        role,
        [{"role": "user", "content": "Reply with the single word: OK"}],
        max_tokens=16,
        use_cache=False,
    )
    assert result.content is not None
    assert "ok" in result.content.lower()
    assert result.usage.input_tokens > 0


class Capital(BaseModel):
    country: str
    capital: str


def test_structured_output(live_settings: Settings) -> None:
    client = LLMClient(live_settings, ledger=Ledger(live_settings.paths.ledger))
    out = client.structured(
        "reader", [{"role": "user", "content": "新加坡的首都是哪里？"}], Capital
    )
    assert "singapore" in out.value.capital.lower() or "新加坡" in out.value.capital


def test_embedding_dense_and_sparse(live_settings: Settings) -> None:
    embedder = build_embedder(
        live_settings, ledger=Ledger(live_settings.paths.ledger), use_cache=False
    )
    batch = embedder.embed(["我领养了一只金毛", "我的狗是金毛犬", "笔记本预算 6000 元"], "document")
    assert batch.dense.shape == (3, live_settings.embedding.dim)
    assert batch.sparse is not None
    assert all(len(s) > 0 for s in batch.sparse)
    similarity = batch.dense @ batch.dense.T
    assert similarity[0, 1] > similarity[0, 2]
