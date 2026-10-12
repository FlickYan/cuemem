from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeAPI
from pydantic import BaseModel

from cuemem.config import Settings
from cuemem.errors import CacheMissError, MissingAPIKeyError, StructuredOutputError
from cuemem.llm import Ledger, LLMClient, ResponseCache, Usage, chat_cost

HELLO = [{"role": "user", "content": "hi"}]


# 固定在 DeepSeek 的高峰时段（周一 10:00，北京时间）：花费按标价计，结果不依赖运行测试的时刻
PEAK_TIME = datetime(2026, 10, 12, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def make_client(settings: Settings, fake_api: FakeAPI, ledger: Ledger | None = None) -> LLMClient:
    return LLMClient(
        settings,
        cache=ResponseCache(None),
        ledger=ledger or Ledger(None),
        http_client=fake_api.client(),
        clock=lambda: PEAK_TIME,
    )


def test_deepseek_usage_and_cost(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.cached_tokens = 40
    ledger = Ledger(None)
    result = make_client(settings, fake_api, ledger).chat("reader", HELLO)

    assert result.content == "OK"
    assert result.usage == Usage(input_tokens=120, output_tokens=30, cached_input_tokens=40)
    nominal, discounted = chat_cost(settings.models["deepseek-flash"].price, result.usage)
    assert nominal == pytest.approx((120 * 0.30 + 30 * 1.20) / 1e6)
    assert discounted < nominal
    [row] = ledger.summary()
    assert (row["model"], row["role"], row["calls"]) == ("deepseek-flash", "reader", 1)
    assert row["nominal_cost"] == pytest.approx(nominal)


def test_provider_specific_request_fields(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.cached_tokens = 10
    client = make_client(settings, fake_api)

    client.chat("reader", HELLO)
    deepseek_body = fake_api.chat_requests[-1]
    assert deepseek_body["model"] == "deepseek-flash"
    assert deepseek_body["thinking"] == {"type": "disabled"}
    assert deepseek_body["temperature"] == 0.0

    result = client.chat("judge", HELLO)
    qwen_body = fake_api.chat_requests[-1]
    assert qwen_body["model"] == "qwen3.8-max"
    assert qwen_body["enable_thinking"] is False
    assert result.usage.cached_input_tokens == 10  # 通义的 prompt_tokens_details.cached_tokens


def test_response_cache_avoids_repeat_calls(settings: Settings, fake_api: FakeAPI) -> None:
    ledger = Ledger(None)
    client = make_client(settings, fake_api, ledger)
    first = client.chat("reader", HELLO)
    second = client.chat("reader", HELLO)

    assert len(fake_api.chat_requests) == 1
    assert not first.from_cache
    assert second.from_cache
    assert second.content == first.content
    [row] = ledger.summary()
    assert (row["calls"], row["cache_hits"]) == (2, 1)
    # 只有真实调用计入花费
    assert row["spent"] == pytest.approx(
        chat_cost(settings.models["deepseek-flash"].price, first.usage)[1]
    )


def test_read_only_cache_raises_on_miss(settings: Settings, fake_api: FakeAPI) -> None:
    read_only = settings.model_copy(update={"llm_cache_mode": "read_only"})
    with pytest.raises(CacheMissError):
        make_client(read_only, fake_api).chat("reader", HELLO)
    assert fake_api.chat_requests == []


def test_empty_reply_is_not_cached(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = ["", "OK"]
    client = make_client(settings, fake_api)
    assert client.chat("reader", HELLO).content == ""
    assert client.chat("reader", HELLO).content == "OK"
    assert len(fake_api.chat_requests) == 2


class Pet(BaseModel):
    name: str
    breed: str


def test_structured_output_repairs_invalid_json(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = [
        '{"name": "Max"}',  # 缺少 breed 字段
        '```json\n{"name": "Max", "breed": "golden retriever"}\n```',
    ]
    out = make_client(settings, fake_api).structured(
        "extractor", [{"role": "user", "content": "Max 是金毛"}], Pet
    )

    assert out.value == Pet(name="Max", breed="golden retriever")
    assert out.attempts == 2
    first, second = fake_api.chat_requests
    assert first["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in first["messages"][0]["content"]
    assert "breed" in second["messages"][-1]["content"]  # 第二次请求带上了校验错误


def test_structured_output_gives_up(settings: Settings, fake_api: FakeAPI) -> None:
    fake_api.chat_replies = ["not json"]
    with pytest.raises(StructuredOutputError):
        make_client(settings, fake_api).structured("extractor", HELLO, Pet, max_attempts=2)
    assert len(fake_api.chat_requests) == 2


def test_missing_api_key(
    settings: Settings, fake_api: FakeAPI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    with pytest.raises(MissingAPIKeyError, match="DEEPSEEK_API_KEY"):
        make_client(settings, fake_api).chat("reader", HELLO)


def test_content_filter_falls_back_to_other_provider(settings: Settings, fake_api: FakeAPI) -> None:
    from fakes import REJECT

    from cuemem.errors import ContentFilteredError

    class Out(BaseModel):
        ok: bool

    # 通义的内容审核误拒，DeepSeek 正常回答
    fake_api.chat_replies = [
        lambda body: REJECT if body["model"] == "qwen3.8-flash" else '{"ok": true}'
    ]
    client = LLMClient(settings, http_client=fake_api.client())
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(ContentFilteredError):
        client.structured("extractor", messages, Out)
    result = client.structured("extractor", messages, Out, fallback_model="deepseek-flash")
    assert result.value.ok and result.result.model_alias == "deepseek-flash"
    assert [r["model"] for r in fake_api.chat_requests] == [
        "qwen3.8-flash",
        "qwen3.8-flash",
        "deepseek-flash",
    ]  # 被拒的请求不重试、不缓存


def test_timeout_falls_back_to_other_provider(settings: Settings, fake_api: FakeAPI) -> None:
    from fakes import TIMEOUT
    from openai import APITimeoutError

    class Out(BaseModel):
        ok: bool

    # 通义在这条输入上生成失控、一直超时，DeepSeek 正常回答（不让 SDK 再重试，测试才快）
    providers = {
        name: p.model_copy(update={"max_retries": 0}) for name, p in settings.providers.items()
    }
    fake_api.chat_replies = [
        lambda body: TIMEOUT if body["model"] == "qwen3.8-flash" else '{"ok": true}'
    ]
    client = LLMClient(
        settings.model_copy(update={"providers": providers}), http_client=fake_api.client()
    )
    messages = [{"role": "user", "content": "hello"}]
    with pytest.raises(APITimeoutError):
        client.structured("extractor", messages, Out)
    result = client.structured("extractor", messages, Out, fallback_model="deepseek-flash")
    assert result.value.ok and result.result.model_alias == "deepseek-flash"


def test_identical_concurrent_requests_are_coalesced(settings: Settings, fake_api: FakeAPI) -> None:
    """两个线程同时发出同样的请求：只调用一次 API，另一个等结果（single-flight）。"""
    import threading
    import time

    started, release = threading.Event(), threading.Event()

    def slow(body: dict[str, object]) -> str:
        started.set()
        release.wait(5)
        return "ok"

    fake_api.chat_replies = [slow]
    client = make_client(settings, fake_api)
    results = []
    first = threading.Thread(target=lambda: results.append(client.chat("reader", HELLO)))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=lambda: results.append(client.chat("reader", HELLO)))
    second.start()
    time.sleep(0.1)  # 让第二个线程进入等待
    release.set()
    first.join(5)
    second.join(5)
    assert len(fake_api.chat_requests) == 1
    assert sorted(r.from_cache for r in results) == [False, True]


def test_structured_retries_when_provider_aborts_json(settings: Settings) -> None:
    """QwenCloud 在 JSON 模式下中止生成（400）：补一句提示换个请求重试，而不是直接报错。"""
    from fakes import ABORT_JSON

    api = FakeAPI()
    api.chat_replies = [ABORT_JSON, '{"answer": "ok"}']
    client = LLMClient(settings, http_client=api.client())

    class Out(BaseModel):
        answer: str

    result = client.structured("extractor", [{"role": "user", "content": "hi"}], Out)
    assert result.value.answer == "ok" and result.attempts == 2
    assert api.chat_requests[1]["messages"][-1]["content"].startswith("Reply with one short")
