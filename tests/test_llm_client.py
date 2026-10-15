from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeAPI
from pydantic import BaseModel

from cuemem.config import Settings
from cuemem.errors import CacheMissError, MissingAPIKeyError, StructuredOutputError
from cuemem.llm import ChatResult, Ledger, LLMClient, ResponseCache, ToolCall, Usage, chat_cost
from cuemem.llm.cache import request_key
from cuemem.llm.client import unparsable_json

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


# ---------- 解析不了的 JSON 回复与响应缓存（N12，评测网关） ----------

JSON_MODE = {"type": "json_object"}
BROKEN = '{"name": "Ana", "city": "Ber'  # 生成失控、被截断的 JSON


def test_identical_retry_after_bad_json_bypasses_the_cache(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """外部系统拿到解析不了的 JSON 后把同样的请求再发一次：不读缓存、重新调用，并覆盖缓存里的
    坏回复；之后同样的请求照常命中缓存（拿到的是新的、能解析的回复）。"""
    fake_api.chat_replies = [BROKEN, '{"name": "Ana", "city": "Berlin"}']
    client = make_client(settings, fake_api)

    def ask() -> ChatResult:
        return client.chat("extractor", HELLO, response_format=JSON_MODE, retry_bad_json=True)

    assert ask().content == BROKEN
    retried = ask()  # 外部系统的重试：重新调用
    assert not retried.from_cache and retried.content == '{"name": "Ana", "city": "Berlin"}'
    assert len(fake_api.chat_requests) == 2
    third = ask()
    assert third.from_cache and third.content == '{"name": "Ana", "city": "Berlin"}'
    assert len(fake_api.chat_requests) == 2


def test_concurrent_identical_retries_count_one_refresh(
    settings: Settings, fake_api: FakeAPI
) -> None:
    """拿到坏 JSON 后两个同样的重试同时到：只有一个重新调用，另一个等它、读缓存里的新回复；
    llm_cache_refresh（网关的 json_refreshes）只记真的重新调用的那一次。"""
    import threading

    from cuemem.core.trace import Trace, start_trace

    started, release, waiting = threading.Event(), threading.Event(), threading.Event()

    def slow(body: dict[str, object]) -> str:
        started.set()
        release.wait(5)
        return '{"name": "Ana", "city": "Berlin"}'

    fake_api.chat_replies = [BROKEN, slow]
    client = make_client(settings, fake_api)
    claim = client._claim

    def watched_claim(key: str) -> threading.Event | None:
        event = claim(key)
        if event is not None:  # 这个线程要等别的线程的同样请求
            waiting.set()
        return event

    client._claim = watched_claim  # type: ignore[method-assign]
    assert client.chat("extractor", HELLO, response_format=JSON_MODE, retry_bad_json=True).content
    traces: list[Trace] = []
    results: list[ChatResult] = []

    def retry() -> None:
        with start_trace("retry") as trace:
            traces.append(trace)
            results.append(
                client.chat("extractor", HELLO, response_format=JSON_MODE, retry_bad_json=True)
            )

    first = threading.Thread(target=retry)
    first.start()
    assert started.wait(5)
    second = threading.Thread(target=retry)
    second.start()
    assert waiting.wait(5)
    release.set()
    first.join(5)
    second.join(5)

    assert len(fake_api.chat_requests) == 2  # 第一次的坏回复 + 一次重新调用
    assert sorted(r.from_cache for r in results) == [False, True]
    assert {r.content for r in results} == {'{"name": "Ana", "city": "Berlin"}'}
    refreshes = [e for t in traces for e in t.events if e["kind"] == "llm_cache_refresh"]
    assert len(refreshes) == 1


def test_bad_json_replay_is_unchanged_without_opt_in(settings: Settings, fake_api: FakeAPI) -> None:
    """不要求的调用方照旧：解析不了的回复也缓存，同样的请求从缓存里拿到同一个回复。"""
    fake_api.chat_replies = [BROKEN, '{"name": "Ana"}']
    client = make_client(settings, fake_api)
    client.chat("extractor", HELLO, response_format=JSON_MODE)
    again = client.chat("extractor", HELLO, response_format=JSON_MODE)
    assert again.from_cache and again.content == BROKEN
    assert len(fake_api.chat_requests) == 1


def test_existing_cache_entries_still_hit(settings: Settings, fake_api: FakeAPI) -> None:
    """缓存键不变：以前写进缓存的回复（能解析的、解析不了的）照常命中；解析不了的那条要等同样的
    请求再来一次才重新调用。"""
    spec = settings.models[settings.roles["extractor"].model]
    client = make_client(settings, fake_api)
    params = client._prepare("extractor", None, None, None, JSON_MODE, None, None)[2]

    def key(messages: list[dict[str, str]]) -> str:  # 与改动之前同样的写法
        return request_key(
            {
                "provider": spec.provider,
                "model": spec.model_id,
                "messages": messages,
                "params": params,
                "extra_body": spec.extra_body,
            }
        )

    good = [{"role": "user", "content": "good"}]
    bad = [{"role": "user", "content": "bad"}]
    assert client.cache is not None
    for messages, content in ((good, '{"ok": true}'), (bad, BROKEN)):
        cached = {"content": content, "model_alias": spec.model_id, "model_id": spec.model_id}
        client.cache.put(key(messages), spec.model_id, cached)

    hit = client.chat("extractor", good, response_format=JSON_MODE, retry_bad_json=True)
    assert hit.from_cache and hit.content == '{"ok": true}'
    first = client.chat("extractor", bad, response_format=JSON_MODE, retry_bad_json=True)
    assert first.from_cache and first.content == BROKEN  # 第一次照常命中，重跑可复现
    assert fake_api.chat_requests == []
    fake_api.chat_replies = ['{"ok": false}']
    retried = client.chat("extractor", bad, response_format=JSON_MODE, retry_bad_json=True)
    assert not retried.from_cache and retried.content == '{"ok": false}'
    assert len(fake_api.chat_requests) == 1


def test_read_only_cache_serves_bad_json_instead_of_calling(
    settings: Settings, fake_api: FakeAPI
) -> None:
    fake_api.chat_replies = [BROKEN]
    client = make_client(settings, fake_api)
    client.chat("extractor", HELLO, response_format=JSON_MODE, retry_bad_json=True)
    client.settings = settings.model_copy(update={"llm_cache_mode": "read_only"})
    again = client.chat("extractor", HELLO, response_format=JSON_MODE, retry_bad_json=True)
    assert again.from_cache and again.content == BROKEN
    assert len(fake_api.chat_requests) == 1


def test_structured_output_still_replays_from_cache(settings: Settings, fake_api: FakeAPI) -> None:
    """CueMem 自己的 JSON 调用（抽取、对账等都走 structured）不受影响：第一次的坏回复与带校验
    错误的重试都缓存，重跑整个从缓存复现，不再调用 API。"""
    fake_api.chat_replies = [BROKEN, '{"name": "Max", "breed": "corgi"}']
    client = make_client(settings, fake_api)
    messages = [{"role": "user", "content": "Max 是柯基"}]
    first = client.structured("extractor", messages, Pet)
    assert first.attempts == 2 and len(fake_api.chat_requests) == 2
    again = client.structured("extractor", messages, Pet)
    assert again.value == first.value and again.attempts == 2
    assert again.result.from_cache and len(fake_api.chat_requests) == 2


def test_unparsable_json() -> None:
    def reply(content: str | None, tool_calls: list[ToolCall] | None = None) -> ChatResult:
        return ChatResult(
            content=content, tool_calls=tool_calls or [], model_alias="m", model_id="m"
        )

    assert not unparsable_json(reply('{"a": [1, 2]}'))
    assert not unparsable_json(reply('```json\n{"a": 1}\n```'))  # 去掉 markdown 围栏
    assert unparsable_json(reply(BROKEN))
    assert unparsable_json(reply(""))
    assert unparsable_json(reply(None))
    assert unparsable_json(reply('{"a": "line\nbreak"}'))  # 字符串里原样的换行：严格解析不过
    call = ToolCall(id="c1", name="save", arguments="{}")
    assert not unparsable_json(reply(None, [call]))
