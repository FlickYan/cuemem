"""GET /metrics（api/metrics.py，第十轮体检 Q18）：默认不提供；打开后只给管理令牌，标签的取值
出自封闭的集合，读它不碰数据库。全部离线。"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from fakes import PROJECT_ROOT, FakeAPI
from fastapi.testclient import TestClient
from scripted import ScriptedModel, echo_extraction

from cuemem import __version__
from cuemem.agent import Agent
from cuemem.api.app import create_app
from cuemem.api.auth import Keys
from cuemem.api.limits import BodyLimit
from cuemem.api.metrics import (
    BUCKETS,
    CONTENT_TYPE,
    Metrics,
    MetricsMiddleware,
    metrics_enabled,
)
from cuemem.api.proxy import MemoryProxy, create_proxy_app
from cuemem.api.service import ChatService
from cuemem.config import Settings
from cuemem.errors import CueMemError
from cuemem.llm.types import Usage
from cuemem.memory import MemorySystem

TOKEN = "admin-token-0123456789"
SCOPED_TOKEN = "scoped-token-0123456789"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
SCOPED = {"Authorization": f"Bearer {SCOPED_TOKEN}"}
KEYS = f"""
keys:
  - name: SECRET-KEY
    token: {SCOPED_TOKEN}
    scopes: [{{user: alice}}]
    max_spaces: 1
    rate_per_minute: 6
"""
needs_gradio = pytest.mark.skipif(
    importlib.util.find_spec("gradio") is None, reason="界面是可选依赖 ui"
)


@dataclass
class Served:
    service: ChatService
    proxy: MemoryProxy
    tmp_path: Path

    @property
    def system(self) -> MemorySystem:
        return self.service.system

    def app(self, **kwargs: Any) -> Any:
        options = {"ui": False, "keys": Keys.parse(KEYS, TOKEN), "metrics": True, **kwargs}
        return create_app(
            self.service, upload_dir=self.tmp_path / "uploads", proxy=self.proxy, **options
        )


@pytest.fixture
def served(n3_settings: Settings, fake_api: FakeAPI, tmp_path: Path) -> Iterator[Served]:
    fake_api.chat_replies = [ScriptedModel(echo_extraction)]
    system = MemorySystem.from_settings(n3_settings, http_client=fake_api.client())
    service = ChatService(system, Agent(system, router="rule"), default_space="alice")
    proxy = MemoryProxy(system, http_client=fake_api.client(), mode="l1")
    try:
        yield Served(service, proxy, tmp_path)
    finally:
        proxy.close_all()
        service.close()


def _samples(text: str) -> dict[str, float]:
    """指标的各行：'名字{标签}' → 值。"""
    rows = [line.rsplit(" ", 1) for line in text.splitlines() if not line.startswith("#")]
    return {name: float(value) for name, value in rows}


def _scrape(http: TestClient) -> dict[str, float]:
    response = http.get("/metrics", headers=AUTH)
    assert response.status_code == 200, response.text
    return _samples(response.text)


def test_metrics_are_off_by_default(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    """不开时没有路由、没有中间件，LLMClient.observer 与各处的 on_throttle 都是 None。"""
    for app in (
        served.app(metrics=False),
        create_proxy_app(served.proxy, keys=Keys.parse(KEYS, TOKEN)),
    ):
        assert all(m.cls is not MetricsMiddleware for m in app.user_middleware)
        (limit,) = (m for m in app.user_middleware if m.cls is BodyLimit)
        assert limit.kwargs["on_throttle"] is None
        assert TestClient(app).get("/metrics", headers=AUTH).status_code == 404
    assert served.system.llm.observer is None and served.proxy.on_throttle is None

    assert metrics_enabled(environ={}) is False and metrics_enabled(True, {}) is True
    for raw, expected in (("1", True), ("TRUE", True), ("0", False), ("false", False), ("", False)):
        assert metrics_enabled(environ={"CUEMEM_METRICS": raw}) is expected
    with pytest.raises(CueMemError, match="CUEMEM_METRICS"):
        metrics_enabled(environ={"CUEMEM_METRICS": "yes"})


def test_metrics_are_for_the_admin_token_only(served: Served) -> None:
    """没带令牌 401，限定作用域的令牌 403，管理令牌 200（Prometheus 的文本格式）；记忆代理单独
    运行时一样。不出现在接口描述里，/api/v1 的快照不变；没配令牌时与别的接口一样不鉴权。"""
    from cuemem.api.memories import openapi_v1

    for app in (
        served.app(),
        create_proxy_app(served.proxy, keys=Keys.parse(KEYS, TOKEN), metrics=True),
    ):
        with TestClient(app) as http:
            assert http.get("/metrics").status_code == 401
            assert http.get("/metrics", headers=SCOPED).status_code == 403
            ok = http.get("/metrics", headers=AUTH)
            assert ok.status_code == 200 and ok.headers["content-type"] == CONTENT_TYPE
            assert f'cuemem_build_info{{version="{__version__}"}} 1' in ok.text.splitlines()
            assert "# TYPE cuemem_http_requests_total counter" in ok.text
            assert "/metrics" not in http.get("/openapi.json").json()["paths"]
    snapshot = json.loads((PROJECT_ROOT / "docs" / "openapi-v1.json").read_text(encoding="utf-8"))
    assert openapi_v1() == snapshot and "/metrics" not in snapshot["paths"]
    with TestClient(served.app(keys=None)) as http:
        assert http.get("/metrics").status_code == 200


def test_labels_never_carry_ids(served: Served, fake_api: FakeAPI) -> None:
    """route 是路由的模板，不是请求的路径；用户、空间、记忆 ID、令牌的名字与模型名都不进标签。
    没匹配到的路径是 other，/mcp 是 "/mcp"。"""
    with TestClient(served.app()) as http:
        who = {**AUTH, "X-CueMem-User": "SECRET-USER"}
        assert http.get("/api/v1/memories/SECRET-ID", headers=who).status_code == 404
        assert http.get("/api/spaces/alice-SECRET/facts", headers=who).status_code == 200
        assert http.get("/nope/SECRET", headers=who).status_code == 404
        assert http.post("/mcp", json={}, headers=who).status_code != 404
        assert http.request("BREW", "/nope/SECRET", headers=who).status_code in (404, 405)
        listed = http.get("/api/v1/memories", params={"user_id": "alice"}, headers=SCOPED)
        assert listed.status_code == 200
        chat = {
            "model": "SECRET-MODEL",
            "user": "SECRET-USER",
            "messages": [{"role": "user", "content": "SECRET-TEXT"}],
        }
        assert http.post("/v1/chat/completions", json=chat, headers=AUTH).status_code == 200
        text = http.get("/metrics", headers=AUTH).text
    assert "SECRET" not in text and "alice" not in text
    samples = _samples(text)
    requests = "cuemem_http_requests_total"
    assert samples[
        f'{requests}{{method="GET",route="/api/v1/memories/{{memory_id}}",status="404"}}'
    ]
    assert samples[
        f'{requests}{{method="GET",route="/api/spaces/{{space:path}}/facts",status="200"}}'
    ]
    assert samples[f'{requests}{{method="GET",route="other",status="404"}}'] == 1
    assert any(k.startswith(f'{requests}{{method="POST",route="/mcp"') for k in samples)
    assert any(k.startswith(f'{requests}{{method="OTHER",route="other"') for k in samples)
    # 转发的那次补全：模型名不在目录里时按配置换成了目录里的别名，标签里是别名
    forwarded = [k for k in samples if k.startswith('cuemem_llm_requests_total{role="proxy"')]
    assert len(forwarded) == 1 and samples[forwarded[0]] == 1
    assert samples["cuemem_http_requests_in_flight"] == 1  # 正在处理的就是这次 /metrics


def test_values_outside_a_label_set_become_other(served: Served) -> None:
    """集合之外的标签值记成 other，不抛异常；标签值里的引号、反斜杠与换行转义。"""
    with TestClient(served.app()) as http:
        registry: Metrics = http.app.state.metrics  # type: ignore[attr-defined]
        spec = served.system.settings.models[served.system.settings.roles["agent"].model]
        usage = Usage(input_tokens=7, output_tokens=3, cached_input_tokens=2)
        observer = served.system.llm.observer
        assert observer is not None
        observer("SECRET-ROLE", "SECRET-MODEL", spec, usage, False)
        observer("agent", served.system.settings.roles["agent"].model, spec, usage, True)
        registry.throttled("SECRET-REASON")
        registry.inc(
            "cuemem_http_requests_total", {"method": "GET", "route": "/SECRET", "status": "9"}
        )
        text = http.get("/metrics", headers=AUTH).text
    samples = _samples(text)
    assert "SECRET" not in text
    assert samples['cuemem_llm_requests_total{role="other",model="other",cached="false"}'] == 1
    tokens = 'cuemem_llm_tokens_total{role="other",model="other",direction='
    assert [samples[f'{tokens}"{d}"}}'] for d in ("input", "output", "cached_input")] == [7, 3, 2]
    cached = [k for k in samples if k.endswith('cached="true"}')]
    assert len(cached) == 1 and samples[cached[0]] == 1  # 命中缓存的只算次数，不算 token
    assert not [k for k in samples if "tokens_total" in k and 'role="agent"' in k]
    assert samples['cuemem_throttled_total{reason="other"}'] == 1
    assert samples['cuemem_http_requests_total{method="GET",route="other",status="other"}'] == 1

    plain = Metrics()
    plain.declare("x_total", "counter", 'a "help"\nline \\ here', v={'q"\\\n'})
    plain.inc("x_total", {"v": 'q"\\\n'})
    out = plain.render()
    assert '# HELP x_total a "help"\\nline \\\\ here' in out
    assert 'x_total{v="q\\"\\\\\\n"} 1' in out


def test_a_failing_observer_never_fails_the_call(
    served: Served, caplog: pytest.LogCaptureFixture
) -> None:
    """observer 抛出异常时，已经完成的调用照常返回、照常记账，只记一条警告。"""

    def broken(*args: Any) -> None:
        raise RuntimeError("boom")

    llm = served.system.llm
    llm.observer = broken
    try:
        with caplog.at_level("WARNING", logger="cuemem.llm.client"):
            result = llm.chat("agent", [{"role": "user", "content": "hi"}])
    finally:
        llm.observer = None
    assert result.content and "LLMClient.observer" in caplog.text


@needs_gradio
@pytest.mark.filterwarnings("ignore::pandas.errors.Pandas4Warning")
def test_ui_requests_are_labelled_ui(served: Served, monkeypatch: pytest.MonkeyPatch) -> None:
    """界面挂在根路径上：落到它的请求（它自己的接口、没匹配到的路径）都是 route="ui"，不是
    Gradio 写进 scope 的它里面的路由模板。"""
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(served.tmp_path / "gradio"))
    with TestClient(served.app(ui=True, keys=None)) as http:
        assert http.get("/config").status_code == 200
        http.get("/gradio_api/file=SECRET-FILE")
        http.get("/nope/SECRET")
        assert http.get("/api/health").status_code == 200
        text = http.get("/metrics").text
    assert "SECRET" not in text and "gradio" not in text and "{p:path}" not in text
    samples = _samples(text)
    ui = [k for k in samples if k.startswith("cuemem_http_requests_total") and 'route="ui"' in k]
    assert sum(samples[k] for k in ui) == 3
    assert samples['cuemem_http_requests_total{method="GET",route="/api/health",status="200"}'] == 1


def test_throttles_are_counted_by_reason(served: Served) -> None:
    """被上限挡回去的请求按原因计数：次数（rate）、请求体太大（body_too_large）、令牌的空间数
    （spaces，/api/v1、/api 与记忆代理各一次）、一个空间进行中的会话数（open_sessions，/api/v1 与
    /api 各一次）。第一次拒绝之前每个原因都是 0。"""
    served.service.max_open_sessions = 1
    with TestClient(served.app(max_body_bytes=4096)) as http:
        before = _scrape(http)
        assert before['cuemem_throttled_total{reason="rate"}'] == 0
        assert before['cuemem_throttled_total{reason="budget"}'] == 0

        big = http.post("/api/v1/memories", content=b"x" * 5000, headers=AUTH)
        assert big.status_code == 413
        chunked = http.post("/api/v1/memories", content=iter([b"x" * 3000] * 2), headers=AUTH)
        assert chunked.status_code == 413

        note = {"messages": "I like tea", "user_id": "alice", "infer": False}
        assert http.post("/api/v1/memories", json=note, headers=SCOPED).status_code == 200
        other = {"user_id": "alice", "agent_id": "a2"}
        assert http.post("/api/v1/runs", json=other, headers=SCOPED).status_code == 429
        start = http.post("/api/sessions", json={"space": "a2/alice"}, headers=SCOPED)
        assert start.status_code == 429, start.text
        chat = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": "hi"}]}
        proxied = http.post(
            "/v1/chat/completions",
            json={**chat, "user": "alice"},
            headers={**SCOPED, "X-CueMem-Agent": "a2"},
        )
        assert proxied.status_code == 429, proxied.text

        run = {"user_id": "alice", "run_id": "r1"}
        assert http.post("/api/v1/runs", json=run, headers=AUTH).status_code == 200
        full = http.post("/api/v1/runs", json={**run, "run_id": "r2"}, headers=AUTH)
        assert full.status_code == 429 and full.json()["error"]["type"] == "SessionLimitError"
        first = http.post("/api/sessions", json={"space": "alice"}, headers=AUTH).json()
        said = http.post(
            f"/api/sessions/{first['session_id']}/messages",
            data={"text": "hi", "mode": "l0"},
            headers=AUTH,
        )
        assert said.status_code == 200, said.text
        assert http.post("/api/sessions", json={"space": "alice"}, headers=AUTH).status_code == 429

        statuses = [
            http.post("/api/v1/memories/search", json={"query": "tea", "user_id": "alice"},
                      headers=SCOPED).status_code
            for _ in range(4)
        ]  # fmt: skip
        assert statuses[-1] == 429 and 200 in statuses  # 每分钟 6 个：前面用掉了 4 个
        after = _scrape(http)
    counts = {
        reason: after[f'cuemem_throttled_total{{reason="{reason}"}}']
        for reason in ("rate", "body_too_large", "spaces", "open_sessions", "concurrency")
    }
    assert counts == {
        "rate": statuses.count(429),
        "body_too_large": 2,
        "spaces": 3,
        "open_sessions": 2,
        "concurrency": 0,
    }
    refused = sum(
        v for k, v in after.items() if k.startswith("cuemem_http") and 'status="429"' in k
    )
    assert refused == statuses.count(429) + 3 + 2


def test_body_limit_reports_concurrency_and_bytes_in_flight() -> None:
    """BodyLimit 的 on_throttle：同时在处理的请求数（concurrency）与请求体的字节数
    （body_bytes）。"""
    keys = Keys.parse(KEYS, TOKEN)
    reasons: list[str] = []
    release: anyio.Event | None = None

    async def app(scope: Any, receive: Any, send: Any) -> None:
        assert release is not None
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    limit = BodyLimit(
        app, keys=keys, max_concurrent=2, max_body_bytes=100, on_throttle=reasons.append
    )

    async def call(length: int) -> int:
        statuses: list[int] = []

        async def receive() -> dict[str, Any]:
            return {"type": "http.request", "body": b""}

        async def send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                statuses.append(message["status"])

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/memories",
            "headers": [
                (b"authorization", f"Bearer {SCOPED_TOKEN}".encode()),
                (b"content-length", str(length).encode()),
            ],
        }
        await limit(scope, receive, send)
        return statuses[0]

    async def main() -> None:
        nonlocal release
        release = anyio.Event()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(call, 60)
            await anyio.wait_all_tasks_blocked()
            assert await call(60) == 429 and reasons == ["body_bytes"]
            tasks.start_soon(call, 0)
            await anyio.wait_all_tasks_blocked()
            assert await call(0) == 429 and reasons == ["body_bytes", "concurrency"]
            release.set()

    anyio.run(main)


def test_request_durations_are_a_histogram(served: Served) -> None:
    """每个请求进一次直方图：_count 是请求数，+Inf 那一格等于 _count，各格累加。"""
    now = [0.0]
    registry = Metrics(clock=lambda: now[0])
    registry.allow_routes({"/x"})

    async def app(scope: Any, receive: Any, send: Any) -> None:
        now[0] += scope["took"]
        if scope["took"] == 7.0:
            raise RuntimeError("boom")  # 没有送出响应头：记成 500
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    middleware = MetricsMiddleware(app, metrics=registry, routes=frozenset({"/x"}))

    async def send(message: dict[str, Any]) -> None:
        return None

    async def call(took: float) -> None:
        scope = {"type": "http", "method": "GET", "path": "/x", "took": took}
        with contextlib.suppress(RuntimeError):
            await middleware(scope, send, send)  # type: ignore[arg-type]

    async def main() -> None:
        for took in (0.005, 0.04, 0.04, 3.0, 7.0, 500.0):
            await call(took)

    anyio.run(main)
    samples = _samples(registry.render())
    name = "cuemem_http_request_duration_seconds"
    buckets = [samples[f'{name}_bucket{{route="other",le="{edge:g}"}}'] for edge in BUCKETS]
    assert buckets == [1, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5]
    assert samples[f'{name}_bucket{{route="other",le="+Inf"}}'] == 6
    assert samples[f'{name}_count{{route="other"}}'] == 6
    assert samples[f'{name}_sum{{route="other"}}'] == pytest.approx(510.085)
    requests = "cuemem_http_requests_total"
    assert samples[f'{requests}{{method="GET",route="other",status="204"}}'] == 5
    assert samples[f'{requests}{{method="GET",route="other",status="500"}}'] == 1
    assert samples["cuemem_http_requests_in_flight"] == 0

    with TestClient(served.app()) as http:
        for _ in range(3):
            assert http.get("/api/health").status_code == 200
        text = http.get("/metrics", headers=AUTH).text
    count = re.search(rf'{name}_count\{{route="/api/health"\}} (\d+)', text)
    top = re.search(rf'{name}_bucket\{{route="/api/health",le="\+Inf"\}} (\d+)', text)
    assert count and top and count.group(1) == top.group(1) == "3"


def test_a_scrape_never_waits_for_the_database(served: Served) -> None:
    """读 /metrics 不查数据库：数据库的锁被占着（遗忘重放时很久）也马上返回。任务数、空间数与
    向量库的状态由清理线程刷新，idle_minutes 为 None、没有别的清理任务时这个线程也起。"""
    served.service.remember("alice", "I live in Lisbon.")
    served.service.remember("bob", "I live in Porto.")
    with TestClient(served.app(idle_minutes=None)) as http:
        deadline = time.monotonic() + 10
        while "cuemem_stats_age_seconds" not in (samples := _scrape(http)):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        assert samples["cuemem_spaces"] == 2 and samples["cuemem_vector_store_up"] == 1
        assert samples['cuemem_consolidation_jobs{state="failed"}'] == 0
        assert 'cuemem_consolidation_jobs{state="pending"}' in samples
        assert samples["cuemem_stats_age_seconds"] < 10
        assert samples["cuemem_open_sessions"] == 0 and samples["cuemem_open_runs"] == 0
        assert samples["cuemem_write_backs_waiting"] == 0
        assert samples["cuemem_write_backs_dropped_total"] == 0
        assert ("cuemem_worker_alive" in samples) == (served.system.worker is not None)

        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with served.system.db._lock:
                held.set()
                release.wait(30)

        holder = threading.Thread(target=hold)
        holder.start()
        try:
            assert held.wait(10)
            started = time.monotonic()
            assert http.get("/metrics", headers=AUTH).status_code == 200
            assert time.monotonic() - started < 1
        finally:
            release.set()
            holder.join()


def test_the_standalone_proxy_has_metrics(served: Served, fake_api: FakeAPI) -> None:
    """cuemem proxy --metrics：转发的请求、它的路由模板与进行中的会话数。"""
    fake_api.chat_replies = ["Hello!"]
    app = create_proxy_app(served.proxy, token=TOKEN, metrics=True)
    with TestClient(app) as http:
        chat = {
            "model": "qwen3.8-flash",
            "user": "alice",
            "messages": [{"role": "user", "content": "hi"}],
        }
        assert http.post("/v1/chat/completions", json=chat, headers=AUTH).status_code == 200
        samples = _scrape(http)
    requests = "cuemem_http_requests_total"
    assert samples[f'{requests}{{method="POST",route="/v1/chat/completions",status="200"}}'] == 1
    assert samples["cuemem_open_runs"] == 1 and "cuemem_open_sessions" not in samples
    llm = 'cuemem_llm_requests_total{role="proxy",model="qwen3.8-flash",cached="false"}'
    assert samples[llm] == 1
    tokens = 'cuemem_llm_tokens_total{role="proxy",model="qwen3.8-flash",direction="output"}'
    assert samples[tokens] > 0
