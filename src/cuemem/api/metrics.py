"""GET /metrics：Prometheus 文本格式的运行指标（第十轮体检 Q18）。默认不开。

打开：cuemem serve --metrics / cuemem proxy --metrics，或环境变量 CUEMEM_METRICS=1
（create_app / create_proxy_app 的 metrics=True）。不开时没有这个路由、没有中间件，
LLMClient.observer 与各处的 on_throttle 都是 None，什么都不多做。

谁能读：只有管理令牌（限定作用域的令牌 403，没带令牌 401）；没配令牌的本机演示与别的接口一样
不鉴权。不算进令牌的次数与并发上限（不在 limits.RATE_PATHS 里），不出现在接口描述里。

标签的规则：每个标签的取值都出自启动时定下的一个封闭集合，集合之外的值一律记成 "other"。
标签里因此不会有用户 ID、Agent ID、空间、会话、记忆 ID、令牌的名字、提问或请求的原始路径：

- route：应用自己的路由的模板（/api/v1/memories/{memory_id}），不是请求的路径。/mcp 是
  "/mcp"，落到界面（Gradio，挂在根路径上）的是 "ui"，别的是 "other"。只认应用自己的路由模板：
  挂载的 Gradio 会把它里面的路由（/gradio_api/file={p:path} 之类）写进 scope["route"]，/mcp 那个
  路由又不留下 scope["route"]；
- method：GET、POST、PUT、PATCH、DELETE、HEAD、OPTIONS，别的是 OTHER；
- status：三位数的状态码；
- reason（cuemem_throttled_total）：REASONS 里的一个；
- role、model（cuemem_llm_*）：配置里的角色名（加上记忆代理的 proxy、proxy-unmetered）与模型
  目录里的别名。

读这个接口不查数据库、不探向量库（/metrics 在事件循环上，遗忘重放时数据库的锁要等很久，见
app._readiness）：任务数、空间数与向量库能不能用由清理线程每分钟刷新一次（refresh_stats），
cuemem_stats_age_seconds 是离上次刷新过了多久。计数都是这个进程自己的，重启后从 0 开始。

没有新的依赖：一把锁、几个字典，自己写出文本格式（0.0.4）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from cuemem import __version__
from cuemem.api.auth import AnyKeys, guard, principal_of
from cuemem.errors import CueMemError

if TYPE_CHECKING:
    from cuemem.config import ModelSpec
    from cuemem.llm.types import Usage
    from cuemem.memory import MemorySystem

logger = logging.getLogger(__name__)

METRICS_ENV = "CUEMEM_METRICS"
CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
OTHER = "other"
BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 120.0)  # 秒；另有 +Inf
METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"})
# 请求被上限挡回去（429、413）的原因。rate、concurrency、body_bytes、body_too_large 在
# limits.BodyLimit；open_sessions 是一个空间进行中的会话数，spaces 是令牌的 max_spaces，budget 是
# 令牌的 max_daily_tokens（记忆代理）。MCP 的工具调用与界面里的同类拒绝不是 HTTP 429，不计
REASONS = frozenset(
    {"rate", "concurrency", "body_bytes", "body_too_large", "open_sessions", "spaces", "budget"}
)
JOB_STATES = ("pending", "failed")  # 任务表里没有"进行中"这个状态：在做的也是 pending
PROXY_ROLES = ("proxy", "proxy-unmetered")  # 记忆代理转发的请求记账用的角色（api/proxy.py）

Kind = Literal["counter", "gauge", "histogram"]
Labels = Mapping[str, str]


def metrics_enabled(flag: bool = False, environ: Mapping[str, str] | None = None) -> bool:
    """开不开 /metrics：命令行的 --metrics，或环境变量 CUEMEM_METRICS（1 / true 开，0 / false
    或不设不开，不分大小写）。别的值 CueMemError（cuemem serve / proxy 打印原因、退出码 1）。"""
    env = os.environ if environ is None else environ
    raw = (env.get(METRICS_ENV) or "").strip()
    if raw.lower() not in ("", "0", "1", "true", "false"):
        raise CueMemError(f"{METRICS_ENV} 只能是 1、true、0 或 false，现在是 {raw!r}")
    return flag or raw.lower() in ("1", "true")


class _Metric:
    def __init__(self, kind: Kind, text: str, labels: dict[str, frozenset[str]]) -> None:
        self.kind = kind
        self.text = text
        self.labels = labels  # 标签名 → 允许的取值（声明的顺序就是输出的顺序）
        self.values: dict[tuple[str, ...], float] = {}
        # 直方图：标签 → 各个桶里的个数（不累加；最后一格是 +Inf）与总和
        self.buckets: dict[tuple[str, ...], list[int]] = {}
        self.sums: dict[tuple[str, ...], float] = {}


class Metrics:
    """一个应用的指标（create_app / create_proxy_app 各建一个，不是模块里的全局对象：测试里
    建很多个应用）。一把锁；inc / set / observe 都只是改字典，不做 I/O。

    每个指标声明时写明标签与每个标签允许的取值，之外的值记成 "other"（不抛异常：指标不能让
    请求失败）。roles、models：cuemem_llm_* 的 role 与 model 的取值。"""

    def __init__(
        self,
        *,
        roles: Iterable[str] = (),
        models: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.clock = clock
        self._lock = threading.Lock()
        self._metrics: dict[str, _Metric] = {}
        self._stats_at: float | None = None  # refresh_stats 上次做完的时刻
        role_set = frozenset(roles) | frozenset(PROXY_ROLES)
        model_set = frozenset(models)
        flag = frozenset({"true", "false"})
        self.declare("cuemem_build_info", "gauge", "CueMem version.", version={__version__})
        self.declare(
            "cuemem_http_requests_total",
            "counter",
            "HTTP requests by method, route template and status code.",
            method=METHODS | {"OTHER"},
            route=(),  # 装中间件时填（allow_routes）
            status={str(code) for code in range(100, 600)},
        )
        self.declare(
            "cuemem_http_request_duration_seconds",
            "histogram",
            "HTTP request duration by route template, to the end of the response body.",
            route=(),
        )
        self.declare("cuemem_http_requests_in_flight", "gauge", "HTTP requests being handled.")
        self.declare(
            "cuemem_throttled_total",
            "counter",
            "Requests refused by a limit (HTTP 429 or 413), by reason.",
            reason=REASONS,
        )
        self.declare("cuemem_open_sessions", "gauge", "Open chat sessions of /api and the UI.")
        self.declare("cuemem_open_runs", "gauge", "Open sessions of the memory proxy.")
        self.declare(
            "cuemem_write_backs_waiting", "gauge", "Memory proxy write-backs kept for a retry."
        )
        self.declare(
            "cuemem_write_backs_dropped_total",
            "counter",
            "Memory proxy write-backs dropped since the process started.",
        )
        self.declare(
            "cuemem_worker_alive", "gauge", "1 when every background consolidation thread is up."
        )
        self.declare(
            "cuemem_consolidation_jobs",
            "gauge",
            "Consolidation jobs by state, refreshed once a minute.",
            state=JOB_STATES,
        )
        self.declare("cuemem_spaces", "gauge", "Memory spaces, refreshed once a minute.")
        self.declare(
            "cuemem_vector_store_up",
            "gauge",
            "1 when the vector store answered the last check, refreshed once a minute.",
        )
        self.declare(
            "cuemem_stats_age_seconds",
            "gauge",
            "Seconds since the jobs, spaces and vector store gauges were refreshed.",
        )
        self.declare(
            "cuemem_llm_requests_total",
            "counter",
            "Chat completions by role and model alias; cached=true were answered from the "
            "response cache.",
            role=role_set,
            model=model_set,
            cached=flag,
        )
        self.declare(
            "cuemem_llm_tokens_total",
            "counter",
            "Tokens of chat completions sent to a provider (response-cache hits not counted); "
            "cached_input is the part of input the provider served from its prompt cache.",
            role=role_set,
            model=model_set,
            direction={"input", "output", "cached_input"},
        )
        self.set("cuemem_build_info", {"version": __version__}, 1)
        self.set("cuemem_http_requests_in_flight", {}, 0)
        for reason in sorted(REASONS):  # 从 0 开始：rate() 在第一次拒绝之前也有值
            self.inc("cuemem_throttled_total", {"reason": reason}, 0)

    def declare(self, name: str, kind: Kind, text: str, **labels: Iterable[str]) -> None:
        self._metrics[name] = _Metric(kind, text, {k: frozenset(v) for k, v in labels.items()})

    def allow_routes(self, routes: Iterable[str]) -> None:
        """route 标签允许的取值：应用自己的顶层路由的模板，加上 /mcp、ui。"""
        allowed = frozenset(routes) | {"/mcp", "ui"}
        with self._lock:
            for metric in self._metrics.values():
                if "route" in metric.labels:
                    metric.labels["route"] = allowed

    def _key(self, metric: _Metric, labels: Labels) -> tuple[str, ...]:
        return tuple(
            value if (value := str(labels.get(name, OTHER))) in allowed else OTHER
            for name, allowed in metric.labels.items()
        )

    def inc(self, name: str, labels: Labels | None = None, by: float = 1.0) -> None:
        with self._lock:
            metric = self._metrics[name]
            key = self._key(metric, labels or {})
            metric.values[key] = metric.values.get(key, 0.0) + by

    def set(self, name: str, labels: Labels | None, value: float) -> None:
        with self._lock:
            metric = self._metrics[name]
            metric.values[self._key(metric, labels or {})] = float(value)

    def observe(self, name: str, labels: Labels | None, seconds: float) -> None:
        with self._lock:
            metric = self._metrics[name]
            key = self._key(metric, labels or {})
            counts = metric.buckets.setdefault(key, [0] * (len(BUCKETS) + 1))
            counts[
                next((i for i, edge in enumerate(BUCKETS) if seconds <= edge), len(BUCKETS))
            ] += 1
            metric.sums[key] = metric.sums.get(key, 0.0) + seconds

    def throttled(self, reason: str) -> None:
        """一个请求被上限挡了回去（BodyLimit、/api、/api/v1 与记忆代理的 on_throttle）。"""
        self.inc("cuemem_throttled_total", {"reason": reason})

    def llm(self, role: str, alias: str, spec: ModelSpec, usage: Usage, from_cache: bool) -> None:
        """LLMClient.observer：一次聊天补全记完账之后（含记忆代理转发的）。命中响应缓存的只算
        次数，不算 token：没有发给供应商。"""
        labels = {"role": role, "model": alias}
        self.inc(
            "cuemem_llm_requests_total", {**labels, "cached": "true" if from_cache else "false"}
        )
        if from_cache:
            return
        for direction, tokens in (
            ("input", usage.input_tokens),
            ("output", usage.output_tokens),
            ("cached_input", usage.cached_input_tokens),
        ):
            self.inc("cuemem_llm_tokens_total", {**labels, "direction": direction}, tokens)

    def stats_refreshed(self) -> None:
        with self._lock:
            self._stats_at = self.clock()

    def render(self) -> str:
        """Prometheus 文本格式（0.0.4）：每个有值的指标一段 HELP、TYPE 与它的各行，按标签排序。"""
        with self._lock:
            if self._stats_at is not None:  # 没刷新过时不写这一项
                age = self._metrics["cuemem_stats_age_seconds"]
                age.values[()] = max(0.0, self.clock() - self._stats_at)
            lines: list[str] = []
            for name, metric in self._metrics.items():
                if not metric.values and not metric.buckets:
                    continue
                lines.append(f"# HELP {name} {_escape(metric.text, quote=False)}")
                lines.append(f"# TYPE {name} {metric.kind}")
                names = tuple(metric.labels)
                for key in sorted(metric.values):
                    lines.append(f"{name}{_labels(names, key)} {_number(metric.values[key])}")
                for key in sorted(metric.buckets):
                    seen = 0
                    for edge, count in zip((*BUCKETS, None), metric.buckets[key], strict=True):
                        seen += count
                        le = "+Inf" if edge is None else _number(edge)
                        lines.append(f"{name}_bucket{_labels((*names, 'le'), (*key, le))} {seen}")
                    lines.append(f"{name}_sum{_labels(names, key)} {_number(metric.sums[key])}")
                    lines.append(f"{name}_count{_labels(names, key)} {seen}")
        return "\n".join(lines) + "\n"


def _escape(text: str, *, quote: bool = True) -> str:
    text = text.replace("\\", "\\\\").replace("\n", "\\n")
    return text.replace('"', '\\"') if quote else text


def _labels(names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not names:
        return ""
    pairs = ",".join(f'{n}="{_escape(v)}"' for n, v in zip(names, values, strict=True))
    return "{" + pairs + "}"


def _number(value: float) -> str:
    return str(int(value)) if value == int(value) else repr(float(value))


class MetricsMiddleware:
    """整个应用最外层的中间件（install）：每个 HTTP 请求记一次个数、用时与正在处理的请求数。
    在最外层，被 Host 检查、请求体上限挡回去的请求也算。用时算到应用返回为止：流式的响应算到
    流结束。没有送出响应头就出错的记成 500。route 标签的取法见模块说明。"""

    def __init__(
        self, app: ASGIApp, *, metrics: Metrics, routes: frozenset[str], ui: bool = False
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.routes = routes
        self.ui = ui

    def _route(self, scope: Scope) -> str:
        template = getattr(scope.get("route"), "path", None)
        if template in self.routes:
            return str(template)
        if scope["path"] == "/mcp":
            return "/mcp"
        return "ui" if self.ui else OTHER

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        metrics = self.metrics
        status = 500
        start = metrics.clock()

        async def watched(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
            await send(message)

        metrics.inc("cuemem_http_requests_in_flight", by=1)
        try:
            await self.app(scope, receive, watched)
        finally:
            route = self._route(scope)
            method = scope["method"] if scope["method"] in METHODS else "OTHER"
            metrics.inc("cuemem_http_requests_in_flight", by=-1)
            metrics.inc(
                "cuemem_http_requests_total",
                {"method": method, "route": route, "status": str(status)},
            )
            metrics.observe(
                "cuemem_http_request_duration_seconds", {"route": route}, metrics.clock() - start
            )


def _templates(routes: Iterable[Any]) -> Iterator[str]:
    """应用自己的路由的模板：Route（含 APIRoute）的 path，不含 Mount（挂载的界面）。新的 FastAPI
    里 include_router 不把路由摊平，app.routes 里是一个包着原来那个 APIRouter 的对象：往里走。
    请求匹配到的 scope["route"] 就是这里走到的路由（CueMem 的 include_router 都不另加前缀，
    path 就是完整的模板）。认不出的对象跳过：它的路由记成 other。"""
    for route in routes:
        if isinstance(route, Route):
            yield route.path
        elif (inner := getattr(route, "original_router", None)) is not None:
            yield from _templates(inner.routes)


def install(app: FastAPI, metrics: Metrics, *, ui: bool = False) -> None:
    """装上 MetricsMiddleware。要在路由都注册完、app._guard 之后调用：最后加的中间件在最外层，
    route 的允许取值取这时应用自己的路由（Route，不含 Mount）。"""
    routes = frozenset(_templates(app.routes))
    metrics.allow_routes(routes)
    app.add_middleware(MetricsMiddleware, metrics=metrics, routes=routes, ui=ui)


def add_route(
    app: FastAPI, metrics: Metrics, keys: AnyKeys, live: Callable[[], Mapping[str, float]]
) -> None:
    """GET /metrics（不出现在接口描述里）：只给管理令牌。live：这时从内存里读出的值（指标名 →
    值，进行中的会话数之类）；不查数据库、不探向量库，见模块说明。指标对象记在 app.state.metrics
    上（嵌入这个应用的程序与测试用）。"""
    app.state.metrics = metrics

    @app.get("/metrics", include_in_schema=False, dependencies=guard(keys))
    async def scrape(http: Request) -> PlainTextResponse:
        if not principal_of(http).admin:
            raise HTTPException(403, "Metrics are served to the admin token only.")
        for name, value in live().items():
            metrics.set(name, None, value)
        return PlainTextResponse(metrics.render(), media_type=CONTENT_TYPE)


def for_system(system: MemorySystem) -> Metrics:
    """这个记忆系统的指标：role、model 的取值来自它的配置，聊天补全记账之后通知它
    （LLMClient.observer）。"""
    settings = system.settings
    metrics = Metrics(roles=settings.roles, models=settings.models)
    system.llm.observer = metrics.llm
    return metrics


def worker_alive(system: MemorySystem) -> dict[str, float]:
    """cuemem_worker_alive（没有后台巩固线程时不写这一项）。"""
    worker: Any = system.worker
    return {} if worker is None else {"cuemem_worker_alive": float(worker.alive())}


def refresh_stats(system: MemorySystem, metrics: Metrics) -> Callable[[], None]:
    """清理线程的一项（每分钟一遍，排在最后：数据库的锁被占着时不耽误别的）：各状态的巩固任务数、
    空间数、向量库能不能用。/metrics 自己不做这些查询。"""

    def refresh() -> None:
        counts = system.db.job_totals()
        for state in JOB_STATES:
            metrics.set("cuemem_consolidation_jobs", {"state": state}, counts.get(state, 0))
        metrics.set("cuemem_spaces", None, len(system.db.space_ids()))
        metrics.set("cuemem_vector_store_up", None, float(system.vectors.ping()))
        metrics.stats_refreshed()

    return refresh
