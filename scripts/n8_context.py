"""N8：稳定前缀值多少——两种上下文布局各跑一遍，比较供应商前缀缓存的命中。

两组对话：演示剧本（三段会话，每轮都召回记忆），以及一段 16 轮的长会话（不召回，只看对话历史
能不能被缓存）。

- stable（默认）：系统提示 + 画像 + 本会话的原话在前，日期与召回的记忆只放在最后一条消息里；
- naive（对照）：日期与召回的记忆放进系统提示（很多实现的写法），每轮请求从第一条消息起就不同。

两遍都关掉本地响应缓存（每次都真实调用），各用一个新的记忆空间，其余完全相同。
比较 Agent 调用的输入 token 里有多少由供应商的前缀缓存提供（qwen3.8-flash 命中价约为原价的 1/9），
以及按缓存折扣算的实际输入成本。结果写到 docs/results/n8-agent/context.json。

用法（通义没有分时价格，什么时候跑都一样；两遍合计约 $0.1）：
    uv run python scripts/n8_context.py
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from cuemem.agent import Agent
from cuemem.agent.shop import SHOP_PERSONA, Catalog, shop_tools
from cuemem.api.replay import DemoScript, TurnResult, replay
from cuemem.api.service import SHOP_CATALOG, ChatService
from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.memory import MemorySystem

SCRIPT = PROJECT_ROOT / "demo" / "shop" / "script.json"
OUT = PROJECT_ROOT / "docs" / "results" / "n8-agent"


def run(layout: str) -> dict[str, Any]:
    settings = load_settings(peak_policy="warn").model_copy(update={"llm_cache_mode": "off"})
    system = MemorySystem.from_settings(settings)
    catalog = Catalog.load(
        SHOP_CATALOG, embedder=system.embedder, image_embedder=system.image_embedder
    )
    agent = Agent(
        system,
        persona=SHOP_PERSONA,
        tools=shop_tools(catalog),
        layout=layout,  # type: ignore[arg-type]
    )
    service = ChatService(system, agent, catalog=catalog)
    space = f"shop-context-{layout}"
    system.db.delete_space(space)
    system.vectors.delete_space(space)
    script = DemoScript.load(SCRIPT)
    try:
        results = replay(service, script, root=SCRIPT.parent, space=space)
    finally:
        service.close()
    return summarize(layout, results)


def summarize(layout: str, results: list[TurnResult]) -> dict[str, Any]:
    turns = [r for r in results if r.reply is not None]
    total = {"input": 0, "cached": 0, "output": 0, "calls": 0}
    rows = []
    for r in turns:
        assert r.reply is not None
        usage = r.reply.trace.usage
        rows.append(
            {
                "session": r.session,
                "index": r.index,
                "level": r.level,
                "passed": r.passed,
                "input": int(usage["input_tokens"]),
                "cached": int(usage["cached_input_tokens"]),
                "output": int(usage["output_tokens"]),
                "llm_ms": r.reply.trace.timings.get("llm", 0.0),
            }
        )
        total["input"] += int(usage["input_tokens"])
        total["cached"] += int(usage["cached_input_tokens"])
        total["output"] += int(usage["output_tokens"])
        total["calls"] += int(usage["llm_calls"])
    price = {"input": 0.15, "cached_input": 0.016}  # qwen3.8-flash，美元 / 百万 tokens
    nominal = total["input"] * price["input"] / 1e6
    fresh = total["input"] - total["cached"]
    discounted = (fresh * price["input"] + total["cached"] * price["cached_input"]) / 1e6
    return {
        "layout": layout,
        "turns": len(turns),
        "passed": sum(r.passed for r in results),
        "checks": len(results),
        **total,
        "cached_share": total["cached"] / total["input"] if total["input"] else 0.0,
        "input_cost_nominal": nominal,
        "input_cost_with_cache": discounted,
        "llm_ms_mean": sum(r["llm_ms"] for r in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }


LONG_SESSION = [
    "Hi! I'm training for my first marathon next spring. Can you help me put together my gear?",
    "Let's start with shoes. I like a soft, cushioned ride and my budget is around $150.",
    "What's the difference between the Stratus Cushion 2 and the Aero Knit Runner?",
    "Which would you pick for long runs over 25 km?",
    "OK. What about socks? I get blisters on long runs.",
    "And for hydration: vest or handheld bottle?",
    "Tell me more about the hydration vest.",
    "Do I need a GPS watch, or is my phone enough?",
    "What would you recommend for running in the rain?",
    "I also run early in the morning when it's dark. Anything for visibility?",
    "Could you sum up a full kit under $400?",
    "Swap the watch for something cheaper, please.",
    "What should I wear on race day?",
    "Any recovery gear you'd suggest after long runs?",
    "Great. Can you list everything we picked with prices?",
    "Thanks, that's really helpful!",
]


def long_session(layout: str) -> dict[str, Any]:
    """一段 16 轮的会话，每轮都不召回记忆（L0）：对话历史越来越长，稳定前缀能把它缓存下来。"""
    settings = load_settings(peak_policy="warn").model_copy(update={"llm_cache_mode": "off"})
    system = MemorySystem.from_settings(settings)
    catalog = Catalog.load(
        SHOP_CATALOG, embedder=system.embedder, image_embedder=system.image_embedder
    )
    agent = Agent(
        system,
        persona=SHOP_PERSONA,
        tools=shop_tools(catalog),
        layout=layout,  # type: ignore[arg-type]
    )
    space = f"shop-long-{layout}"
    system.db.delete_space(space)
    system.vectors.delete_space(space)
    start = datetime(2026, 10, 1, 9, 0)
    session = agent.start(space, at=start, session_id="long")
    rows = []
    try:
        for i, text in enumerate(LONG_SESSION):
            reply = agent.chat(session, text, at=start + timedelta(minutes=2 * i), mode="l0")
            usage = reply.trace.usage
            rows.append(
                {
                    "turn": i + 1,
                    "input": int(usage["input_tokens"]),
                    "cached": int(usage["cached_input_tokens"]),
                    "calls": int(usage["llm_calls"]),
                    "window": reply.trace.window,
                    "evicted": reply.trace.evicted,
                }
            )
    finally:
        agent.end(session)
        system.close()
    total_in = sum(r["input"] for r in rows)
    total_cached = sum(r["cached"] for r in rows)
    late = rows[len(rows) // 2 :]
    return {
        "layout": layout,
        "input": total_in,
        "cached": total_cached,
        "cached_share": total_cached / total_in if total_in else 0.0,
        "late_cached_share": sum(r["cached"] for r in late) / max(1, sum(r["input"] for r in late)),
        "rows": rows,
    }


def main() -> None:
    out: dict[str, Any] = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "runs": {},
        "long": {},
    }
    for layout in ("stable", "naive"):
        summary = run(layout)
        out["runs"][layout] = summary
        print(
            f"剧本 {layout:7s} 通过 {summary['passed']}/{summary['checks']} · Agent 调用 "
            f"{summary['calls']} 次 · 输入 {summary['input']:,} · 前缀缓存 {summary['cached']:,}"
            f"（{summary['cached_share']:.0%}）· 输入成本 ${summary['input_cost_nominal']:.4f} → "
            f"${summary['input_cost_with_cache']:.4f}"
        )
    for layout in ("stable", "naive"):
        summary = long_session(layout)
        out["long"][layout] = summary
        print(
            f"长会话 {layout:7s} 输入 {summary['input']:,} · 前缀缓存 {summary['cached']:,}"
            f"（{summary['cached_share']:.0%}；后 8 轮 {summary['late_cached_share']:.0%}）"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "context.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
