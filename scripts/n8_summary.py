"""N8：把演示剧本的回放报告（cuemem demo --report）压成结果摘要。

写到 docs/results/n8-agent/demo.json（--out 另写一个文件）。
摘要只留每一轮的档位、工具、引用、回答、用量与耗时，不含本地文件路径（报告里图片的路径带着
本机用户名）。

发表的花费与耗时是真实调用的：回放里有命中本地缓存的模型调用时（花费记 0、几乎不耗时），不写
发表的 demo.json、退出码 1，除非加 --allow-cached。复现脚本（scripts/reproduce.py）写到
runs/demo/summary.json。

用法：
    uv run cuemem demo demo/shop/script.json --fresh --report runs/demo/final-full.json
    uv run python scripts/n8_summary.py runs/demo/final-full.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from cuemem.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "docs" / "results" / "n8-agent" / "demo.json"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path, help="cuemem demo --report 写的回放报告")
    parser.add_argument(
        "--out", type=Path, default=OUT, help="摘要写到哪里（默认是发表的 demo.json）"
    )
    parser.add_argument(
        "--allow-cached",
        action="store_true",
        help="回放里有命中本地缓存的调用时也写发表的 demo.json",
    )
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    turns: list[dict[str, Any]] = []
    for t in report["turns"]:
        row: dict[str, Any] = {
            "session": t["session"],
            "index": t["index"],
            "text": t["text"],
            "passed": t["passed"],
            "failures": t["failures"],
        }
        if t.get("forgot") is not None:
            row["forgot"] = t["forgot"]
        reply = t.get("reply")
        if reply:
            trace = reply["trace"]
            usage = trace["usage"]
            row.update(
                level=t["level"],
                expected_level=t["expected_level"],
                p_none=trace["scores"].get("p_none"),
                tools=[{"name": x["name"], "arguments": x["arguments"]} for x in trace["tools"]],
                citations=reply["citations"],
                answer=reply["plain"],
                recalled=len(trace["recalled"]),
                partitions=trace["partitions"],
                usage={
                    k: usage[k]
                    for k in ("llm_calls", "input_tokens", "cached_input_tokens", "output_tokens")
                }
                | {"spent": round(usage["spent"], 6)},
                timings=trace["timings"],
            )
        turns.append(row)
    chats = [r for r in turns if "level" in r]
    levels: dict[str, int] = {}
    for r in chats:
        levels[r["level"]] = levels.get(r["level"], 0) + 1
    total_ms = [r["timings"]["total"] for r in chats]
    llm_ms = [r["timings"]["llm"] for r in chats]
    summary = {
        "checks": len(turns),
        "passed": sum(r["passed"] for r in turns),
        "chat_turns": len(chats),
        "levels": levels,
        "memory_search_turns": sum(
            any(x["name"] == "memory_search" for x in r["tools"]) for r in chats
        ),
        "shop_tool_turns": sum(
            any(x["name"] in ("search_products", "find_similar_products") for x in r["tools"])
            for r in chats
        ),
        "llm_calls_per_turn": statistics.mean(r["usage"]["llm_calls"] for r in chats),
        "input_tokens": sum(r["usage"]["input_tokens"] for r in chats),
        "cached_input_tokens": sum(r["usage"]["cached_input_tokens"] for r in chats),
        "spent_per_turn": statistics.mean(r["usage"]["spent"] for r in chats),
        "total_s_median": statistics.median(total_ms) / 1000,
        "total_s_mean": statistics.mean(total_ms) / 1000,
        "llm_s_mean": statistics.mean(llm_ms) / 1000,
        "write_ms_median": statistics.median(r["timings"]["write"] for r in chats),
    }
    cached = sum(
        int(t["reply"]["trace"]["usage"].get("cache_hits", 0))
        for t in report["turns"]
        if t.get("reply")
    )
    if cached and args.out.resolve() == OUT.resolve() and not args.allow_cached:
        raise SystemExit(
            f"回放里有 {cached} 次模型调用命中本地缓存（花费记 0、几乎不耗时），不覆盖发表的 "
            f"{OUT.name}；用 --out 另写一个文件，或加 --allow-cached"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"summary": summary, "turns": turns}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
