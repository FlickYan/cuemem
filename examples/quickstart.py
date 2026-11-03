"""CueMem 的 Python SDK：十几行代码走一遍写入、检索、回答、版本链与遗忘（N12）。

    uv run python examples/quickstart.py                    # 默认配置（DeepSeek + QwenCloud）
    uv run python examples/quickstart.py --preset openai    # OPENAI_API_KEY
    uv run python examples/quickstart.py --preset ollama    # 本机 Ollama，不要 Key

数据写在临时目录里，跑完即删，不碰你的记忆库。
"""

from __future__ import annotations

import argparse
import json
import tempfile

from cuemem import Memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default=None, help="openai、ollama，或不填用默认配置")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as home, Memory(preset=args.preset, home=home) as m:
        # 写入：先在线写入原始日志，wait=True 时同步抽取事实
        m.add(
            "I live in Berlin and I run every Sunday.",
            user_id="alice",
            timestamp="2026-08-01T10:00",
            wait=True,
        )
        moved = m.add(
            [
                {"role": "user", "content": "Big news: I moved to Munich last week!"},
                {"role": "assistant", "content": "Congratulations on the move!"},
            ],
            user_id="alice",
            timestamp="2026-09-01T10:00",
            wait=True,
        )
        print("events:", json.dumps(moved["results"], indent=1, ensure_ascii=False)[:800])

        # 检索：控制器决定回忆多深
        found = m.search("Where do I live now?", user_id="alice")
        print("level:", found["level"], found["reasons"])
        for item in found["results"][:5]:
            print(f"  [{item['type']}] {item['memory'][:80]}  ({item.get('status', '')})")
        # search 不低于 L1（调用它就是决定要查）；min_level="l0" 时照控制器的决定，可以不查
        gate = m.search("What is 2 + 2?", user_id="alice", min_level="l0")
        print("no memory needed:", gate["level"])

        # 回答（带引用）与版本链
        print("answer:", m.answer("Where do I live now?", user_id="alice")["answer"])
        update = next((e for e in moved["results"] if e["event"] == "UPDATE"), None)
        if update is not None:
            for version in m.history(update["id"]):
                print(f"  {version['event']:6} {version['memory']}  [{version['status']}]")

        # 遗忘：先预演，再真的删
        if update is not None:
            plan = m.delete(update["id"], dry_run=True)
            print("would delete:", [t["text"] for t in plan["turns"]])
            m.delete(update["id"])
            print("after forgetting:", [f["memory"] for f in m.get_all(user_id="alice")["results"]])


if __name__ == "__main__":
    main()
