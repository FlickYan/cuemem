"""不要 Key 试一试：离线哈希向量，写入一句话、再检索出来，不调用任何模型。

    uv run python examples/keyless.py

数据写在临时目录里，跑完即删，不碰你的记忆库：数据目录会记住向量模型（命令失败了也一样），哈希向量
只在单独的目录里试。infer=False 只写原始日志、不抽取事实；抽取、回答（m.answer、cuemem ask）与
巩固（cuemem consolidate）要 LLM 的 Key。哈希向量只看字面，检索质量远不如真正的向量模型。
"""

from __future__ import annotations

import tempfile

from cuemem import Memory


def main() -> None:
    with (
        tempfile.TemporaryDirectory() as home,
        Memory(home=home, config={"embedding": {"backend": "hash"}}, background=False) as m,
    ):
        m.add(
            [{"role": "user", "content": "I just started a new job at Acme in Berlin."}],
            user_id="alice",
            infer=False,
        )
        found = m.search("Where do I work?", user_id="alice")
        print(found["level"], found["results"][0]["memory"])


if __name__ == "__main__":
    main()
