"""pip 安装后的冒烟测试（CI 的 package 任务）：在源码仓库之外、用装好的 wheel 跑一遍。

离线、不需要 API Key：向量用哈希向量器（CUEMEM_EMBEDDING__BACKEND=hash），数据写到 CUEMEM_HOME。
覆盖 SDK 的写入、检索与导出，导购演示的商品目录不在包里时的报错，以及 HTTP 接口能起来。

    cd "$(mktemp -d)"                # 不在仓库里：import cuemem 拿到的是装好的包
    CUEMEM_HOME=$PWD/home CUEMEM_EMBEDDING__BACKEND=hash python /path/to/scripts/smoke_wheel.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from fastapi.testclient import TestClient

import cuemem
from cuemem import Memory
from cuemem.api.app import create_app
from cuemem.api.service import ChatService
from cuemem.config import load_settings, source_checkout
from cuemem.errors import CueMemError


def main() -> None:
    assert not source_checkout(), f"应该测装好的包，实际导入的是 {cuemem.__file__}"
    assert os.environ.get("CUEMEM_EMBEDDING__BACKEND") == "hash", "离线冒烟测试要用哈希向量器"
    home = Path(os.environ["CUEMEM_HOME"])

    memory = Memory(background=False)
    try:
        added = memory.add(
            [
                {"role": "user", "content": "I adopted a golden retriever named Max last spring."},
                {"role": "assistant", "content": "Congratulations on adopting Max!"},
            ],
            user_id="ci",
            infer=False,
        )
        assert added["turn_ids"], added
        found = memory.search("golden retriever", user_id="ci", mode="l1")
        assert "golden retriever" in str(found), found
        exported = memory.export(user_id="ci", derived=False)
        assert exported["sessions"], exported
    finally:
        memory.close()
    assert (home / "data").is_dir(), "数据应该写在 CUEMEM_HOME 下"

    settings = load_settings()
    try:  # 默认的导购演示要用源码仓库里的商品目录：包里没有时给出可读的报错
        ChatService.create(settings).close()
    except CueMemError as exc:
        assert "--scenario plain" in str(exc), exc
    else:
        raise AssertionError("包里不该有导购演示的商品目录")
    service = ChatService.create(settings, scenario="plain")
    try:
        # 只装了核心依赖时没有 Gradio：要界面也照常提供接口（界面在可选依赖 ui 里）
        app = create_app(service, upload_dir=home / "uploads")
        client = TestClient(app)
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json()["ok"], health.text
        if importlib.util.find_spec("gradio") is None:
            assert client.get("/").status_code == 404, "没装 Gradio 时根路径上不该有界面"
    finally:
        service.close()
    print(f"wheel 冒烟测试通过：{cuemem.__file__}")


if __name__ == "__main__":
    main()
