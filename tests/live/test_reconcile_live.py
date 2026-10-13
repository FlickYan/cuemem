"""联网测试：对账能不能分清"范围不同""现实变了""说错了"（N11 修订的对账提示）。

N9 的错误分析里，"周五 6 点起床"取代了"工作日 6 点半起床"（两者可以同时成立）；英文示例里
"搬到新加坡"被标成了"更正"（应是现实变了）。这里用一段新编的小对话核对修订后的提示，
不用测试集里的原题。抽取与对账用 qwen3.8-flash，一次只花几分钱。

运行：uv run pytest -m live tests/live/test_reconcile_live.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cuemem.config import PathsConfig, Settings, load_settings
from cuemem.core.models import Conversation, SessionIn, TurnIn
from cuemem.memory import MemorySystem

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    loaded = load_settings()
    missing = [p.api_key_env for p in loaded.providers.values() if p.api_key() is None]
    if missing:
        pytest.skip(f"缺少 API Key：{missing}")
    return loaded


def _session(sid: str, when: datetime, text: str) -> SessionIn:
    return SessionIn(
        session_id=sid,
        started_at=when,
        turns=[TurnIn(role="user", text=text), TurnIn(role="assistant", text="Got it, thanks!")],
    )


CONVERSATION = Conversation(
    space_id="reconcile-live",
    sessions=[
        _session(
            "s1",
            datetime(2025, 3, 3, 20),
            "I live in Shanghai. On weekdays I wake up at 6:30. My sister Mia is 30.",
        ),
        _session(
            "s2",
            datetime(2025, 4, 10, 20),
            "On Fridays I get up at 6:00 now, because I go to a spin class before work.",
        ),
        _session(
            "s3",
            datetime(2025, 6, 2, 20),
            "Big news: I moved to Singapore last month for a new job.",
        ),
        _session(
            "s4", datetime(2025, 6, 20, 20), "Oops, I got that wrong before: Mia is 31, not 30."
        ),
    ],
)


def test_scope_change_and_correction(live_settings: Settings, tmp_path: Path) -> None:
    paths = (
        PathsConfig()
        .resolve_against(tmp_path)
        .model_copy(
            update={"ledger": live_settings.paths.ledger}  # 花费记进项目的账本
        )
    )
    system = MemorySystem.from_settings(live_settings.model_copy(update={"paths": paths}))
    try:
        system.ingest(CONVERSATION, consolidate=True)
        facts = system.facts("reconcile-live")
    finally:
        system.close()
    lines = "\n".join(
        f"{f.statement} | current={f.is_current} | reason={f.supersede_reason}" for f in facts
    )

    def find(*words: str) -> list[str]:
        return [
            line for line in lines.splitlines() if all(w.lower() in line.lower() for w in words)
        ]

    # 范围不同：周五的新习惯不取代工作日的习惯
    assert any("current=True" in line for line in find("6:30")), lines
    assert any("current=True" in line for line in find("6:00")), lines
    # 现实变了：上海 → 新加坡是 changed，不是 corrected
    assert find("Shanghai") and all("reason=changed" in line for line in find("Shanghai")), lines
    assert any("current=True" in line for line in find("Singapore")), lines
    # 说错了：30 岁 → 31 岁是 corrected
    old_age = [line for line in find("Mia", "30") if "31" not in line]
    assert old_age and all("reason=corrected" in line for line in old_age), lines
