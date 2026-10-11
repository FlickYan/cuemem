"""分时价格：高峰判断、批量任务在高峰时段暂停、非高峰的实际花费按折扣计。"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from fakes import FakeAPI

from cuemem.config import PeakHours, Settings
from cuemem.llm import Ledger, LLMClient, ResponseCache
from cuemem.llm.peak import in_peak, next_peak_start, peak_end

SH = ZoneInfo("Asia/Shanghai")
CFG = PeakHours()  # 工作日 09:00–12:00、14:00–18:00


def at(day: int, hour: int, minute: int = 0) -> datetime:
    """2026-10-12 是周一；day=0 表示周一。"""
    return datetime(2026, 10, 12 + day, hour, minute, tzinfo=SH)


def test_peak_windows() -> None:
    assert in_peak(CFG, at(0, 9, 0)) and in_peak(CFG, at(0, 11, 59))
    assert not in_peak(CFG, at(0, 12, 0))  # 结束时刻不算高峰
    assert not in_peak(CFG, at(0, 13, 30)) and in_peak(CFG, at(0, 17, 30))
    assert not in_peak(CFG, at(5, 10, 0))  # 周六
    assert peak_end(CFG, at(0, 10, 15)) == at(0, 12, 0)
    assert next_peak_start(CFG, at(0, 12, 30)) == at(0, 14, 0)
    assert next_peak_start(CFG, at(4, 19, 0)) == at(7, 9, 0)  # 周五晚上 → 下周一
    # 不同时区的时间先换算到北京时间：UTC 02:00 = 北京 10:00
    assert in_peak(CFG, datetime(2026, 10, 12, 2, 0, tzinfo=ZoneInfo("UTC")))


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start
        self.slept: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += timedelta(seconds=seconds)


def client(settings: Settings, fake_api: FakeAPI, clock: FakeClock, **kwargs: object) -> LLMClient:
    return LLMClient(
        settings,
        cache=ResponseCache(None),
        ledger=Ledger(None),
        http_client=fake_api.client(),
        clock=clock,
        sleep=clock.sleep,
        **kwargs,  # type: ignore[arg-type]
    )


def test_batch_calls_wait_for_off_peak(settings: Settings, fake_api: FakeAPI) -> None:
    clock = FakeClock(at(0, 10, 0))
    notices: list[tuple[str, datetime]] = []
    llm = client(
        settings,
        fake_api,
        clock,
        peak_policy="wait",
        on_peak_wait=lambda p, until: notices.append((p, until)),
    )
    messages = [{"role": "user", "content": "hi"}]
    llm.chat("reader", messages)  # deepseek-flash：暂停到 12:00 再调用
    assert clock.now == at(0, 12, 0) and sum(clock.slept) == 2 * 3600
    assert notices == [("deepseek", at(0, 12, 0))]
    assert len(fake_api.chat_requests) == 1

    clock.now = at(0, 15, 0)
    llm.chat("reader", messages)  # 命中缓存：不等待、不发请求
    assert clock.now == at(0, 15, 0) and len(fake_api.chat_requests) == 1
    llm.chat("judge", messages)  # 通义没有分时价格：不等待
    assert clock.now == at(0, 15, 0) and len(fake_api.chat_requests) == 2


@pytest.mark.parametrize("policy", ["allow", "warn"])
def test_interactive_policies_do_not_wait(
    settings: Settings, fake_api: FakeAPI, policy: str
) -> None:
    clock = FakeClock(at(0, 10, 0))
    llm = client(settings, fake_api, clock, peak_policy=policy)
    llm.chat("reader", [{"role": "user", "content": "hi"}])
    assert clock.slept == [] and len(fake_api.chat_requests) == 1


def test_off_peak_spend_is_discounted(settings: Settings, fake_api: FakeAPI) -> None:
    clock = FakeClock(at(0, 20, 0))  # 晚上：非高峰
    llm = client(settings, fake_api, clock)
    llm.chat("reader", [{"role": "user", "content": "hi"}])
    assert llm.ledger is not None
    [row] = llm.ledger.summary()
    assert row["spent"] == pytest.approx(row["nominal_cost"] * 0.5)
