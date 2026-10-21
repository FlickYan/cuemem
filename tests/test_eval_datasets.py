from __future__ import annotations

import re
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest

from cuemem.evals.datasets import (
    DATASETS,
    RAW_SESSION_IDS_ENV,
    DatasetSpec,
    download,
    evidence_turns,
    load_locomo,
    load_longmemeval,
    protocol_samples,
    sample_evidence_turns,
    to_conversation,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_longmemeval_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RAW_SESSION_IDS_ENV, raising=False)
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    by_id = {s.sample_id: s for s in evalset.samples}
    assert set(by_id) == {"q-ssu", "q-tr", "q-cat_abs"}

    ssu = by_id["q-ssu"]
    assert ssu.question_type == "single-session-user"
    assert ssu.ability == "information_extraction"
    assert not ssu.is_abstention
    assert ssu.question_time == datetime(2023, 6, 1, 10, 0)
    # 会话 ID 按时间顺序改写（原始 ID 会泄漏证据会话，见 test_eval_leak.py），原 ID 留作追溯
    assert evalset.session_ids == "anon"
    assert ssu.evidence_session_ids == ["session_1"]
    sessions = evalset.sessions_of(ssu)
    assert [s.session_id for s in sessions] == ["session_1", "session_2", "session_3"]
    assert [s.source_id for s in sessions] == ["ssu-a", "ssu-b", "ssu-c"]
    assert sessions[0].started_at == datetime(2023, 5, 20, 9, 0)
    assert evidence_turns(sessions) == {("session_1", 0)}

    assert by_id["q-tr"].answer == "12"  # 整数答案转成字符串
    assert by_id["q-tr"].ability == "temporal_reasoning"
    assert by_id["q-cat_abs"].is_abstention

    conversation = to_conversation("space", sessions)
    assert [t.role for t in conversation.sessions[0].turns] == ["user", "assistant"]


def test_longmemeval_loader_filters_ids() -> None:
    evalset = load_longmemeval(FIXTURES / "longmemeval_mini.json", ids={"q-tr"})
    assert [s.sample_id for s in evalset.samples] == ["q-tr"]
    assert set(evalset.haystacks) == {"q-tr"}


def test_locomo_loader() -> None:
    evalset = load_locomo(FIXTURES / "locomo_mini.json")
    by_type = {s.question_type: s for s in evalset.samples}
    assert set(by_type) == {"single-hop", "temporal", "adversarial", "multi-hop"}
    assert {s.space_key for s in evalset.samples} == {"conv-test"}  # 同一段对话的题共享历史

    sessions = evalset.haystacks["conv-test"]
    first = sessions[0].turns
    assert [t.speaker for t in first] == ["Alice", "Bob", "Alice"]
    assert {t.role for t in first} == {"user"}
    assert first[2].text.endswith("[shares a photo: a photo of a golden retriever puppy]")
    assert sessions[1].started_at == datetime(2023, 5, 20, 10, 0)

    single = by_type["single-hop"]
    assert single.question_time == datetime(2023, 5, 20, 10, 0)  # 取最后一个会话的时间
    assert sample_evidence_turns(single, sessions) == {("session_1", 2)}
    assert single.evidence_session_ids == ["session_1"]

    adversarial = by_type["adversarial"]
    assert adversarial.is_abstention
    assert "It was hard" in adversarial.answer

    multi = by_type["multi-hop"]
    assert multi.answer == "2023"
    # "D1:1; D2:1" 这种合并写法会被拆开
    assert sample_evidence_turns(multi, sessions) == {("session_1", 0), ("session_2", 0)}


def test_protocol_samples() -> None:
    """评测协议选题（eval run、eval probe 与两个混合流脚本共用）：LoCoMo 默认去掉对抗题，点名题型
    时只留这些题型（对抗题也可以点名）；其余数据集默认全留。顺序与数据集里相同。"""
    locomo = load_locomo(FIXTURES / "locomo_mini.json")
    ids = [s.sample_id.removeprefix("conv-test-") for s in protocol_samples(locomo)]
    assert ids == ["q0", "q1", "q3"]
    picked = protocol_samples(locomo, {"adversarial", "single-hop"})
    assert [s.sample_id.removeprefix("conv-test-") for s in picked] == ["q0", "q2"]
    assert protocol_samples(locomo, set()) == protocol_samples(locomo)
    assert len(locomo.samples) == 4  # 不改动传入的题目集合

    longmemeval = load_longmemeval(FIXTURES / "longmemeval_mini.json")
    assert protocol_samples(longmemeval) == longmemeval.samples
    assert [s.sample_id for s in protocol_samples(longmemeval, ["temporal-reasoning"])] == ["q-tr"]


def test_mixed_stream_scripts_use_the_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """两个混合流脚本（N7 开发集、N9 测试集）选记忆题也经 protocol_samples：LoCoMo 不含对抗题。
    去掉对抗题的规则只写在 datasets 里，命令行与脚本里不再各抄一份。"""
    import importlib
    import shutil

    root = Path(__file__).resolve().parents[1]
    shutil.copy(FIXTURES / "locomo_mini.json", tmp_path / "locomo10.json")
    monkeypatch.syspath_prepend(str(root / "scripts"))
    n7 = importlib.import_module("n7_mixed")
    n9 = importlib.import_module("n9_mixed")
    monkeypatch.setattr(n7, "DEV_SETS", (("locomo", None, 2),))  # 像 LoCoMo 的前 385 题
    assert [s.sample_id for s in n7.dev_samples(tmp_path)] == ["conv-test-q0", "conv-test-q1"]
    monkeypatch.setattr(n9, "TEST_SETS", (("locomo", "test7"),))
    split = ["conv-test-q1", "conv-test-q2", "conv-test-q3"]
    monkeypatch.setattr(n9, "load_split", lambda dataset, name: split)
    assert [s.sample_id for s in n9.test_samples(tmp_path)] == ["conv-test-q1", "conv-test-q3"]
    for path in ("src/cuemem/cli.py", "scripts/n7_mixed.py", "scripts/n9_mixed.py"):
        assert '"adversarial"' not in (root / path).read_text(encoding="utf-8"), path


def test_dataset_downloads_are_pinned() -> None:
    """每个下载地址都固定在上游的一个提交上（不是 main），每个文件都有要核对的 sha256。"""
    files = [(s.url, s.sha256) for s in DATASETS.values()]
    files += [(url, sha) for s in DATASETS.values() for _, url, sha in s.extras]
    for url, sha in files:
        assert "/main/" not in url and re.search(r"/[0-9a-f]{40}/", url), url
        assert re.fullmatch(r"[0-9a-f]{64}", sha), url


class _Served:
    """httpx2.stream 的替身：按 URL 返回预先放好的内容（分两块），可以中途断开。"""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.broken: set[str] = set()

    def stream(self, method: str, url: str, **kwargs: Any) -> Any:
        served = self

        class Response:
            def __enter__(self) -> Response:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def raise_for_status(self) -> None:
                return None

            def iter_bytes(self) -> Iterator[bytes]:
                body = served.files[url]
                yield body[:3]
                if url in served.broken:
                    raise httpx2.ReadError("connection reset")
                yield body[3:]

        return Response()


def test_download_verifies_the_pinned_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """下载后核对 sha256：一致才保存、压缩包才解压；上游内容变了（或下载被改写、截断）时报
    DatasetChecksumError，什么都不留下（没有半个文件，压缩包不解压）。以前不核对，照样保存。"""
    import hashlib
    import io
    import tarfile

    from cuemem.errors import DatasetChecksumError
    from cuemem.evals import datasets

    def sha(body: bytes) -> str:
        return hashlib.sha256(body).hexdigest()

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("toy_images/a.txt")
        info.size = 5
        archive.addfile(info, io.BytesIO(b"hello"))
    tar, index, data = buffer.getvalue(), b'{"ids": ["q1"]}', b'[{"question": "q1"}]'
    served = _Served()
    served.files = {"u/tar": tar, "u/index": index, "u/data": data}
    monkeypatch.setattr(datasets.httpx2, "stream", served.stream)
    extras = (
        ("toy/toy_images.tar.gz", "u/tar", sha(tar)),
        ("toy/index.json", "u/index", sha(index)),
    )
    spec = DatasetSpec("toy", "toy/data.json", "u/data", sha(data), "测试用", extras)
    monkeypatch.setitem(DATASETS, "toy", spec)

    good = tmp_path / "good"
    assert download("toy", good) == good / "toy" / "data.json"
    assert (good / "toy" / "data.json").read_bytes() == data
    assert (good / "toy" / "toy_images" / "a.txt").read_bytes() == b"hello"
    assert sorted(p.name for p in (good / "toy").iterdir()) == [
        "data.json",
        "index.json",
        "toy_images",
    ]

    served.files["u/data"] = data.replace(b"q1", b"q2")  # 上游改了题目
    changed = tmp_path / "changed"
    with pytest.raises(DatasetChecksumError, match=f"sha256 应为 {sha(data)}"):
        download("toy", changed)
    assert sorted(p.name for p in (changed / "toy").iterdir()) == ["index.json", "toy_images"]

    served.files["u/tar"] = tar[:-8]  # 压缩包被截断：不解压
    tampered = tmp_path / "tampered"
    with pytest.raises(DatasetChecksumError, match=re.escape("toy_images.tar.gz")):
        download("toy", tampered)
    assert list((tampered / "toy").iterdir()) == []

    served.files["u/tar"] = tar
    served.broken.add("u/index")  # 下载到一半断开：不留下 .part
    broken = tmp_path / "broken"
    with pytest.raises(httpx2.ReadError):
        download("toy", broken)
    assert [p.name for p in (broken / "toy").iterdir()] == ["toy_images"]
