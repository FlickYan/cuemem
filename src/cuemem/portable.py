"""记忆空间的导出与导入（N12，D-063）。

对标 Letta 的 Agent File（.af）、Cognee 的 COGX、MemOS 的 dump / load。

CueMem 的派生记忆（事实、实体、摘要、画像）是原始日志的函数（D-021），所以可移植的只需要
原始日志：会话、发言（含时间、说话人、遗忘留下的序号空缺）与图片原文件。导入就是把日志写回去、
再巩固一遍——换了抽取模型、换了机器也能得到一致的结构，而不是搬一份可能过期的派生结果。

文件格式（JSON，format = "cuemem.space"，version = 1）：

    {"format": "cuemem.space", "version": 1, "exported_at": "...", "space_id": "alice",
     "sessions": [{"session_id": "...", "started_at": "...",
                   "turns": [{"idx": 0, "role": "user", "text": "... <image> ...", "ts": "...",
                              "speaker": null, "images": [{"sha256": "...", "mime": "image/jpeg",
                                                           "data": "<base64>"}]}]}],
     "derived": {"facts": [...], "profile": "..."}}

derived 只是给别的系统读的快照（带版本链的事实与画像），导入时不用它。没有导出的：回复引用了
哪些记忆的溯源表（导入后事实 ID 会重新生成，对不上）；遗忘时沿引用扩散因此对导入的空间不起作用，
按内容找转述（D-051）照常工作。
"""

from __future__ import annotations

import base64
import mimetypes
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any

from cuemem.core.models import SessionIn, Turn, TurnIn
from cuemem.memory import MemorySystem

FORMAT = "cuemem.space"
VERSION = 1


def export_space(system: MemorySystem, space_id: str, *, derived: bool = True) -> dict[str, Any]:
    """导出一个记忆空间：原始日志与图片原文件（base64），可选附上派生记忆的快照。"""
    sessions = system.db.get_sessions(space_id)
    if not sessions:
        raise KeyError(f"No memory space {space_id}")
    media = {m.id: m for m in system.db.get_media(space_id)}
    out_sessions = []
    for session_id, started_at in sorted(sessions.items(), key=lambda item: item[1]):
        turns = system.db.get_turns(space_id, session_id)
        out_sessions.append(
            {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "turns": [_turn_out(t, media) for t in turns],
            }
        )
    data: dict[str, Any] = {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": datetime.now().replace(microsecond=0).isoformat(),
        "space_id": space_id,
        "sessions": out_sessions,
    }
    if derived:
        core = system.core_memory(space_id)
        data["derived"] = {
            "facts": [
                f.model_dump(mode="json", exclude={"space_id", "entity_ids"})
                for f in system.facts(space_id)
            ],
            "profile": core.text if core is not None else "",
        }
    return data


def import_space(
    system: MemorySystem,
    data: Mapping[str, Any],
    *,
    space_id: str | None = None,
    consolidate: bool = False,
) -> dict[str, Any]:
    """导入导出的文件：把原始日志写回去（序号与空缺照原样），派生记忆由巩固重新生成。

    space_id 不填用文件里的；目标空间里已有同名会话时，相同序号的发言跳过（导入是幂等的）。
    consolidate=True 时立即同步巩固，否则只登记任务（后台线程或下次 consolidate 时完成）。
    """
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a CueMem export (format must be {FORMAT!r})")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"Export version {data.get('version')} is newer than this CueMem")
    target = space_id or str(data["space_id"])
    turns_added = images_added = 0
    with tempfile.TemporaryDirectory(prefix="cuemem-import-") as scratch:
        for session in data.get("sessions", []):
            started = datetime.fromisoformat(session["started_at"])
            # 遗忘留下的序号空缺照原样保留（重新抽取时要标出来，D-059）：按连续的序号分段写入
            ordered = sorted(session["turns"], key=lambda t: int(t["idx"]))
            for _, run in groupby(enumerate(ordered), key=lambda p: int(p[1]["idx"]) - p[0]):
                chunk = [turn for _, turn in run]
                turns = [_turn_in(t, Path(scratch)) for t in chunk]
                report = system.append(
                    target,
                    SessionIn(session_id=session["session_id"], started_at=started, turns=turns),
                    int(chunk[0]["idx"]),
                )
                turns_added += report.turns_added
                images_added += report.images_added
    out: dict[str, Any] = {
        "space_id": target,
        "sessions": len(data.get("sessions", [])),
        "turns_added": turns_added,
        "images_added": images_added,
    }
    if consolidate:
        out["consolidation"] = system.consolidate(target).as_dict()
    return out


def _turn_out(turn: Turn, media: Mapping[str, Any]) -> dict[str, Any]:
    images = []
    for media_id in turn.media_ids:
        item = media.get(media_id)
        if item is None or not Path(item.path).is_file():
            continue  # 原图已不在（例如被遗忘删掉）：只留文本里的 <image> 标记
        mime = mimetypes.guess_type(item.path)[0] or "image/jpeg"
        payload = base64.b64encode(Path(item.path).read_bytes()).decode("ascii")
        images.append({"sha256": item.sha256, "mime": mime, "data": payload})
    return {
        "idx": turn.idx,
        "role": turn.role,
        "text": turn.text,
        "ts": turn.ts.isoformat(),
        "speaker": turn.speaker,
        "images": images,
    }


def _turn_in(turn: Mapping[str, Any], scratch: Path) -> TurnIn:
    paths = list(_image_files(turn.get("images") or [], scratch))
    return TurnIn(
        role=turn["role"],
        text=turn["text"],
        ts=datetime.fromisoformat(turn["ts"]) if turn.get("ts") else None,
        speaker=turn.get("speaker"),
        images=[str(p) for p in paths],
    )


def _image_files(images: Sequence[Mapping[str, Any]], scratch: Path) -> Iterator[Path]:
    for image in images:
        suffix = mimetypes.guess_extension(image.get("mime") or "") or ".jpg"
        path = scratch / f"{image.get('sha256') or len(list(scratch.iterdir()))}{suffix}"
        path.write_bytes(base64.b64decode(image["data"]))
        yield path
