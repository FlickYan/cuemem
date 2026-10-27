"""记忆空间的导出与导入（N12，D-063）。

对标 Letta 的 Agent File（.af）、Cognee 的 COGX、MemOS 的 dump / load。

CueMem 的派生记忆（事实、实体、摘要、画像）是原始日志的函数（D-021），所以可移植的只需要
原始日志：会话、发言（含时间、说话人、遗忘留下的序号空缺）与图片原文件。导入就是把日志写回去、
再巩固一遍——换了抽取模型、换了机器也能得到一致的结构，而不是搬一份可能过期的派生结果。

文件格式（JSON，format = "cuemem.space"，version = 2）：

    {"format": "cuemem.space", "version": 2, "exported_at": "...", "space_id": "alice",
     "sessions": [{"session_id": "...", "started_at": "...",
                   "turns": [{"idx": 0, "role": "user", "text": "... <image> ...", "ts": "...",
                              "speaker": null, "images": [{"sha256": "...", "mime": "image/jpeg",
                                                           "data": "<base64>"}],
                              "metadata": {...}, "expires_at": "...", "edit_of": "..."}]}],
     "derived": {"facts": [...], "profile": "..."}}

version 2（N12 之后）：发言可以带 SDK 写入时给的附加信息（core.models.TurnMeta）——metadata、
expires_at（过期时间）与 edit_of（Memory.update 写下的更正要取代的事实）；没有时省略这几个键。
version 1 的文件照样能导入（没有附加信息）。edit_of 指向的事实 ID 由空间与抽取结果派生：导入
到同名的空间、抽取结果相同时对得上，更正照样取代原来那条；导入到别的空间或换了抽取模型时对不上，
那句 "Correction: …" 就按普通的话对账（单值的事实交给 LLM 判断是不是更正）。

derived 只是给别的系统读的快照（带版本链的事实与画像），导入时不用它。没有导出的：回复引用了
哪些记忆的溯源表（导入后事实 ID 会重新生成，对不上）；遗忘时沿引用扩散因此对导入的空间不起作用，
按内容找转述（D-051）照常工作。

导入的文件不可信（可能是别人发来的，HTTP 接口也收）：图片的 sha256 与 mime 不拿来拼文件名，
临时文件按重新算出的内容哈希命名、后缀按 PIL 认出的格式取；不是图片、哈希对不上时整个导入报错，
一条发言也不写。
"""

from __future__ import annotations

import base64
import hashlib
import io
import mimetypes
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any

from PIL import Image

from cuemem.core.models import SessionIn, Turn, TurnIn, TurnMeta
from cuemem.core.timeutil import normalize_expiry
from cuemem.memory import MemorySystem

FORMAT = "cuemem.space"
VERSION = 2  # 2：发言带附加信息（metadata、过期时间、更正目标）；1 照样能导入
# PIL 认出的格式 → 临时文件的后缀（媒体库按后缀存放）；没列出的格式用 .img（同 SDK 的 data URL）
_FORMAT_SUFFIX = {
    "JPEG": ".jpg",
    "MPO": ".jpg",  # 手机拍的 JPEG 常被认成 MPO
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tif",
}


def export_space(system: MemorySystem, space_id: str, *, derived: bool = True) -> dict[str, Any]:
    """导出一个记忆空间：原始日志与图片原文件（base64），可选附上派生记忆的快照。"""
    sessions = system.db.get_sessions(space_id)
    if not sessions:
        raise KeyError(f"No memory space {space_id}")
    media = {m.id: m for m in system.db.get_media(space_id)}
    meta = system.db.get_turn_meta(space_id)
    out_sessions = []
    for session_id, started_at in sorted(sessions.items(), key=lambda item: item[1]):
        turns = system.db.get_turns(space_id, session_id)
        out_sessions.append(
            {
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "turns": [_turn_out(t, media, meta.get(t.id)) for t in turns],
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

    文件不合格式时（缺键、类型不对）一律 ValueError("Malformed export: …")，什么都不写：以前是
    KeyError、TypeError、AttributeError，本地与 MemoryClient 抛出的不同，/api 回 404 或 500，
    /api/v1 回 400 或没有说明的 500。
    """
    turns_added = images_added = 0
    with tempfile.TemporaryDirectory(prefix="cuemem-import-") as scratch:
        try:
            target, chunks, meta = _parse_export(data, space_id, Path(scratch))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Malformed export: {type(exc).__name__}: {exc}") from exc
        # 整个文件在一个事务里提交：向量化中途出错时一条都不写，重试从头开始（每段各向量化一次，
        # 与以前逐段写入时的请求相同）
        if chunks:
            extra = meta if any(m is not None for m in meta.values()) else None
            with system.appending(target, chunks, meta=extra) as report:
                turns_added, images_added = report.turns_added, report.images_added
    out: dict[str, Any] = {
        "space_id": target,
        "sessions": len(data.get("sessions", [])),
        "turns_added": turns_added,
        "images_added": images_added,
    }
    if consolidate:
        out["consolidation"] = system.consolidate(target).as_dict()
    return out


def _parse_export(
    data: Mapping[str, Any], space_id: str | None, scratch: Path
) -> tuple[str, list[tuple[SessionIn, int]], dict[tuple[str, int], TurnMeta | None]]:
    """导入的第一步：检查格式、读完整个文件（图片写进 scratch），还什么都不写。返回（目标空间，
    按连续序号分好的段，（会话，序号）→ 附加信息）。文件不可信：缺键、类型不对时抛出 KeyError、
    TypeError 或 AttributeError，由 import_space 统一改成 ValueError。"""
    if data.get("format") != FORMAT:
        raise ValueError(f"Not a CueMem export (format must be {FORMAT!r})")
    if int(data.get("version", 0)) > VERSION:
        raise ValueError(f"Export version {data.get('version')} is newer than this CueMem")
    if not space_id and not data.get("space_id"):  # 以前是 KeyError，经 /api/v1 成了没有说明的 500
        raise ValueError("The export has no space_id: pass user_id or agent_id to import it.")
    target = space_id or str(data["space_id"])
    # 先读完、校验完整个文件再写：有一张坏图时一条发言也不写，不留下导入了一半的空间
    chunks: list[tuple[SessionIn, int]] = []
    meta: dict[tuple[str, int], TurnMeta | None] = {}  # （会话，序号）→ 附加信息
    for session in data.get("sessions", []):
        started = datetime.fromisoformat(session["started_at"])
        # 遗忘留下的序号空缺照原样保留（重新抽取时要标出来，D-059）：按连续的序号分段写入
        ordered = sorted(session["turns"], key=lambda t: int(t["idx"]))
        for _, run in groupby(enumerate(ordered), key=lambda p: int(p[1]["idx"]) - p[0]):
            chunk = [turn for _, turn in run]
            turns = [_turn_in(t, scratch) for t in chunk]
            piece = SessionIn(session_id=session["session_id"], started_at=started, turns=turns)
            chunks.append((piece, int(chunk[0]["idx"])))
            for t in chunk:  # 序号重复时与发言一样先写的算（INSERT OR IGNORE）
                meta.setdefault((piece.session_id, int(t["idx"])), _meta_in(t))
    return target, chunks, meta


def _turn_out(turn: Turn, media: Mapping[str, Any], meta: TurnMeta | None) -> dict[str, Any]:
    images = []
    for media_id in turn.media_ids:
        item = media.get(media_id)
        if item is None or not Path(item.path).is_file():
            continue  # 原图已不在（例如被遗忘删掉）：只留文本里的 <image> 标记
        mime = mimetypes.guess_type(item.path)[0] or "image/jpeg"
        payload = base64.b64encode(Path(item.path).read_bytes()).decode("ascii")
        images.append({"sha256": item.sha256, "mime": mime, "data": payload})
    out: dict[str, Any] = {
        "idx": turn.idx,
        "role": turn.role,
        "text": turn.text,
        "ts": turn.ts.isoformat(),
        "speaker": turn.speaker,
        "images": images,
    }
    if meta is not None:  # version 2：只写有的那几项
        out.update(meta.model_dump(exclude_defaults=True))
    return out


def _turn_in(turn: Mapping[str, Any], scratch: Path) -> TurnIn:
    paths = list(_image_files(turn.get("images") or [], scratch))
    return TurnIn(
        role=turn["role"],
        text=turn["text"],
        ts=datetime.fromisoformat(turn["ts"]) if turn.get("ts") else None,
        speaker=turn.get("speaker"),
        images=[str(p) for p in paths],
    )


def _meta_in(turn: Mapping[str, Any]) -> TurnMeta | None:
    """导入文件里一句发言的附加信息（version 2）；没有时是 None。文件不可信：类型不对、过期时间
    认不出来时整个导入报错（与坏图相同，在写入任何东西之前）。"""
    metadata, expires, edit_of = turn.get("metadata"), turn.get("expires_at"), turn.get("edit_of")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("Export contains turn metadata that is not a JSON object")
    if edit_of is not None and not isinstance(edit_of, str):
        raise ValueError("Export contains an edit_of that is not a memory id")
    if not metadata and expires is None and edit_of is None:
        return None
    return TurnMeta(metadata=metadata or {}, expires_at=normalize_expiry(expires), edit_of=edit_of)


def _image_files(images: Sequence[Mapping[str, Any]], scratch: Path) -> Iterator[Path]:
    """导出文件里的图片 → 临时目录里的文件。文件名不用文件里的任何字段（sha256 写成 "../x" 或
    绝对路径就能写到临时目录外面）；先认出是图片再落盘，声明的 mime 不可靠（未知后缀导出成
    image/jpeg），后缀按 PIL 认出的格式取。"""
    for image in images:
        try:
            data = base64.b64decode(image["data"], validate=True)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format or ""
                probe.verify()
        except Exception as exc:  # 坏图时 PIL 抛的异常五花八门：OSError、SyntaxError、struct.error…
            raise ValueError("Export contains an attachment that is not a valid image") from exc
        sha = hashlib.sha256(data).hexdigest()
        if image.get("sha256") and image["sha256"] != sha:
            raise ValueError("Image sha256 does not match its data")
        path = scratch / f"{sha}{_FORMAT_SUFFIX.get(fmt, '.img')}"
        path.write_bytes(data)
        yield path
