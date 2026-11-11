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

会话可以带 "infer": false（只在为 false 时写）：源空间里只写了原始日志、没有登记巩固任务的会话
（SDK 的 add(infer=False)）。导入时同样不登记任务，后台线程不去抽取它，显式的巩固（consolidate、
load(wait=True)、cuemem import --consolidate）照样抽取，与源空间相同。进行中的会话不带这个键。
version 仍是 2：导入时只拒绝更新的 version，旧版本的 CueMem 忽略这个键（照样抽取）。

derived 只是给别的系统读的快照（带版本链的事实与画像），导入时不用它。没有导出的：回复引用了
哪些记忆的溯源表（导入后事实 ID 会重新生成，对不上）；遗忘时沿引用扩散因此对导入的空间不起作用，
按内容找转述（D-051）照常工作。

导入的文件不可信（可能是别人发来的，HTTP 接口也收）：图片的 sha256 与 mime 不拿来拼文件名，
临时文件按重新算出的内容哈希命名、后缀按 PIL 认出的格式取；不是图片、哈希对不上时整个导入报错，
一条发言也不写。

Mem0 的导出（第八轮体检）：from_mem0 把 Mem0 get_all 的结果（列表、{"results": [...]}，或平台
接口分页的 {"count", "next", "results"}）换成上面的格式，交给同一个 import_space：一条记忆是一个
会话（session_id 由 Mem0 的 id 与更新时间派生），重复导入什么都不加，Mem0 里改过的记忆作为新会话
进来、由对账取代旧的事实。见 from_mem0。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby
from pathlib import Path
from typing import Any

from PIL import Image

from cuemem.core.models import MediaItem, SessionIn, Turn, TurnIn, TurnMeta
from cuemem.core.timeutil import naive_local, normalize_expiry
from cuemem.memory import MemorySystem, _open_mark
from cuemem.store.media import ACCEPTED_FORMATS
from cuemem.store.sqlite import MemoryDB

FORMAT = "cuemem.space"
MEM0_PREFIX = "mem0-"  # Mem0 导入的会话 ID 的前缀（from_mem0）
# Mem0 记忆的这几个顶层键是字符串或者没有（mem0/memory/main.py 的 promoted_payload_keys）
_MEM0_STR_KEYS = ("user_id", "agent_id", "run_id", "actor_id", "role", "attributed_to")
VERSION = 2  # 2：发言带附加信息（metadata、过期时间、更正目标）；1 照样能导入
# iter_export_json 攒到这么多个字符就写出一段（只有文字的空间不必每句发言一段）
EXPORT_CHUNK_BYTES = 1 << 16
# iter_export_json 每次编码原图的这么多个字节（3 的倍数，编出 1 MiB 的 base64）
EXPORT_IMAGE_SLICE = 3 << 18
# PIL 认出的格式 → 临时文件的后缀（媒体库按后缀存放）；没列出的格式用 .img（同 SDK 的 data URL）。
# 导入只认 store/media.py 的 ACCEPTED_FORMATS（MPO 经 JPEG 的解码器打开）
_FORMAT_SUFFIX = {
    "JPEG": ".jpg",
    "MPO": ".jpg",  # 手机拍的 JPEG 常被认成 MPO
    "PNG": ".png",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tif",
}


@dataclass(frozen=True)
class ExportPlan:
    """export_plan 的结果：导出一个空间要的、不带原图的那些（会话按开始时间排好，发言、附加
    信息与原图在写出时再读）。bounds 是每个会话这时最大的发言序号：之后写进来的发言不在这次导出
    里（见 iter_export_json）。"""

    db: MemoryDB
    space_id: str
    sessions: list[tuple[str, datetime]]
    bounds: dict[str, int]
    log_only: set[str]
    derived: dict[str, Any] | None


def export_plan(
    system: MemorySystem | MemoryDB, space_id: str, *, derived: bool = True
) -> ExportPlan:
    """导出的第一步：会话、每个会话这时最大的发言序号与派生记忆的快照，不读发言与原图（便宜）。
    没有这个空间时 KeyError（HTTP 接口据此在开始写响应之前回 404）。"""
    db = system if isinstance(system, MemoryDB) else system.db
    sessions = db.get_sessions(space_id)
    if not sessions:
        raise KeyError(f"No memory space {space_id}")
    snapshot: dict[str, Any] | None = None
    if derived:
        core = db.get_core_memory(space_id)
        snapshot = {
            "facts": [
                f.model_dump(mode="json", exclude={"space_id", "entity_ids"})
                for f in db.get_facts(space_id)
            ],
            "profile": core.text if core is not None else "",
        }
    return ExportPlan(
        db=db,
        space_id=space_id,
        sessions=sorted(sessions.items(), key=lambda item: item[1]),
        bounds=db.turn_bounds(space_id),
        log_only=_log_only(system, space_id),
        derived=snapshot,
    )


def export_space(
    system: MemorySystem | MemoryDB, space_id: str, *, derived: bool = True
) -> dict[str, Any]:
    """导出一个记忆空间：原始日志与图片原文件（base64），可选附上派生记忆的快照。

    只读数据库与原图，不用向量库：cuemem export 只打开数据库（向量库打不开、或者 serve 正开着
    它的时候也能导出）。整个导出在内存里；HTTP 接口改用 iter_export_json 边读边写，两者读发言的
    方式相同（计划那一刻的发言，见 iter_export_json）。"""
    plan = export_plan(system, space_id, derived=derived)
    out_sessions = []
    for session_id, started_at in plan.sessions:
        turns, meta, media = _session_rows(plan, session_id)
        out: dict[str, Any] = {
            "session_id": session_id,
            "started_at": started_at.isoformat(),
            "turns": [_turn_out(t, media, meta.get(t.id)) for t in turns],
        }
        if turns and session_id in plan.log_only:
            out["infer"] = False
        out_sessions.append(out)
    data = _header(plan, _now())
    data["sessions"] = out_sessions
    if plan.derived is not None:
        data["derived"] = plan.derived
    return data


def iter_export_json(plan: ExportPlan, exported_at: str | None = None) -> Iterator[bytes]:
    """export_space 的 JSON（UTF-8），一段一段地写出：与 Starlette 的 JSONResponse 渲染
    export_space 的结果逐字节相同（ensure_ascii=False、紧凑的分隔符、同样的键序），但同一时间只有
    一张原图在内存里，它的 base64 一段一段地编出、写出（_turn_pieces；文字每段最多攒到
    EXPORT_CHUNK_BYTES）。exported_at 不给时是现在。

    第八轮体检 P7：以前 /api/v1/export 与 /api/spaces/{space}/export 先把整个空间（每张原图的
    base64）放进一个 dict 再渲染成一个响应体，峰值约是响应体的三倍；同一张图写了 N 次时媒体库里
    只有一份，导出里却有 N 份，一个只能写自己空间的令牌就能让一次导出占几 GB。写出途中出错时响应
    已经开始，只能截断（调用方解析 JSON 时出错）；写出途中被遗忘删掉的原图照样跳过。

    第九轮体检 P6：导出的是 export_plan 那一刻有的发言。会话的发言在写到它时才读，客户端读得慢
    时这段时间就是整个下载：以前这期间写进来的发言也被导出，附加信息与图片记录却是开始时读的，
    于是它没有 metadata、过期时间与图片（导回去之后永远不过期）。现在计划记下每个会话最大的序号
    （bounds），之后写进来的发言不导出，之后新开的会话本来就不在计划里；每个会话的发言、附加信息
    与图片记录在同一次持锁里读（MemoryDB.export_rows），写出的发言带的总是它自己的那一份——整个
    会话被遗忘、又用同一个 run_id 重新写时序号从 0 重来，这样的发言也是完整的。写出途中的遗忘：
    还没写到的会话里被遗忘的发言不再写出，已经写出的收不回来；derived 与会话的 infer 是开始时的。"""
    stamp = exported_at if exported_at is not None else _now()
    head = _dumps({**_header(plan, stamp), "sessions": []})
    pending = [head[:-2]]  # 去掉 "]}"：后面接各个会话
    size = len(pending[0])
    for i, (session_id, started_at) in enumerate(plan.sessions):
        opening = {"session_id": session_id, "started_at": started_at.isoformat(), "turns": []}
        pending.append(("," if i else "") + _dumps(opening)[:-2])
        turns, meta, media = _session_rows(plan, session_id)
        for j, turn in enumerate(turns):
            if j:
                pending.append(",")
            for piece in _turn_pieces(turn, media, meta.get(turn.id)):
                if len(piece) >= EXPORT_CHUNK_BYTES:  # 一段 base64 或很长的文字：不再拼接
                    if pending:
                        yield "".join(pending).encode("utf-8")
                        pending, size = [], 0
                    yield piece.encode("utf-8")
                    continue
                pending.append(piece)
                size += len(piece)
                if size >= EXPORT_CHUNK_BYTES:
                    yield "".join(pending).encode("utf-8")
                    pending, size = [], 0
        tail = "," + _dumps({"infer": False})[1:-1] if turns and session_id in plan.log_only else ""
        pending.append("]" + tail + "}")
    pending.append("]")
    if plan.derived is not None:
        pending.append("," + _dumps({"derived": plan.derived})[1:-1])
    pending.append("}")
    yield "".join(pending).encode("utf-8")


def _session_rows(
    plan: ExportPlan, session_id: str
) -> tuple[list[Turn], dict[str, TurnMeta], dict[str, MediaItem]]:
    """一个会话在计划之内的发言，连同它们的附加信息与图片记录（一次持锁读出）。计划之后才有
    发言的会话什么都不给。"""
    return plan.db.export_rows(plan.space_id, session_id, plan.bounds.get(session_id, -1))


def _header(plan: ExportPlan, exported_at: str) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "version": VERSION,
        "exported_at": exported_at,
        "space_id": plan.space_id,
    }


def _now() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _dumps(value: Any) -> str:
    """与 Starlette 的 JSONResponse.render 相同的写法（只是还没编码成 UTF-8）。"""
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, indent=None, separators=(",", ":")
    )


def _log_only(system: MemorySystem | MemoryDB, space_id: str) -> set[str]:
    """只写了原始日志、没有登记过巩固任务的会话（SDK 的 add(infer=False)）。进行中的会话与崩溃时
    留下标记的会话除外：它们在导入的空间里不再进行中，照常登记任务、抽取。只给了数据库时（cuemem
    export）进行中的会话只看数据库里的标记：别的进程开着的 run 都留了标记。"""
    db = system if isinstance(system, MemoryDB) else system.db
    jobbed = {job["session_id"] for job in db.get_jobs(space_id)}
    prefix = _open_mark(space_id)
    held = {key.removeprefix(prefix) for key in db.meta_keys(prefix)}
    if isinstance(system, MemorySystem):
        held |= system._held(space_id)
    return set(db.get_sessions(space_id)) - jobbed - held


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
    if not isinstance(data, Mapping):  # 与 Memory.load 相同；以前说 AttributeError（第八轮复审）
        raise ValueError("Malformed export: a CueMem export is a JSON object")
    turns_added = images_added = 0
    with tempfile.TemporaryDirectory(prefix="cuemem-import-") as scratch:
        try:
            target, chunks, meta, log_only = _parse_export(data, space_id, Path(scratch))
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError(f"Malformed export: {type(exc).__name__}: {exc}") from exc
        # 整个文件在一个事务里提交：向量化中途出错时一条都不写，重试从头开始。几段合成一批一起
        # 向量化（每批最多 256 句发言，见 MemorySystem.appending），不是每段一个请求
        if chunks:
            extra = meta if any(m is not None for m in meta.values()) else None
            # 目标空间里已经有任务的会话照常登记（又多了发言时要重新抽取）
            if log_only:
                log_only -= {job["session_id"] for job in system.db.get_jobs(target)}
            with system.appending(target, chunks, meta=extra, log_only=log_only) as report:
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
) -> tuple[str, list[tuple[SessionIn, int]], dict[tuple[str, int], TurnMeta | None], set[str]]:
    """导入的第一步：检查格式、读完整个文件（图片写进 scratch），还什么都不写。返回（目标空间，
    按连续序号分好的段，（会话，序号）→ 附加信息，只写日志、不登记巩固任务的会话）。文件不可信：
    缺键、类型不对时抛出 KeyError、TypeError 或 AttributeError，由 import_space 统一改成
    ValueError。"""
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
    log_only: set[str] = set()
    for session in data.get("sessions", []):
        infer = session.get("infer", True)
        if not isinstance(infer, bool):  # 文件不可信："no" 之类不当成 False
            raise TypeError(f"session infer must be a boolean, not {type(infer).__name__}")
        if infer is False:
            log_only.add(session["session_id"])
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
    return target, chunks, meta, log_only


def _turn_out(turn: Turn, media: Mapping[str, Any], meta: TurnMeta | None) -> dict[str, Any]:
    images = []
    for media_id in turn.media_ids:
        item = media.get(media_id)
        if item is None or not Path(item.path).is_file():
            continue  # 原图已不在（例如被遗忘删掉）：只留文本里的 <image> 标记
        mime = mimetypes.guess_type(item.path)[0] or "image/jpeg"
        try:  # 读出的原图只是 b64encode 的参数：编码完就放掉（导出时内存里少一份）
            payload = base64.b64encode(Path(item.path).read_bytes()).decode("ascii")
        except FileNotFoundError:  # 刚被遗忘删掉（边读边写的导出途中）：同上
            continue
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


def _turn_pieces(turn: Turn, media: Mapping[str, Any], meta: TurnMeta | None) -> Iterator[str]:
    """_dumps(_turn_out(...)) 一段一段地给出（iter_export_json）：拼起来逐字符相同，但原图的
    base64 每段只有 EXPORT_IMAGE_SLICE 个原始字节编出的那么多（base64 不用 JSON 转义，直接写）。
    第八轮复审：以前整句发言先 json.dumps，一张图同时在内存里有原始字节、base64、转义后与拼接后
    的字符串、UTF-8 共约五份 base64 那么多；现在只有原始字节（约 0.75 份）加一段。"""
    head = _dumps(
        {
            "idx": turn.idx,
            "role": turn.role,
            "text": turn.text,
            "ts": turn.ts.isoformat(),
            "speaker": turn.speaker,
        }
    )
    yield head[:-1] + ',"images":['
    sep = ""
    for media_id in turn.media_ids:
        item = media.get(media_id)
        if item is None or not Path(item.path).is_file():
            continue  # 原图已不在（例如被遗忘删掉）：只留文本里的 <image> 标记
        mime = mimetypes.guess_type(item.path)[0] or "image/jpeg"
        try:
            raw = memoryview(Path(item.path).read_bytes())
        except FileNotFoundError:  # 刚被遗忘删掉（边读边写的导出途中）：同上
            continue
        yield f'{sep}{{"sha256":{_dumps(item.sha256)},"mime":{_dumps(mime)},"data":"'
        for start in range(0, len(raw), EXPORT_IMAGE_SLICE):  # 3 的倍数：拼起来同一个 base64
            yield base64.b64encode(raw[start : start + EXPORT_IMAGE_SLICE]).decode("ascii")
        del raw
        yield '"}'
        sep = ","
    extra = meta.model_dump(exclude_defaults=True) if meta is not None else {}
    yield "]" + ("," + _dumps(extra)[1:-1] if extra else "") + "}"


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
            with Image.open(io.BytesIO(data), formats=ACCEPTED_FORMATS) as probe:
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


def mem0_records(records: object) -> list[Mapping[str, Any]]:
    """Mem0 导出的记忆列表：get_all 的列表、{"results": [...]} 或平台接口分页的
    {"count", "next", "results"}。别的形状 ValueError("Malformed Mem0 export: …")。"""
    items = records.get("results") if isinstance(records, Mapping) else records
    if isinstance(items, str | bytes) or not isinstance(items, Sequence):
        raise ValueError('Malformed Mem0 export: expected a list of memories or {"results": [...]}')
    for n, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"Malformed Mem0 export: item {n} is not a JSON object")
    return list(items)


def mem0_groups(records: object) -> dict[tuple[str | None, str | None], list[Mapping[str, Any]]]:
    """按（user_id，agent_id）分组 Mem0 的记忆：cuemem import --from mem0 每组导入一次，各是一个
    事务。这里只检查形状与这两个键，别的由 from_mem0 检查。"""
    groups: dict[tuple[str | None, str | None], list[Mapping[str, Any]]] = {}
    for n, item in enumerate(mem0_records(records)):
        groups.setdefault(_mem0_pair(item, n), []).append(item)
    return groups


def from_mem0(
    records: object, *, user_id: str | None = None, agent_id: str | None = None
) -> dict[str, Any]:
    """Mem0 导出的记忆 → 一份 cuemem.space（version 2）文件，交给 import_space（第八轮体检）。

    - 目标空间：给了 user_id 或 agent_id 时是 space_for(user_id, agent_id)；否则所有记忆的
      （user_id，agent_id）必须相同，混着几个用户时 ValueError 并列出各组（一次导入是一个事务、
      只查一次作用域；cuemem import 按组逐个导入）；
    - 一条记忆一个会话：session_id 是 "mem0-" 加上 sha256(id、更新时间) 的前 16 位。同一条记忆
      再导入时会话已在、发言跳过，什么都不加；在 Mem0 里改过的记忆更新时间不同，作为新会话进来，
      对账取代旧的事实，旧的留作历史；
    - 一句发言：text 是记忆的文字；时间是更新时间（没有时用创建时间），Mem0 写的带时区的时间换算
      成本机时间；role 按 Mem0 的 role / attributed_to，是 assistant 时为 assistant，否则 user；
      speaker 是 actor_id；metadata 是 Mem0 的 metadata 加上 mem0_id、mem0_run_id 与
      mem0_updated_at；expires_at 是 expiration_date；
    - 文字为空的记忆跳过，数目记在返回值的 "skipped" 里（import_space 不看这个键）。

    先检查完所有记忆再返回：哪一条不合格式都 ValueError("Malformed Mem0 export: …")，什么都不写。
    Mem0 的历史版本（history）不导入；事实由 CueMem 的抽取重新得出，措辞与 Mem0 的不同。"""
    from cuemem.sdk import space_for

    items = mem0_records(records)
    pairs = {_mem0_pair(item, n) for n, item in enumerate(items)}
    sessions = [_mem0_session(item, n) for n, item in enumerate(items)]
    if user_id or agent_id:
        target = space_for(user_id, agent_id)
    elif len(pairs) > 1:
        listed = ", ".join(f"(user_id={u!r}, agent_id={a!r})" for u, a in sorted(pairs, key=str))
        raise ValueError(
            "The Mem0 export mixes several users or agents; import each with its user_id or "
            f"agent_id: {listed}"
        )
    elif not pairs or pairs == {(None, None)}:
        raise ValueError(
            "The Mem0 export names no user_id or agent_id: pass user_id or agent_id to import it."
        )
    else:
        target = space_for(*next(iter(pairs)))
    kept = [session for session in sessions if session is not None]
    return {
        "format": FORMAT,
        "version": VERSION,
        "space_id": target,
        "sessions": kept,
        "skipped": len(sessions) - len(kept),
    }


def _mem0_pair(item: Mapping[str, Any], n: int) -> tuple[str | None, str | None]:
    """一条 Mem0 记忆的（user_id，agent_id）；顶层的几个 ID 键不是字符串时 ValueError。"""
    for key in _MEM0_STR_KEYS:
        if item.get(key) is not None and not isinstance(item[key], str):
            raise ValueError(f"Malformed Mem0 export: item {n} has a {key} that is not a string")
    return item.get("user_id") or None, item.get("agent_id") or None


def _mem0_session(item: Mapping[str, Any], n: int) -> dict[str, Any] | None:
    """一条 Mem0 记忆 → 导入文件里的一个会话（见 from_mem0）；文字为空时 None。"""
    memory_id, text, metadata = item.get("id"), item.get("memory"), item.get("metadata")
    if not isinstance(memory_id, str) or not memory_id:
        raise ValueError(f"Malformed Mem0 export: item {n} has no id")
    if not isinstance(text, str):
        raise ValueError(f"Malformed Mem0 export: memory {memory_id!r} has no text")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError(f"Malformed Mem0 export: memory {memory_id!r} has a non-object metadata")
    try:  # 与 SDK 的 add 相同：metadata 要能写成 JSON
        extra = json.loads(json.dumps(dict(metadata or {}), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Malformed Mem0 export: memory {memory_id!r} has metadata that is not JSON: {exc}"
        ) from exc
    stamps: dict[str, datetime] = {}
    for key in ("created_at", "updated_at"):
        value = item.get(key)
        if value is None:
            continue
        try:
            if not isinstance(value, str):
                raise ValueError(f"{value!r} is not a string")
            # 换算成本机时间也在这里：带时区、在日期范围边上的（0001-01-01T00:00+05:00）
            # OverflowError，以前原样抛出（第八轮复审）
            stamps[key] = naive_local(datetime.fromisoformat(value))
        except (ValueError, OverflowError) as exc:
            raise ValueError(
                f"Malformed Mem0 export: memory {memory_id!r} has a bad {key}: {exc}"
            ) from exc
    if not stamps:
        raise ValueError(f"Malformed Mem0 export: memory {memory_id!r} has no created_at")
    try:
        expires = normalize_expiry(item.get("expiration_date"))
    except ValueError as exc:
        raise ValueError(f"Malformed Mem0 export: memory {memory_id!r}: {exc}") from exc
    if not text.strip():
        return None
    version = item.get("updated_at") or item["created_at"]
    ts = (stamps.get("updated_at") or stamps["created_at"]).isoformat()
    who = item.get("role") or item.get("attributed_to") or ""
    extra["mem0_id"] = memory_id
    if item.get("run_id"):
        extra["mem0_run_id"] = item["run_id"]
    if item.get("updated_at"):
        extra["mem0_updated_at"] = item["updated_at"]
    turn: dict[str, Any] = {
        "idx": 0,
        "role": "assistant" if who.lower() == "assistant" else "user",
        "text": text,
        "ts": ts,
        "speaker": item.get("actor_id") or None,
        "images": [],
        "metadata": extra,
    }
    if expires is not None:
        turn["expires_at"] = expires
    digest = hashlib.sha256(f"{memory_id}\0{version}".encode()).hexdigest()[:16]
    return {"session_id": f"{MEM0_PREFIX}{digest}", "started_at": ts, "turns": [turn]}
