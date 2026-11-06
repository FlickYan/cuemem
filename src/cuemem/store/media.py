"""媒体库：原图按内容哈希存放（N6）。

同一张图无论在多少次对话里出现，磁盘上只存一份：路径由 SHA-256 决定
（data/media/ab/abcdef….jpg）。描述与向量都是派生数据，原图是"回答时再感知"的依据，
属于原始层，遗忘时随发言一起删除（见 MemorySystem.forget）；写入失败、图片记录没有提交时也删掉
（见 OnlineWriter._discard），否则没有任何记录指向它，遗忘与删除空间都找不到它。

发给模型的图片统一缩放到最长边不超过 max_side 像素、转成 JPEG：图片 token 大致与像素数成正比，
缩放是控制成本最直接的手段。

解压炸弹（第三轮体检）：几十 KB 的 PNG 可以有一亿多像素，解码时要几百 MB。像素数超过 Pillow 的
Image.MAX_IMAGE_PIXELS（默认约 8950 万）的图片不进媒体库（put 报 ValueError）；同时解码的大图
（超过 LARGE_PIXELS）最多 LARGE_DECODES 张，几个请求一起解码也撑不爆共用的服务。

图片格式（第四轮体检）：进媒体库、导入的图片只认 ACCEPTED_FORMATS，别的格式不交给 Pillow 的
解码器去试；认不出时 put 报 ValueError（/api/v1 是 400）。

进程被杀时的原图（第七轮体检）：put 存下原图、图片记录还没提交时进程被 kill -9、OOM 杀掉或断电，
_discard 来不及删，没有任何记录指向这个文件，遗忘、delete_all 与 reset 都找不到它；遗忘提交之后、
删原图之前被杀掉同样。给了 owner（主数据库的路径，MemorySystem.from_settings）时，这两种操作先在
.journal/ 下写一个标记（<哈希>.<随机串>.json：文件名、所属数据库、进程号、主机名、时间），操作
做完删掉它。recover 按标记收拾进程已经不在了的操作留下的原图：只删标记点名的、数据库里没有记录
引用的、本进程里也没有正在写入的文件。不按"没有记录引用"扫整个媒体库：媒体库与评测空间共用，
评测空间的图片记录在各自的数据库里（或只在内存里），主数据库查不到，扫一遍会删掉它们；评测
（ephemeral、on_disk）也不写标记。
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shutil
import socket
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from cuemem.envfile import atomic_write_text

# 收进来的图片（/api/v1、界面、记忆代理与 MCP 的写回、导入，都是不可信的字节）只试这几种格式的
# 解码器。Pillow 默认挨个试四十来种格式，冷门格式的文件头解析出过不少安全问题：Pillow 12 的 XBM
# 正则 512 字节就能占住 GIL 几分钟，整个服务（所有令牌、所有接口）跟着停住。手机拍的 MPO 经 JPEG
# 的解码器打开；与 portable._FORMAT_SUFFIX 相同。只用在入口：image_data_url 不限，存下的图照常读
ACCEPTED_FORMATS = ("JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF")

# 像素数超过这个数的图片，解码、转换与缩放要排队，同时最多 LARGE_DECODES 张（一张 1600 万像素的
# RGB 图解码约 50 MB，8900 万像素的约 270 MB）；小图不受限制
LARGE_PIXELS = 16_000_000
LARGE_DECODES = 2
_large_decodes = threading.BoundedSemaphore(LARGE_DECODES)

JOURNAL_DIR = ".journal"  # 进行中的写入与遗忘的标记（见模块说明）
# 标记写下这么久之后，即使写它的进程看起来还在（别的主机上的、进程号被复用的），recover 也收拾
RECOVER_AFTER_S = 3600.0
# 本进程里还在进行的操作的标记，所有 MediaStore 共用：recover 不碰它们
_live_markers: set[Path] = set()
_live_lock = threading.Lock()


@dataclass(frozen=True)
class StoredImage:
    sha256: str
    path: Path
    width: int
    height: int
    marker: Path | None = None  # 这次 put 新存了文件时的标记（见模块说明），release 删掉它


class MediaStore:
    def __init__(self, root: Path, *, owner: str | None = None) -> None:
        """owner：图片记录所在的数据库（路径），给了时写标记、recover 只收拾它的（见模块说明）。"""
        self.root = root
        self.owner = owner
        self._lock = threading.Lock()
        # 存进来、图片记录还没提交（或还没放弃）的写入，按内容哈希计数：这期间不删这个文件
        self._writing: Counter[str] = Counter()

    @property
    def journal(self) -> Path:
        return self.root / JOURNAL_DIR

    def put(self, source: Path) -> StoredImage:
        """把图片存进媒体库（已存在则跳过），返回内容哈希、存放路径与尺寸。调用方的写入结束
        （图片记录提交了或放弃了）时调用 release；在那之前 delete_unused 不删这个文件。

        新存一个文件时先写标记（给了 owner 时），release 删掉它；文件已经在了时不写：别的记录
        或别的写入拥有它。复制出错时删掉复制了一半的文件与标记。"""
        data = source.read_bytes()
        width, height = _checked_size(data)  # 先认图：不是图片、像素太多的文件不进媒体库
        sha = hashlib.sha256(data).hexdigest()
        suffix = source.suffix.lower() or ".jpg"
        target = self.root / sha[:2] / f"{sha}{suffix}"
        marker: Path | None = None
        with self._lock:
            self._writing[sha] += 1
            try:
                if not target.exists():
                    if self.owner is not None:
                        marker = self._mark("put", sha, target.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    partial = target.with_suffix(target.suffix + ".part")
                    try:
                        shutil.copyfile(source, partial)
                        partial.replace(target)
                    finally:
                        partial.unlink(missing_ok=True)
            except BaseException:
                self._done(sha)
                self._unmark(marker, delete=True)
                raise
        return StoredImage(sha256=sha, path=target, width=width, height=height, marker=marker)

    def release(self, image: StoredImage, *, in_use: Callable[[str], bool] | None = None) -> bool:
        """一次 put 的写入结束了（见 put）。in_use 给了时（写入失败、图片记录没有提交）顺带按
        delete_unused 的规则删掉这张原图，返回删了没有。之后删掉 put 的标记；删原图出错时标记
        留着，recover 再试。"""
        try:
            with self._lock:
                self._done(image.sha256)
                removed = in_use is not None and self._delete_unused(
                    image.path, image.sha256, in_use
                )
        except BaseException:
            self._unmark(image.marker, delete=False)
            raise
        self._unmark(image.marker, delete=True)
        return removed

    def mark_deleting(self, images: Iterable[tuple[str, Path]]) -> list[Path]:
        """遗忘提交之前，为要删的原图（内容哈希，路径）写下标记：提交之后、删原图之前进程被杀掉
        时，recover 按它删掉没有记录再引用的原图。没给 owner 时什么都不写。删完之后调用
        unmark_deleting。写到一半出错时删掉已经写的，再抛出。"""
        markers: list[Path] = []
        if self.owner is None:
            return markers
        try:
            for sha, path in images:
                markers.append(self._mark("forget", sha, path.name))
        except BaseException:
            self.unmark_deleting(markers, done=True)
            raise
        return markers

    def unmark_deleting(self, markers: Iterable[Path], *, done: bool) -> None:
        """遗忘的原图删完了，或者遗忘没有提交（done=True）：删掉 mark_deleting 的标记。
        done=False（提交了、有原图没删掉）时标记留着，交给 recover。"""
        for marker in markers:
            self._unmark(marker, delete=done)

    def recover(
        self, in_use: Callable[[str], bool], *, older_than_s: float = RECOVER_AFTER_S
    ) -> tuple[int, int]:
        """收拾进程已经不在了的写入与遗忘留下的原图（见模块说明）：标记点名的文件（连同复制了
        一半的 .part）没有图片记录引用（in_use：owner 那个数据库）、本进程里也没有正在写入时
        删掉，有记录引用时留着；然后删掉标记。本进程里正在写入同一张图时标记留到下次。只看
        owner 是自己的标记，从不列举、删除没有标记的文件。返回（删掉的原图，留着的原图）张数。"""
        if self.owner is None:
            return 0, 0
        deleted = kept = 0
        for marker, record in self._stale(older_than_s):
            with self._lock:
                target = _target(self.root, record)
                if target is not None:
                    sha = str(record["sha256"])
                    if self._writing.get(sha):
                        continue
                    if in_use(sha):
                        kept += 1
                    else:
                        deleted += int(self.delete(target))
                        target.with_suffix(target.suffix + ".part").unlink(missing_ok=True)
                marker.unlink(missing_ok=True)
        cutoff = time.time() - older_than_s
        for tmp in self.journal.glob(".*.tmp"):  # 写标记时被杀掉留下的临时文件
            try:
                if tmp.stat().st_mtime <= cutoff:
                    tmp.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
        return deleted, kept

    def pending(self, *, older_than_s: float = RECOVER_AFTER_S) -> int:
        """recover 现在会收拾的标记个数（cuemem doctor）。"""
        return len(self._stale(older_than_s)) if self.owner is not None else 0

    def _done(self, sha: str) -> None:
        self._writing[sha] -= 1
        if self._writing[sha] <= 0:
            del self._writing[sha]

    def _mark(self, kind: str, sha: str, name: str) -> Path:
        """写一个标记（写完落盘再换名：不会只有半个），记进本进程的 _live_markers。"""
        self.journal.mkdir(parents=True, exist_ok=True)
        marker = self.journal / f"{sha}.{uuid.uuid4().hex}.json"
        record = {
            "kind": kind,
            "sha256": sha,
            "name": name,
            "owner": self.owner,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "ts": time.time(),
        }
        with _live_lock:
            _live_markers.add(marker)
        try:
            atomic_write_text(marker, json.dumps(record), fsync=True)
        except BaseException:
            self._unmark(marker, delete=True)
            raise
        return marker

    @staticmethod
    def _unmark(marker: Path | None, *, delete: bool) -> None:
        """操作结束了：本进程不再护着这个标记；delete 时删掉它（否则留给 recover）。"""
        if marker is None:
            return
        with _live_lock:
            _live_markers.discard(marker)
        if delete:
            marker.unlink(missing_ok=True)

    def _stale(self, older_than_s: float) -> list[tuple[Path, dict[str, object]]]:
        """该收拾的标记与内容：owner 是自己的、本进程里没在进行的；同一台主机上写它的进程不在了
        （或者就是本进程：容器重启后 cuemem 还是 1 号进程），或者已经写下 older_than_s 秒。读不
        出的标记（不是这里写的）超过 older_than_s 秒也收拾，只删标记。"""
        if not self.journal.is_dir():
            return []
        now = time.time()
        host = socket.gethostname()
        found: list[tuple[Path, dict[str, object]]] = []
        for marker in sorted(self.journal.glob("*.json")):
            with _live_lock:
                if marker in _live_markers:
                    continue
            try:
                record = json.loads(marker.read_text("utf-8"))
                if not isinstance(record, dict):
                    raise ValueError(marker)
            except FileNotFoundError:
                continue
            except (OSError, ValueError):
                try:
                    if now - marker.stat().st_mtime >= older_than_s:
                        found.append((marker, {}))
                except OSError:
                    pass
                continue
            if record.get("owner") != self.owner:
                continue
            ts, pid = record.get("ts"), record.get("pid")
            old = isinstance(ts, int | float) and now - ts >= older_than_s
            gone = record.get("host") == host and isinstance(pid, int) and not _alive(pid)
            if old or gone:
                found.append((marker, record))
        return found

    def delete_unused(self, path: Path, sha256: str, in_use: Callable[[str], bool]) -> bool:
        """没有图片记录引用（in_use：数据库里还有没有这个哈希的记录）、本进程里也没有进行中的
        写入要用它时，删掉这张原图。同一张图可能是别的记录、别的空间的。"""
        with self._lock:
            return self._delete_unused(path, sha256, in_use)

    def _delete_unused(self, path: Path, sha256: str, in_use: Callable[[str], bool]) -> bool:
        if self._writing.get(sha256) or in_use(sha256):
            return False
        return self.delete(path)

    def delete(self, path: Path) -> bool:
        """删除一张原图（调用方确认没有别的记录再引用它，通常用 delete_unused）。"""
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        return True


def _target(root: Path, record: dict[str, object]) -> Path | None:
    """标记点名的原图（按现在的媒体库位置）；不像 put 存的（哈希、文件名不对）时 None。"""
    sha, name = record.get("sha256"), record.get("name")
    if not isinstance(sha, str) or not isinstance(name, str):
        return None
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        return None
    if not name.startswith(sha) or Path(name).name != name or "\\" in name:
        return None
    return root / sha[:2] / name


def _alive(pid: int) -> bool:
    """同一台主机上这个进程还在不在。本进程算不在：本进程里在进行的标记在 _live_markers 里，
    不在那里的是上一个同号的进程（容器重启后 cuemem 还是 1 号进程）留下的。不是 POSIX 时不查
    （Windows 的 os.kill 会结束进程），只按时间。"""
    if pid == os.getpid():
        return False
    if os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:  # PermissionError：别的用户的进程
        return True
    return True


def _checked_size(data: bytes) -> tuple[int, int]:
    """图片的尺寸（只读文件头，不解码）。认不出、不是 ACCEPTED_FORMATS 里的格式时 ValueError（以前
    是 PIL 的 UnidentifiedImageError，/api/v1 回 500，冷门格式照样收下）；像素数超过 Pillow 的
    Image.MAX_IMAGE_PIXELS 时 ValueError。上限在调用时读、不改它：它是整个进程的设置，SDK 用户的
    程序也在用；为 None（关掉了）时不查。Pillow 自己只在超过两倍时报 DecompressionBombError（不是
    ValueError：记忆代理会当成值得重试，挡住这个会话之后的写回），一到两倍之间只警告、照样解码。"""
    try:
        with Image.open(io.BytesIO(data), formats=ACCEPTED_FORMATS) as image:
            width, height = image.size
    except Image.DecompressionBombError as exc:
        raise ValueError(f"Image too large: {exc}") from exc
    except UnidentifiedImageError as exc:
        accepted = ", ".join(ACCEPTED_FORMATS)
        raise ValueError(f"Not a valid image (accepted formats: {accepted}).") from exc
    limit = Image.MAX_IMAGE_PIXELS
    if limit is not None and width * height > limit:
        raise ValueError(f"Image too large: {width}x{height} pixels (limit {limit}).")
    return width, height


def check_image(path: Path | str) -> tuple[int, int]:
    """图片文件的尺寸（只读文件头，不解码），像素数的规则与 put 相同（_checked_size）：超过上限时
    ValueError。对话（Agent.chat）在召回、解码与调用模型之前先查上传的图片。"""
    return _checked_size(Path(path).read_bytes())


@contextmanager
def decoding(image: Image.Image) -> Iterator[None]:
    """解码 image（已经打开、只读了文件头）的期间：像素数超过 LARGE_PIXELS 时占一个大图解码的
    名额，名额用完时等着；小图直接解码。只排队，解码的结果不变。"""
    if image.width * image.height > LARGE_PIXELS:
        with _large_decodes:
            yield
    else:
        yield


def image_data_url(path: Path | str, *, max_side: int = 768, quality: int = 85) -> str:
    """读图、缩放、编码成 data URL（OpenAI 兼容接口的图片输入格式）。大图的解码与缩放排队
    （decoding），输出的字节与不排队时相同；不用 draft()、thumbnail() 之类的捷径，它们会改变
    发给模型的字节（以及缓存的键）。

    缩放分两次（先宽后高），钉住 Pillow 12 一次 resize 时的顺序：Pillow 13 起高度缩得比宽度多
    两倍以上时先缩高度（13.0.0 发布说明的 "Resizing vertically first"），手机截图之类的高图
    像素会变、缓存的键跟着变。两次缩放在 Pillow 12 上与一次 resize 逐字节相同，在 13 上
    还原 12 的字节（tests/test_store.py 钉着各种高图的摘要）。"""
    with Image.open(path) as source, decoding(source):
        image = source.convert("RGB")
        scale = max_side / max(image.size)
        if scale < 1:
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            if size[0] != image.width:
                image = image.resize((size[0], image.height), Image.Resampling.LANCZOS)
            if size[1] != image.height:
                image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
