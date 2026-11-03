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
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import threading
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

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


@dataclass(frozen=True)
class StoredImage:
    sha256: str
    path: Path
    width: int
    height: int


class MediaStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.Lock()
        # 存进来、图片记录还没提交（或还没放弃）的写入，按内容哈希计数：这期间不删这个文件
        self._writing: Counter[str] = Counter()

    def put(self, source: Path) -> StoredImage:
        """把图片存进媒体库（已存在则跳过），返回内容哈希、存放路径与尺寸。调用方的写入结束
        （图片记录提交了或放弃了）时调用 release；在那之前 delete_unused 不删这个文件。"""
        data = source.read_bytes()
        width, height = _checked_size(data)  # 先认图：不是图片、像素太多的文件不进媒体库
        sha = hashlib.sha256(data).hexdigest()
        suffix = source.suffix.lower() or ".jpg"
        target = self.root / sha[:2] / f"{sha}{suffix}"
        with self._lock:
            self._writing[sha] += 1
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".part")
                shutil.copyfile(source, partial)
                partial.replace(target)
        return StoredImage(sha256=sha, path=target, width=width, height=height)

    def release(self, image: StoredImage, *, in_use: Callable[[str], bool] | None = None) -> bool:
        """一次 put 的写入结束了（见 put）。in_use 给了时（写入失败、图片记录没有提交）顺带按
        delete_unused 的规则删掉这张原图，返回删了没有。"""
        with self._lock:
            self._writing[image.sha256] -= 1
            if self._writing[image.sha256] <= 0:
                del self._writing[image.sha256]
            return in_use is not None and self._delete_unused(image.path, image.sha256, in_use)

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
