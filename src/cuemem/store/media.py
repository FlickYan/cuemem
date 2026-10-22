"""媒体库：原图按内容哈希存放（N6）。

同一张图无论在多少次对话里出现，磁盘上只存一份：路径由 SHA-256 决定
（data/media/ab/abcdef….jpg）。描述与向量都是派生数据，原图是"回答时再感知"的依据，
属于原始层，遗忘时随发言一起删除（见 MemorySystem.forget）；写入失败、图片记录没有提交时也删掉
（见 OnlineWriter._discard），否则没有任何记录指向它，遗忘与删除空间都找不到它。

发给模型的图片统一缩放到最长边不超过 max_side 像素、转成 JPEG：图片 token 大致与像素数成正比，
缩放是控制成本最直接的手段。
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


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
        with Image.open(io.BytesIO(data)) as image:  # 先认图：不是图片的文件不进媒体库
            width, height = image.size
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


def image_data_url(path: Path | str, *, max_side: int = 768, quality: int = 85) -> str:
    """读图、缩放、编码成 data URL（OpenAI 兼容接口的图片输入格式）。"""
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = max_side / max(image.size)
    if scale < 1:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
