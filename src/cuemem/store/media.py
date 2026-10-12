"""媒体库：原图按内容哈希存放（N6）。

同一张图无论在多少次对话里出现，磁盘上只存一份：路径由 SHA-256 决定
（data/media/ab/abcdef….jpg）。描述与向量都是派生数据，原图是"回答时再感知"的依据，
属于原始层，遗忘时随发言一起删除（见 MemorySystem.forget）。

发给模型的图片统一缩放到最长边不超过 max_side 像素、转成 JPEG：图片 token 大致与像素数成正比，
缩放是控制成本最直接的手段。
"""

from __future__ import annotations

import base64
import hashlib
import io
import shutil
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

    def put(self, source: Path) -> StoredImage:
        """把图片存进媒体库（已存在则跳过），返回内容哈希、存放路径与尺寸。"""
        data = source.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        suffix = source.suffix.lower() or ".jpg"
        target = self.root / sha[:2] / f"{sha}{suffix}"
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".part")
            shutil.copyfile(source, partial)
            partial.replace(target)
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
        return StoredImage(sha256=sha, path=target, width=width, height=height)

    def delete(self, path: Path) -> bool:
        """删除一张原图（调用方确认没有别的记录再引用它）。"""
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
