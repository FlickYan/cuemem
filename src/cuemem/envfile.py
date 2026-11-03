""".env 文件的读写：只替换指定的键，保留其余内容与注释；新内容只在仅本人可读写的文件里。"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values

_KEY_RE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")
_SAFE_VALUE_RE = re.compile(r"[A-Za-z0-9_\-./:+@%=,]*")


def read_env(path: Path) -> dict[str, str]:
    """读取 .env 中已设置（非空）的值。"""
    if not path.exists():
        return {}
    return {k: v for k, v in dotenv_values(path).items() if v}


def update_env(path: Path, updates: Mapping[str, str | None]) -> None:
    """按键更新 .env：值为 None 表示删除该键；不在 updates 中的行原样保留。"""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    pending = dict(updates)
    output: list[str] = []
    for line in lines:
        match = _KEY_RE.match(line)
        key = match.group(1) if match else None
        if key is not None and key in pending:
            value = pending.pop(key)
            if value is not None:
                output.append(f"{key}={_quote(value)}")
            continue
        output.append(line)
    output.extend(f"{key}={_quote(value)}" for key, value in pending.items() if value is not None)
    _replace_private(path, "\n".join(output) + "\n")


def _replace_private(path: Path, text: str) -> None:
    """把 text 写进同一目录下 0600 的临时文件，再换名替换 path（第四轮体检）。

    以前就地重写之后才 chmod 0600：cp .env.example .env 得到的是 0644，别的本地用户在这之前打开
    的文件描述符照样读到新写的 Key。换名之后 Key 从来不在别人能读的 inode 里，之前打开的描述符
    只看得到旧内容。path 是符号链接时替换它指向的文件，链接留着。"""
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    os.chmod(target, 0o600)


def mask(secret: str) -> str:
    """只显示首尾几位，用于确认填的是哪一个 Key。"""
    return f"{secret[:3]}…{secret[-4:]}" if len(secret) > 10 else "***"


def _quote(value: str) -> str:
    if _SAFE_VALUE_RE.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
