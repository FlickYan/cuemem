""".env 文件的读写：只替换指定的键，保留其余内容与注释；写入后把权限设为仅本人可读写。"""

from __future__ import annotations

import os
import re
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(output) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def mask(secret: str) -> str:
    """只显示首尾几位，用于确认填的是哪一个 Key。"""
    return f"{secret[:3]}…{secret[-4:]}" if len(secret) > 10 else "***"


def _quote(value: str) -> str:
    if _SAFE_VALUE_RE.fullmatch(value):
        return value
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
