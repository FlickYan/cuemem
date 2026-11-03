"""数据目录的权限（第四轮体检）：记忆系统新建的目录只给本人（0700）。

数据库、LLM 缓存（完整的抽取提示词）、trace（用户的提问）与原图都是用户的个人信息。以前目录按
常见的 umask 022 建成 0755、文件 0644，同一台机器上别的账户（或被攻破的服务）都能读；源码仓库
的数据目录默认就是仓库根目录（例如 VPS 上的 /opt/cuemem）。目录是 0700 时，里面的文件即使是
0644 也没人进得去，所以 SDK 不必改进程的 umask（那是用户自己的应用）；命令行另外把 umask 收紧到
077（cli._private_umask），它建的文件是 0600。

已经存在的目录一律不改权限：运营者可能有意给备份用户留了读权限，cuemem doctor 只提醒。
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuemem.config import Settings


def private_dirs(*dirs: Path) -> None:
    """建好这些目录与缺的上级目录，新建的都是 0700；已经存在的不动（不 chmod）。

    从第一个已经存在的上级目录往下逐级 os.mkdir(…, 0o700)：Path.mkdir(parents=True) 建的上级
    目录用默认权限（0777 减去 umask）。建不了时（没有权限、挡在路上的是文件）不报错，留给各组件
    自己的 mkdir 照旧在原来的地方报错：trace 目录写不进去只记一条警告（TraceSink.write），不该
    让记忆系统打不开。"""
    for directory in dict.fromkeys(dirs):
        missing: list[Path] = []
        current = directory
        try:
            while not current.exists() and current != current.parent:
                missing.append(current)
                current = current.parent
            for path in reversed(missing):
                with contextlib.suppress(FileExistsError):  # 别的进程同时建好了
                    os.mkdir(path, 0o700)
        except OSError:
            continue


def storage_dirs(settings: Settings) -> list[Path]:
    """记忆系统写的目录：数据库、账本、两个缓存、trace、原图，向量库在本地时还有它的目录。"""
    paths = settings.paths
    dirs = [
        paths.db.parent,
        paths.ledger.parent,
        paths.llm_cache.parent,
        paths.embed_cache.parent,
        paths.traces,
        paths.media,
    ]
    if settings.vector_store.url is None:
        dirs.append(paths.qdrant)
    return list(dict.fromkeys(dirs))
