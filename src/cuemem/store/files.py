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
import logging
import os
import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, TYPE_CHECKING

from cuemem.errors import CueMemError, DatabaseOpenError, StorageLockedError

if TYPE_CHECKING:
    from cuemem.config import Settings

logger = logging.getLogger(__name__)

# 设为 1 时 lock_database 不加锁（几个进程共用一个数据库，风险自负，见 lock_database）
SHARED_DB_ENV = "CUEMEM_ALLOW_SHARED_DB"


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


@contextlib.contextmanager
def opening(path: Path) -> Iterator[None]:
    """打开数据库、账本或缓存（MemorySystem.from_settings）：SQLite 打不开（目录只读、磁盘满了）
    或者建不了上级目录（挡在路上的是文件，OSError）时换成 DatabaseOpenError，一行说明哪个文件，
    格式与 cli._unopenable 相同。以前原样抛出，cuemem serve、proxy、chat、demo 与 mcp 打印调用栈，
    也看不出是哪个文件（第七轮体检）。只包打开：用的时候出的错照常抛出。

    文件损坏或者根本不是 SQLite 数据库（SQLITE_NOTADB、SQLITE_CORRUPT*，抛的是 OperationalError
    的父类 sqlite3.DatabaseError）也一样，换成从备份恢复的说明：以前原样抛出 "file is not a
    database"，不说是四个文件里的哪一个（第八轮体检）。"""
    try:
        yield
    except CueMemError:  # 已经说明了（DatabaseOpenError、StorageLockedError、配置不符）：不再包一层
        raise
    except (sqlite3.DatabaseError, OSError) as exc:
        if corrupt_database(exc):
            raise DatabaseOpenError(
                f"{path} 打不开（{exc}）：文件损坏或不是 SQLite 数据库，从备份恢复（见 README 的"
                "备份与恢复）；缓存与账本可以删掉重建"
            ) from exc
        raise DatabaseOpenError(f"{path} 打不开（{exc}）：检查目录权限与磁盘空间") from exc


def corrupt_database(exc: BaseException) -> bool:
    """SQLite 认不出这个文件（SQLITE_NOTADB）或者文件损坏了（SQLITE_CORRUPT*）。"""
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    name = getattr(exc, "sqlite_errorname", "") or ""
    return name.startswith(("SQLITE_NOTADB", "SQLITE_CORRUPT"))


def lock_database(db: Path) -> IO[str] | None:
    """给数据库加独占锁（MemorySystem.from_settings，第七轮体检）：数据库旁边的 <数据库>.lock，
    不等待，拿不到时 StorageLockedError，说明拿着锁的进程（pid 与命令，写在锁文件里）。

    一个数据库同一时间只能有一个进程写（D-068 起的前提）：接手进行中的会话、reindex 删孤儿向量、
    分配发言序号与重新读令牌文件都按这个前提来。以前只有 Qdrant 本地模式的锁挡着；用 Qdrant
    服务端（docker-compose 的默认）时第二个进程（cuemem mcp、reindex、SDK 的 Memory()）照样
    打开，接手第一个进程还在写的会话、把它刚写的向量当孤儿删掉。锁按数据库文件而不是数据目录：
    paths.db 与 paths.qdrant 可以分开配置。只读数据库的命令（export、spaces、inspect、doctor、keys）
    不经过 from_settings，不加锁。

    返回拿着锁的文件，关掉记忆系统时交给 unlock_database；环境变量 CUEMEM_ALLOW_SHARED_DB=1 时
    不加锁、记一条警告，返回 None。flock 在同一进程里的两个文件描述符之间也互斥：同一进程再打开
    一次也被拒绝，与 Qdrant 本地模式相同。NFS 上 flock 可能不起作用。"""
    if os.environ.get(SHARED_DB_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.warning(
            "%s=1：不给数据库 %s 加锁。几个进程同时写它时，一个进程会接手另一个进程进行中的会话、"
            "reindex 删掉另一个进程刚写的向量、同一会话的发言序号相撞",
            SHARED_DB_ENV,
            db,
        )
        return None
    import portalocker

    path = db.with_name(db.name + ".lock")
    with opening(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = os.fdopen(os.open(path, os.O_RDWR | os.O_CREAT, 0o600), "r+", encoding="utf-8")
    flags = portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING
    try:
        portalocker.lock(handle, flags)
    except portalocker.exceptions.LockException as exc:
        try:  # 拿着锁的进程写下的 pid 与命令；读不出（正好在写）时不说
            pid, _, command = handle.read().strip().partition("\t")
        except (OSError, ValueError):
            pid = command = ""
        finally:
            handle.close()
        holder = f"（pid {pid}，{command}）" if pid else ""
        raise StorageLockedError(
            f"数据库 {db} 正被另一个进程使用{holder}：先停掉它，或者通过 MemoryClient、HTTP"
            "（/api/v1、/mcp）连到那个进程；确实要几个进程共用时设置 "
            f"{SHARED_DB_ENV}=1（风险自负）"
        ) from exc
    try:
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\t{' '.join(sys.argv[:2])}\n")
        handle.flush()
    except OSError:  # 只是给别的进程看的说明，写不进去也照样拿着锁
        pass
    return handle


def unlock_database(handle: IO[str] | None) -> None:
    """放掉 lock_database 加的锁（锁文件留着：删掉它时另一个进程可能正要锁它）。"""
    if handle is None:
        return
    import portalocker

    with contextlib.suppress(OSError, ValueError):
        portalocker.unlock(handle)
    handle.close()


def disk_trouble(exc: sqlite3.OperationalError) -> bool:
    """SQLite 写不进去是磁盘的问题：满了（SQLITE_FULL）、只读（SQLITE_READONLY*）、读写出错
    （SQLITE_IOERR*）。"database is locked"（SQLITE_BUSY，别的连接拿着写锁）不算：那是并发，
    不是磁盘。缓存与账本的 best_effort 只对这几种不抛出（第七轮体检）。"""
    name = getattr(exc, "sqlite_errorname", "") or ""
    return name == "SQLITE_FULL" or name.startswith(("SQLITE_READONLY", "SQLITE_IOERR"))


def spare_write(
    conn: sqlite3.Connection,
    exc: sqlite3.OperationalError,
    *,
    best_effort: bool,
    failing: bool,
    what: str,
    path: Path | None,
) -> None:
    """缓存或账本的一次写入出错之后（调用方持有它的锁、在 except 里调用）：先回滚，否则失败的
    INSERT 或 commit 留着打开的事务，一直拿着 WAL 的写锁，之后的写入与删除（包括遗忘的清理）
    都等不到。best_effort 而且是磁盘的问题（disk_trouble）时记一条警告就返回（之前已经失败过、
    failing 时不再记），调用方接着走；否则原样抛出。"""
    with contextlib.suppress(sqlite3.Error):
        conn.rollback()
    if not (best_effort and disk_trouble(exc)):
        raise exc
    if not failing:
        logger.warning("%s写不进 %s，这次与之后的不记，直到写入恢复：%s", what, path, exc)
