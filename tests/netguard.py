"""测试不访问网络的守卫（tests/conftest.py 在整个测试会话里装上）。

只放行本机：别的主机名的 DNS 查询、连到别的地址都拒绝（像连不上一样抛出 OSError），并记进
ATTEMPTS；conftest 让发生了这些尝试的测试失败（只抛出异常不够：Gradio 的后台线程把异常吞掉了）。

环境变量里配了 HTTP(S)_PROXY 时，HTTP 客户端（httpx2 默认 trust_env）连的是本机的代理、由代理
去连外网：只看目标地址的话，这样的请求像是连本机。所以连到代理的端口也算访问网络、照样拒绝：
装上守卫时环境变量里的代理（conftest 随后删掉它们，删之前建好的客户端还记着它），与之后又设了的
（测试或代码自己设的）。

管不到的：测试起的子进程（它们继承删掉了代理与 Key 的环境变量，但没有这个守卫）。
"""

from __future__ import annotations

import errno
import ipaddress
import os
import socket
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import pytest

PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


@dataclass(frozen=True)
class Attempt:
    kind: str  # dns：查别的主机名；connect：连别的地址；proxy：连代理
    host: str
    port: Any
    test: str  # 当时在跑的测试（PYTEST_CURRENT_TEST）


ATTEMPTS: list[Attempt] = []
_reported = 0


def unreported() -> list[Attempt]:
    """上次取过之后新记下的尝试（取过的不再给）。"""
    global _reported
    new = ATTEMPTS[_reported:]
    _reported = len(ATTEMPTS)
    return new


def _address(host: Any) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    name = host.decode() if isinstance(host, bytes) else str(host)
    try:
        address = ipaddress.ip_address(name.split("%")[0])
    except ValueError:
        return None
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped  # ::ffff:127.0.0.1
    return address


def local(host: Any) -> bool:
    """本机：localhost、回环地址与"任意地址"（绑定端口时用）。"""
    if host is None or host in ("", b"", "localhost", b"localhost"):
        return True
    address = _address(host)
    return address is not None and (address.is_loopback or address.is_unspecified)


def proxy_ports() -> set[int]:
    """环境变量里现在配在本机的代理的端口。"""
    ports = set()
    for name in PROXY_ENV:
        value = os.environ.get(name)
        if not value:
            continue
        parts = urlsplit(value if "://" in value else f"http://{value}")
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
        except ValueError:
            continue
        if local(parts.hostname):
            ports.add(port)
    return ports


def _record(kind: str, host: Any, port: Any) -> None:
    ATTEMPTS.append(Attempt(kind, str(host), port, os.environ.get("PYTEST_CURRENT_TEST", "?")))


def install(patch: pytest.MonkeyPatch) -> None:
    """换上拒绝外网的 getaddrinfo 与 connect（patch 撤销时还原）。要在删掉代理的环境变量之前
    调用：记下那时的代理。"""
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    proxies_at_start = proxy_ports()

    def getaddrinfo(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        # 写成 IP 的地址不查 DNS（连过去时再拦）；别的主机名一律不查
        if not local(host) and _address(host) is None:
            _record("dns", host, port)
            raise socket.gaierror(socket.EAI_NONAME, f"netguard: no network in tests ({host})")
        return real_getaddrinfo(host, port, *args, **kwargs)

    def check(sock: socket.socket, address: Any) -> None:
        if sock.family not in (socket.AF_INET, socket.AF_INET6):
            return  # Unix 套接字
        host, port = address[0], address[1]
        if not local(host):
            _record("connect", host, port)
        elif port in proxies_at_start or port in proxy_ports():
            _record("proxy", host, port)
        else:
            return
        raise ConnectionRefusedError(
            errno.ECONNREFUSED, f"netguard: no network in tests ({host}:{port})"
        )

    def guarded(real: Callable[..., Any]) -> Callable[..., Any]:
        def connect(sock: socket.socket, address: Any) -> Any:
            check(sock, address)
            return real(sock, address)

        return connect

    patch.setattr(socket, "getaddrinfo", getaddrinfo)
    patch.setattr(socket.socket, "connect", guarded(real_connect))
    patch.setattr(socket.socket, "connect_ex", guarded(real_connect_ex))
