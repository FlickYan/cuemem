"""测试的网络守卫（tests/netguard.py，conftest 装上）：外网的查询与连接被拒绝并记下；环境里配了
本机的代理时，经它去外网的请求也被拒绝，代理什么都收不到。本机照常。"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import httpx2
import netguard
import pytest


def test_requests_out_are_refused() -> None:
    with pytest.raises(httpx2.ConnectError, match="netguard"):
        httpx2.Client(timeout=3, trust_env=False).get("https://api.deepseek.invalid/v1/models")
    assert [(a.kind, a.host) for a in netguard.unreported()] == [("dns", "api.deepseek.invalid")]
    with pytest.raises(ConnectionRefusedError, match="netguard"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)  # 写死的地址也不行
    assert [(a.kind, a.host) for a in netguard.unreported()] == [("connect", "192.0.2.1")]


@pytest.fixture
def listener() -> Iterator[socket.socket]:
    """本机的一个端口，当作代理（只看它有没有收到连接）。"""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    sock.setblocking(False)
    yield sock
    sock.close()


def _proxy_at(listener: socket.socket, monkeypatch: pytest.MonkeyPatch) -> int:
    port: int = listener.getsockname()[1]
    monkeypatch.setenv("HTTPS_PROXY", f"http://127.0.0.1:{port}")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    return port


def test_a_loopback_proxy_does_not_carry_requests_out(
    listener: socket.socket, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTPS_PROXY 指向本机的代理：只放行本机的守卫看不出这是去外网（httpx2 先连本机的代理，
    代理收到 "CONNECT upstream.invalid:443" 再去连）。D-069 记下的那次真的发到上游的请求，在
    HTTPS_PROXY 指向本机代理的环境里就是这样出去的。"""
    port = _proxy_at(listener, monkeypatch)
    with pytest.raises(httpx2.ConnectError, match="netguard"):
        httpx2.Client(timeout=3).get("https://upstream.invalid/")
    assert [(a.kind, a.port) for a in netguard.unreported()] == [("proxy", port)]
    with pytest.raises(BlockingIOError):
        listener.accept()  # 代理一个连接都没收到


def test_the_proxy_stays_refused_after_it_is_unset(
    listener: socket.socket, monkeypatch: pytest.MonkeyPatch
) -> None:
    """装上守卫之后才删掉代理的环境变量（conftest 就是这样做的）：删之前建好的客户端还记着代理，
    经它的请求照样拒绝。"""
    port = _proxy_at(listener, monkeypatch)
    client = httpx2.Client(timeout=3)
    with pytest.MonkeyPatch.context() as patch:
        netguard.install(patch)
        monkeypatch.delenv("HTTPS_PROXY")
        with pytest.raises(httpx2.ConnectError, match="netguard"):
            client.get("https://upstream.invalid/")
    assert [(a.kind, a.port) for a in netguard.unreported()] == [("proxy", port)]
    with pytest.raises(BlockingIOError):
        listener.accept()


def test_loopback_still_works() -> None:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        with socket.create_connection(listener.getsockname(), timeout=3):
            conn, _ = listener.accept()
            conn.close()
        assert socket.getaddrinfo("localhost", 80)
    finally:
        listener.close()
    assert netguard.unreported() == []
