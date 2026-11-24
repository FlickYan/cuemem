"""/docs（api/docs.py，第九轮体检 P15）：cuemem serve 与 cuemem proxy 的接口文档页。

以前是 FastAPI 自带的页面：/docs 从 cdn.jsdelivr.net 加载 swagger-ui-dist@5（跟着 5.x 浮动、不校验
内容），图标来自 fastapi.tiangolo.com；/redoc 加载 redoc@2 与 fonts.googleapis.com。现在 /docs 固定
一个确切的版本并带 integrity，只联系 cdn.jsdelivr.net；/redoc 与 /docs/oauth2-redirect 不再提供。"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cuemem.api.app import create_app
from cuemem.api.docs import (
    SWAGGER_CSS_SRI,
    SWAGGER_JS_SRI,
    SWAGGER_UI_VERSION,
    swagger_html,
)
from cuemem.api.proxy import create_proxy_app

TOKEN = "docs-token-0123456789"


def serve_app(tmp_path: Path) -> FastAPI:
    service = MagicMock()
    service.max_open_sessions = 4
    return create_app(service, upload_dir=tmp_path / "uploads", ui=False, token=TOKEN, mcp=False)


def proxy_app(tmp_path: Path) -> FastAPI:
    return create_proxy_app(MagicMock(), token=TOKEN)


APPS = pytest.mark.parametrize("build", [serve_app, proxy_app])


@APPS
def test_docs_pins_swagger_ui_with_integrity(
    build: Callable[[Path], FastAPI], tmp_path: Path
) -> None:
    """脚本与样式来自确切的版本，带 integrity 与 crossorigin；页面里没有别的主机。"""
    assert re.fullmatch(r"5\.\d+\.\d+", SWAGGER_UI_VERSION)
    page = TestClient(build(tmp_path)).get("/docs")
    assert page.status_code == 200 and page.headers["content-type"].startswith("text/html")
    tags = re.findall(r"<script[^>]*\bsrc=[^>]*>|<link[^>]*\brel=\"stylesheet\"[^>]*>", page.text)
    assert len(tags) == 2, tags
    for tag, name, sri in zip(
        tags,
        ("swagger-ui.css", "swagger-ui-bundle.js"),
        (SWAGGER_CSS_SRI, SWAGGER_JS_SRI),
        strict=True,
    ):
        url = f"https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}/{name}"
        assert f'"{url}"' in tag, tag
        assert sri.startswith("sha384-") and f'integrity="{sri}"' in tag, tag
        assert 'crossorigin="anonymous"' in tag, tag
    hosts = set(re.findall(r"(?:https?:)?//([A-Za-z0-9.-]+)", page.text))
    assert hosts == {"cdn.jsdelivr.net"}
    assert 'url: "/openapi.json",' in page.text and "oauth2RedirectUrl" not in page.text


@APPS
def test_redoc_and_the_oauth2_redirect_are_gone(
    build: Callable[[Path], FastAPI], tmp_path: Path
) -> None:
    """/redoc 与 /docs/oauth2-redirect 是 404；/openapi.json 照常，/docs 不在接口描述里。"""
    http = TestClient(build(tmp_path))
    assert http.get("/redoc").status_code == 404
    assert http.get("/docs/oauth2-redirect").status_code == 404
    spec = http.get("/openapi.json")
    assert spec.status_code == 200 and spec.json()["paths"]
    assert not any(path.startswith(("/docs", "/redoc")) for path in spec.json()["paths"])


def test_the_title_and_the_spec_url_are_escaped() -> None:
    page = bytes(swagger_html("<b>&", "/x</script><script>alert(1)").body).decode()
    assert "<title>&lt;b&gt;&amp;</title>" in page
    assert "</script><script>alert" not in page and "\\u003c/script>" in page
