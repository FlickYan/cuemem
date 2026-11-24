"""/docs：接口文档页（Swagger UI），cuemem serve 与 cuemem proxy 共用（第九轮体检 P15）。

FastAPI 自带的 /docs 从 cdn.jsdelivr.net 加载 swagger-ui-dist@5（跟着 5.x 浮动的版本，不校验内容），
图标来自 fastapi.tiangolo.com；/redoc 还加载 redoc@2 与 fonts.googleapis.com 的字体。这些页面不要
令牌，与登录之后的界面同源。现在：

- /docs 是这里手写的页面：Swagger UI 固定在一个确切的版本（SWAGGER_UI_VERSION），脚本与样式带
  子资源完整性校验（integrity=，内容与这里记的散列不同时浏览器不执行），图标是空的 data: URL。
  除了 cdn.jsdelivr.net 不联系别的主机；fastapi.openapi.docs.get_swagger_ui_html 写不出 integrity；
- /redoc 与 /docs/oauth2-redirect 不再提供（接口用的是 Bearer 令牌，没有 OAuth2）；
- /openapi.json 不变，不需要联网。

没有出口网络的机器上 /docs 是空白页（要能访问 cdn.jsdelivr.net）：把 Swagger UI 的文件放进包里
才能离线使用，那要往仓库里加第三方文件，这里没有做。
"""

from __future__ import annotations

import html
import json

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

# 升级 Swagger UI：三个常量一起改。散列从那个确切版本的文件算出来，不要从别处抄：
#   npm pack swagger-ui-dist@<版本> && tar xzf swagger-ui-dist-<版本>.tgz
#   openssl dgst -sha384 -binary package/swagger-ui-bundle.js | openssl base64 -A
#   openssl dgst -sha384 -binary package/swagger-ui.css | openssl base64 -A
# （再与 https://cdn.jsdelivr.net/npm/swagger-ui-dist@<版本>/ 下的同名文件核对一遍。）
# Dependabot 不管这里的版本
SWAGGER_UI_VERSION = "5.34.1"
SWAGGER_JS_SRI = "sha384-XFYN47VzPiDY6qPtxYKg0nqUeW/upo20xf1j2SUE7gT/i35FJB4cuS/71Iegzk8Z"
SWAGGER_CSS_SRI = "sha384-4c3vsE5f+trhFyHrFrLM1fFGUn531NIaC7dcqEWIJdoRQJ5CkJbSuSCwXLpaS1KF"
SWAGGER_UI_CDN = f"https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}"

# SwaggerUIBundle 的参数与 FastAPI 自带的页面相同（没有 oauth2RedirectUrl）
_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="stylesheet" href="{cdn}/swagger-ui.css" integrity="{css}" crossorigin="anonymous">
<link rel="icon" href="data:,">
<title>{title}</title>
</head>
<body>
<div id="swagger-ui"></div>
<script src="{cdn}/swagger-ui-bundle.js" integrity="{js}" crossorigin="anonymous"></script>
<script>
const ui = SwaggerUIBundle({{
    url: {url},
    dom_id: '#swagger-ui',
    layout: 'BaseLayout',
    deepLinking: true,
    showExtensions: true,
    showCommonExtensions: true,
    presets: [
        SwaggerUIBundle.presets.apis,
        SwaggerUIBundle.SwaggerUIStandalonePreset
    ],
}})
</script>
</body>
</html>
"""


def swagger_html(title: str, openapi_url: str) -> HTMLResponse:
    """Swagger UI 的页面：固定版本、带 integrity 的脚本与样式，读 openapi_url 的接口描述。"""
    url = json.dumps(openapi_url).replace("<", "\\u003c")  # 写在 <script> 里：不让它提前结束
    page = _PAGE.format(
        cdn=SWAGGER_UI_CDN,
        css=SWAGGER_CSS_SRI,
        js=SWAGGER_JS_SRI,
        title=html.escape(title),
        url=url,
    )
    return HTMLResponse(page)


def add_docs(app: FastAPI) -> None:
    """给应用加上 GET /docs（不出现在接口描述里）。应用要用 docs_url=None、redoc_url=None、
    swagger_ui_oauth2_redirect_url=None 创建，FastAPI 自带的三个页面才不提供。"""

    @app.get("/docs", include_in_schema=False)
    async def docs(request: Request) -> HTMLResponse:
        # 与 FastAPI 自带的页面一样带上 root_path（反向代理把应用放在子路径下时）
        root = str(request.scope.get("root_path", "")).rstrip("/")
        return swagger_html(f"{app.title} - Swagger UI", root + (app.openapi_url or ""))
