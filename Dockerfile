# CueMem 的容器镜像（N12）：HTTP 接口 + 记忆浏览界面 + OpenAI 兼容的记忆代理（/v1），一个进程。
#   docker build -t cuemem .
#   docker run -p 127.0.0.1:7860:7860 -v cuemem-data:/data --env-file .env cuemem
# .env 里要有 CUEMEM_API_TOKEN：容器里监听 0.0.0.0，没有访问令牌时拒绝启动。
# 多进程 / 大规模部署见 docker-compose.yml（Qdrant 服务端）。
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
WORKDIR /src
# 先只装依赖（锁定版本），源码改动时这一层可以复用
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable

FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
# 数据目录（数据库、本地向量库、缓存、原图、.env）放在卷里；安装后的包不在源码仓库里，
# 不设 CUEMEM_HOME 时会用 ~/.cuemem
ENV PATH=/opt/venv/bin:$PATH CUEMEM_HOME=/data PYTHONUNBUFFERED=1
RUN useradd --create-home --uid 1000 cuemem && mkdir -p /data && chown cuemem /data
USER cuemem
VOLUME /data
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/api/health')"
ENTRYPOINT ["cuemem"]
# 演示的导购场景要用仓库里的商品目录，镜像里用通用助手；访问令牌 CUEMEM_API_TOKEN 必须设置
# （不鉴权在可信网络里跑：命令后面加 --insecure-no-auth）
CMD ["serve", "--host", "0.0.0.0", "--port", "7860", "--scenario", "plain", "--space", "default"]
