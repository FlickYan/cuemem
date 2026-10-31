# CueMem 的容器镜像（N12）：HTTP 接口 + 记忆浏览界面 + OpenAI 兼容的记忆代理（/v1），一个进程。
#   docker build -t cuemem .
#   docker run -p 127.0.0.1:7860:7860 -v cuemem-data:/data --env-file .env cuemem
# .env 里要有 CUEMEM_API_TOKEN（至少 16 个字符，openssl rand -hex 32）：容器里监听 0.0.0.0，没有
# 访问令牌或令牌太短时拒绝启动。别的机器要访问时在前面放 TLS 反向代理（README 的 Public deployment）。
# 多进程 / 大规模部署见 docker-compose.yml（Qdrant 服务端）。
# 默认带上可选依赖 ui（记忆浏览界面，Gradio）与 local（本机向量 fastembed，预设 deepseek 要用）；
# 只要接口与记忆代理时
#   docker build --build-arg EXTRAS= -t cuemem .   # 镜像小得多（不装 Gradio、fastembed 与 onnxruntime）
FROM python:3.12-slim AS build
ARG EXTRAS="--extra ui --extra local"
# uv 按版本与镜像摘要固定，与 CI 相同、在 pyproject.toml 的 uv_build 范围里：构建用 uv 自带的
# uv_build，不从 PyPI 现取一份（tests/test_ci_pins.py 核对）
COPY --from=ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never
WORKDIR /src
# 先只装依赖（锁定版本），源码改动时这一层可以复用
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project $EXTRAS
COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable $EXTRAS

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
# （不鉴权在可信网络里跑：命令后面加 --insecure-no-auth。这时防 DNS 重绑定，只接受 IP 地址与
# localhost 作为 Host；用主机名访问时，例如别的容器用 compose 的服务名，把它加进 CUEMEM_ALLOWED_HOSTS）
CMD ["serve", "--host", "0.0.0.0", "--port", "7860", "--scenario", "plain", "--space", "default"]
