"""写出 /api/v1 记忆接口的 OpenAPI 快照 docs/openapi-v1.json（cuemem.api.memories.openapi_v1）。

改了 /api/v1 的接口或返回值的模型之后运行一次（tests/test_memory_api.py 核对快照没有过时）：

    uv run python scripts/openapi_v1.py            # 写出
    uv run python scripts/openapi_v1.py --check    # 只检查，过时了退出码 1

别的语言的客户端按快照生成，例如 TypeScript：
    npx openapi-typescript docs/openapi-v1.json -o cuemem.d.ts
离线，不打开记忆系统。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cuemem.api.memories import openapi_v1

TARGET = Path(__file__).resolve().parent.parent / "docs" / "openapi-v1.json"


def render() -> str:
    return json.dumps(openapi_v1(), indent=2, ensure_ascii=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="只检查快照是不是最新的")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current != text:
            print(f"{TARGET} 过时了：运行 uv run python scripts/openapi_v1.py", file=sys.stderr)
            raise SystemExit(1)
        return
    TARGET.write_text(text, encoding="utf-8")
    print(f"已写出 {TARGET}")


if __name__ == "__main__":
    main()
