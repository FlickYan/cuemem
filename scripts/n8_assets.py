"""N8：生成电商演示的图片素材——商品图与"用户分享的照片"（通义生图，QwenCloud）。

演示的商店与商品都是虚构的；用生成的图片而不是网上的照片，避免图片的版权与来源问题。
提示词写在 demo/shop/catalog.json 里（每件商品一条，再加统一的"棚拍商品图"风格），
生成后缩放到最长边 768 像素存成 JPEG，提交进仓库，之后不必重新生成。

用法：
    uv run python scripts/n8_assets.py                # 只生成还没有的
    uv run python scripts/n8_assets.py --only P01,P02 --force
    uv run python scripts/n8_assets.py --dry-run      # 只列出要生成的

每张约 $0.03（2026-10 国际站标价），全部 25 张约 $0.8；花费记入用量账本（kind=image）。
"""

from __future__ import annotations

import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx2
from PIL import Image

from cuemem.config import PROJECT_ROOT, load_settings
from cuemem.llm.ledger import Ledger, UsageRecord

CATALOG = PROJECT_ROOT / "demo" / "shop" / "catalog.json"
ENDPOINT = "https://maas.qwencloudapi.com/api/v1/services/aigc/multimodal-generation/generation"
# 依次尝试的模型（第一个能用的为准）与每张图的标价（美元）
MODELS = {"qwen-image-3.0": 0.03, "qwen-image-2.0": 0.035, "qwen-image": 0.035}
NEGATIVE = "text, letters, logo, brand name, watermark, label, people's faces"
MAX_SIDE = 768


def generate(client: httpx2.Client, key: str, model: str, prompt: str) -> bytes:
    body = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": prompt}]}]},
        "parameters": {
            "size": "1024*1024",
            "negative_prompt": NEGATIVE,
            "prompt_extend": False,
            "watermark": False,
        },
    }
    for attempt in range(1, 7):
        response = client.post(ENDPOINT, json=body, headers={"Authorization": f"Bearer {key}"})
        if response.status_code == 429 or response.status_code >= 500:  # 限流：退避后重试
            time.sleep(10 * attempt)
            continue
        if response.status_code != 200:
            raise RuntimeError(f"{model}: HTTP {response.status_code} {response.text[:300]}")
        data = response.json()
        url = data["output"]["choices"][0]["message"]["content"][0]["image"]
        image = client.get(url, timeout=120)
        image.raise_for_status()
        return image.content
    raise RuntimeError(f"{model}: 重试后仍失败")


def save(raw: bytes, target: Path) -> None:
    with Image.open(io.BytesIO(raw)) as source:
        image = source.convert("RGB")
    scale = MAX_SIDE / max(image.size)
    if scale < 1:
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=85)


def jobs(catalog: dict[str, Any], only: set[str], force: bool) -> list[tuple[str, str, Path]]:
    style = catalog["image_style"]
    root = CATALOG.parent
    out: list[tuple[str, str, Path]] = []
    for product in catalog["products"]:
        out.append((product["id"], f"{product['prompt']}. {style}", root / product["image"]))
    for photo in catalog["photos"]:
        out.append((Path(photo["file"]).stem, photo["prompt"], root / photo["file"]))
    return [
        (name, prompt, path)
        for name, prompt, path in out
        if (not only or name in only) and (force or not path.exists())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", default="", help="逗号分隔的商品 ID 或照片名")
    parser.add_argument("--force", action="store_true", help="已有的也重新生成")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)  # 并发高了会被限流
    args = parser.parse_args()

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    todo = jobs(catalog, {s for s in args.only.split(",") if s}, args.force)
    print(f"要生成 {len(todo)} 张")
    if args.dry_run or not todo:
        for name, prompt, path in todo:
            print(f"  {name} → {path.relative_to(PROJECT_ROOT)}: {prompt[:80]}…")
        return

    settings = load_settings()
    key = settings.providers["qwen"].api_key()
    if key is None:
        raise SystemExit("缺少 DASHSCOPE_API_KEY，先运行 uv run cuemem configure")
    ledger = Ledger(settings.paths.ledger)
    client = httpx2.Client(timeout=180)

    # 先用第一张图确定可用的模型
    model = ""
    first_name, first_prompt, first_path = todo[0]
    for candidate in MODELS:
        try:
            raw = generate(client, key, candidate, first_prompt)
        except RuntimeError as exc:
            print(f"  {candidate} 不可用：{exc}")
            continue
        model = candidate
        save(raw, first_path)
        print(f"  模型 {model}；{first_name} 完成")
        break
    if not model:
        raise SystemExit("没有可用的生图模型")

    def record() -> None:
        ledger.record(
            UsageRecord(
                ts=datetime.now(),
                run_id=ledger.run_id,
                kind="image",
                role="assets",
                provider="qwen",
                model=model,
                input_tokens=0,
                cost=MODELS[model],
                cost_cached=MODELS[model],
                currency="USD",
            )
        )

    record()

    def one(job: tuple[str, str, Path]) -> str:
        name, prompt, path = job
        try:
            save(generate(client, key, model, prompt), path)
        except RuntimeError as exc:  # 失败的留到下次运行再补（只生成还没有的）
            return f"{name} 失败：{exc}"
        record()
        return f"{name} 完成"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for line in pool.map(one, todo[1:]):
            print(f"  {line}")
    ledger.close()
    print(f"本次请求 {len(todo)} 张（失败的不计费），每张 ${MODELS[model]}")


if __name__ == "__main__":
    main()
