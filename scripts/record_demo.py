"""录 README 里的演示动图：用无头浏览器按演示剧本操作界面，截下关键状态，合成 GIF。

需要本机装有 Google Chrome，并先建好演示记忆、启动界面：

    uv run cuemem demo demo/shop/script.json --fresh --until 2
    uv run cuemem serve
    uv run --with playwright python scripts/record_demo.py          # 录制并合成
    uv run python scripts/record_demo.py --assemble-only            # 只用已有的帧重新合成

Playwright 只在录制时临时装（不进项目依赖），用的是本机的 Chrome（channel="chrome"）。
帧存在 runs/demo-frames/（不提交），动图写到 docs/assets/demo.gif。回答由模型现场生成，
每次录的内容会略有不同；录完记得用 `--until 2` 恢复演示前的记忆。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "runs" / "demo-frames"
GIF = ROOT / "docs" / "assets" / "demo.gif"
URL = "http://127.0.0.1:7860/"
VIEWPORT = {"width": 1440, "height": 1180}
CROP = (100, 0, 1340, 1180)  # 去掉两侧留白
WIDTH = 1000

# （帧名前缀, 字幕）：每一段的说明，显示在画面顶部
CAPTIONS = [
    ("start", "Third chat with Maya on Oct 10 · her August and September chats are in memory"),
    ("m1", "1  Remembers her across sessions · the trace shows what the controller chose and why"),
    ("m2", "2  Knows what changed · EU 40 for running shoes since September; her usual size is 39"),
    ("m3", '3  "The pair I showed you in August" · recalls the photo, re-looks at the original'),
    ("m4", "4  How her budget changed · $120 → $180; the old fact is kept, marked no longer true"),
    ("memory", "5  Memory browser · profile, fact timeline with validity, photos, sessions"),
    ("forget", "6  Forget a fact · its exchange and the replies that restated it go, nothing else"),
    ("m5", "7  Gone from context and memory · Cue no longer knows the birthday"),
]
# 界面的对话区不一定自动滚到底（长回答之后会停在中间），截图前自己滚
LATEST_JS = """() => {
  const box = document.querySelector('[aria-label="chatbot conversation"]');
  if (!box) return;
  box.scrollTop = box.scrollHeight;
  const asked = box.querySelectorAll('.user-row');
  const last = asked[asked.length - 1];
  const gap = last ? last.getBoundingClientRect().top - box.getBoundingClientRect().top : 0;
  if (gap < 0) box.scrollTop += gap - 12;
}"""
MESSAGES = [
    "Hi Cue, I'm back!",
    "What size running shoes do I wear now?",
    "Can you find shoes that look like the pair I showed you in August?",
    "How has my shoe budget changed since we first talked?",
]


# ---------- 录制 ----------


def record() -> list[dict[str, Any]]:
    from playwright.sync_api import Page, sync_playwright

    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    frames: list[dict[str, Any]] = []

    def shot(page: Page, name: str, seconds: float) -> None:
        path = FRAMES / f"{len(frames):02d}-{name}.png"
        page.screenshot(path=str(path))
        frames.append({"file": path.name, "seconds": seconds, "name": name})
        print(f"frame {len(frames):02d} {name}", flush=True)

    def chat_box(page: Page) -> Any:
        return page.locator('textarea[placeholder^="Message Cue"]')

    def wait_answer(page: Page, timeout: float = 180) -> None:
        start = time.time()
        page.wait_for_timeout(800)
        while time.time() - start < timeout:
            if not page.locator("text=Thinking…").count() and chat_box(page).is_enabled():
                page.wait_for_timeout(1200)  # 等图片与轨迹渲染完
                return
            page.wait_for_timeout(500)
        raise TimeoutError("the answer did not finish")

    def to_latest(page: Page) -> None:
        """滚到对话底部；回答太长、顶出了这一问时，把这一问滚回顶上（问与答的开头都在画面里）。"""
        page.evaluate(LATEST_JS)
        page.wait_for_timeout(400)

    def send(page: Page, text: str, tag: str) -> None:
        box = chat_box(page)
        box.click()
        box.type(text[: len(text) // 2], delay=25)
        shot(page, f"{tag}-typing", 0.35)
        box.type(text[len(text) // 2 :], delay=25)
        shot(page, f"{tag}-typed", 0.7)
        box.press("Enter")
        page.wait_for_timeout(700)
        shot(page, f"{tag}-thinking", 1.0)
        wait_answer(page)
        to_latest(page)
        shot(page, f"{tag}-answer", 5.0)

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.goto(URL, wait_until="load")
        page.wait_for_selector('textarea[placeholder^="Message Cue"]', timeout=60000)
        page.wait_for_timeout(1500)
        page.locator('textarea[placeholder^="e.g. 2026-10-10"]').fill("2026-10-10 10:00")
        page.get_by_role("button", name="New session").click()
        page.wait_for_timeout(2500)
        shot(page, "start", 3.0)
        for i, message in enumerate(MESSAGES, 1):
            send(page, message, f"m{i}")

        page.get_by_role("tab", name="Memory").click()
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Refresh").click()
        page.wait_for_timeout(3000)
        shot(page, "memory-top", 4.0)
        page.mouse.wheel(0, 650)
        page.wait_for_timeout(1200)
        shot(page, "memory-facts", 4.5)

        page.mouse.wheel(0, 2000)
        page.wait_for_timeout(1000)
        dropdown = page.get_by_label("Which one")
        dropdown.click()
        dropdown.type("turns 30", delay=40)  # "…Ana who turns 30 on October 3rd."
        page.wait_for_timeout(1200)
        shot(page, "forget-pick", 2.5)
        page.locator("li", has_text="turns 30").first.click()
        page.wait_for_timeout(800)
        page.get_by_role("button", name="Forget", exact=True).click()
        for _ in range(180):
            if page.locator("text=Forgot").count():
                break
            page.wait_for_timeout(1000)
        page.wait_for_timeout(1500)
        shot(page, "forget-done", 4.5)

        page.get_by_role("tab", name="Chat").click()
        page.wait_for_timeout(1500)
        send(page, "When is my sister's birthday?", "m5")
        browser.close()
    (FRAMES / "frames.json").write_text(json.dumps(frames, indent=1), encoding="utf-8")
    return frames


# ---------- 合成 ----------


def _font(size: int) -> Any:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _caption(name: str) -> str:
    return next(text for prefix, text in CAPTIONS if name.startswith(prefix))


def assemble(frames: list[dict[str, Any]]) -> Path:
    font = _font(19)
    images: list[Image.Image] = []
    durations: list[int] = []
    for frame in frames:
        shot = Image.open(FRAMES / frame["file"]).convert("RGB").crop(CROP)
        shot = shot.resize((WIDTH, round(shot.height * WIDTH / shot.width)), Image.LANCZOS)
        bar = 44
        canvas = Image.new("RGB", (WIDTH, shot.height + bar), "#0d1117")
        canvas.paste(shot, (0, bar))
        ImageDraw.Draw(canvas).text(
            (18, bar // 2), _caption(frame["name"]), font=font, fill="#f0f6fc", anchor="lm"
        )
        images.append(canvas.quantize(colors=256, dither=Image.Dither.NONE))
        durations.append(round(frame["seconds"] * 1000))
    GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        GIF, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True
    )
    return GIF


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assemble-only", action="store_true", help="只用已有的帧重新合成")
    args = parser.parse_args()
    if args.assemble_only:
        frames = json.loads((FRAMES / "frames.json").read_text(encoding="utf-8"))
    else:
        frames = record()
    path = assemble(frames)
    seconds = sum(f["seconds"] for f in frames)
    print(f"{path}（{len(frames)} 帧，约 {seconds:.0f} 秒，{path.stat().st_size / 1e6:.1f} MB）")


if __name__ == "__main__":
    main()
