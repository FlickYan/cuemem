"""把 demo/video/cuemem-60s.html 渲染成 MP4（无声，1920 × 1080，30 fps）。

    uv run --with playwright --with imageio-ffmpeg python scripts/render_video.py
    uv run --with playwright python scripts/render_video.py --still 12.5 --still 27   # 只截静帧检查
    （渲染命令加上 --start 22 --end 32，只渲染一段）
    uv run --with playwright python scripts/render_video.py --thumbnail   # README 里的视频封面

页面里所有动画挂在一条暂停的 GSAP 时间线上：逐帧 seek 到 t = i / fps 再截图，输出与机器快慢无关。
用本机 Chrome（channel="chrome"）；字体与 GSAP 从 CDN 加载，需要联网。ffmpeg 用 imageio-ffmpeg
自带的那个，不往系统里装东西。输出写到 runs/video/（不提交）。
"""

from __future__ import annotations

import argparse
import functools
import http.server
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAGE = "demo/video/cuemem-60s.html"
OUT = ROOT / "runs" / "video"
SIZE = {"width": 1920, "height": 1080}
POSTER_AT = 2.6  # 片名完整出现的时刻
ENCODE = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p"]
THUMBNAIL = ROOT / "docs" / "assets" / "video-thumbnail.png"
# 封面：片名帧下方加一个播放按钮，提示这是视频
PLAY_BUTTON = """() => {
  const b = document.createElement('div');
  b.style.cssText = 'position:absolute;left:0;right:0;bottom:118px;margin:0 auto;width:fit-content;'
    + 'display:flex;align-items:center;gap:18px;padding:16px 36px 16px 20px;border-radius:999px;'
    + 'background:#0F172A;color:#fff;font:600 34px Inter,sans-serif;'
    + 'box-shadow:0 16px 40px rgba(15,23,42,.25)';
  b.innerHTML = '<svg width="52" height="52" viewBox="0 0 52 52"><circle cx="26" cy="26" r="26"'
    + ' fill="#14B8A6"/><path d="M21 15 L37 26 L21 37 Z" fill="#fff"/></svg>'
    + 'Watch the 60-second intro';
  document.querySelector('#stage').append(b);
}"""


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass


def serve() -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(_Quiet, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def open_page(playwright: Any, port: int) -> tuple[Any, Any, list[str]]:
    browser = playwright.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport=SIZE, device_scale_factor=1)
    problems: list[str] = []
    page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
    page.on(  # 资源加载失败按地址记；浏览器自己要的 favicon 不算
        "response",
        lambda r: (
            r.status >= 400 and "favicon" not in r.url and problems.append(f"{r.status} {r.url}")
        ),
    )
    page.goto(f"http://127.0.0.1:{port}/{PAGE}?render", wait_until="load")
    page.wait_for_function("window.cuemem && window.cuemem.ready", timeout=60_000)
    if problems:
        raise RuntimeError("\n".join(problems))
    return browser, page, problems


def shot(page: Any, t: float) -> bytes:
    page.evaluate("t => window.cuemem.seek(t)", t)
    return page.screenshot(type="png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--still", type=float, action="append", help="只截这些时刻的静帧（秒），可重复"
    )
    parser.add_argument("--start", type=float, default=0.0, help="从第几秒开始渲染")
    parser.add_argument("--end", type=float, help="渲染到第几秒（默认到结尾）")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--out", type=Path, default=OUT / "cuemem-60s.mp4")
    parser.add_argument("--thumbnail", action="store_true", help="只生成 README 里的视频封面")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    server, port = serve()
    try:
        with sync_playwright() as p:
            browser, page, problems = open_page(p, port)
            duration = float(page.evaluate("window.cuemem.duration"))
            if args.thumbnail:
                thumbnail(page)
            elif args.still:
                stills = OUT / "stills"
                stills.mkdir(exist_ok=True)
                for t in args.still:
                    path = stills / f"t{t:05.2f}.png"
                    path.write_bytes(shot(page, t))
                    print(path.relative_to(ROOT))
            else:
                render(page, args, duration)
            browser.close()
            if problems:
                raise RuntimeError("\n".join(problems))
    finally:
        server.shutdown()


def thumbnail(page: Any) -> None:
    import io

    from PIL import Image

    page.evaluate("t => window.cuemem.seek(t)", POSTER_AT)
    page.evaluate(PLAY_BUTTON)
    image = Image.open(io.BytesIO(page.screenshot(type="png"))).convert("RGB")
    image.resize((1280, 720), Image.Resampling.LANCZOS).save(THUMBNAIL, optimize=True)
    print(f"{THUMBNAIL.relative_to(ROOT)}  {THUMBNAIL.stat().st_size / 1e3:.0f} KB")


def render(page: Any, args: argparse.Namespace, duration: float) -> None:
    import imageio_ffmpeg

    end = min(args.end or duration, duration)
    first, last = round(args.start * args.fps), round(end * args.fps)
    out: Path = args.out
    if args.start or args.end:
        out = out.with_name(f"{out.stem}-{args.start:g}-{end:g}{out.suffix}")
    source = ["-f", "image2pipe", "-framerate", str(args.fps), "-c:v", "png", "-i", "-"]
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", *source]
    cmd += [*ENCODE, "-movflags", "+faststart", str(out)]  # 网页上可以边下边播
    ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert ffmpeg.stdin is not None
    started = time.time()
    for i in range(first, last):
        ffmpeg.stdin.write(shot(page, i / args.fps))
        if (i - first) % (args.fps * 5) == 0:
            elapsed = time.time() - started
            print(f"  {i / args.fps:5.1f} s / {end:.0f} s  ({elapsed:.0f} s elapsed)", flush=True)
    ffmpeg.stdin.close()
    if ffmpeg.wait():
        raise RuntimeError("ffmpeg failed")
    if not (args.start or args.end):
        (OUT / f"{out.stem}-poster.png").write_bytes(shot(page, POSTER_AT))
    size, elapsed = out.stat().st_size / 1e6, time.time() - started
    print(f"{out.relative_to(ROOT)}  {last - first} frames, {size:.1f} MB, {elapsed:.0f} s")


if __name__ == "__main__":
    main()
