"""两份 README 与部署、排查问题两页不走样（第十轮体检 Q20）。

不联网、不启动进程，只读仓库里的文件：
- 出错信息与部署文件的注释里点名的 README 小节，两份 README 里都还有这个标题（部署的四节在 0.10.0
  搬到了 docs/deployment.md，README 在原来的标题下留着摘要与链接，所以标题不能改名）；
- README 与新的四页里指向仓库内文件的链接都找得到文件，带 # 的找得到那个标题；
- 版本号、ADR 的条数跟着代码与决策记录走，安装命令不从 PyPI 按名字装。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import cuemem

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "cuemem"
READMES = {"en": ROOT / "README.md", "zh": ROOT / "README.zh-CN.md"}
DEPLOYMENT = {"en": ROOT / "docs" / "en" / "deployment.md", "zh": ROOT / "docs" / "deployment.md"}
TROUBLESHOOTING = {
    "en": ROOT / "docs" / "en" / "troubleshooting.md",
    "zh": ROOT / "docs" / "troubleshooting.md",
}
# 搬走正文、留下摘要的四节：英文标题、中文标题
MOVED = (
    ("Deployment", "部署"),
    ("Public deployment", "公网部署"),
    ("Backup and recovery", "备份与恢复"),
    ("Upgrading", "升级"),
)
# 点名 README 小节的地方：文件、文件里的原话、它指的那一节（英文标题、中文标题）
NAMED = (
    (SRC / "cli.py", "README 的 Public deployment / 公网部署", MOVED[1]),
    (SRC / "store" / "files.py", "备份与恢复）；缓存与账本可以删掉重建", MOVED[2]),
    (SRC / "store" / "sqlite.py", "README 的“升级”", MOVED[3]),
    (ROOT / "Dockerfile", "README 的 Public deployment", MOVED[1]),
    (ROOT / "docker-compose.yml", "README 的 Public", MOVED[1]),
    (ROOT / ".env.example", "README 的 Public deployment", MOVED[1]),
)
# src 里提到 README 的文件就这几个：多出一个时，先看它点名的小节在不在，再加进来
MENTIONS = {"cli.py", "api/app.py", "store/files.py", "store/sqlite.py", "evals/baselines/amem.py"}


def _headings(path: Path) -> list[tuple[int, str]]:
    """（级别，标题文字）；代码块里以 # 开头的行不算。"""
    found: list[tuple[int, str]] = []
    fenced = False
    for line in path.read_text("utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (match := re.match(r"(#{1,6}) (.+)", line)):
            found.append((len(match[1]), match[2].strip()))
    return found


def _anchors(path: Path) -> set[str]:
    """GitHub 给标题生成的锚点（小写、去掉标点、空格换成连字符，重名的加 -1、-2），以及手写的
    <a id="...">。"""
    anchors: set[str] = set(re.findall(r'<a id="([^"]+)">', path.read_text("utf-8")))
    seen: dict[str, int] = {}
    for _, text in _headings(path):
        text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
        slug = re.sub(r"[^\w\- ]", "", text.lower()).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def _section(path: Path, heading: str) -> list[str]:
    """这个标题下面、到下一个标题为止的非空行。"""
    lines = path.read_text("utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(rf"#+ {re.escape(heading)}", line)]
    assert len(starts) == 1, f"{path.name} 里“{heading}”这个标题应当正好有一个"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("#"):
            break
        if line.strip():
            body.append(line)
    return body


def test_sections_named_in_messages_still_exist() -> None:
    titles = {lang: {text for _, text in _headings(path)} for lang, path in READMES.items()}
    for path, phrase, (en, zh) in NAMED:
        assert phrase in path.read_text("utf-8"), f"{path.name} 里点名 README 的那句话变了"
        assert en in titles["en"], f"README.md 里没有“{en}”一节（{path.name} 点名了它）"
        assert zh in titles["zh"], f"README.zh-CN.md 里没有“{zh}”一节（{path.name} 点名了它）"
    mentioning = {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "README" in path.read_text("utf-8")
    }
    assert mentioning == MENTIONS


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_moved_sections_keep_a_stub_that_links_to_the_page(lang: str) -> None:
    page = DEPLOYMENT[lang]
    target = page.relative_to(ROOT).as_posix()
    page_titles = [text for level, text in _headings(page) if level == 2]
    assert page_titles == [pair[lang == "zh"] for pair in MOVED]
    for pair in MOVED:
        title = pair[lang == "zh"]
        stub = _section(READMES[lang], title)
        assert 1 <= len(stub) <= 6, f"{READMES[lang].name} 的“{title}”应当只是几行摘要"
        anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        assert f"]({target}#{anchor})" in "\n".join(stub)
        assert len(_section(page, title)) >= 4  # 正文在那一页


@pytest.mark.parametrize(
    "path",
    [*READMES.values(), *DEPLOYMENT.values(), *TROUBLESHOOTING.values()],
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_local_links_resolve(path: Path) -> None:
    text = path.read_text("utf-8")
    links = re.findall(r"\]\(([^)\s]+)\)", text) + re.findall(r'(?:src|srcset)="([^"]+)"', text)
    assert links
    for link in links:
        if re.match(r"[a-z]+:", link):  # http、https、mailto
            continue
        target, _, anchor = link.partition("#")
        file = (path.parent / target).resolve() if target else path
        assert file.exists(), f"{path.name}：{link} 指向的文件不存在"
        if anchor and file.suffix == ".md":
            assert anchor in _anchors(file), f"{path.name}：{link} 指向的标题不存在"


def test_version_and_adr_count_follow_the_code() -> None:
    version = cuemem.__version__
    en, zh = (path.read_text("utf-8") for path in READMES.values())
    assert f"({version} now)" in en and f"`...@v{version}`" in en
    assert f"（现在是 {version}）" in zh and f"`...@v{version}`" in zh
    decisions = (ROOT / "docs" / "en" / "decisions.md").read_text("utf-8")
    assert f"Decision records ({len(re.findall(r'^## D-', decisions, flags=re.M))} ADRs)" in en


@pytest.mark.parametrize(
    "path",
    [*READMES.values(), *DEPLOYMENT.values(), *TROUBLESHOOTING.values()],
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_install_commands_never_name_the_pypi_package(path: Path) -> None:
    """PyPI 上名叫 cuemem 的包不是这个项目：安装命令都带着 GitHub 的地址。"""
    for command in re.findall(r"pip install[^`\n]*", path.read_text("utf-8")):
        if "cuemem" in command:
            assert "@ git+https://github.com/FlickYan/cuemem" in command, command
