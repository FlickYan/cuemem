"""配置参考（docs/configuration.md 与 docs/en/configuration.md）不与配置模型走样（第十轮体检 Q15）。

每个配置模型在两个页面里各有一张表：字段一个不多、一个不少、顺序相同，默认值一栏就是实际生效的
默认值。加了字段、改了默认值而没有改文档，这里会失败。页面里的示例预设也要真的能加载。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from cuemem import config
from cuemem.config import Settings, load_settings, unknown_keys

DOCS = Path(__file__).resolve().parents[1] / "docs"
PAGES = {"zh": DOCS / "configuration.md", "en": DOCS / "en" / "configuration.md"}
REQUIRED = {"zh": "必填", "en": "required"}

# 标题里反引号中的名字 → 配置模型。条目由用户命名的段（providers、models、roles）与它们的子模型
# 没有"实际生效的默认值"，默认值一栏是字段不写时的值
SECTIONS: dict[str, type[BaseModel]] = {
    "Settings": Settings,
    "paths": config.PathsConfig,
    "providers": config.ProviderConfig,
    "providers.<name>.peak_hours": config.PeakHours,
    "models": config.ModelSpec,
    "models.<alias>.price": config.Price,
    "roles": config.RoleConfig,
    "embedding": config.EmbeddingConfig,
    "multimodal_embedding": config.MultimodalEmbeddingConfig,
    "rerank": config.RerankConfig,
    "vector_store": config.VectorStoreConfig,
    "retrieval": config.RetrievalConfig,
    "consolidation": config.ConsolidationConfig,
    "proxy": config.ProxyConfig,
}


def _sub_model(annotation: Any) -> type[BaseModel] | None:
    """字段的子模型：X、X | None 或 dict[str, X] 里的配置模型。"""
    args = getattr(annotation, "__args__", ())
    candidates = [annotation, *args]
    for candidate in candidates:
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            return candidate
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _sections(text: str) -> dict[str, list[tuple[str, str]]]:
    """页面里的表：标题中反引号里的名字 → [(字段, 默认值一栏)]。"""
    found: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("#"):
            name = re.search(r"`([^`]+)`", line)
            current = name.group(1) if line.startswith("### ") and name else None
        elif current and (row := re.match(r"\| `(\w+)` \| (.*?) \|", line)):
            found.setdefault(current, []).append((row.group(1), row.group(2)))
    return found


def _expected(lang: str, loaded: Settings) -> dict[str, list[tuple[str, str]]]:
    expected: dict[str, list[tuple[str, str]]] = {}
    for key, model in SECTIONS.items():
        instance: BaseModel | None = None
        if model is Settings:
            instance = loaded
        elif key in Settings.model_fields and isinstance(getattr(loaded, key), BaseModel):
            instance = getattr(loaded, key)
        rows = []
        for name, field in model.model_fields.items():
            if model is Settings and _sub_model(field.annotation) is not None:
                continue  # 各个段有自己的表
            if instance is not None:
                cell = f"`{_fmt(getattr(instance, name))}`"
            elif field.is_required():
                cell = REQUIRED[lang]
            else:
                default = field.default
                if default is PydanticUndefined and field.default_factory is not None:
                    default = field.default_factory()  # type: ignore[call-arg]
                cell = f"`{_fmt(default)}`"
            rows.append((name, cell))
        expected[key] = rows
    return expected


def test_every_config_model_has_a_section() -> None:
    """从 Settings 出发能走到的配置模型都在 SECTIONS 里：加了一个段，这里先失败。"""
    reachable: set[type[BaseModel]] = set()
    todo: list[type[BaseModel]] = [Settings]
    while todo:
        model = todo.pop()
        if model in reachable:
            continue
        reachable.add(model)
        for field in model.model_fields.values():
            if (sub := _sub_model(field.annotation)) is not None:
                todo.append(sub)
    assert reachable == set(SECTIONS.values())


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_field_tables_match_the_config_models(clean_env: None, lang: str) -> None:
    """两个页面的每张表与配置模型一致：字段、顺序、默认值（默认配置文件实际生效的值）。"""
    tables = _sections(PAGES[lang].read_text("utf-8"))
    expected = _expected(lang, Settings())  # 相对路径不解析：文档里写的是配置文件里的样子
    assert list(tables) == list(expected)
    for key, rows in expected.items():
        assert tables[key] == rows, key


def _preset(lang: str) -> str:
    blocks = re.findall(r"```yaml\n(.*?)```", PAGES[lang].read_text("utf-8"), flags=re.S)
    assert len(blocks) == 1
    return str(blocks[0])


def test_the_worked_preset_loads_cleanly(clean_env: None, tmp_path: Path) -> None:
    """页面里的示例预设两种语言逐字相同、没有不认识的键，加载之后不再用到任何默认的供应商。"""
    text = _preset("en")
    assert _preset("zh") == text
    assert unknown_keys(yaml.safe_load(text)) == []
    path = tmp_path / "my-preset.yaml"
    path.write_text(text, "utf-8")
    settings = load_settings(preset=str(path), home=tmp_path)
    assert settings.used_providers() == ["gateway"]
    assert {rc.model for rc in settings.roles.values()} == {"my-chat"}
    assert len(settings.roles) == 7
    assert settings.multimodal_embedding.backend == "none"
    assert settings.rerank.provider is None
    assert settings.consolidation.fallback_model is None
    assert settings.consolidation.caption_images is False


@pytest.mark.parametrize("language", ["zh", "en"])
def test_the_retention_rows_name_every_command_that_sweeps(language: str) -> None:
    """第十轮复审（Q15）：两个保留期限由 cuemem serve、cuemem proxy 与（Q12 之后）stdio 的
    cuemem mcp 执行，页面里的两行要都写上。"""
    rows = [
        line
        for line in PAGES[language].read_text(encoding="utf-8").splitlines()
        if line.startswith(("| `traces_retention_days`", "| `expired_retention_days`"))
    ]
    assert len(rows) == 2
    for row in rows:
        assert all(f"`cuemem {name}`" in row for name in ("serve", "proxy", "mcp")), row
