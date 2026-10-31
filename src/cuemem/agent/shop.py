"""电商导购演示的领域部分（N8）：商品目录与两个工具。

- search_products：按文字找商品（商品描述的向量与查询的余弦相似度），可按类目、价格上限过滤；
- find_similar_products：按图找同款——图片可以是用户这次发的（u1），也可以是从记忆里召回的
  （i1，例如一个月前分享的照片），与商品图在同一个图文向量空间里比较。

商品目录是一个 JSON 文件（demo/shop/catalog.json），商品图放在它旁边。演示用的商店与商品
都是虚构的，图片由 scripts/n8_assets.py 用通义的生图模型生成。
"""

from __future__ import annotations

import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from cuemem.agent.runtime import Persona
from cuemem.agent.tools import Tool, ToolContext, ToolResult
from cuemem.embed.base import FloatArray, TextEmbedder
from cuemem.embed.multimodal import ImageEmbedder

SHOP_PERSONA = Persona(
    name="Cue",
    role="the personal shopping assistant of CueMart, an online store for running and outdoor "
    "gear (a demo store with made-up products)",
    instructions="""Shopping:
- Use search_products to find items in the catalog, and find_similar_products to find items that look like a picture: one the user sends now (u1, u2, …) or one from memory (i1, i2, …).
- Recommend only products returned by these tools, by name and price, with each recommended product name in bold; two or three good options are better than a long list.
- Take what you know about the user into account: budget, sizes, preferences and values (for example vegan materials). If something changed, follow the latest and say so briefly.
- Check sizes against the product's size range when you know the user's size.
- You cannot check stock, place orders or track deliveries; don't offer to.
- Keep replies short: about 120 words at most, unless the user asks for detail.""",  # noqa: E501
)


class Product(BaseModel):
    id: str
    name: str
    category: str
    price: float
    colors: list[str] = Field(default_factory=list)
    materials: list[str] = Field(default_factory=list)
    sizes: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    image: str | None = None  # 相对目录文件的路径
    prompt: str = ""  # 生成商品图用的提示词

    def document(self) -> str:
        """检索用的商品描述。"""
        parts = [
            self.name,
            self.category,
            f"colors: {', '.join(self.colors)}" if self.colors else "",
            f"materials: {', '.join(self.materials)}" if self.materials else "",
            ", ".join(self.tags),
            self.description,
        ]
        return ". ".join(p for p in parts if p)

    def line(self, score: float | None = None) -> str:
        """给模型看的一行。"""
        # 不给编号：模型用不着，给了它会照抄进回答（商品卡片按名字对应）
        fields = [self.name, f"${self.price:g}", self.category]
        if self.colors:
            fields.append("/".join(self.colors))
        if self.materials:
            fields.append("materials: " + ", ".join(self.materials))
        if self.sizes:
            fields.append(f"sizes {self.sizes[0]}–{self.sizes[-1]}")
        if self.tags:
            fields.append(", ".join(self.tags))
        if score is not None:
            fields.append(f"match {score:.2f}")
        return " — ".join(fields) + (f"\n    {self.description}" if self.description else "")

    def card(self, root: Path, score: float | None = None) -> dict[str, Any]:
        """界面上的商品卡片。"""
        image = str(root / self.image) if self.image else None
        return {
            "id": self.id,
            "name": self.name,
            "price": self.price,
            "category": self.category,
            "image": image,
            "score": None if score is None else round(score, 3),
        }


class Catalog:
    """商品目录：文字检索用文本向量，按图找同款用图片向量（都走向量缓存，只算一次）。"""

    def __init__(
        self,
        products: Sequence[Product],
        root: Path,
        *,
        embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.products = list(products)
        self.root = root
        self.embedder = embedder
        self.image_embedder = image_embedder
        self._text: FloatArray | None = None
        self._images: tuple[list[Product], FloatArray] | None = None
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        embedder: TextEmbedder,
        image_embedder: ImageEmbedder | None = None,
    ) -> Catalog:
        data = json.loads(path.read_text(encoding="utf-8"))
        products = [Product.model_validate(p) for p in data["products"]]
        return cls(products, path.parent, embedder=embedder, image_embedder=image_embedder)

    def get(self, product_id: str) -> Product | None:
        return next((p for p in self.products if p.id == product_id), None)

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        max_price: float | None = None,
        limit: int = 5,
    ) -> list[tuple[Product, float]]:
        with self._lock:
            if self._text is None:
                docs = [p.document() for p in self.products]
                self._text = self.embedder.embed(docs, "document").dense
            matrix = self._text
        vector = self.embedder.embed([query], "query").dense[0]
        return self._rank(self.products, matrix @ vector, category, max_price, limit)

    def similar(
        self,
        image: Path,
        *,
        category: str | None = None,
        max_price: float | None = None,
        limit: int = 5,
    ) -> list[tuple[Product, float]]:
        if self.image_embedder is None:
            return []
        with self._lock:
            if self._images is None:
                pictured = [p for p in self.products if p.image]
                paths = [self.root / str(p.image) for p in pictured]
                self._images = (pictured, self.image_embedder.embed_images(paths))
            pictured, matrix = self._images
        vector = self.image_embedder.embed_images([image])[0]
        return self._rank(pictured, matrix @ vector, category, max_price, limit)

    @staticmethod
    def _rank(
        products: Sequence[Product],
        scores: FloatArray,
        category: str | None,
        max_price: float | None,
        limit: int,
    ) -> list[tuple[Product, float]]:
        wanted = (category or "").strip().lower()
        ranked: list[tuple[Product, float]] = []
        for i in np.argsort(-scores):
            product = products[int(i)]
            if wanted and wanted not in product.category.lower():
                continue
            if max_price is not None and product.price > max_price:
                continue
            ranked.append((product, float(scores[int(i)])))
            if len(ranked) >= limit:
                break
        return ranked


FILTERS: dict[str, Any] = {
    "category": {
        "type": "string",
        "description": "optional: only this category, e.g. running shoes, hiking, apparel",
    },
    "max_price": {"type": "number", "description": "optional: highest price in USD"},
}


def shop_tools(catalog: Catalog) -> list[Tool]:
    def search(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        query = str(args.get("query", "")).strip()
        if not query:
            return ToolResult("Give a non-empty query.")
        ctx.queries.append(query)  # 查询向量进共用的向量缓存：trace 记下它，遗忘时一起删
        found = catalog.search(
            query, category=_text(args.get("category")), max_price=_number(args.get("max_price"))
        )
        return _result(catalog, found)

    def similar(args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        label = str(args.get("image", "")).strip()
        path = ctx.image_path(label)
        if path is None or not path.exists():
            return ToolResult(
                f"No picture called {label!r} here. Use u1, u2… for pictures the user sent in "
                "this conversation, or i1, i2… for images in the recalled memories."
            )
        found = catalog.similar(
            path, category=_text(args.get("category")), max_price=_number(args.get("max_price"))
        )
        if not found:
            return ToolResult("Searching by picture is not available.")
        return _result(catalog, found)

    return [
        Tool(
            "search_products",
            "Search the store catalog by description. Returns products with price, colors, "
            "materials and sizes.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what the user is looking for"},
                    **FILTERS,
                },
                "required": ["query"],
            },
            search,
        ),
        Tool(
            "find_similar_products",
            "Find catalog products that look like a picture: u1, u2… are pictures the user sent "
            "in this conversation; i1, i2… are images in the recalled memories.",
            {
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "picture id, e.g. u1 or i1"},
                    **FILTERS,
                },
                "required": ["image"],
            },
            similar,
        ),
    ]


def _result(catalog: Catalog, found: list[tuple[Product, float]]) -> ToolResult:
    if not found:
        return ToolResult("No matching products.")
    lines = [p.line(score) for p, score in found]
    cards = [p.card(catalog.root, score) for p, score in found]
    return ToolResult(f"{len(found)} products:\n" + "\n".join(lines), {"products": cards})


def _text(value: Any) -> str | None:
    return str(value).strip() or None if value is not None else None


def _number(value: Any) -> float | None:
    try:
        return float(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None
