"""N11：离线延迟基准——记忆空间从小到大，在线写入、探针、控制器与 L1 打包各要多久。

不调用任何 API：向量用离线哈希向量器（维度与线上的 qwen3.7-text-embedding 相同，1024），
也不需要 LLM。会话与轮走真实的在线写入；事实与会话摘要按巩固写库的格式直接写入（内容是合成的），
让每一路检索的规模接近真实空间：事实约为轮数的 1.2 倍，每个会话一条摘要。

线上每次查询还要调一次向量接口（实测约 110 ms），延迟表里不含，是本机计算的部分。
并发压测（--load-size 那个规模）里用 110 ms 的等待模拟这次网络调用：等待期间不占 GIL，
多线程能把它重叠起来，这才是线上的形态。

默认的空间没有实体、也没有图片（发表的延迟表就是这样测的）。--entities-per-round 按每轮多少个
写入合成的实体（名字与别名，LongMemEval 约每轮 1 个、LoCoMo 约 0.12 个），--photos 在均匀分布的
会话里附上几张图片（媒体记录与图片向量，以文搜图那一路会跑起来）。用了这两个参数时结果写进
latency-extended.json，不覆盖 latency.json。

结果写进 docs/results/n11-improvements/ 下（按 local / server 各占一组，替换本机这一组）；
--out 换一个目录。复现脚本（scripts/reproduce.py）写到 runs/n11-latency/：延迟随机器而变，
发表的表是在另一台机器上测的。

用法：
    uv run python scripts/n11_latency.py                     # Qdrant 本地模式
    docker run -d --name cuemem-qdrant -p 6333:6333 qdrant/qdrant:v1.19.1
    uv run python scripts/n11_latency.py --qdrant-url http://localhost:6333
    uv run python scripts/n11_latency.py --entities-per-round 0.6 --photos 30
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import tempfile
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from cuemem.config import PathsConfig, Settings, load_settings
from cuemem.core.ids import stable_id
from cuemem.core.models import (
    Conversation,
    Entity,
    EntityType,
    Episode,
    Fact,
    MemoryUnit,
    SessionIn,
    TurnIn,
)
from cuemem.core.timeutil import to_epoch
from cuemem.embed.base import EmbeddingBatch, EmbedKind, TextEmbedder
from cuemem.memory import MemorySystem
from cuemem.read.l1 import L1Config
from cuemem.write import build_rounds, build_turns

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "results" / "n11-improvements"
SIZES = (500, 2_000, 8_000, 20_000)  # 轮数：一年每天聊一次约 3,600 轮
ROUNDS_PER_SESSION = 10
FACTS_PER_SESSION = 12
REPEATS = 40  # 每种操作测多少次（另有 3 次预热不计）
BATCH_SESSIONS = 200  # 建库时一次写入多少个会话
LOAD_SIZE = 2_000  # 并发压测用的空间规模（轮数）
LOAD_THREADS = (1, 4, 16)
LOAD_READS = 64  # 每种并发度共发多少次读
EMBED_DELAY_S = 0.11  # 线上查询向量化的网络耗时（N7 实测约 110 ms）

PEOPLE = ["Ana", "Ben", "Chloe", "Dev", "Emma", "Farid", "Grace", "Hiro", "Iris", "Jon"]
SYLLABLES = ["ka", "lo", "mi", "ren", "sa", "tor", "vel", "wen", "dri", "osh", "pal", "quin"]
ENTITY_TYPES: list[EntityType] = ["person", "pet", "place", "organization", "product", "event"]
PLACES = ["Singapore", "Lisbon", "the office", "the gym", "Tokyo", "the park", "home", "Paris"]
TOPICS = [
    "running shoes",
    "a birthday dinner",
    "the quarterly report",
    "a vegan recipe",
    "my sister's wedding",
    "a camping trip",
    "the new apartment",
    "piano lessons",
    "a job interview",
    "the dentist",
    "a road trip",
    "my budget",
]
VERBS = ["talked about", "planned", "bought", "cancelled", "booked", "finished", "started"]
QUESTIONS = [
    "What did I say about {topic} with {person}?",
    "When did {person} and I go to {place}?",
    "How many times have I mentioned {topic}?",
    "Which happened first, {topic} or the trip to {place}?",
    "What is my current plan for {topic}?",
    "Can you recommend something for {topic}?",
]


def build_settings(directory: Path, qdrant_url: str | None) -> Settings:
    """默认配置换成哈希向量器，数据写进临时目录；不读项目的 .env，所有 Key 都是假的。"""
    os.environ["CUEMEM_ENV_FILE"] = str(directory / "none.env")
    os.environ["DEEPSEEK_API_KEY"] = "fake-deepseek"
    os.environ["DASHSCOPE_API_KEY"] = "fake-dashscope"
    settings = load_settings(env_file=directory / "none.env")
    embedding = settings.embedding.model_copy(update={"backend": "hash", "dim": 1024})
    vector_store = settings.vector_store.model_copy(update={"url": qdrant_url})
    return settings.model_copy(
        update={
            "paths": PathsConfig().resolve_against(directory),
            "embedding": embedding,
            "vector_store": vector_store,
            "peak_policy": "allow",
        }
    )


def sentence(rng: random.Random) -> str:
    return (
        f"{rng.choice(PEOPLE)} and I {rng.choice(VERBS)} {rng.choice(TOPICS)} "
        f"at {rng.choice(PLACES)} on day {rng.randrange(1, 365)}."
    )


def session(rng: random.Random, index: int, start: datetime) -> SessionIn:
    turns = []
    for _ in range(ROUNDS_PER_SESSION):
        turns.append(TurnIn(role="user", text=f"{sentence(rng)} {sentence(rng)}"))
        turns.append(TurnIn(role="assistant", text=f"Got it. {sentence(rng)}"))
    return SessionIn(
        session_id=f"s{index:05d}", started_at=start + timedelta(days=index), turns=turns
    )


def derived(space: str, sess: SessionIn, rng: random.Random) -> tuple[list[Fact], Episode]:
    """一个会话的巩固结果（合成）：12 条事实与 1 条摘要，来源指向会话里真实的轮。"""
    turns = build_turns(space, sess)
    rounds = build_rounds(turns)
    facts = []
    for j in range(FACTS_PER_SESSION):
        source = rounds[j % len(rounds)]
        statement = f"The user {sentence(rng)}"
        facts.append(
            Fact(
                id=stable_id(space, "fact", sess.session_id, str(j)),
                space_id=space,
                subject_id="user",
                subject="User",
                predicate=f"p{rng.randrange(40)}",
                object=rng.choice(TOPICS),
                statement=statement,
                recorded_at=sess.started_at,
                session_id=sess.session_id,
                source_turn_ids=source.source_turn_ids,
            )
        )
    episode = Episode(
        id=stable_id(space, "episode", sess.session_id),
        space_id=space,
        session_id=sess.session_id,
        start_ts=sess.started_at,
        end_ts=sess.started_at,
        summary=f"The user and the assistant discussed {rng.choice(TOPICS)} and {sentence(rng)}",
        turn_ids=[t.id for t in turns],
    )
    return facts, episode


def entities_for(space: str, sess: SessionIn, count: int, rng: random.Random) -> list[Entity]:
    """一个会话里新出现的实体（合成，--entities-per-round）：名 + 随机拼的姓，别名是姓，
    三成还有只叫名字的别名（问题里点名的人就会命中）。"""
    out = []
    for _ in range(count):
        first = rng.choice(PEOPLE)
        last = "".join(rng.choice(SYLLABLES) for _ in range(rng.randrange(2, 4))).capitalize()
        name, etype = f"{first} {last}", rng.choice(ENTITY_TYPES)
        out.append(
            Entity(
                id=stable_id(space, "entity", name.lower(), etype),
                space_id=space,
                name=name,
                type=etype,
                aliases=[last, *([first] if rng.random() < 0.3 else [])],
                description=f"mentioned with {rng.choice(TOPICS)}",
                first_seen=sess.started_at,
                last_seen=sess.started_at,
            )
        )
    return out


def write_derived(
    system: MemorySystem,
    space: str,
    items: list[tuple[list[Fact], Episode]],
    entities: Sequence[Entity] = (),
) -> None:
    units: list[MemoryUnit] = []
    payloads: list[dict[str, Any]] = []
    unique = list({e.id: e for e in entities}.values())  # 重名的合成实体只留一个
    for entity in unique:  # 与巩固一样：实体的描述也进向量索引
        units.append(
            MemoryUnit(
                id=entity.id,
                space_id=space,
                kind="entity",
                session_id="",
                text=entity.descriptor,
                ts=entity.first_seen,
                source_turn_ids=[],
                meta={"name": entity.name, "type": entity.type},
            )
        )
        payloads.append({"kind": "entity"})
    for facts, episode in items:
        for fact in facts:
            units.append(
                MemoryUnit(
                    id=fact.id,
                    space_id=space,
                    kind="fact",
                    session_id=fact.session_id,
                    text=fact.statement,
                    ts=fact.recorded_at,
                    source_turn_ids=fact.source_turn_ids,
                    meta={"subject_id": fact.subject_id, "predicate": fact.predicate},
                )
            )
            payloads.append({"kind": "fact", "current": True, "subject_id": fact.subject_id})
        units.append(
            MemoryUnit(
                id=episode.id,
                space_id=space,
                kind="summary",
                session_id=episode.session_id,
                text=episode.summary,
                ts=episode.start_ts,
                source_turn_ids=episode.turn_ids,
            )
        )
        payloads.append({"kind": "summary"})
    batch = system.embedder.embed([u.text for u in units], "document")
    system.vectors.upsert(
        space,
        [u.id for u in units],
        batch.dense,
        batch.sparse,
        [
            {"session_id": u.session_id, "ts": to_epoch(u.ts), **p}
            for u, p in zip(units, payloads, strict=True)
        ],
    )
    system.db.apply_consolidation(
        facts=[f for facts, _ in items for f in facts],
        entities=unique,
        episodes=[e for _, e in items],
        units=units,
    )


def photo(directory: Path, index: int) -> Path:
    """一张小图片（每张颜色不同，内容哈希不同）。"""
    from PIL import Image

    path = directory / f"photo-{index:04d}.png"
    color = (index * 37 % 256, index * 91 % 256, index * 53 % 256)
    Image.new("RGB", (32, 24), color).save(path)
    return path


def with_photos(sessions: list[SessionIn], photos: int, directory: Path) -> list[SessionIn]:
    """在均匀分布的 photos 个会话的第一句用户发言里附一张图片。"""
    directory.mkdir(parents=True, exist_ok=True)
    out = list(sessions)
    for k in range(min(photos, len(out))):
        i = k * len(out) // photos
        turns = list(out[i].turns)
        turns[0] = turns[0].model_copy(update={"images": [str(photo(directory, k))]})
        out[i] = out[i].model_copy(update={"turns": turns})
    return out


def build_space(
    system: MemorySystem,
    space: str,
    rounds: int,
    seed: int,
    *,
    entities_per_round: float = 0.0,
    photos: int = 0,
    media_dir: Path | None = None,
) -> float:
    """entities_per_round 与 photos 为 0 时，合成的数据与以前逐字相同（发表的延迟表可比）。"""
    rng = random.Random(seed)
    start = datetime(2024, 1, 1, 9)
    sessions = [session(rng, i, start) for i in range(rounds // ROUNDS_PER_SESSION)]
    if photos:
        assert media_dir is not None
        sessions = with_photos(sessions, photos, media_dir / space)
    entity_rng = random.Random(seed + 7)  # 另用一个随机数发生器：不改动事实与摘要
    per_session = entities_per_round * ROUNDS_PER_SESSION
    started = time.perf_counter()
    for i in range(0, len(sessions), BATCH_SESSIONS):
        chunk = sessions[i : i + BATCH_SESSIONS]
        system.ingest(Conversation(space_id=space, sessions=chunk))
        entities = [
            e
            for j, s in enumerate(chunk, start=i)
            for e in entities_for(
                space, s, int((j + 1) * per_session) - int(j * per_session), entity_rng
            )
        ]
        write_derived(system, space, [derived(space, s, rng) for s in chunk], entities)
    return time.perf_counter() - started


def percentiles(samples: list[float]) -> dict[str, float]:
    return {
        "p50": round(float(np.percentile(samples, 50)), 1),
        "p95": round(float(np.percentile(samples, 95)), 1),
    }


def timed(fn: Callable[[int], Any], repeats: int = REPEATS) -> list[float]:
    for i in range(3):
        fn(-1 - i)
    samples = []
    for i in range(repeats):
        started = time.perf_counter()
        fn(i)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def measure(
    system: MemorySystem, space: str, rounds: int, seed: int, trial: int = 0
) -> dict[str, Any]:
    rng = random.Random(seed + 1 + 1000 * trial)
    when = datetime(2024, 1, 1) + timedelta(days=rounds // ROUNDS_PER_SESSION + 1)
    questions = [
        rng.choice(QUESTIONS).format(
            topic=rng.choice(TOPICS), person=rng.choice(PEOPLE), place=rng.choice(PLACES)
        )
        for _ in range(REPEATS + 3)
    ]
    live = SessionIn(session_id=f"live-{trial}", started_at=when, turns=[])
    system.open_session(space, live.session_id)

    def append(i: int) -> None:
        turns = [
            TurnIn(role="user", text=sentence(rng), ts=when),
            TurnIn(role="assistant", text=f"Noted. {sentence(rng)}", ts=when),
        ]
        system.append(space, live.model_copy(update={"turns": turns}), 2 * (i + 3))

    cfg = L1Config()
    parts: dict[str, list[float]] = {"recall": [], "probe": [], "decide": [], "pack": []}
    assert system.learned is not None
    images = system.db.has_media(space)  # 只有 --photos 时为真

    def read(i: int) -> None:
        question = questions[i]
        probe = system.probe.run(space, question, when, cfg)
        started = time.perf_counter()
        system.learned.decide(probe.features, has_images=images)
        decided = time.perf_counter()
        system.l1.pack(space, probe.candidates, cfg)
        packed = time.perf_counter()
        if i >= 0:
            parts["recall"].append(probe.candidates.timings_ms.get("recall", 0.0))
            parts["probe"].append(probe.ms)
            parts["decide"].append((decided - started) * 1000)
            parts["pack"].append((packed - decided) * 1000)

    write = timed(append)
    total = timed(read)
    units = system.vectors.count(space)
    return {
        "rounds": rounds,
        "units": units,
        "append_ms": percentiles(write),
        **{f"{name}_ms": percentiles(values) for name, values in parts.items()},
        "read_ms": percentiles(total),
    }


class DelayedEmbedder:
    """查询向量化前先等一段时间，模拟线上向量接口的网络延迟（sleep 期间释放 GIL）。"""

    def __init__(self, inner: TextEmbedder, delay_s: float) -> None:
        self.inner = inner
        self.delay_s = delay_s
        self.name = inner.name
        self.dim = inner.dim
        self.supports_sparse = inner.supports_sparse

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbeddingBatch:
        if kind == "query":
            time.sleep(self.delay_s)
        return self.inner.embed(texts, kind)


def load(system: MemorySystem, space: str, rounds: int, seed: int) -> list[dict[str, Any]]:
    """同一进程里多个线程同时读（探针 + 控制器 + L1 打包，含模拟的向量接口延迟）。"""
    rng = random.Random(seed + 2)
    when = datetime(2024, 1, 1) + timedelta(days=rounds // ROUNDS_PER_SESSION + 1)
    cfg = L1Config()
    learned = system.learned
    assert learned is not None
    original = system.l1.embedder
    system.l1.embedder = DelayedEmbedder(original, EMBED_DELAY_S)

    images = system.db.has_media(space)

    def read(question: str) -> float:
        started = time.perf_counter()
        probe = system.probe.run(space, question, when, cfg)
        learned.decide(probe.features, has_images=images)
        system.l1.pack(space, probe.candidates, cfg)
        return (time.perf_counter() - started) * 1000

    rows = []
    try:
        for threads in LOAD_THREADS:
            questions = [
                f"{rng.choice(QUESTIONS)} #{i}".format(
                    topic=rng.choice(TOPICS), person=rng.choice(PEOPLE), place=rng.choice(PLACES)
                )
                for i in range(LOAD_READS)
            ]
            with ThreadPoolExecutor(threads) as pool:
                list(pool.map(read, questions[:threads]))  # 预热
                started = time.perf_counter()
                latencies = list(pool.map(read, questions))
                elapsed = time.perf_counter() - started
            rows.append(
                {
                    "threads": threads,
                    "qps": round(LOAD_READS / elapsed, 1),
                    "latency_ms": percentiles(latencies),
                }
            )
    finally:
        system.l1.embedder = original
    return rows


def median_row(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """几次测量取中位数（每个分位数分别取）：服务端跑在 Docker 的虚拟机里，单次波动可达 2 倍。"""
    row = dict(trials[-1])
    for key, value in trials[0].items():
        if isinstance(value, dict):
            row[key] = {q: round(float(np.median([t[key][q] for t in trials])), 1) for q in value}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qdrant-url", help="Qdrant 服务端地址；不填用本地模式")
    parser.add_argument("--sizes", type=int, nargs="*", default=list(SIZES))
    parser.add_argument("--load-size", type=int, default=LOAD_SIZE, help="并发压测的规模；0 = 不做")
    parser.add_argument("--trials", type=int, default=3, help="每个规模测几次，取中位数")
    parser.add_argument(
        "--entities-per-round",
        type=float,
        default=0.0,
        help="每轮写入多少个合成实体（如 0.6）；默认 0，与发表的延迟表一致",
    )
    parser.add_argument("--photos", type=int, default=0, help="每个空间附几张图片（如 30）；默认 0")
    parser.add_argument(
        "--out", type=Path, default=OUT, help="结果写进哪个目录（默认是发表结果的目录）"
    )
    args = parser.parse_args()
    mode = "server" if args.qdrant_url else "local"
    extended = bool(args.entities_per_round or args.photos)
    rows = []
    concurrency: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="cuemem-latency-") as tmp:
        system = MemorySystem.from_settings(build_settings(Path(tmp), args.qdrant_url))
        try:
            for size in args.sizes:
                space = f"bench-{mode}-{size}"
                system.vectors.delete_space(space)
                build_s = build_space(
                    system,
                    space,
                    size,
                    seed=size,
                    entities_per_round=args.entities_per_round,
                    photos=args.photos,
                    media_dir=Path(tmp) / "photos",
                )
                waited = time.perf_counter()
                while not system.vectors.indexed(space):  # 服务端：等后台建完索引再测
                    time.sleep(0.5)
                index_s = time.perf_counter() - waited
                trials = [measure(system, space, size, size, t) for t in range(args.trials)]
                row = {
                    **median_row(trials),
                    "build_s": round(build_s, 1),
                    "index_s": round(index_s, 1),
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                if size == args.load_size:
                    concurrency = load(system, space, size, seed=size)
                    print(json.dumps(concurrency, ensure_ascii=False))
                system.vectors.delete_space(space)
        finally:
            system.close()
    args.out.mkdir(parents=True, exist_ok=True)
    # 带实体或图片的结果另存一个文件：发表的 latency.json 与以前的合成数据可比
    path = args.out / ("latency-extended.json" if extended else "latency.json")
    existing = json.loads(path.read_text("utf-8")) if path.exists() else {}
    existing[mode] = {
        "machine": f"{platform.platform()}, {os.cpu_count()} CPUs",
        "embedder": "hash-1024 (no API call; production adds ~110 ms per query)",
        **(
            {"entities_per_round": args.entities_per_round, "photos": args.photos}
            if extended
            else {}
        ),
        "repeats": REPEATS,
        "trials": args.trials,
        "rows": rows,
        "concurrency": {
            "rounds": args.load_size,
            "reads": LOAD_READS,
            "embed_delay_ms": EMBED_DELAY_S * 1000,
            "rows": concurrency,
        },
    }
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"→ {path}")


if __name__ == "__main__":
    main()
