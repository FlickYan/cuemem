"""一键复现 README 与 docs/results/ 里的数字和图（N9 验收）。

按节点依次执行：评测运行（固定的运行名与方法列表；同名运行断点续跑、请求命中本地缓存时不花钱）、
生成报告、把报告与图复制到 docs/results/，再跑各节点的分析脚本。

在已有缓存的机器上重跑几乎免费，只是重新生成报告与图；在新机器上从零跑，需要数据集
（uv run cuemem data download）与 API Key，全部重跑的名义成本约 $150（各节点的花费见路线图）。

注意：
- N2、N3 的数字用的是 N1 选的向量模型 text-embedding-v4（N4 起换成 qwen3.7-text-embedding），
  这两步会自动切换向量模型（评测记忆空间按向量模型分目录存放，互不干扰）；
- N8 的上下文布局实验（scripts/n8_context.py）关掉了本地缓存、每次真实调用，不在这里重跑；
- 测试集（test400、LoCoMo 后 7 段对话、MemLens 官方 195 题）只在 N9 使用；
- 标了"可选"的步骤还没有真正跑过、没有缓存，默认跳过，加 --with-optional 才执行
  （N9 的大模型回答档 deepseek-v4-pro，名义成本约 $17）。

用法：
    uv run python scripts/reproduce.py --dry-run          # 只列出要执行的命令
    uv run python scripts/reproduce.py --only N9          # 只复现某个节点
    uv run python scripts/reproduce.py                    # 全部（跳过可选步骤）
    uv run python scripts/reproduce.py --only N9 --with-optional
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "eval"
RESULTS = ROOT / "docs" / "results"
V4 = {"CUEMEM_EMBEDDING__MODEL": "text-embedding-v4"}
QFLASH = {"CUEMEM_ROLES__READER__MODEL": "qwen3.8-flash"}
QSMALL = {"CUEMEM_ROLES__READER__MODEL": "qwen3.6-35b-a3b"}
DSPRO = {"CUEMEM_ROLES__READER__MODEL": "deepseek-v4-pro"}
# N7 训练、N8 演示与 N9 冻结评测用的都是 v1 控制器；N11 起默认是 v2
# （src/cuemem/read/controller_model.json）
V1 = {"CUEMEM_RETRIEVAL__CONTROLLER": "controller_model_v1.json"}
NODE_ENV = {"N7": V1, "N8": V1, "N9": V1}
NO_CACHE = {"CUEMEM_LLM_CACHE_MODE": "off"}
# 外部基线 Mem0（N11）：mem0ai 不是依赖，临时装上；spaCy 与 fastembed 让它的混合检索完整生效
SPACY_SM = (
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
MEM0 = ["uv", "run", "--with", "mem0ai", "--with", "spacy", "--with", "fastembed",
        "--with", SPACY_SM]  # fmt: skip


@dataclass
class Step:
    node: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    copy: tuple[str, str] | None = None  # （运行名，docs/results 下的目录）：复制报告与图
    optional: bool = False  # 没有缓存、要真花钱的步骤：默认跳过

    def describe(self) -> str:
        env = " ".join(f"{k}={v}" for k, v in {**NODE_ENV.get(self.node, {}), **self.env}.items())
        return f"[{self.node}] {env + ' ' if env else ''}{' '.join(self.command)}"


def run(
    node: str,
    name: str,
    dataset: str,
    split: str,
    methods: str,
    *,
    limit: int | None = None,
    env: dict[str, str] | None = None,
    copy: str | None = None,
    workers: int = 8,
    optional: bool = False,
) -> Step:
    command = [
        "uv", "run", "cuemem", "eval", "run",
        "--dataset", dataset, "--split", split, "--methods", methods,
        "--name", name, "--workers", str(workers), "--max-cost", "60",
    ]  # fmt: skip
    if limit is not None:
        command += ["--limit", str(limit)]
    return Step(node, command, env or {}, (name, copy) if copy else None, optional)


def probe(node: str, name: str, dataset: str, split: str, *, limit: int | None = None) -> Step:
    command = ["uv", "run", "cuemem", "eval", "probe", "--dataset", dataset, "--split", split]
    command += ["--name", name]
    if limit is not None:
        command += ["--limit", str(limit)]
    return Step(node, command)


def script(node: str, path: str, *args: str) -> Step:
    return Step(node, ["uv", "run", "python", path, *args])


N7_LME = "B0,B1,B3,B4-nobm25,B4-note,B5,B6,B6h,B6r,B6z,L0,ORACLE"
N7_LOCOMO = "B0,B3,B4-nobm25,B4-note,B5,B6,B6h,B6r,B6z,L0,ORACLE"
N7_MEMLENS = "B0,B4-mm,B4-see,B5,B5-see,B6,B6h,B6z,L0,ORACLE"
N6_MEMLENS = "B0,B1,B4-text,B4-cap,B4-iso,B4-mm,B4-see,B4-novis,B4-see-novis,B5-see,ORACLE"
N6_TEST = "B0,B1,B4-text,B4-mm,B4-see,B4-see-novis,B5-see,ORACLE"
N9_LME = (
    "B0,L0,B1,B2,B2F,B2F-current,B2F-plain,B4-nobm25,B4-note,B5,B5-nonote,B5-noiter,"
    "B6r,B6,B6h,B6z,B3,B6a,ORACLE"
)
N9_LOCOMO = "B0,L0,B1,B2,B4-nobm25,B4-note,B5,B5-ppr,B6r,B6,B6h,B6z,B3,B6a,ORACLE"
N9_MEMLENS32 = "B0,B1,L0,B4-mm,B4-see,B5,B5-see,B6,B6h,B6z,B3,B6a,ORACLE"
N9_MEMLENS128 = "B0,B1,L0,B4-mm,B4-see,B5,B5-see,B6,B6h,B6z,ORACLE"
READER_METHODS = "L0,B4-nobm25,B5,B3,B6a"

STEPS: list[Step] = [
    # N2 基线（text-embedding-v4）
    run("N2", "dev100-baselines", "longmemeval_s", "dev100", "B0,B1,B2,ORACLE", env=V4,
        copy="n2-dev100-baselines"),
    # N3 写入路径（text-embedding-v4）
    run("N3", "dev100-n3", "longmemeval_s", "dev100",
        "B0,B1,B2,B2F,B2F-keys,B2F-current,B2F-plain,B2F-noprofile,ORACLE", env=V4,
        copy="n3-dev100-facts"),
    run("N3", "locomo-3conv-n3", "locomo", "all", "B2,B2F,ORACLE", limit=385, env=V4,
        copy="n3-locomo-3conv"),
    # N4 L1 快速召回
    run("N4", "dev100-n4", "longmemeval_s", "dev100",
        "B0,B1,B2,B2F,B4,B4-k10,B4-k5,B4-b2k,B4-notime,B4-norel,B4-nobm25,ORACLE",
        copy="n4-dev100-l1"),
    run("N4", "locomo-3conv-n4", "locomo", "all",
        "B2,B2F,B4,B4-k10,B4-b2k,B4-notime,B4-norel,B4-nobm25,ORACLE", limit=385,
        copy="n4-locomo-3conv"),
    # N5 L2 深度检索
    run("N5", "dev100-n5", "longmemeval_s", "dev100",
        "B0,B1,B2F,B4,B4-nobm25,B4-b6k,B4-note,B5,B5-noplan,B5-noiter,B5-nonote,B5-ppr,B6r,B3,"
        "ORACLE", copy="n5-dev100-l2"),
    run("N5", "locomo-3conv-n5", "locomo", "all",
        "B2F,B4,B4-nobm25,B4-b6k,B4-note,B5,B5-nonote,B5-ppr,B6r,ORACLE", limit=385,
        copy="n5-locomo-3conv"),
    # N6 多模态
    run("N6", "memlens32-dev100-n6", "memlens_32k", "dev100", N6_MEMLENS,
        copy="n6-memlens/dev100-32k"),
    run("N6", "memlens32-agent195-n6", "memlens_32k", "agent195", N6_TEST,
        copy="n6-memlens/agent195-32k"),
    run("N6", "memlens128-agent195-n6", "memlens_128k", "agent195", N6_TEST,
        copy="n6-memlens/agent195-128k"),
    # N7 检索控制器：全信息标注、探针、混合流、训练与评测、换回答模型
    run("N7", "dev100-n7", "longmemeval_s", "dev100", N7_LME),
    run("N7", "locomo-3conv-n7", "locomo", "all", N7_LOCOMO, limit=385),
    run("N7", "memlens32-dev100-n7", "memlens_32k", "dev100", N7_MEMLENS),
    probe("N7", "dev100-n7", "longmemeval_s", "dev100"),
    probe("N7", "locomo-3conv-n7", "locomo", "all", limit=385),
    probe("N7", "memlens32-dev100-n7", "memlens_32k", "dev100"),
    script("N7", "scripts/n7_mixed.py"),
    script("N7", "scripts/n7_controller.py"),
    run("N7", "dev100-n7-qflash", "longmemeval_s", "dev100", "B3,B4-nobm25,B5,L0", env=QFLASH),
    run("N7", "locomo-3conv-n7-qflash", "locomo", "all", "B3,B4-nobm25,B5,L0", limit=385,
        env=QFLASH),
    script("N7", "scripts/n7_h3.py"),
    # N8 Agent 与演示：剧本回放与核对（只用通义）
    Step("N8", ["uv", "run", "cuemem", "demo", "demo/shop/script.json", "--fresh", "--report",
                "runs/demo/final-full.json"]),
    script("N8", "scripts/n8_summary.py", "runs/demo/final-full.json"),
    # N9 测试集上的最终评测
    run("N9", "test400-n9", "longmemeval_s", "test400", N9_LME),
    run("N9", "locomo-test7-n9", "locomo", "test7", N9_LOCOMO),
    run("N9", "memlens32-test-n9", "memlens_32k", "agent195", N9_MEMLENS32),
    run("N9", "memlens128-test-n9", "memlens_128k", "agent195", N9_MEMLENS128),
    probe("N9", "test400-n9", "longmemeval_s", "test400"),
    probe("N9", "locomo-test7-n9", "locomo", "test7"),
    probe("N9", "memlens32-test-n9", "memlens_32k", "agent195"),
    probe("N9", "memlens128-test-n9", "memlens_128k", "agent195"),
    # qwen3.6-35b-a3b 的调用限额很紧（8 路并发时约三分之一的请求 429），用 2 路
    run("N9", "test100r-qsmall", "longmemeval_s", "test100r", READER_METHODS, env=QSMALL,
        workers=2),
    run("N9", "locomo-test200r-qsmall", "locomo", "test200r", READER_METHODS, env=QSMALL,
        workers=2),
    run("N9", "test100r-qflash", "longmemeval_s", "test100r", READER_METHODS, env=QFLASH),
    run("N9", "locomo-test200r-qflash", "locomo", "test200r", READER_METHODS, env=QFLASH),
    # 大模型一档：DeepSeek 账户余额不足，还没有跑过（报告里记为未跑）
    run("N9", "test100r-dspro", "longmemeval_s", "test100r", READER_METHODS, env=DSPRO,
        optional=True),
    run("N9", "locomo-test200r-dspro", "locomo", "test200r", READER_METHODS, env=DSPRO,
        optional=True),
    script("N9", "scripts/n9_mixed.py"),
    script("N9", "scripts/n9_errors.py", "summarize"),  # 人工错误标注（已提交）的汇总
    script("N9", "scripts/n9_report.py"),  # results.json、图与 tables.md
    # N11：测试集之后的补做（控制器 v2 只在开发集上选，测试集只作事后重放）
    script("N11", "scripts/n11_controller_v2.py"),  # 导出 v2 并写 controller_v2.json
    script("N11", "scripts/n11_latency.py"),  # 离线；服务端那一组要先起 Qdrant（见脚本说明）
    Step("N11", [*MEM0, "cuemem", "eval", "run", "--dataset", "locomo", "--split", "test200r",
                 "--methods", "M0", "--name", "locomo-test200r-mem0", "--workers", "12",
                 "--max-cost", "5"]),
    Step("N11", [*MEM0, "cuemem", "eval", "run", "--dataset", "longmemeval_s", "--split",
                 "test100r", "--methods", "M0", "--name", "test100r-mem0", "--workers", "16",
                 "--max-cost", "5"]),
    script("N11", "scripts/n11_mem0.py"),
    script("N11", "scripts/n11_second_judge.py", "run"),
    Step("N11", ["uv", "run", "cuemem", "data", "download", "wildchat"]),
    Step("N11", ["uv", "run", "--with", "pyarrow", "python", "scripts/n11_wildchat.py", "prepare"]),
    script("N11", "scripts/n11_wildchat.py", "probe"),
    script("N11", "scripts/n11_wildchat.py", "report"),
    # 重复运行：关掉缓存、真实调用，按定义不能从缓存复现
    run("N11", "test400-n11-rerun", "longmemeval_s", "test400", "B4-nobm25,B6",
        env={**NO_CACHE, **V1}, optional=True),
    run("N11", "locomo-test7-n11-rerun", "locomo", "test7", "B4-nobm25", env=NO_CACHE,
        optional=True),
    script("N11", "scripts/n11_variance.py"),
    script("N11", "scripts/n11_audit.py"),  # 导出人工抽检的样本（已存在时不覆盖）
]  # fmt: skip


def copy_report(name: str, target: str) -> None:
    source = RUNS / name
    dest = RESULTS / target
    dest.mkdir(parents=True, exist_ok=True)
    for file in ("report.md", "run.json", "pareto-light.png", "pareto-dark.png"):
        if (source / file).exists():
            shutil.copy2(source / file, dest / file)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="只复现某个节点，如 N9")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-optional", action="store_true", help="也执行标了可选的步骤")
    args = parser.parse_args()
    steps = [s for s in STEPS if args.only is None or s.node == args.only]
    for step in steps:
        if step.optional and not args.with_optional:
            print(f"（跳过可选步骤）{step.describe()}", flush=True)
            continue
        print(step.describe(), flush=True)
        if args.dry_run:
            continue
        env = {**os.environ, **NODE_ENV.get(step.node, {}), **step.env}
        done = subprocess.run(step.command, cwd=ROOT, env=env, check=False)
        if done.returncode != 0:
            sys.exit(f"失败：{step.describe()}")
        if step.copy is not None:
            copy_report(*step.copy)


if __name__ == "__main__":
    main()
