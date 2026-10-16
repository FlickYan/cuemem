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
  （N9 的大模型回答档 deepseek-v4-pro，名义成本约 $17）；
- 这里的 LongMemEval / MemLens 数字都是 N12 修复会话 ID 泄漏之前跑的：证据会话的原始 ID
  （answer_…）出现在回答模型看到的证据抬头里。所有步骤都设了 CUEMEM_EVAL_RAW_SESSION_IDS=1
  （LEAK_ERA）：提示与当时逐字相同、缓存照常命中，续跑旧运行时会话 ID 的写法也对得上（LoCoMo
  不受影响）。修复后的数字要另起运行名重跑，不在这里。

不会悄悄改动发表的结果：
- 控制器的三个模型文件（v1、v2 与默认的 v3）是冻结评测用的：训练脚本只重新评测、核对
  （--no-export --check），重新训练的结果与仓库里的文件对不上时整个脚本停下，不覆盖；
- 评测运行都加了 --fail-on-error：达到花费上限提前停止、或者有组合出错时退出码非零，脚本停下，
  不把不完整的报告复制到 docs/results/。不复制报告的步骤里，出错的组合与这一步之前完全相同时
  照常继续：那是发表时就有、每次重跑都一样的出错（例如被内容审核拒绝的题）；
- 续跑没有新答的题时不复制 run.json（续跑只在里面追加时间与环境）；
- --cache-only 让每一步的 LLM 请求都只读本地缓存（CUEMEM_LLM_CACHE_MODE=read_only）：未命中
  即报错、不调用 API，要重新算的请求有一个字节变了，也会在复制报告之前停下（向量请求照常，
  未命中时花费很小）。关掉缓存的步骤（按定义不能从缓存复现）跳过；N8 的演示回放未命中时只跳过
  N8（Agent 的请求随产品改动，不是发表的评测数字）。跑完用 git diff docs/results src/cuemem/read
  核对结果。

用法：
    uv run python scripts/reproduce.py --dry-run          # 只列出要执行的命令
    uv run python scripts/reproduce.py --only N9          # 只复现某个节点
    uv run python scripts/reproduce.py                    # 全部（跳过可选步骤）
    uv run python scripts/reproduce.py --only N9 --with-optional
    uv run python scripts/reproduce.py --cache-only       # LLM 只读本地缓存，不调用 API
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from cuemem.evals.report import pending_errors

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "eval"
RESULTS = ROOT / "docs" / "results"
V4 = {"CUEMEM_EMBEDDING__MODEL": "text-embedding-v4"}
QFLASH = {"CUEMEM_ROLES__READER__MODEL": "qwen3.8-flash"}
QSMALL = {"CUEMEM_ROLES__READER__MODEL": "qwen3.6-35b-a3b"}
DSPRO = {"CUEMEM_ROLES__READER__MODEL": "deepseek-v4-pro"}
# N7 训练、N8 演示与 N9 冻结评测用的都是 v1 控制器；N11 用 v2；之后默认是 v3
# （src/cuemem/read/controller_model.json，D-058）
V1 = {"CUEMEM_RETRIEVAL__CONTROLLER": "controller_model_v1.json"}
V2 = {"CUEMEM_RETRIEVAL__CONTROLLER": "controller_model_v2.json"}
NODE_ENV = {"N7": V1, "N8": V1, "N9": V1, "N11": V2}
NO_CACHE = {"CUEMEM_LLM_CACHE_MODE": "off"}
CACHE_ONLY = {"CUEMEM_LLM_CACHE_MODE": "read_only"}  # --cache-only
# 文档里的数字都是修复会话 ID 泄漏之前的（见模块说明）：每一步都用原始会话 ID
LEAK_ERA = {"CUEMEM_EVAL_RAW_SESSION_IDS": "1"}
# 外部基线 Mem0（N11）：mem0ai 不是依赖，临时装上（N11、N12 用的都是 2.2.1，各版本的算法不同）；
# spaCy 与 fastembed 让它的混合检索完整生效
SPACY_SM = (
    "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)
MEM0 = ["uv", "run", "--with", "mem0ai==2.2.1", "--with", "spacy", "--with", "fastembed",
        "--with", SPACY_SM]  # fmt: skip
# 评测运行有组合出错（退出码 3）或达到花费上限提前停止（4）时退出码非零：脚本停下，不复制
# 不完整的报告（cuemem eval run --fail-on-error）
FAIL_ON_ERROR = "--fail-on-error"
ERRORS_EXIT, BUDGET_EXIT = 3, 4
# 控制器的训练脚本只评测、核对冻结的模型文件，不覆盖（见模块说明）
VERIFY_ONLY = ("--no-export", "--check")
REPORT_FILES = ("report.md", "run.json", "pareto-light.png", "pareto-dark.png")


@dataclass
class Step:
    node: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    copy: tuple[str, str] | None = None  # （运行名，docs/results 下的目录）：复制报告与图
    optional: bool = False  # 没有缓存、要真花钱的步骤：默认跳过
    # --cache-only 时失败（未命中缓存）只跳过这个节点余下的步骤，不让整个脚本停下
    best_effort: bool = False

    def environment(self) -> dict[str, str]:
        """这一步在当前环境之上另外设的环境变量（LEAK_ERA、节点的、这一步自己的）。"""
        return {**LEAK_ERA, **NODE_ENV.get(self.node, {}), **self.env}

    def uncached(self) -> bool:
        """关掉了缓存的步骤：按定义不能从缓存复现。"""
        return self.environment().get("CUEMEM_LLM_CACHE_MODE") == "off"

    def run_name(self) -> str | None:
        """评测运行的运行名（其他步骤为 None）。"""
        if FAIL_ON_ERROR not in self.command:
            return None
        return self.command[self.command.index("--name") + 1]

    def describe(self) -> str:
        env = " ".join(f"{k}={v}" for k, v in self.environment().items() if k not in LEAK_ERA)
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
        "--name", name, "--workers", str(workers), "--max-cost", "60", FAIL_ON_ERROR,
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
    script("N7", "scripts/n7_controller.py", *VERIFY_ONLY),  # 核对 v1，不覆盖
    run("N7", "dev100-n7-qflash", "longmemeval_s", "dev100", "B3,B4-nobm25,B5,L0", env=QFLASH),
    run("N7", "locomo-3conv-n7-qflash", "locomo", "all", "B3,B4-nobm25,B5,L0", limit=385,
        env=QFLASH),
    script("N7", "scripts/n7_h3.py"),
    # N8 Agent 与演示：剧本回放与核对（只用通义）
    Step("N8", ["uv", "run", "cuemem", "demo", "demo/shop/script.json", "--fresh", "--report",
                "runs/demo/final-full.json"], best_effort=True),
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
    # 核对 v2 并写 controller_v2.json；核对 v3（再感知的门槛，D-058）并写 see_gate.json
    script("N11", "scripts/n11_controller_v2.py", *VERIFY_ONLY),
    script("N11", "scripts/n11_see_gate.py", *VERIFY_ONLY),
    script("N11", "scripts/n11_latency.py"),  # 离线；服务端那一组要先起 Qdrant（见脚本说明）
    Step("N11", [*MEM0, "cuemem", "eval", "run", "--dataset", "locomo", "--split", "test200r",
                 "--methods", "M0", "--name", "locomo-test200r-mem0", "--workers", "12",
                 "--max-cost", "5", FAIL_ON_ERROR]),
    Step("N11", [*MEM0, "cuemem", "eval", "run", "--dataset", "longmemeval_s", "--split",
                 "test100r", "--methods", "M0", "--name", "test100r-mem0", "--workers", "16",
                 "--max-cost", "5", FAIL_ON_ERROR]),
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


def answered(name: str) -> int:
    """运行目录里 results.jsonl 的大小：一步前后不变，说明这次续跑没有新答的题。"""
    path = RUNS / name / "results.jsonl"
    return path.stat().st_size if path.exists() else -1


def failed(name: str) -> set[tuple[str, str]]:
    """运行里出错、还没重跑成功的（题目，方法）组合。"""
    return {(e.get("sample_id"), e.get("method")) for e in pending_errors(RUNS / name)}


def copy_report(name: str, target: str, *, run_info: bool = True) -> None:
    """把报告与图复制到 docs/results/。run_info=False（续跑没有新答的题）时留着那里的 run.json：
    续跑只在里面追加了时间与环境。"""
    source = RUNS / name
    dest = RESULTS / target
    dest.mkdir(parents=True, exist_ok=True)
    for file in REPORT_FILES:
        if file == "run.json" and not run_info and (dest / file).exists():
            continue
        if (source / file).exists():
            shutil.copy2(source / file, dest / file)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="只复现某个节点，如 N9")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--with-optional", action="store_true", help="也执行标了可选的步骤")
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="只读本地缓存（CUEMEM_LLM_CACHE_MODE=read_only）：未命中即报错、不调用 API",
    )
    args = parser.parse_args(argv)
    steps = [s for s in STEPS if args.only is None or s.node == args.only]
    leak_era = " ".join(f"{k}={v}" for k, v in LEAK_ERA.items())
    print(f"（每一步都设了 {leak_era}：复现修复会话 ID 泄漏之前的数字）", flush=True)
    extra = CACHE_ONLY if args.cache_only else {}
    if args.cache_only:
        cache_only = " ".join(f"{k}={v}" for k, v in CACHE_ONLY.items())
        print(f"（--cache-only：每一步都设了 {cache_only}，未命中缓存即报错）", flush=True)
    skipped: set[str] = set()  # --cache-only 时演示回放未命中、余下步骤跳过的节点
    for step in steps:
        if step.optional and not args.with_optional:
            print(f"（跳过可选步骤）{step.describe()}", flush=True)
            continue
        if args.cache_only and step.uncached():
            print(f"（关掉缓存的步骤，--cache-only 时跳过）{step.describe()}", flush=True)
            continue
        if step.node in skipped:
            print(f"（{step.node} 的演示回放没有复现，跳过）{step.describe()}", flush=True)
            continue
        print(step.describe(), flush=True)
        if args.dry_run:
            continue
        name = step.run_name()
        before = (answered(name), failed(name)) if name else None
        env = {**os.environ, **step.environment(), **extra}
        code = subprocess.run(step.command, cwd=ROOT, env=env, check=False).returncode
        reported_errors = code == ERRORS_EXIT and name is not None and step.copy is None
        if reported_errors and before and before[1] and failed(str(name)) <= before[1]:
            # 出错的组合与这一步之前完全相同：发表时就有、每次重跑都一样的出错（例如 N9 里被
            # qwen3.8-flash 的内容审核拒绝的那道题，报告按三个回答模型都答了的 99 题比较）
            print("（出错的组合与这一步之前相同：发表时就有的出错，继续）", flush=True)
            code = 0
        if code != 0 and args.cache_only and step.best_effort:
            print(f"（只读缓存下没有复现，跳过 {step.node} 余下的步骤）", flush=True)
            skipped.add(step.node)
            continue
        if code != 0 and reported_errors:
            sys.exit(
                f"失败：{step.describe()} 有组合出错（报错见上，明细在 errors.jsonl）。确认是每次"
                "重跑都一样的出错（如内容审核拒绝）后再运行一次：出错与之前相同的步骤会照常继续"
            )
        if code != 0:
            sys.exit(f"失败（退出码 {code}）：{step.describe()}")
        if step.copy is not None:
            assert before is not None
            copy_report(*step.copy, run_info=answered(step.copy[0]) != before[0])


if __name__ == "__main__":
    main()
