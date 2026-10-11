# CueMem

**线索驱动、可自我调控的多模态 Agent 长期记忆系统**
*Cue-driven, metacognitive, multimodal long-term memory for LLM agents*

> 进行中：N2 评测框架与三条基线已完成，下一步是 N3 写入路径。进度见下方表格与 [路线图](docs/roadmap.md)。

## 它要解决什么

让 LLM Agent 在跨越数月、上百个会话的交互中**记得住、想得起、记得准**，而且这种能力来自记忆系统本身，不依赖"模型足够聪明"。

- **记得住**：写入时把对话整理成带有效期的事实、实体图和多粒度索引。原始日志不可变，派生记忆可重建、可溯源。
- **想得起**：由一个**解耦、可训练、可度量的检索控制器**决定这一轮要不要想起、想多深（L0 不检索 / L1 快速召回 / L2 深度检索），证据不足时自动升级。
- **记得准**：信息更新时旧事实失效但保留历史；图片在索引时压缩成描述和向量，回答时取回原图再看一眼。

## 为什么不只是把上下文做得更长

上下文窗口还会继续变大，但下面这些问题不会因此消失。它们是**信息管理**问题，不是**容量**问题。

| 问题 | 只靠长上下文 | 记忆系统的做法 |
|---|---|---|
| 成本与延迟 | 每轮重读全部历史 | 只取需要的部分，按需加深检索 |
| 有效上下文远小于名义长度 | 越长越容易漏看、误用（lost in the middle、context rot） | 只把相关证据放进上下文 |
| 知识过期 | 新旧说法并存，靠模型临场判断 | 写入时显式失效，同时保留历史版本 |
| 跨会话、跨设备 | 会话结束即消失 | 持久存储，可共享 |
| 隐私与可控 | 无法查看、修改、删除 | 每条记忆可查看、可删除、可溯源 |
| 多模态 | 图片和视频 token 昂贵，无法常驻 | 压缩索引 + 回答时再感知 |

## 架构

```mermaid
flowchart LR
    Q["用户消息<br/>文本 / 图片"] --> P["记忆探针<br/>无 LLM · 毫秒级"]
    P --> G{"检索控制器"}
    G -- L0 不检索 --> C["上下文组装"]
    G -- L1 --> F["快速召回<br/>混合检索 · 时间过滤 · 重排"]
    G -- L2 --> D["深度检索<br/>查询规划 · 图扩散 · 迭代检查"]
    F -. 证据不足时升级 .-> D
    F --> C
    D --> C
    C --> R["Reader（LLM / VLM）<br/>带引用回答 · 原图再感知"]
    Q --> W["在线写入"]
    W -. 异步 .-> K["巩固<br/>事实 · 实体 · 有效期 · 画像"]
    W --> M[("记忆存储")]
    K --> M
    M --> P
    M --> F
    M --> D
```

## 快速开始

需要 Python 3.12 与 [uv](https://docs.astral.sh/uv/)。模型使用 DeepSeek 与通义千问的 API（OpenAI 兼容接口；通义经 QwenCloud 接入）。

```bash
uv sync                                  # 安装依赖，创建 .venv
uv run cuemem configure                  # 交互式填写 API Key，写入 .env（需在真实终端里运行）
uv run cuemem doctor --live              # 检查配置与各模型的连通性
uv run cuemem ingest examples/toy_conversation.json
uv run cuemem ask "我现在在哪里工作？" --space demo --at 2025-10-10 --show-context
uv run cuemem usage --today              # 调用量与成本（名义成本 / 估算花费）
uv run cuemem data download              # 下载评测数据集（LongMemEval_S、LoCoMo）
uv run cuemem eval run --split pilot10   # 10 题试跑全部基线（约 $0.4）；dev100 为正式开发集
```

没有 Key 时也可以离线体验写入与检索，向量改用哈希近似：在命令前加 `CUEMEM_EMBEDDING__BACKEND=hash`，回答仍然需要 LLM。

开发：

```bash
uv run pytest                            # 离线单元测试，不需要 Key
uv run pytest -m live                    # 联网冒烟测试，需要 Key，费用极低
uv run ruff check . && uv run mypy       # 代码风格与类型检查
```

## 文档

| 文档 | 内容 |
|---|---|
| [设计文档](docs/design.md) | 动机、架构、检索控制器、多模态、评测方案、技术选型 |
| [路线图](docs/roadmap.md) | 实施节点、验收标准、进度 |
| [相关工作](docs/related-work.md) | 与 MemGPT、Mem0、Zep、HippoRAG、Adaptive-RAG、MemLens 等的对比与定位 |
| [设计决策记录](docs/decisions.md) | 每个重要取舍的背景、备选方案与理由 |

## 进度

| 节点 | 内容 | 状态 |
|---|---|---|
| N0 | 项目定位与总体设计 | 已完成 |
| N1 | 工程骨架与基础设施 | 已完成 |
| N2 | 评测框架与基线 | 已完成 |
| N3 | 写入路径：结构化、带有效期的记忆 | 下一步 |
| N4 | L1 快速召回 | 未开始 |
| N5 | L2 深度检索 | 未开始 |
| N6 | 多模态记忆 | 未开始 |
| N7 | 检索控制器（核心） | 未开始 |
| N8 | Agent Runtime 与 Demo | 未开始 |
| N9 | 完整评测与分析 | 未开始 |
| N10 | 包装 | 未开始 |

## 结果

主线是统一评测框架下的**准确率–成本帕累托曲线**。所有方法共用同一个回答模型（deepseek-flash）与同一个判分模型（qwen3.8-max，LongMemEval 官方判分模板）。

### N2 基线：LongMemEval_S 开发集（100 题）

| 方法 | 准确率（95% 置信区间） | 平均输入 tokens | 每题名义成本 |
|---|---|---|---|
| B0 无记忆 | 6%（2–11） | 458 | $0.0002 |
| B1 全上下文 | 72%（63–80） | 105,262 | $0.0317 |
| B2 朴素检索（dense top-10） | 65%（56–74） | 4,830 | $0.0016 |
| ORACLE 只给证据会话（上限参照） | 83%（76–90） | 6,645 | $0.0021 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/results/n2-dev100-baselines/pareto-dark.png">
  <img alt="各方法的准确率与每题名义成本（对数坐标）" src="docs/results/n2-dev100-baselines/pareto-light.png" width="720">
</picture>

三点观察：

1. **长上下文不等于记忆**：B1 能看到全部信息，却比只看证据的 ORACLE 低 11 个百分点（置信区间部分重叠）：关键信息被淹没在约 10 万 tokens 里。
2. **检索很划算，但还不够好**：朴素检索只用 B1 约 1/20 的成本，准确率 65% 对 72%（置信区间重叠）。
3. **错误主要出在检索**：B2 答错的 35 道题里，12 道检索没命中证据，14 道命中了但证据不全或干扰多（ORACLE 能答对），只有 9 道是连 ORACLE 也答错的阅读推理问题。这正是 N3–N5 的改进空间。

完整报告见 [docs/results/n2-dev100-baselines/report.md](docs/results/n2-dev100-baselines/report.md)。复现：`uv run cuemem eval run --split dev100 --name dev100-baselines`，约 15 分钟、约 $4（缓存命中后重跑几乎免费）。

N1 冒烟测试里还有一个典型失败：问"我第一次提到想去旅行是什么时候"，问题中的问法词稀释了关键词"旅行"，相关记忆只排第 7（共 9 条）；单独搜"旅行"时排第 1。查询改写（N5）、关键词检索（N4）与检索控制器（N7）都针对这类问题。
