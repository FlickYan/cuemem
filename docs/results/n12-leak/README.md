# N12 结果：LongMemEval 会话 ID 泄漏——修复与影响

N12 查出一处评测泄漏：LongMemEval 证据会话的 ID 都以 `answer_` 开头，而回答模型看到的每条检索证据都标着会话 ID。N2–N11 发表的 LongMemEval 数字里，检索类方法都是在这个条件下跑的。修复之后，在同一批 100 道题上做了三臂消融，量它对数字有多大影响。

原始数字：[summary.json](summary.json)（`scripts/n12_leak.py`，离线汇总 `runs/eval/` 下的运行，不调用 API）；对照的发表数字：[../n11-improvements/mem0.json](../n11-improvements/mem0.json)（`test100r-mem0`）。决策记录：D-067。

三臂：**raw** 是修复之前的条件（原始 ID，发表数字的条件），**anon** 是修复之后（`session_<i>`），**label** 是修复之后、只在证据抬头里放回原始 ID（§3.1）。两个协议：**原协议**是 N9 / N11 的配置（回答 deepseek-flash），**只用通义**的回答模型是 qwen3.8-flash（§3.3）。

## 结论

1. **没有测出泄漏的效应。** 只差证据抬头里会话名的对比（label 对 anon）：原协议合并 +21 / −12（+1.8 个点，p = 0.1628，含 Mem0），只用通义的协议 +10 / −14（−1.0 个点，p = 0.5413）。没有一个方法单独显著；CueMem 的三个方法在两个协议上方向也不一致。每臂只跑一次、每个方法 100 题，一两个点的效应在这个规模上测不出来。
2. **换 ID 本身带来的波动与泄漏同一量级。** 抬头相同、只差存储里的 ID（raw 对 label）：原协议合并 +5 / −14（−2.25 个点，p = 0.0636），原始 ID 一臂反而更低；只用通义的协议是 +8 / −6（+0.5 个点，p = 0.7905）。所以"修复前对修复后"（raw 对 anon）的差不能直接读成泄漏的效应。
3. **CueMem 发表的 LongMemEval 数字没有因泄漏虚高的证据，但 95% 区间仍容许单个方法虚高约 8 个点（L1 +2.0，区间 −3.7 到 +8.1）、L1、L2 与控制器平均约 3–4 个点；这 100 题里只有 5 道拒答题，`_abs` 那部分泄漏几乎测不到。** 原协议修复前对修复后（raw 对 anon，即发表的数字比修复后高多少）：L1 92.0% → 90.0%（+4 / −2，p = 0.6875，差 +2.0 个点，95% 区间 [−3.7, +8.1]）、L2 93.0% → 96.0%（+0 / −3，p = 0.25，−3.0 [−8.5, +1.5]）、控制器 88.0% → 89.0%（+2 / −3，p = 1.0，−1.0 [−6.6, +4.4]），方向不一，都不显著；L1、L2 与控制器逐题取平均是 −0.67 个点 [−4.00, +2.67]；检索类方法（含朴素检索）合并 +8 / −14（−1.5 个点，p = 0.2863）。只差抬头的 label 对 anon：L1 +2.0 [−3.0, +7.5]、L2 −1.0 [−6.9, +4.7]、控制器 +1.0 [−4.4, +6.6]、Mem0 +6.0 [−1.1, +13.4]，三个 CueMem 方法平均 +0.67 [−2.33, +3.67]；只用通义的协议合并 raw 对 anon −0.33 [−3.33, +2.33]、label 对 anon −1.67 [−5.33, +2.00]。区间都在 [summary.json](summary.json) 里（单个方法是 Newcombe 的配对差区间 `ci95`，三个方法的平均 `pooled_cuemem` 是按题 bootstrap，第三轮体检之后补上，D-070）。全上下文不受影响（两臂逐题相同），"检索比全上下文好"的结论不变。
4. **Mem0 发表的 83.0% 可能高了几个点。** 原协议上 Mem0 的 label 对 anon 是 +9 / −3（anon 79.0%，label 85.0%，p = 0.146），是所有方法里最大的，但不显著。修复后（原协议 anon）CueMem L1 90.0%、Mem0 79.0%，逐题 +15 / −4（名义 p = 0.0192）；N11 发表的是 92.0% 对 83.0%（+13 / −4，名义 p = 0.049）。修复后 L1 领先 Mem0 的差距没有缩小（§4.5），但两次都只是名义显著：这一臂里对 Mem0 的 4 个比较一起做 Holm 校正后是 0.0576，N11 那次运行里对 Mem0 的 3 个比较一起校正后是 0.147（[summary-holm.json](summary-holm.json)、[../n11-improvements/mem0-holm.json](../n11-improvements/mem0-holm.json)，D-069）。
5. **审计查出第二处泄漏**：外部基线 Cognee 的会话蒸馏把题号写进了记忆，拒答题的题号带 `_abs`。已修复；没有发表的数字用到它。

## 1. 泄漏

### 1.1 是什么

- LongMemEval_S（`longmemeval_s_cleaned.json`）里 948 个证据会话的 ID 全部以 `answer_` 开头；30 道拒答题的证据会话 ID 还带 `_abs`。其他会话是 `sharegpt_…`、`ultrachat_…` 或一串十六进制。
- `src/cuemem/read/reader.py` 的 `format_memories` 给每条检索证据写一行抬头，最后一项是会话 ID：

  ```text
  [m1] <日期> · <距提问几天> · session answer_<哈希>_<k>
  ```

  回答模型因此能直接看出哪几条证据来自证据会话；拒答题的抬头还写着 `_abs`，等于告诉它"这题该拒答"。
- 修复提交于 2026-10-15（`00f676a`）。

### 1.2 影响了哪些数字

| | 受影响 | 原因 |
|---|---|---|
| 朴素检索 B2；CueMem L1 / L2 / 控制器；模型自己调检索工具的 B3、B6a；Mem0（M0） | 是 | 证据都经过 `format_memories`，抬头带会话 ID |
| 外部基线 Graphiti、LangMem、Cognee、A-MEM（MG / ML / MC / MA） | 会，但没有数字 | 同一个排版；它们只在 LoCoMo 上跑过 |
| 不检索（B0、L0）、全上下文（B1）、上限参照（ORACLE） | 否 | 不显示会话 ID（B1 按 1..n 给会话编号） |
| LoCoMo | 否 | 会话 ID 本来就是按时间编号的 `session_1..N` |
| MemLens | 否 | 原始 ID 是 `sess_<哈希>`，看不出哪个是证据 |

所以 N2–N11 发表的所有 LongMemEval 数字里，检索类方法都是在泄漏条件下跑的。LoCoMo 与 MemLens 的数字不受影响。

## 2. 修复与审计

### 2.1 修复（D-067）

- **加载时改写会话 ID。** LongMemEval 与 MemLens 的会话按在这份历史里的时间顺序改成 `session_<i>`（与 LoCoMo 自带的写法相同）。证据会话、证据发言随之改写，检索指标不变；原始 ID 留在 `EvalSession.source_id`，只用于追溯，不进任何提示。同一份历史里重复出现的原 ID 共用一个新 ID。MemLens 不泄漏，也一并改写；所以修复后重跑 MemLens 也会带上换名的噪声（§3.2），与发表的数字比较时要用 `=1`（`scripts/reproduce.py` 就是这样做的）。
- **两种写法的写入分开存、不混用。** CueMem 的评测空间放在 `<数据集>-anon` 下，Mem0 与外部基线的写入放在 `<数据集>-anon_<历史>` 下。写法记在空间的数据库、`build.json` 与 `run.json` 里：续跑时写法不同直接拒绝，放错地方的空间按记下的写法认出来、重建。原始 ID 沿用修复之前的位置，旧的空间、写入与响应缓存照常命中。
- **两个开关**（环境变量 `CUEMEM_EVAL_RAW_SESSION_IDS`，只影响 LongMemEval / MemLens；设了之后命令行会给出提示）：
  - `=1`：恢复原始 ID，提示与修复之前逐字节相同；
  - `=label`：会话 ID 照常改写、共用改写后的空间，只在证据抬头里显示原始 ID（消融用，见 §3）。
- **历史数字照旧复现。** `scripts/reproduce.py` 的每一步都设 `=1`：文档里修复之前的数字照旧复现，提示与缓存都对得上；修复后的数字要另起运行名重跑。分析已有运行的脚本（`n9_errors`、`n12_baselines`）按 `run.json` 记下的写法加载，不看当前的环境变量。
- **LoCoMo 逐字节不变。** 测试把一次小评测的全部请求与修复之前的提交（67226a8）比对哈希。

### 2.2 审计：还有哪些路会把 ID 交给模型

`tests/test_eval_leak.py`（全部离线）逐项检查：

- 交给记忆系统的对话里没有原始 ID；
- 全上下文、朴素检索、L1、L2、上限参照的全部请求（抽取、对账、规划、回答、判分）里都没有原始 ID；
- 拒答题的题号（空间 ID 里带 `_abs`）只用于目录与缓存的作用域，不进任何请求；
- Mem0 的写入按写法分开存；
- label 消融的回答提示，把抬头里的会话名换回去之后与默认逐字相同。

**第二处泄漏：Cognee（MC）的会话蒸馏。**

- 对用户与助手的对话（LongMemEval），Cognee 照它自己的 `local_ingest` 把每个会话蒸馏出的经验作为文档写回同一个数据集。
- 蒸馏用的会话 ID 原来带着数据集名，也就是题号。经验文档要交给 cognify 的模型抽图谱，Cognee 还按这个 ID 建节点集（`session_learnings:<会话 ID>`，实体的关系里看得到）。
- 拒答题的题号以 `_abs` 结尾，等于把"这题该拒答"写进了记忆。
- 修复：蒸馏的会话 ID 只用会话编号（`s<编号>_session`），早先带题号的蒸馏写入不再复用。
- LoCoMo 不做蒸馏，不受影响；没有发表的数字用到它。

## 3. 消融设计

### 3.1 三臂

同一批 100 道题（LongMemEval test100r，N11 外部基线用的同一批，含 5 道拒答题）。同一个协议里，三臂的回答模型与判分模型相同。

| 臂 | 开关 | 存储、检索与排序 | 证据抬头里的会话名 | 代表 |
|---|---|---|---|---|
| raw | `CUEMEM_EVAL_RAW_SESSION_IDS=1` | 原始 ID 的空间 | 原始 ID（`answer_…`） | 修复之前，即发表数字的条件 |
| anon | 不设（默认） | 改写后的空间 | `session_<i>` | 修复之后 |
| label | `CUEMEM_EVAL_RAW_SESSION_IDS=label` | 与 anon 相同（直接复用） | 原始 ID（`answer_…`） | 只有抬头泄漏 |

### 3.2 三组对比

- **label 对 anon：泄漏本身。** 两臂共用同一份空间，检索结果、排序与提示都相同，只差抬头里的会话名（`tests/test_eval_leak.py` 用脚本化的模型在朴素检索、L1（B4）、L2 上逐字核对过；控制器与 Mem0 没有单独测）。这是干净的比较。
- **raw 对 label：换名的噪声（对照）。** 两臂的抬头相同，差在存储里的 ID：
  - 轮、事实的 ID 是含会话 ID 的哈希，排序遇到并列时按 ID 定先后；
  - 同一会话的几轮时间相同，回答模型看到它们的先后就由哈希决定，检索分数并列时也一样；
  - 换了会话 ID，这些先后跟着变。这部分与泄漏无关，按设计没有固定的偏向，但会算进修复前后的差里。它有多大，决定了 raw 对 anon 能不能读成泄漏。
- **raw 对 anon：修复前对修复后**，两者都有。

全上下文（B1）不显示会话 ID，只跑 raw 与 anon 两臂，当作检查：两臂应逐题相同。

### 3.3 两个协议

- **原协议（orig）**：N9 / N11 发表数字的配置，即默认配置。抽取与规划 qwen3.8-flash，向量 qwen3.7-text-embedding，回答 deepseek-flash，判分 qwen3.8-max，巩固的备用模型 deepseek-flash。
  - 写入放在 `data/eval-spaces-default`：巩固指纹含备用模型。pq 的 LongMemEval 空间抽取模型与向量相同，备用模型却是 qwen3.7-plus，放在一起会被判为过期、重建。
  - 三臂都在同一份代码（`2b276f9`）上跑。raw 一臂是在 `data/eval-spaces-default` 里重新写入、重新回答的，不是 N11 那次运行（见 §4.1 的说明）。
- **只用通义（pq）**：`CUEMEM_PRESET=qwen`，回答 qwen3.8-flash，判分 qwen3.8-max，其余见 [N12 外部基线记录](../n12-baselines/README.md)（D-065）。
  - 写入放在 `data/eval-spaces-qwen`。
  - anon 与 label 在 `2b276f9` 上跑；raw 一臂用修复之前的代码（全上下文 `5bc8fd2`，其余 `4775454`）。按设计它与 `=1` 等价：`src/cuemem/evals/datasets.py` 的模块说明写明 `=1`"恢复原始 ID，与修复之前逐字节相同"；测试只对 LoCoMo 的请求做了逐字节快照，LongMemEval 上测的是 `=1` 恢复原始 ID。

两个协议的回答模型来自两家，可以看泄漏的效应是否依赖回答模型。

方法：全上下文 B1、朴素检索 B2、CueMem L1（B4-nobm25）、L2（B5）、控制器（B6），以及 Mem0 2.2.1（M0）。Mem0 不是每臂都有：原协议只跑了 anon 与 label（raw 的参照是 N11 发表的那次运行），pq 只跑了 raw。

统计：同一题上的精确符号检验，只数一臂答对、另一臂答错的题。"合并"把检索类方法的胜负加起来；这些是同一批题，彼此不独立，合并的 p 值只作参考。

## 4. 结果

"甲 对 乙"一栏是逐题胜负：甲对、乙错的题数 / 甲错、乙对的题数。

### 4.1 原协议（回答 deepseek-flash）

| 方法 | N11 发表（原始 ID） | raw | anon | label | label 对 anon | raw 对 label | raw 对 anon |
|---|---|---|---|---|---|---|---|
| 全上下文 B1 | 80.0% | 81.0% | 81.0% | — | — | — | 0 / 0 |
| 朴素检索 B2 | 86.0% | 82.0% | 86.0% | 87.0% | +4 / −3（p = 1.0） | +0 / −5（p = 0.0625） | +2 / −6（p = 0.2891） |
| CueMem L1 | 92.0% | 92.0% | 90.0% | 92.0% | +3 / −1（p = 0.625） | +1 / −1（p = 1.0） | +4 / −2（p = 0.6875） |
| CueMem L2 | 96.0% | 93.0% | 96.0% | 95.0% | +2 / −3（p = 1.0） | +2 / −4（p = 0.6875） | +0 / −3（p = 0.25） |
| CueMem 控制器 | 92.0% | 88.0% | 89.0% | 90.0% | +3 / −2（p = 1.0） | +2 / −4（p = 0.6875） | +2 / −3（p = 1.0） |
| Mem0 2.2.1 | 83.0% | — | 79.0% | 85.0% | +9 / −3（p = 0.146） | — | — |

- "N11 发表"取自 [mem0.json](../n11-improvements/mem0.json) 的 `test100r-mem0`：Mem0 是 N11 那次运行，其余方法是 N9 测试运行里同一批题的结果。
- N11 发表的数字与 raw 一臂都是原始 ID，两者之间的差（如 B2 86.0% 对 82.0%、控制器 92.0% 对 88.0%）与泄漏无关。raw 一臂是重新回答的：全上下文、朴素检索、L1、L2 的回答调用都没有命中缓存（`runs/eval/orig-lme-raw-cuemem/results.jsonl`）；空间是重新写入的，但写入的 7,869 次模型调用里 7,867 次命中缓存（`ingest.jsonl`），接近回放。控制器的版本不是原因：`controller_model.json`（v3）与发表时的 v1 只在有图的空间里不同，文字数据集上逐题相同（N11 §1.3、§1.6）。差在哪里没有单独查清，见 §6。看泄漏要看同一批运行的三臂。

### 4.2 只用通义（回答 qwen3.8-flash）

| 方法 | raw | anon | label | label 对 anon | raw 对 label | raw 对 anon |
|---|---|---|---|---|---|---|
| 全上下文 B1 | 77.0% | 77.0% | — | — | — | 0 / 0 |
| 朴素检索 B2 | 82.0% | 83.0% | 84.0% | +4 / −3（p = 1.0） | +2 / −4（p = 0.6875） | +3 / −4（p = 1.0） |
| CueMem L1 | 90.0% | 91.0% | 89.0% | +1 / −3（p = 0.625） | +2 / −1（p = 1.0） | +2 / −3（p = 1.0） |
| CueMem L2 | 94.0% | 93.0% | 93.0% | +3 / −3（p = 1.0） | +2 / −1（p = 1.0） | +2 / −1（p = 1.0） |
| CueMem 控制器 | 89.0% | 90.0% | 87.0% | +2 / −5（p = 0.4531） | +2 / −0（p = 0.5） | +2 / −3（p = 1.0） |
| Mem0 2.2.1 | 83.0% | — | — | — | — | — |

N11 §1.4 发表过同一批 100 题上回答模型换成 qwen3.8-flash 的数字（泄漏期，原始 ID）：总是 L1 93.0%、总是 L2 93.0%、控制器（λ = 30）93.0%；这里 raw 一臂是 90.0%、94.0%、89.0%。两者的模型只差巩固的备用模型（N11 那次是默认的 deepseek-flash，pq 是 qwen3.7-plus），但空间不是同一份（pq 的空间是 N12 重新写入的），代码版本也不同，N11 那几个数字又是换回答模型的事后分析。所以两边的差不能读成泄漏的效应。

### 4.3 合并（检索类方法的胜负相加）

| 协议 | 对比 | 计入的方法 | 胜 / 负 | 净值 | p |
|---|---|---|---|---|---|
| 原协议 | label 对 anon（泄漏） | B2、L1、L2、控制器、Mem0 | +21 / −12 | +1.8 个点 | 0.1628 |
| | raw 对 label（换名噪声） | B2、L1、L2、控制器 | +5 / −14 | −2.25 个点 | 0.0636 |
| | raw 对 anon（修复前对修复后） | B2、L1、L2、控制器 | +8 / −14 | −1.5 个点 | 0.2863 |
| 只用通义 | label 对 anon（泄漏） | B2、L1、L2、控制器 | +10 / −14 | −1.0 个点 | 0.5413 |
| | raw 对 label（换名噪声） | B2、L1、L2、控制器 | +8 / −6 | +0.5 个点 | 0.7905 |
| | raw 对 anon（修复前对修复后） | B2、L1、L2、控制器 | +9 / −11 | −0.5 个点 | 0.8238 |

- 净值 = 100 ×（胜 − 负）÷ 计入的题次（每个方法 100 题）。
- 原协议的泄漏效应（+1.8 个点）的绝对值比换名噪声（−2.25 个点）还小；只用通义时泄漏一项是负的。
- 计入的方法以有这组对比为准：Mem0 只在原协议的 label 对 anon 里有两臂。

### 4.4 拒答题

原始 ID 的抬头里，拒答题的证据会话还带 `_abs`。5 道拒答题上，label 并没有比 anon 答得更好：

- 两个协议的 L2 都是 label 少答对 1 道（原协议 anon 100.0% → label 80.0%，只用通义 80.0% → 60.0%）；
- 其余方法 label 与 anon 相同。

### 4.5 CueMem 对 Mem0（原协议，同一臂里逐题比较）

N11 的结论之一是 LongMemEval 上 L1 比 Mem0 好。Mem0 又是泄漏效应最大的方法，所以在每一臂里照 N11 的写法重算了一遍（summary.json 的 `vs_mem0`）。

| 对比 | N11 发表（原始 ID） | 修复后（anon） | label |
|---|---|---|---|
| 朴素检索 对 Mem0 | +8 / −5（p = 0.581） | +12 / −5（p = 0.1435） | +10 / −8（p = 0.8145） |
| CueMem L1 对 Mem0 | +13 / −4（p = 0.049） | +15 / −4（p = 0.0192） | +9 / −2（p = 0.0654） |
| CueMem L2 对 Mem0 | — | +19 / −2（p = 0.0002） | +12 / −2（p = 0.0129） |
| CueMem 控制器 对 Mem0 | +15 / −6（p = 0.078） | +17 / −7（p = 0.0639） | +12 / −7（p = 0.3593） |

- "N11 发表"取自 [mem0.json](../n11-improvements/mem0.json) 的 `test100r-mem0`（N11 没有发表 L2 对 Mem0）。
- 修复后 L1 对 Mem0 的差距没有缩小，原始 p 仍小于 0.05。表里的 p 都没有校正：每一臂里对 Mem0 的 4 个比较做 Holm 校正后（[summary-holm.json](summary-holm.json)），修复后 L1 对 Mem0 是 0.0576，label 一臂与只用通义的 raw 一臂里 L2 对 Mem0（+12 / −2、+14 / −3）是 0.0518、0.0509，都只是名义显著；修复后 L2 对 Mem0（+19 / −2）校正后 0.0009，仍显著。N11 那一列的 L1 对 Mem0 按 N11 那次运行的 3 个比较校正后是 0.147。label 一臂里各组差距都最小，与 Mem0 的 label 对 anon（+9 / −3）比 CueMem 各方法大一致；这些臂间的差本身都不显著。
- 每个配置只跑了一次；N11 那一列的 CueMem 来自 N9 的测试运行，Mem0 来自 N11 的运行，不是同一批运行。
- 只用通义的协议只有 raw 一臂有 Mem0（两边都有泄漏）：L1 对 Mem0 +12 / −5（p = 0.1435），控制器 +14 / −8（p = 0.2863）。

## 5. 对已发表结论的影响

- **CueMem**：修复前后的变化两个方向都有（原协议 L1 92.0% → 90.0%、L2 93.0% → 96.0%、控制器 88.0% → 89.0%），量级与换名噪声相同；只差抬头的 label 对 anon 也都不显著（L1 +3 / −1、L2 +2 / −3、控制器 +3 / −2）。没有证据表明泄漏抬高了 L1、L2 或控制器的数字，但 95% 区间仍容许单个方法虚高约 8 个点（L1 +2.0，区间 −3.7 到 +8.1）、L1、L2 与控制器平均约 3–4 个点；这 100 题里只有 5 道拒答题，`_abs` 那部分泄漏几乎测不到。
- **检索对全上下文**：全上下文两臂逐题相同；修复后原协议 L1 90.0%、L2 96.0%，全上下文 81.0%（只用通义：91.0%、93.0% 对 77.0%）。
- **CueMem 对 Mem0**：N11 的结论是 L1 92.0% 对 Mem0 83.0%（+13 / −4，名义 p = 0.049）、控制器 92.0%（+15 / −6，p = 0.078）。这是受泄漏影响最大的一组对比（Mem0 的 label 对 anon +9 / −3）。修复后（原协议 anon）是 L1 90.0%、控制器 89.0% 对 Mem0 79.0%，逐题 L1 +15 / −4（名义 p = 0.0192）、控制器 +17 / −7（p = 0.0639）：差距没有缩小，修复没有削弱"L1 比 Mem0 好"（§4.5）；但做多重比较校正之后，修复前后这一条都只是名义显著（0.147、0.0576，D-069）。两边都只跑了一次。
- **文档里的历史数字保留原样**：`scripts/reproduce.py` 照旧复现它们，引用时要注明是修复之前的条件。修复之后的 LongMemEval 数字以这里与之后的运行为准。

## 6. 局限

- **控制器是在泄漏期的结果上标定的。** 控制器的各档准确率表与 λ 是按 N7 开发集的运行结果统计和选定的，那些运行的 LongMemEval 部分有泄漏。这里没有重新标定：L1、L2 的泄漏效应测不出来，准确率表的偏差应该也小，但没有直接验证。
- **同一会话内的先后由哈希决定。** `format_memories` 按（时间，ID）排证据；同一会话的几轮时间相同，先后就由含会话 ID 的哈希决定，不是对话里的顺序。这是修复之前就有的，LoCoMo 也一样；这次没有改，它正是 raw 对 label 那部分噪声的来源。
- **拒答题的 `_abs`。** 原始 ID 里拒答题的证据会话带 `_abs`，raw 与 label 两臂的抬头都泄漏了"这是拒答题"。只有 5 道拒答题，§4.4 只能说明没有看到它帮忙。
- **每臂只跑一次。** 100 道题上 1 道题就是 1 个点，单次运行本身也有波动（N11 §5.2 的重复运行）。各方法单独的差都不显著。
- **同一配置重跑，与发表的数字差了 3–4 个点。** 原协议的 raw 一臂与 N11 发表的数字是同一个配置，同样用原始 ID（代码与空间是新的），朴素检索却是 82.0% 对 86.0%、L2 93.0% 对 96.0%、控制器 88.0% 对 92.0%。N11 §5.2 关掉缓存重跑时，准确率变化不超过 0.9 个点（LongMemEval 400 题、LoCoMo 1,155 题，逐题翻转 3–4%）。这里每个方法 100 题，差得更多；回答是重新调用的，写入几乎是缓存回放（§4.1），更具体的原因没有单独查清。所以本文只在同一批运行的三臂之间比较，发表数字与修复后数字之间的差不能读成泄漏的效应。
- **Mem0 的 +6 个点不显著**（+9 / −3，p = 0.146）。原协议没有重跑 Mem0 的 raw 一臂，与发表的 83.0% 比较时混着重跑的差异；只用通义的协议只有 Mem0 的 raw 一臂，看不出它的泄漏效应。
- **合并的检验不独立**：同一批题上几个方法的胜负相加，p 值偏乐观。
- **只测了一部分受影响的方法。** 只在 LongMemEval test100r 上，用两个回答模型（deepseek-flash、qwen3.8-flash）测了 B1、B2、L1、L2、控制器与 Mem0。模型自己调检索工具的 B3、B6a 同样受影响，N9 / N11 换回答模型时小模型（qwen3.6-35b-a3b）那一档也是，都没有做消融。MemLens 的原始 ID 不泄漏，也没有做消融。

## 7. 复现

汇总（离线，不调用 API）：

```bash
uv run python scripts/n12_leak.py   # 读 runs/eval/ 下的运行，写 docs/results/n12-leak/summary.json
```

运行名与臂的对应在 `scripts/n12_leak.py` 的 `ARMS` 里，模块说明写了三臂与两个协议。没有跑过的三个运行（`orig-lme-raw-mem0`、`pq-lme-mem0`、`pq-lme-label-mem0`，§3.3"Mem0 不是每臂都有"）记在 `NOT_RUN` 里，汇总时跳过；`ARMS` 里别的运行缺了结果时，脚本说明缺了哪些、退出码 1，什么都不写（第三轮体检之后，D-070：以前在没有 `runs/` 的克隆里会用空的结果改写发表的 summary.json 与 summary-holm.json）。

原协议：默认配置（不设 `CUEMEM_PRESET`），写入另放一处。回答模型是 DeepSeek，所有运行都加 `--peak wait`（高峰时段暂停）。

```bash
unset CUEMEM_PRESET
export CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-default
LME="--dataset longmemeval_s --split test100r --max-cost 8 --peak wait"
CUEMEM_EVAL_RAW_SESSION_IDS=1 uv run cuemem eval run $LME \
    --methods B1,B2,B4-nobm25,B5,B6 --name orig-lme-raw-cuemem
uv run cuemem eval run $LME --methods B1,B2,B4-nobm25,B5,B6 --name orig-lme-cuemem
CUEMEM_EVAL_RAW_SESSION_IDS=label uv run cuemem eval run $LME \
    --methods B2,B4-nobm25,B5,B6 --name orig-lme-label-cuemem

# Mem0：mem0ai 不是依赖，临时装上。与 scripts/reproduce.py 相同，只是这里固定了版本（N11、N12 用的是 2.2.1）
SPACY_SM="en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
MEM0=(uv run --with mem0ai==2.2.1 --with spacy --with fastembed --with "$SPACY_SM")
"${MEM0[@]}" cuemem eval run $LME --methods M0 --name orig-lme-mem0
CUEMEM_EVAL_RAW_SESSION_IDS=label "${MEM0[@]}" cuemem eval run $LME \
    --methods M0 --name orig-lme-label-mem0
```

只用通义（不调用 DeepSeek，不需要 `--peak`；`MEM0` 见上）：

```bash
export CUEMEM_PRESET=qwen CUEMEM_PATHS__EVAL_SPACES=data/eval-spaces-qwen
LME="--dataset longmemeval_s --split test100r --max-cost 8"
uv run cuemem eval run $LME --methods B1,B2,B4-nobm25,B5,B6 --name pq-lme-cuemem
CUEMEM_EVAL_RAW_SESSION_IDS=label uv run cuemem eval run $LME \
    --methods B2,B4-nobm25,B5,B6 --name pq-lme-label-cuemem
# raw 一臂当时用修复之前的代码跑（B1 在 pq-lme-warm-b1 里）；用修复之后的代码等价于：
CUEMEM_EVAL_RAW_SESSION_IDS=1 uv run cuemem eval run $LME \
    --methods B2,B4-nobm25,B5,B6 --name pq-lme-raw-cuemem
CUEMEM_EVAL_RAW_SESSION_IDS=1 uv run cuemem eval run --dataset longmemeval_s --split test100r \
    --max-cost 6 --methods B1 --name pq-lme-warm-b1   # 当时的花费上限是 6；只是上限，100 题都跑完了
CUEMEM_EVAL_RAW_SESSION_IDS=1 "${MEM0[@]}" cuemem eval run $LME \
    --methods M0 --name pq-lme-raw-mem0
```

同名运行会断点续跑，请求命中本地缓存时不花钱。复现 N2–N11 文档里修复之前的数字用 `scripts/reproduce.py`，它的每一步都已设 `CUEMEM_EVAL_RAW_SESSION_IDS=1`。
