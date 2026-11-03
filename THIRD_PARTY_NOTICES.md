# Third-party notices

CueMem's own code and documentation are released under the MIT License (see [LICENSE](LICENSE)). The repository also contains, or is evaluated on, the third-party material listed below. The notices that their licences require are reproduced here.

## Included in this repository

### LongMemEval judge template

- **Where:**
  - `src/cuemem/evals/judge.py`: the answer-check prompts, copied verbatim from `src/evaluation/evaluate_qa.py` (`get_anscheck_prompt`).
  - `tests/fixtures/longmemeval_official.py`: a copy of that function, reformatted by ruff (double quotes in four conditions, one trailing space removed); the template strings and syntax tree are unchanged. A regression test uses it.
- **Source:** <https://github.com/xiaowu0162/LongMemEval>
- **Licence:**

```text
MIT License

Copyright (c) 2024 Di Wu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### MemLens judge prompts

- **Where:** `src/cuemem/evals/memlens_prompts/`: the prompt modules of upstream `judge_prompts/` (imported by the official `llm_judge.py`), byte-identical; `__init__.py` differs only by a header docstring CueMem added.
- **Source:** <https://github.com/xrenaf/MEMLENS/tree/77f3ab9a52fa2d6a17978e2dffe80438a4ecced2/judge_prompts>
- **Licence:**
  - MIT (upstream `LICENSE-CODE`), Copyright (c) 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors. The full text is in [src/cuemem/evals/memlens_prompts/LICENSE](src/cuemem/evals/memlens_prompts/LICENSE).
  - The upstream README's Licenses section lists "prompt templates, judge artefacts" as part of the MemLens dataset under CC BY 4.0. To the extent that covers these prompts: MemLens, Ren et al. 2026 (arXiv:2605.14906), <https://github.com/xrenaf/MEMLENS>, licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>), unmodified, provided as is without warranties.

### A-MEM keyword prompt

- **Where:** `src/cuemem/evals/baselines/amem.py`, `QUERY_PROMPT` (from `generate_query_llm` in `test_advanced.py`; indentation and one trailing space removed).
- **Source:** <https://github.com/WujiangXu/AgenticMemory> (commit f9a173d)
- **Licence:**

```text
MIT License

Copyright (c) 2025 Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets (not redistributed)

`cuemem data download` fetches the evaluation datasets from their publishers. They are not part of this repository and are not covered by its MIT License.

| Dataset | Licence |
|---|---|
| LongMemEval_S (cleaned) | MIT |
| LoCoMo | CC BY-NC 4.0 |
| MemLens | CC BY 4.0; images remain under their source sites' licences |
| databricks-dolly-15k | CC BY-SA 3.0 |
| WildChat-1M (one shard, optional, N11) | ODC-BY |

The repository does contain some files derived from these datasets.

- **`evals/splits/`**: question and conversation IDs that define the development and test splits.
- **`evals/mixed/dolly_lines*.json`**: line numbers into databricks-dolly-15k.
- **`evals/mixed/wildchat_test.json`**: conversation hashes of WildChat-1M (allenai/WildChat-1M, ODC-BY). The messages themselves are not stored; `cuemem data download wildchat` fetches the shard, and `scripts/n11_wildchat.py` reads the first user turns from it.
- **`evals/mixed/topic_negatives*.jsonl`** (not covered by CueMem's MIT License):
  - one question per benchmark question (dev 585: LoCoMo 385, LongMemEval 100, MemLens 100; test 1,750: 1,155 / 400 / 195), written by qwen3.8-flash from that question; most are new text, but 4 test rows repeat their source question verbatim (LoCoMo conv-43-q71, conv-47-q36; LongMemEval gpt4_1a1dc16d, e493bb7c) and about 40 closely paraphrase it (MemLens rows not checked);
  - LoCoMo-derived rows: derived from LoCoMo, Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents" (ACL 2024), <https://github.com/snap-research/locomo> (commit 3eb6f2c), licensed under CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>), provided as is without warranties; modified: rewritten by an LLM as above; non-commercial use only;
  - MemLens-derived rows: derived from MemLens, Ren et al. 2026 (arXiv:2605.14906), <https://huggingface.co/datasets/xiyuRenBill/MEMLENS> (revision afa101a), licensed under CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>), provided as is without warranties; modified: rewritten by an LLM as above;
  - LongMemEval-derived rows: derived from LongMemEval_S (cleaned), <https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned> (revision 98d7416), MIT (notice above).
- **The reports** (`docs/`): these quote a few short benchmark questions and answers to illustrate the error analysis. The datasets are credited where they are used.

## Generated assets

`demo/shop/images/` and `demo/shop/photos/` were generated with qwen-image-3.0 for this project (`scripts/n8_assets.py`, prompts in `demo/shop/catalog.json`). The store, its products and the people in the demo are fictional.
