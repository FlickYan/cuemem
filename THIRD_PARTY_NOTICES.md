# Third-party notices

CueMem's own code and documentation are released under the MIT License (see [LICENSE](LICENSE)). The repository also contains, or is evaluated on, the third-party material listed below. The notices that their licences require are reproduced here.

## Included in this repository

### LongMemEval judge template

- **Where:**
  - `src/cuemem/evals/judge.py`: the answer-check prompts, copied verbatim from `src/evaluation/evaluate_qa.py` (`get_anscheck_prompt`).
  - `tests/fixtures/longmemeval_official.py`: an unmodified copy of that function, used by a regression test.
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

- **Where:** `src/cuemem/evals/memlens_prompts/`, copied verbatim from the official `llm_judge.py` (commit 77f3ab9).
- **Licence:** MIT, Copyright (c) 2026 Xiyu Ren, Zhaowei Wang, and the MemLens Authors. The full text is in [src/cuemem/evals/memlens_prompts/LICENSE](src/cuemem/evals/memlens_prompts/LICENSE).

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
- **`evals/mixed/topic_negatives*.jsonl`**:
  - one general-knowledge question per benchmark question, written by qwen3.8-flash from that question's topic;
  - they do not reproduce benchmark text, but they are derived from the benchmarks' questions;
  - treat them as subject to the corresponding dataset licences (LoCoMo's is non-commercial).
- **The reports** (`docs/`): these quote a few short benchmark questions and answers to illustrate the error analysis. The datasets are credited where they are used.

## Generated assets

`demo/shop/images/` and `demo/shop/photos/` were generated with qwen-image-3.0 for this project (`scripts/n8_assets.py`, prompts in `demo/shop/catalog.json`). The store, its products and the people in the demo are fictional.
