"""判分。

LongMemEval 的判分模板逐字照搬官方脚本 src/evaluation/evaluate_qa.py（get_anscheck_prompt），
调用方式也一致：只发一条 user 消息，temperature=0，max_tokens=10，回复中含 "yes" 即判为正确。
LoCoMo 没有官方的 LLM 判分，沿用同一套模板（D-015），另外按原论文口径计算 token F1 作参考。
MemLens 用官方 llm_judge.py 的提示词（按题型给评分标准与例子，逐字照搬，见 memlens_prompts/），
判分模型输出评分理由与 {"answer_score": 0/1}；超过 500 词的回答按官方做法直接判 0。
官方判分模型是 Qwen3-VL-235B，QwenCloud 上没有，这里用与其他数据集相同的 qwen3.8-max。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from cuemem.evals.datasets import Sample
from cuemem.evals.memlens_prompts import build_judge_prompt, get_task_key
from cuemem.llm.client import LLMClient

MEMLENS_MAX_WORDS = 500  # 官方 llm_judge.py：解析后的回答超过 500 词直接判 0

# ---------- LongMemEval 官方模板（逐字照搬，包括个别句末多出的空格） ----------

_GENERIC = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response is "
    "equivalent to the correct answer or contains all the intermediate steps to get the correct "
    "answer, you should also answer yes. If the response only contains a subset of the "
    "information required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of days/weeks/months, "
    "etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is "
    "18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\n"
    "Model Response: {}\n\nIs the model response correct? Answer yes or no only."
)
_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. Please answer "
    "yes if the response contains the correct answer. Otherwise, answer no. If the response "
    "contains some previous information along with an updated answer, the response should be "
    "considered as correct as long as the updated answer is the required answer.\n\n"
    "Question: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response "
    "from a model. Please answer yes if the response satisfies the desired response. Otherwise, "
    "answer no. The model does not need to reflect all the points in the rubric. The response "
    "is correct as long as it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)
_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. The "
    "model could say that the information is incomplete, or some other information is given "
    "but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\n"
    "Model Response: {}\n\nDoes the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

TEMPLATES = {
    # LongMemEval（官方）
    "single-session-user": _GENERIC,
    "single-session-assistant": _GENERIC,
    "multi-session": _GENERIC,
    "temporal-reasoning": _TEMPORAL,
    "knowledge-update": _KNOWLEDGE_UPDATE,
    "single-session-preference": _PREFERENCE,
    # LoCoMo（沿用 LongMemEval 的模板）
    "single-hop": _GENERIC,
    "multi-hop": _GENERIC,
    "open-domain": _GENERIC,
    "temporal": _TEMPORAL,
}


def judge_prompt(sample: Sample, response: str) -> str:
    template = _ABSTENTION if sample.is_abstention else TEMPLATES[sample.question_type]
    return template.format(sample.question, sample.answer, response)


@dataclass(frozen=True)
class Verdict:
    correct: bool
    raw: str


class Judge:
    def __init__(self, llm: LLMClient, role: str = "judge") -> None:
        self.llm = llm
        self.role = role

    def judge(self, sample: Sample, response: str) -> Verdict:
        if sample.dataset.startswith("memlens"):
            return self._judge_memlens(sample, response)
        result = self.llm.chat(
            self.role,
            [{"role": "user", "content": judge_prompt(sample, response)}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (result.content or "").strip()
        return Verdict(correct="yes" in raw.lower(), raw=raw)

    def _judge_memlens(self, sample: Sample, response: str) -> Verdict:
        if len(response.split()) > MEMLENS_MAX_WORDS:
            return Verdict(correct=False, raw=f"auto_zero: more than {MEMLENS_MAX_WORDS} words")
        result = self.llm.chat(
            self.role,
            [{"role": "user", "content": memlens_prompt(sample, response)}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = (result.content or "").strip()
        return Verdict(correct=parse_answer_score(raw) == 1, raw=raw[-400:])


def memlens_prompt(sample: Sample, response: str) -> str:
    task = get_task_key(
        sample.question_type, str(sample.meta.get("question_subtype", "")), sample.answer
    )
    old_answer = str(sample.meta.get("old_answer") or "")  # 空串与 None 在官方代码里等价
    return str(build_judge_prompt(task, sample.question, sample.answer, response, old_answer))


def parse_answer_score(text: str) -> int:
    """与官方 llm_judge.py 的解析顺序一致：JSON 里的 answer_score → "deserves x point" →
    "[Score]: x"，都没有时判 0。"""
    for pattern in (
        r'"answer_score"\s*:\s*(\d+)',
        r"deserves\s+(\d+)\s+point",
        r"\[Score\]\s*:\s*(\d+)",
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return min(int(match.group(1)), 1)
    return 0


# ---------- token F1（LoCoMo 原论文口径，SQuAD 风格的归一化） ----------

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = _ARTICLES.sub(" ", text)
    return text.split()


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = _normalize(prediction), _normalize(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)
