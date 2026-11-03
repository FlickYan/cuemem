"""Gradio 界面（N8）：对话 + 记忆轨迹面板 + 记忆浏览器。

- 对话：文字与图片；每一轮右侧展示"记忆轨迹"——检索控制器选了哪一档、为什么、各档的预期
  准确率与效用，上下文各分区的 token，召回并被引用的记忆，工具调用，用量与耗时；
- 记忆：用户画像、事实时间线（含被取代的旧版本）、实体图、图库、会话，以及遗忘；
- 原理：一页说明。

界面文字用英文；与接口共用服务对象（api/service.py）。默认不联系第三方：字体是 Gradio 自带的，
Gradio 的使用统计与版本检查要设 GRADIO_ANALYTICS_ENABLED=True 才打开（第四轮体检）。

访问令牌（api/auth.py）：用限定作用域的令牌登录时，每个事件（经界面或 /gradio_api 直接调用）都
先检查记忆空间在不在它的作用域里，不在时报错、什么都不读不写；页面打开时，默认的记忆空间不在
作用域里就换成作用域里的第一个。
"""

from __future__ import annotations

import base64
import html
import io
import math
import os
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, cast, get_args

import gradio as gr
from PIL import Image

from cuemem.agent import Reply
from cuemem.agent.runtime import RecalledItem, TurnTrace
from cuemem.api.auth import ADMIN, Principal
from cuemem.api.service import ChatService
from cuemem.memory import RecallMode
from cuemem.read.controller import Router, english_reason
from cuemem.store.media import decoding

# 字体用 Gradio 自带的 IBM Plex Sans（三种粗细的文件都在包里，由本服务提供）。以前是
# GoogleFont("Inter")：Gradio 没带它，登录之后每个访客的浏览器都去 fonts.googleapis.com 取，访客的
# IP 发给了 Google（第四轮体检）。等宽字体 Soft 本来就用自带的 IBM Plex Mono
THEME = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate",
    font=[
        gr.themes.LocalFont("IBM Plex Sans", weights=(400, 600, 700)),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

CSS = """
.cm-header h1 {margin: 0; font-size: 1.6rem;}
.cm-header p {margin: 2px 0 0; color: var(--body-text-color-subdued);}
.trace {font-size: 13px; line-height: 1.45;}
.trace h4 {margin: 14px 0 6px; font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--body-text-color-subdued);}
.trace .row {display: flex; gap: 8px; align-items: center; flex-wrap: wrap;}
.badge {display: inline-block; padding: 2px 9px; border-radius: 999px; font-weight: 600;
  font-size: 12px;}
.lvl-L0 {background: rgba(100,116,139,.18); color: var(--body-text-color);}
.lvl-L1 {background: rgba(20,184,166,.20); color: var(--body-text-color);}
.lvl-L2 {background: rgba(99,102,241,.22); color: var(--body-text-color);}
.lvl-see {background: rgba(245,158,11,.25); color: var(--body-text-color);}
.muted {color: var(--body-text-color-subdued);}
.mem {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 6px 9px;
  margin: 5px 0;}
.mem.cited {border-color: #14b8a6; box-shadow: inset 3px 0 0 #14b8a6;}
.mem .ref {font-family: var(--font-mono); font-weight: 700; margin-right: 6px;}
.mem .text {margin-top: 2px; white-space: pre-wrap;}
.mem img {width: 72px; height: 72px; object-fit: cover; border-radius: 6px; float: left;
  margin: 2px 9px 2px 0;}
.mem::after {content: ""; display: block; clear: both;}
.stack {display: flex; height: 12px; border-radius: 6px; overflow: hidden; margin: 4px 0 6px;
  background: var(--background-fill-secondary);}
.stack span {display: block; height: 100%;}
.legend {display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px;}
.legend i {display: inline-block; width: 9px; height: 9px; border-radius: 2px;
  margin-right: 4px;}
.util {width: 100%; border-collapse: collapse; font-size: 12px; border: none !important;}
.util td {padding: 2px 6px; border: none !important;
  border-bottom: 1px solid var(--border-color-primary) !important;}
.trace details summary {cursor: pointer; margin: 6px 0;}
.util .bar {height: 8px; border-radius: 4px; background: #14b8a6;}
.util tr.chosen td {font-weight: 700;}
.tool {font-family: var(--font-mono); font-size: 12px; padding: 4px 0;}
sup.cite {color: #0d9488; font-weight: 600; font-size: .72em; margin-left: 1px;}
.cm-graph svg {width: 100%; height: auto;}
"""

PARTITION_COLORS = {
    "system": "#94a3b8",
    "tools": "#cbd5e1",
    "core": "#f59e0b",
    "working": "#3b82f6",
    "recalled": "#14b8a6",
    "message": "#8b5cf6",
}
PARTITION_NAMES = {
    "system": "system prompt",
    "tools": "tool specs",
    "core": "profile",
    "working": "this chat",
    "recalled": "recalled",
    "message": "message",
}
ACTION_EN = {"L0": "L0 profile only", "L1": "L1 fast recall", "L2": "L2 deep recall"}
TYPE_COLORS = {
    "person": "#f59e0b",
    "pet": "#ec4899",
    "place": "#3b82f6",
    "organization": "#8b5cf6",
    "product": "#14b8a6",
    "event": "#ef4444",
    "other": "#64748b",
}
DATE_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d")
CITATION = re.compile(r"\[\s*([a-z]\d+(?:\s*[,，;；]\s*[a-z]\d+)*)\s*\]")
EMPTY_TRACE = (
    '<div class="trace muted"><h4>Memory trace</h4>Send a message: this panel shows how the '
    "memory controller decided how much to recall, what went into the context, which memories "
    "the answer cites, and what it cost.</div>"
)

HOW_IT_WORKS = """
### What happens on every message

1. **Probe** (no LLM, ~50–250 ms): one hybrid retrieval pass over long-term memory. Its scores,
   plus cues in the message ("I showed you", "last month", "how many"), become 54 features.
2. **Controller**: a small two-stage model (trained offline on 585 labelled questions) estimates
   whether this message needs memory at all and what kind of question it is, then picks the
   action with the best *expected accuracy − λ × cost*:
   **L0** profile only · **L1** fast recall · **L2** deep recall (query planning, follow-up
   searches) · **+see** re-attach original photos.
3. **Context assembly** with a token budget per partition. The system prompt, the profile
   snapshot and this chat's recent turns form a *stable prefix* (cache-friendly); recalled
   memories go only into the latest message and are rebuilt every turn. Turns already in the
   chat are excluded from recall.
4. **Answer** with tools: `memory_search` is the model's escape hatch when the controller missed
   something; shop tools search the catalog by text or by picture. Answers cite memories as
   [f2] / [m3].
5. **Write-back**: each exchange (with photos) is logged and indexed right away, without an LLM.
   When the session ends, a background job extracts facts with validity periods, resolves
   updates ("my budget is now $180" supersedes "$120"), describes images and refreshes the
   profile.

**Forgetting** deletes the raw turns a fact or photo came from, then rebuilds everything derived
from the remaining log, including caches.

The store and products are made up for the demo; product photos are AI-generated.
"""


def build_ui(
    service: ChatService, *, access: Callable[[str | None], Principal | None] | None = None
) -> gr.Blocks:
    """access：登录的用户名 → 令牌（api.auth.Keys.by_username；None 是没登录）。不给时不限制
    （没有访问令牌的本机演示）。"""
    lam_default = _lam_default(service)

    def who(request: gr.Request | None) -> Principal:
        if access is None:
            return ADMIN
        principal = access(request.username if request is not None else None)
        if principal is None:
            raise gr.Error("Sign in first.")
        return principal

    def allowed(request: gr.Request | None, space_id: str) -> str:
        """界面里填的记忆空间（空着是默认空间），先过令牌的作用域检查。"""
        space = space_id.strip() or service.default_space
        if not who(request).allows(space):
            raise gr.Error(f"Your access token cannot open the memory space {space!r}.")
        return space

    def mine(request: gr.Request | None, sid: str | None) -> str | None:
        """这个页面的会话：属于作用域之外的空间时当作没有（会话 ID 只在服务端的页面状态里，
        这里再查一遍）。"""
        if not sid:
            return None
        try:
            space = service.get(sid).space_id
        except KeyError:
            return sid  # 已经结束：照常由 _end、service.chat 处理
        return sid if who(request).allows(space) else None

    # Gradio 把上传的照片、对话与图库里显示的图都复制进自己的临时目录，遗忘时没法删到：每小时清掉
    # 一小时以前的副本（会话读的是记忆里存的那份，不受影响；更早的对话里的图不再显示）。
    # 使用统计默认关：gr.Blocks 一建好（挂载、从不 launch 的也一样）就向 api.gradio.app 查版本、经
    # huggingface_hub 向 huggingface.co 报告；运营者设 GRADIO_ANALYTICS_ENABLED=True（.env 里也行，
    # 这时已经读过）才打开
    analytics = os.environ.get("GRADIO_ANALYTICS_ENABLED") == "True"
    with gr.Blocks(
        title="CueMem demo", delete_cache=(3600, 3600), analytics_enabled=analytics
    ) as demo:
        session_id = gr.State(None)
        gr.HTML(
            '<div class="cm-header"><h1>CueMem</h1><p>A shopping assistant with long-term '
            "memory, and a window into what it remembers and why.</p></div>"
        )
        with gr.Row(equal_height=True):
            space = gr.Textbox(service.default_space, label="Memory space", scale=2)
            when = gr.Textbox(
                label="Session date (simulated)",
                placeholder="e.g. 2026-10-10 10:00 · blank = now",
                scale=2,
            )
            new_button = gr.Button("New session", variant="primary", scale=1)
            end_button = gr.Button("End session", scale=1)
        status = gr.Markdown("No session yet: sending a message starts one.")
        with gr.Accordion("Memory controller settings", open=False), gr.Row():
            mode = gr.Radio(
                ["auto", "l0", "l1", "l2"],
                value="auto",
                label="Recall",
                info="auto = the controller decides; l0/l1/l2 force a level",
            )
            router = gr.Radio(
                list(get_args(Router)),
                value=service.agent.router,
                label="Controller (auto)",
                info="learned: probe + trained model; hybrid: ask an LLM when unsure",
            )
            lam = gr.Slider(
                0, 100, value=lam_default, step=5, label="λ cost knob",
                info="higher = cheaper; lower = recall more",
            )  # fmt: skip

        with gr.Tabs():
            with gr.Tab("Chat"), gr.Row():
                with gr.Column(scale=6):
                    chatbot = gr.Chatbot(
                        height=620,
                        show_label=False,
                        placeholder="Say hi to Cue, or share a photo of something you like.",
                        group_consecutive_messages=False,
                    )
                    box = gr.MultimodalTextbox(
                        file_types=["image"],
                        file_count="multiple",
                        placeholder="Message Cue… (you can attach photos)",
                        show_label=False,
                        sources=["upload"],
                    )
                with gr.Column(scale=5):
                    trace = gr.HTML(EMPTY_TRACE)
            with gr.Tab("Memory") as memory_tab:
                with gr.Row():
                    refresh = gr.Button("Refresh", scale=0)
                    current_only = gr.Checkbox(label="Current facts only", value=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        profile = gr.Markdown()
                    with gr.Column(scale=3):
                        graph = gr.HTML(elem_classes=["cm-graph"])
                facts = gr.Dataframe(
                    headers=["Said on", "Fact", "Status", "Replaced by"],
                    interactive=False,
                    wrap=True,
                    label="Fact timeline",
                    column_widths=["12%", "44%", "16%", "28%"],
                )
                gallery = gr.Gallery(
                    label="Images in memory",
                    columns=6,
                    height=230,
                    preview=False,
                    object_fit="cover",
                    fit_columns=False,
                )
                sessions = gr.Dataframe(
                    headers=["Session", "Started", "Turns", "State"],
                    interactive=False,
                    label="Sessions",
                )
                with gr.Group():
                    with gr.Row(equal_height=True):
                        forget_kind = gr.Radio(
                            ["fact", "image", "session"], value="fact", label="Forget a", scale=1
                        )
                        forget_target = gr.Dropdown(label="Which one", choices=[], scale=4)
                        forget_button = gr.Button("Forget", variant="stop", scale=1)
                    gr.Markdown(
                        "Forgetting deletes the exchange the item came from and any later "
                        "reply that cited it, then rebuilds facts, profile and indexes from "
                        "what is left.",
                        elem_classes=["muted"],
                    )
                forget_result = gr.Markdown()
            with gr.Tab("How it works"):
                gr.Markdown(HOW_IT_WORKS)

        # ---------- 事件 ----------

        def start(
            space_id: str, date: str, sid: str | None, request: gr.Request
        ) -> tuple[Any, ...]:
            space = allowed(request, space_id)
            notice = ""
            if sid := mine(request, sid):
                notice = _end(service, sid)
            session = service.start(space, at=_parse_date(date))
            return (
                session.session_id,
                [],
                EMPTY_TRACE,
                _status_text(service, session.session_id, notice),
            )

        new_button.click(start, [space, when, session_id], [session_id, chatbot, trace, status])

        def end(sid: str | None, request: gr.Request) -> tuple[Any, ...]:
            sid = mine(request, sid)
            if not sid:
                return None, "No session to end."
            return None, _end(service, sid)

        end_button.click(end, [session_id], [session_id, status])

        def send(
            message: dict[str, Any] | None,
            history: list[dict[str, Any]],
            sid: str | None,
            space_id: str,
            date: str,
            recall_mode: str,
            controller: str,
            lam_value: float,
            request: gr.Request,
        ) -> Iterator[tuple[Any, ...]]:
            space = allowed(request, space_id)  # 还没有会话时在这里开
            sid = mine(request, sid)
            text = str((message or {}).get("text") or "").strip()
            files = [_file_path(f) for f in (message or {}).get("files") or []]
            if not text and not files:
                yield history, gr.MultimodalTextbox(value=None), sid, gr.skip(), gr.skip()
                return
            notice = ""
            if sid and not service.is_open(sid):  # 闲置太久，清理线程结束了它（已经交给抽取）
                sid, notice = None, _ended_notice(service)
            if not sid:
                sid = service.start(space, at=_parse_date(date)).session_id
            user: list[Any] = [text] if text else []
            user += [{"path": f} for f in files]
            history = [*history, {"role": "user", "content": user}]
            yield (
                history,
                gr.MultimodalTextbox(value=None, interactive=False),
                sid,
                '<div class="trace muted"><h4>Memory trace</h4>Thinking…</div>',
                _status_text(service, sid, notice),
            )
            try:
                reply = service.chat(
                    sid,
                    text or "(the user sent a picture)",
                    images=files,
                    mode=cast(RecallMode, recall_mode),
                    router=cast(Router, controller),
                    lam=float(lam_value),
                )
                answer = _assistant(reply)
                panel = render_trace(reply, service)
            except Exception as exc:  # 界面不能崩：把错误显示在对话里
                history = [
                    *history,
                    {"role": "assistant", "content": f"⚠️ {type(exc).__name__}: {exc}"},
                ]
                yield (
                    history,
                    gr.MultimodalTextbox(value=None, interactive=True),
                    sid,
                    EMPTY_TRACE,
                    _status_text(service, sid, notice),
                )
                return
            history = [*history, {"role": "assistant", "content": answer}]
            yield (
                history,
                gr.MultimodalTextbox(value=None, interactive=True),
                sid,
                panel,
                _status_text(service, sid, notice),
            )

        box.submit(
            send,
            [box, chatbot, session_id, space, when, mode, router, lam],
            [chatbot, box, session_id, trace, status],
        )

        def load_memory(
            space_id: str, current: bool, kind: str, request: gr.Request
        ) -> tuple[Any, ...]:
            sid = allowed(request, space_id)
            rows = service.facts(sid, current_only=current)
            table = [
                [r["said_on"][:10], r["statement"], r["status"], r["replaced_by"] or ""]
                for r in rows
            ]
            media = service.media(sid)
            pictures = [
                (m["path"], f"{m['ts'][:10]} · {m['caption'] or 'not described yet'}")
                for m in media
                if Path(m["path"]).exists()
            ]
            listed = [
                [s.session_id, f"{s.started_at:%Y-%m-%d %H:%M}", s.turns, "open" if s.open else ""]
                for s in service.session_list(sid)
            ]
            return (
                _profile_markdown(service.profile(sid)),
                render_graph(service.graph(sid)),
                table,
                pictures,
                listed,
                gr.Dropdown(choices=_forget_choices(service, sid, kind), value=None),
            )

        memory_outputs = [profile, graph, facts, gallery, sessions, forget_target]
        memory_inputs = [space, current_only, forget_kind]
        refresh.click(load_memory, memory_inputs, memory_outputs)
        memory_tab.select(load_memory, memory_inputs, memory_outputs)
        current_only.change(load_memory, memory_inputs, memory_outputs)

        def choices(space_id: str, kind: str, request: gr.Request) -> Any:
            options = _forget_choices(service, allowed(request, space_id), kind)
            return gr.Dropdown(choices=options, value=None)

        forget_kind.change(choices, [space, forget_kind], [forget_target])

        def forget(space_id: str, kind: str, target: str | None, request: gr.Request) -> str:
            sid = allowed(request, space_id)
            if not target:
                return "Pick something to forget first."
            report = service.forget(
                sid,
                fact_ids=[target] if kind == "fact" else [],
                media_ids=[target] if kind == "image" else [],
                session_ids=[target] if kind == "session" else [],
            )
            extra = [
                f"{report.cascaded} from later replies that cited it" if report.cascaded else "",
                f"{report.restated} that restated it without citing" if report.restated else "",
            ]
            cascade = ", including " + " and ".join(e for e in extra if e) if any(extra) else ""
            if report.regrown:
                cascade += f"; {report.regrown} fact(s) grew back after the rebuild and went too"
            if report.rebuild_error:  # 删了，只是重建没做完
                return (
                    f"Forgot {report.turns_deleted} conversation turn(s){cascade}. Rebuilding the "
                    f"memory from the remaining log failed ({report.rebuild_error}); it finishes "
                    "on the next consolidation."
                )
            return (
                f"Forgot {report.turns_deleted} conversation turn(s){cascade}; facts "
                f"{report.facts_before} → {report.facts_after}, rebuilt from the remaining log "
                f"in {report.elapsed_ms / 1000:.1f} s. The chat on screen keeps its history; "
                "Cue's context and memory no longer have it."
            )

        forget_button.click(forget, [space, forget_kind, forget_target], [forget_result]).then(
            load_memory, memory_inputs, memory_outputs
        )

        def opened(request: gr.Request) -> Any:
            """页面打开时：默认的记忆空间不在令牌的作用域里，换成作用域里的第一个。"""
            principal = who(request)
            if principal.allows(service.default_space):
                return gr.skip()
            return principal.home() or ""

        demo.load(opened, None, [space])
    return demo


# ---------- 渲染 ----------


def render_trace(reply: Reply, service: ChatService) -> str:
    t = reply.trace
    parts = ['<div class="trace">', "<h4>Controller</h4>", _route(t)]
    if t.scores:
        parts.append(_utilities(t))
    if t.deep:
        parts.append(_deep(t.deep))
    parts += ["<h4>Context budget</h4>", _partitions(t)]
    parts.append(f"<h4>Recalled memories · {len(t.recalled)}</h4>")
    parts.append(_recalled(t.recalled) or '<div class="muted">None this turn.</div>')
    if t.tools:
        parts.append(f"<h4>Tool calls · {len(t.tools)}</h4>")
        parts += [
            _tool(record.name, record.arguments, record.result, record.ms) for record in t.tools
        ]
    parts += ["<h4>Usage</h4>", _usage(t, service)]
    parts.append("</div>")
    return "".join(parts)


def _route(t: TurnTrace) -> str:
    badges = [f'<span class="badge lvl-{t.level}">{ACTION_EN.get(t.level, t.level)}</span>']
    if t.see:
        badges.append('<span class="badge lvl-see">+ re-look at photos</span>')
    reasons = [english_reason(r) for r in t.reasons]
    details = []
    if t.router:
        details.append(f"controller: {t.router}")
    if "p_none" in t.scores:
        details.append(f"P(no memory needed) = {t.scores['p_none']:.0%}")
    if "probe_ms" in t.scores:
        details.append(f"probe {t.scores['probe_ms']:.0f} ms")
    if t.scores.get("asked_llm"):
        details.append("asked an LLM (gate unsure)")
    why = f'<div class="muted">why: {html.escape(", ".join(reasons))}</div>' if reasons else ""
    return (
        f'<div class="row">{"".join(badges)}</div>{why}'
        f'<div class="muted">{html.escape(" · ".join(details))}</div>'
    )


def _utilities(t: TurnTrace) -> str:
    actions = [k[2:] for k in t.scores if k.startswith("p:")]
    if not actions:
        return ""
    chosen = t.level + ("+see" if t.see else "")
    rows = ['<table class="util"><tr class="muted"><td></td><td>expected accuracy</td>'
            "<td>after cost</td></tr>"]  # fmt: skip
    for action in actions:
        p = t.scores.get(f"p:{action}", 0.0)
        u = t.scores.get(f"u:{action}", 0.0)
        mark = ' class="chosen"' if action == chosen else ""
        rows.append(
            f"<tr{mark}><td>{html.escape(action)}</td>"
            f'<td><div class="bar" style="width:{max(2, p * 100):.0f}%"></div></td>'
            f"<td>{u:.2f}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def _deep(deep: dict[str, Any]) -> str:
    """L2：规划出的子查询，以及充分性检查后补检索了什么。"""
    lines = []
    if deep.get("sub_queries"):
        queries = "; ".join(str(q) for q in deep["sub_queries"])
        lines.append(f"planned searches: {queries}")
    for i, check in enumerate(deep.get("checks", []), start=1):
        if check.get("sufficient"):
            lines.append(f"check {i}: evidence sufficient")
        else:
            extra = "; ".join(check.get("queries") or [])
            lines.append(
                f"check {i}: missing {check.get('missing') or '?'} → searched again: {extra}"
                f" (+{check.get('added_rounds', 0)} excerpts)"
            )
    return "".join(f'<div class="muted">{html.escape(line)}</div>' for line in lines)


def _partitions(t: TurnTrace) -> str:
    total = sum(t.partitions.values()) or 1
    segments = "".join(
        f'<span title="{PARTITION_NAMES[k]}: {v:,}" '
        f'style="width:{v / total * 100:.1f}%;background:{PARTITION_COLORS[k]}"></span>'
        for k, v in t.partitions.items()
        if v > 0 and k in PARTITION_COLORS
    )
    legend = "".join(
        f'<span><i style="background:{PARTITION_COLORS[k]}"></i>{PARTITION_NAMES[k]} '
        f"{v:,}{_cap(k, t.budget)}</span>"
        for k, v in t.partitions.items()
        if k in PARTITION_COLORS
    )
    evicted = f" · {t.evicted} old turn(s) moved out of the chat window" if t.evicted else ""
    return (
        f'<div class="stack">{segments}</div><div class="legend">{legend}</div>'
        f'<div class="muted">≈{total:,} tokens · stable prefix: system + profile + '
        f"{t.window} turn(s) of this chat{evicted}</div>"
    )


def _cap(key: str, budget: dict[str, int]) -> str:
    return f"/{budget[key]:,}" if key in budget else ""


def _recalled(items: list[RecalledItem]) -> str:
    """被引用的记忆全部展开；其余的折叠起来（召回了、但回答没有用到）。"""
    order = {"profile": 0, "fact": 1, "image": 2, "excerpt": 3}
    ranked = sorted(items, key=lambda i: (not i.cited, order[i.kind], i.ref))
    cited = [_memory(i) for i in ranked if i.cited]
    rest = [_memory(i) for i in ranked if not i.cited]
    out = "".join(cited)
    if rest:
        label = f"{len(rest)} more recalled, not cited" if cited else f"{len(rest)} recalled"
        out += f'<details><summary class="muted">{label}</summary>{"".join(rest)}</details>'
    return out


def _memory(item: RecalledItem) -> str:
    meta: list[str] = [item.kind]
    if item.when is not None:
        meta.append(("said " if item.kind == "fact" else "") + f"{item.when:%Y-%m-%d}")
    if item.status and item.status != "current":
        meta.append(item.status.split(", see")[0])
    elif item.kind in ("fact", "profile"):
        meta.append("current")
    if item.attached:
        meta.append("original photo shown to the model")
    if item.via == "memory_search":
        meta.append("found by memory_search")
    if item.cited:
        meta.append("cited")
    text = item.text if len(item.text) <= 260 else item.text[:257] + "…"
    image = ""
    if item.kind == "image" and item.path and Path(item.path).exists():
        image = f'<img src="{_thumbnail(item.path)}" alt="">'
    return (
        f'<div class="mem{" cited" if item.cited else ""}">{image}'
        f'<span class="ref">{item.ref}</span>'
        f'<span class="muted">{html.escape(" · ".join(meta))}</span>'
        f'<div class="text">{html.escape(text)}</div></div>'
    )


def _tool(name: str, arguments: dict[str, Any], result: str, ms: float) -> str:
    args = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    first = result.splitlines()[0] if result else ""
    return (
        f'<div class="tool">{html.escape(name)}({html.escape(args)}) '
        f'<span class="muted">→ {html.escape(first[:90])} · {ms:.0f} ms</span></div>'
    )


def _usage(t: TurnTrace, service: ChatService) -> str:
    u = t.usage
    model = service.system.settings.roles["agent"].model
    cached = int(u.get("cached_input_tokens", 0))
    lines = [
        f"{model} · {int(u.get('llm_calls', 0))} LLM call(s) · "
        f"{int(u.get('input_tokens', 0)):,} input tokens"
        + (f" ({cached:,} from the provider's prefix cache)" if cached else "")
        + f" · {int(u.get('output_tokens', 0)):,} output",
        f"≈ ${u.get('spent', 0.0):.4f} this turn · recall {t.timings.get('recall', 0):.0f} ms · "
        f"model {t.timings.get('llm', 0) / 1000:.1f} s · write {t.timings.get('write', 0):.0f} ms",
    ]
    return "".join(f'<div class="muted">{html.escape(line)}</div>' for line in lines)


def _assistant(reply: Reply) -> list[Any]:
    """回答：真实存在的引用编号显示成上标（模型编造的去掉）；回答里提到的商品附一排商品图。"""
    valid = set(reply.citations)

    def cite(match: re.Match[str]) -> str:
        refs = [r for r in re.findall(r"[a-z]\d+", match.group(1)) if r in valid]
        return "".join(f'<sup class="cite">{r}</sup>' for r in refs)

    text = CITATION.sub(cite, html.escape(reply.text, quote=False))
    text = text.replace('</sup><sup class="cite">', ", ")  # 相邻的引用合成一个上标
    text = re.sub(r"[ \t]+(<sup|[,.;:!?，。；：！？])", r"\1", text)
    content: list[Any] = [text.strip()]
    shown = [p for p in reply.products if p.get("mentioned") and p.get("image")]
    if shown:
        content.append(
            gr.Gallery(
                value=[(p["image"], f"{p['name']} · ${p['price']:g}") for p in shown[:4]],
                columns=min(4, len(shown)),
                height=190,
                show_label=False,
                allow_preview=False,
            )
        )
    return content


def render_graph(graph: dict[str, Any]) -> str:
    """实体图：画像主体（关联事实最多的实体）居中，其余实体排在一圈上。"""
    nodes = graph["nodes"]
    if not nodes:
        return (
            '<div class="muted">No entities yet: they appear after a session is consolidated.</div>'
        )
    width, height = 560, 380
    center = max(nodes, key=lambda n: n["facts"])
    others = [n for n in nodes if n is not center][:14]
    pos = {center["id"]: (width / 2, height / 2)}
    for i, node in enumerate(others):
        angle = 2 * math.pi * i / max(1, len(others)) - math.pi / 2
        pos[node["id"]] = (width / 2 + 210 * math.cos(angle), height / 2 + 140 * math.sin(angle))
    svg = [f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    seen: set[tuple[str, str]] = set()
    for edge in graph["edges"]:
        key = (edge["source"], edge["target"])
        if key in seen or edge["source"] not in pos or edge["target"] not in pos:
            continue
        seen.add(key)
        (x1, y1), (x2, y2) = pos[edge["source"]], pos[edge["target"]]
        svg.append(
            f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            'stroke="currentColor" stroke-opacity=".25" stroke-width="1.5"/>'
        )
        svg.append(
            f'<text x="{(x1 + x2) / 2:.0f}" y="{(y1 + y2) / 2 - 3:.0f}" font-size="10" '
            f'text-anchor="middle" fill="currentColor" fill-opacity=".55">'
            f"{html.escape(edge['label'].replace('_', ' '))}</text>"
        )
    for node in [center, *others]:
        x, y = pos[node["id"]]
        radius = 9 + min(10, node["facts"])
        color = TYPE_COLORS.get(node["type"], TYPE_COLORS["other"])
        svg.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{radius}" fill="{color}" fill-opacity=".85"/>'
            f'<text x="{x:.0f}" y="{y + radius + 13:.0f}" font-size="12" text-anchor="middle" '
            f'fill="currentColor">{html.escape(node["name"])}</text>'
        )
    svg.append("</svg>")
    legend = " ".join(
        f'<span><i style="background:{c};display:inline-block;width:9px;height:9px;'
        f'border-radius:50%;margin-right:3px"></i>{t}</span>'
        for t, c in TYPE_COLORS.items()
    )
    return (
        '<div class="trace"><h4>Entity graph · current facts</h4>'
        + "".join(svg)
        + f'<div class="legend">{legend}</div></div>'
    )


def _profile_markdown(text: str) -> str:
    if not text:
        return "#### Profile (core memory)\n_Empty: the profile is built when a session ends._"
    return f"#### Profile (core memory)\n{text}"


def _forget_choices(service: ChatService, space_id: str, kind: str) -> list[tuple[str, str]]:
    sid = space_id.strip() or service.default_space
    if kind == "fact":
        return [
            (f"{r['said_on'][:10]} · {r['statement'][:90]}", r["id"])
            for r in service.facts(sid, current_only=True)
        ]
    if kind == "image":
        return [
            (f"{m['ts'][:10]} · {(m['caption'] or 'image')[:80]}", m["id"])
            for m in service.media(sid)
        ]
    return [
        (f"{s.started_at:%Y-%m-%d %H:%M} · {s.turns} turns · {s.session_id}", s.session_id)
        for s in service.session_list(sid)
    ]


def _status_text(service: ChatService, sid: str | None, notice: str = "") -> str:
    parts = []
    if sid:
        try:
            session = service.get(sid)
        except KeyError:
            session = None
        if session is not None:
            clock = " (simulated)" if session.simulated else ""
            parts.append(
                f"**Session** `{session.session_id}` · started "
                f"{session.started_at:%Y-%m-%d %H:%M}{clock} · "
                f"{len(session.window)} turn(s) in the chat window"
            )
            state = service.status(session.space_id)
            if state["pending"]:
                parts.append(f"consolidating {state['pending']} earlier session(s)…")
    if notice:
        parts.append(notice)
    return " · ".join(parts) or "No session yet: sending a message starts one."


def _ended_notice(service: ChatService) -> str:
    """页面里的会话已经不在了（闲置太久，HTTP 应用的清理线程结束了它）：发的这句话开一个新的会话。
    以前页面一直拿着结束了的会话，之后每句话都报 KeyError，直到点 New session。"""
    idle = f" after {service.idle_minutes:g} idle minutes" if service.idle_minutes else ""
    return (
        f"The previous session ended{idle} and is being consolidated; "
        "this message starts a new one."
    )


def _end(service: ChatService, sid: str) -> str:
    try:
        out = service.end(sid)
    except KeyError:
        return "That session had already ended."
    if out["turns"]:
        return (
            "Session ended: consolidating it in the background (facts, updates, photo "
            "descriptions, profile)."
        )
    return "Session ended (it was empty)."


def _parse_date(text: str) -> datetime | None:
    text = (text or "").strip()
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise gr.Error(f"Could not read the date {text!r}; use YYYY-MM-DD HH:MM.")


def _file_path(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("path"))
    return str(getattr(item, "path", item))


def _thumbnail(path: str, size: int = 144) -> str:
    with Image.open(path) as source, decoding(source):  # 大图的解码排队（store/media.py）
        image = source.convert("RGB")
        image.thumbnail((size, size))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _lam_default(service: ChatService) -> float:
    learned = service.system.learned
    if service.agent.lam is not None:
        return float(service.agent.lam)
    return float(learned.lam) if learned is not None else 30.0
