r"""XTML: eXtensible Token Markup Language (paper appendix F).

K3's chat template is redesigned around three goals:

* **Extensibility** -- new capabilities arrive as backward-compatible message
  formats rather than template revisions, so a single template serves the whole
  model generation.
* **A low alignment tax** -- the format is learnable from minimal supervised
  data, so a lightly fine-tuned pre-trained model can proceed directly to
  reinforcement learning.
* **Decoding friendliness** -- the format admits simple encoders, streaming
  parsers, and grammar-constrained enforcers.

The mechanism is to replace XML's angle brackets with three reserved *tokens* --
``[open]``, ``[sep]``, ``[close]`` -- plus an ``[end_of_msg]`` stop marker.  An
element ``[open]tag attr="value"[sep] ... [close]tag[sep]`` is isomorphic to its
XML counterpart, but because every structural boundary is a single special token
there is no tokenisation ambiguity at element edges, and constrained decoding
reduces to masking a handful of token ids.

Layout (Figure 16a).  Global option messages -- the tool declaration and the
reasoning-effort setting -- precede the input messages: they govern the whole
session and rarely change, and modifying them invalidates the KV cache anyway.
One-shot options (``tool_choice``, ``response_format``) are appended *after* the
input messages, so per-request changes leave the history KV cache intact.  A
third kind, the input option message, interleaves with input messages to
override a global option mid-session -- which is also how dynamically loaded
tools are announced, extending the toolset without rebuilding the preceding
context.

Channels (Figure 16b).  An assistant message body is organised into ``think``,
``response``, and ``tools``.  The generation *prefix* alone selects the mode --
``[open]think[sep]`` for thinking, ``[open]response[sep]`` for instruct -- rather
than separate templates.  K3 supports *preserved thinking*: in thinking mode the
think channel is always retained in history, kept even when its content is
empty, so the model observes a consistent message structure across turns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator
from typing_extensions import Literal

__all__ = [
    "OPEN",
    "SEP",
    "CLOSE",
    "END_OF_MSG",
    "ToolCall",
    "Message",
    "Conversation",
    "serialize",
    "parse",
    "StreamingParser",
    "generation_prefix",
    "allowed_next_tokens",
    "loss_mask_spans",
]

OPEN = "[open]"
SEP = "[sep]"
CLOSE = "[close]"
END_OF_MSG = "[end_of_msg]"

EffortLevel = Literal["low", "medium", "high", "max"]
#: The schema reserves four levels; K3 supports a subset of them.
EFFORT_LEVELS: tuple[str, ...] = ("low", "medium", "high", "max")
K3_SUPPORTED_EFFORT: tuple[str, ...] = ("low", "high", "max")

Channel = Literal["think", "response", "tools"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """One call in the ``tools`` channel.

    ``index`` parallels calls within a message so that tool *results* can be
    matched back to their calls: a result message repeats the same
    ``tool``/``index`` pair and follows the order of its call.
    """

    tool: str
    index: int
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Raw JSON fallback, used when arguments cannot be decomposed into typed
    #: blocks.  It occurs only in input tokens and its loss is masked.
    raw_json: str | None = None


@dataclass
class Message:
    role: str
    content: str = ""
    think: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: For tool-result messages: which call this answers.
    tool: str | None = None
    index: int | None = None
    #: Option messages carry a type instead of free text.
    option_type: str | None = None
    option_value: Any = None


@dataclass
class Conversation:
    """A full context, in the zone order of Figure 16a."""

    messages: list[Message] = field(default_factory=list)
    tools: list[dict] | None = None
    thinking_effort: str | None = None
    tool_choice: str | None = None
    response_format: dict | None = None
    #: Tools announced mid-session, after the initial declaration.
    dynamic_tools: list[tuple[int, list[dict]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _element(tag: str, body: str, **attrs: Any) -> str:
    """序列化一个 XTML 元素：[open]tag attr="value"[sep]body[close]tag[sep]"""
    attr = "".join(f' {k}="{v}"' for k, v in attrs.items())
    return f"{OPEN}{tag}{attr}{SEP}{body}{CLOSE}{tag}{SEP}"


def _argument(key: str, value: Any) -> str:
    """序列化一个类型化参数。

    字符串参数以原始文本形式出现，其他 JSON 类型紧凑序列化。
    """
    if isinstance(value, str):
        return _element("argument", value, key=key, type="string")
    kind = {
        bool: "boolean",
        int: "number",
        float: "number",
        list: "array",
        dict: "object",
    }.get(type(value), "object")
    if value is None:
        kind = "null"
    return _element("argument", json.dumps(value, separators=(",", ":")), key=key, type=kind)


def _serialize_tool_calls(calls: list[ToolCall]) -> str:
    """序列化工具调用列表。"""
    parts = []
    for call in calls:
        if call.raw_json is not None:
            body = _element("json", call.raw_json)
        else:
            body = "".join(_argument(k, v) for k, v in call.arguments.items())
        parts.append(_element("call", body, tool=call.tool, index=call.index))
    return _element("tools", "".join(parts))


def _serialize_message(msg: Message, *, thinking: bool, is_history: bool) -> str:
    """序列化单条消息。"""
    if msg.option_type is not None:
        value = (
            msg.option_value
            if isinstance(msg.option_value, str)
            else json.dumps(msg.option_value, separators=(",", ":"))
        )
        return _element("message", value, role="option", type=msg.option_type) + END_OF_MSG

    attrs: dict[str, Any] = {"role": msg.role}
    if msg.tool is not None:
        attrs["tool"] = msg.tool
    if msg.index is not None:
        attrs["index"] = msg.index

    body = ""
    if msg.role == "assistant":
        # Preserved thinking: in thinking mode the channel is always present,
        # even when empty, so the structure the model sees never changes shape.
        if thinking:
            body += _element("think", msg.think or "")
        body += _element("response", msg.content)
        if msg.tool_calls:
            body += _serialize_tool_calls(msg.tool_calls)
    else:
        body = msg.content
    _ = is_history
    return _element("message", body, **attrs) + END_OF_MSG


def serialize(
    conv: Conversation, *, thinking: bool = True, add_generation_prefix: bool = True
) -> str:
    """将 Conversation 渲染为 XTML token 流。"""
    out: list[str] = []

    # --- global option messages: govern the whole session, rarely change ---
    if conv.tools is not None:
        out.append(
            _element(
                "message",
                json.dumps(conv.tools, separators=(",", ":")),
                role="option",
                type="tool-declare",
            )
            + END_OF_MSG
        )
    if conv.thinking_effort is not None:
        if conv.thinking_effort not in EFFORT_LEVELS:
            raise ValueError(
                f"unknown reasoning effort {conv.thinking_effort!r}; "
                f"the schema reserves {EFFORT_LEVELS}"
            )
        out.append(
            _element(
                "message", conv.thinking_effort, role="option", type="thinking-effort"
            )
            + END_OF_MSG
        )

    # --- input messages, with dynamic tool declarations interleaved ---
    dynamic = dict(conv.dynamic_tools)
    for i, msg in enumerate(conv.messages):
        if i in dynamic:
            out.append(
                _element(
                    "message",
                    json.dumps(dynamic[i], separators=(",", ":")),
                    role="option",
                    type="tool-declare",
                )
                + END_OF_MSG
            )
        out.append(_serialize_message(msg, thinking=thinking, is_history=True))

    # --- one-shot options: appended last so history stays cacheable ---
    if conv.tool_choice is not None:
        out.append(
            _element("message", conv.tool_choice, role="option", type="tool_choice")
            + END_OF_MSG
        )
    if conv.response_format is not None:
        out.append(
            _element(
                "message",
                json.dumps(conv.response_format, separators=(",", ":")),
                role="option",
                type="response_format",
            )
            + END_OF_MSG
        )

    if add_generation_prefix:
        out.append(generation_prefix(thinking=thinking))
    return "".join(out)


def generation_prefix(*, thinking: bool) -> str:
    """生成模式选择前缀。"""
    head = f"{OPEN}message role=\"assistant\"{SEP}"
    return head + (f"{OPEN}think{SEP}" if thinking else f"{OPEN}response{SEP}")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_ELEMENT_RE = re.compile(
    re.escape(OPEN) + r"(?P<tag>[a-zA-Z_][\w-]*)(?P<attrs>[^\[]*)" + re.escape(SEP)
)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class Element:
    """解析后的 XTML 元素节点。"""
    tag: str
    attrs: dict[str, str]
    body: str
    children: list["Element"] = field(default_factory=list)


def _parse_elements(text: str, pos: int = 0) -> tuple[list[Element], int]:
    """递归下降解析兄弟节点序列。"""
    out: list[Element] = []
    while pos < len(text):
        if text.startswith(END_OF_MSG, pos):
            pos += len(END_OF_MSG)
            continue
        if text.startswith(CLOSE, pos):
            return out, pos
        m = _ELEMENT_RE.match(text, pos)
        if not m:
            break
        tag = m.group("tag")
        attrs = dict(_ATTR_RE.findall(m.group("attrs")))
        inner_start = m.end()
        close_marker = f"{CLOSE}{tag}{SEP}"

        children, after = _parse_elements(text, inner_start)
        if children:
            end = text.find(close_marker, after)
            body = ""
        else:
            end = text.find(close_marker, inner_start)
            body = text[inner_start:end] if end != -1 else text[inner_start:]
        out.append(Element(tag, attrs, body, children))
        pos = (end + len(close_marker)) if end != -1 else len(text)
    return out, pos


def _decode_argument(el: Element) -> Any:
    """解码参数元素的值。"""
    kind = el.attrs.get("type", "string")
    if kind == "string":
        return el.body
    if kind == "null":
        return None
    try:
        return json.loads(el.body)
    except json.JSONDecodeError:
        return el.body


def parse(text: str) -> Conversation:
    """将 XTML 流解析为 Conversation。"""
    elements, _ = _parse_elements(text)
    conv = Conversation()
    for el in elements:
        if el.tag != "message":
            continue
        role = el.attrs.get("role", "user")
        if role == "option":
            kind = el.attrs.get("type")
            if kind == "tool-declare":
                declared = json.loads(el.body)
                if conv.tools is None:
                    conv.tools = declared
                else:
                    conv.dynamic_tools.append((len(conv.messages), declared))
            elif kind == "thinking-effort":
                conv.thinking_effort = el.body
            elif kind == "tool_choice":
                conv.tool_choice = el.body
            elif kind == "response_format":
                conv.response_format = json.loads(el.body)
            continue

        msg = Message(role=role)
        if "tool" in el.attrs:
            msg.tool = el.attrs["tool"]
        if "index" in el.attrs:
            msg.index = int(el.attrs["index"])

        if el.children:
            for child in el.children:
                if child.tag == "think":
                    msg.think = child.body
                elif child.tag == "response":
                    msg.content = child.body
                elif child.tag == "tools":
                    for call_el in child.children:
                        if call_el.tag != "call":
                            continue
                        call = ToolCall(
                            tool=call_el.attrs.get("tool", ""),
                            index=int(call_el.attrs.get("index", 0)),
                        )
                        for arg in call_el.children:
                            if arg.tag == "argument":
                                call.arguments[arg.attrs["key"]] = _decode_argument(arg)
                            elif arg.tag == "json":
                                call.raw_json = arg.body
                        msg.tool_calls.append(call)
        else:
            msg.content = el.body
        conv.messages.append(msg)
    return conv


class StreamingParser:
    """增量解析器，逐 chunk 解码，输出 (channel, text) 对。"""

    def __init__(self) -> None:
        self.buffer = ""
        self.channel: str | None = None
        self.finished = False

    def feed(self, chunk: str) -> Iterator[tuple[str, str]]:
        """喂入 chunk，yield (channel, text) 对。"""
        self.buffer += chunk
        while True:
            if self.channel is None:
                m = _ELEMENT_RE.search(self.buffer)
                if not m:
                    return
                if m.group("tag") in ("think", "response", "tools"):
                    self.channel = m.group("tag")
                self.buffer = self.buffer[m.end():]
                continue
            close_marker = f"{CLOSE}{self.channel}{SEP}"
            idx = self.buffer.find(close_marker)
            if idx == -1:
                safe = len(self.buffer) - len(close_marker)
                if safe > 0:
                    yield self.channel, self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
                return
            yield self.channel, self.buffer[:idx]
            self.buffer = self.buffer[idx + len(close_marker):]
            self.channel = None
            if self.buffer.startswith(END_OF_MSG):
                self.finished = True
                self.buffer = self.buffer[len(END_OF_MSG):]


def allowed_next_tokens(channel: str | None, *, thinking: bool) -> list[str]:
    """语法约束钩子：当前状态下合法的下一个 token。"""
    if channel is None:
        return [OPEN + ("think" if thinking else "response")]
    if channel == "think":
        return [CLOSE + "think" + SEP]
    if channel == "response":
        return [CLOSE + "response" + SEP, OPEN + "tools"]
    return [CLOSE + "tools" + SEP, OPEN + "call", END_OF_MSG]


def loss_mask_spans(text: str) -> list[tuple[int, int]]:
    """返回需要 mask loss 的字符区间（纯 JSON 回退块）。"""
    spans: list[tuple[int, int]] = []
    marker = f"{OPEN}json{SEP}"
    close = f"{CLOSE}json{SEP}"
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start == -1:
            return spans
        end = text.find(close, start)
        if end == -1:
            return spans
        spans.append((start, end + len(close)))
        pos = end + len(close)
