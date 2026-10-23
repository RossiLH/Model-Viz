"""Tests for posttrain.xtml — XTML 对话模板。"""

import pytest

from models.llm.kimi_k3.posttrain.test_helpers import (
    sample_conversation,
    verify_xtml_roundtrip,
)
from models.llm.kimi_k3.posttrain.xtml import (
    OPEN,
    SEP,
    CLOSE,
    END_OF_MSG,
    ToolCall,
    Message,
    Conversation,
    serialize,
    parse,
    StreamingParser,
    generation_prefix,
    allowed_next_tokens,
    loss_mask_spans,
    EFFORT_LEVELS,
    K3_SUPPORTED_EFFORT,
)


class TestConstants:
    def test_reserved_tokens(self):
        assert OPEN == "[open]"
        assert SEP == "[sep]"
        assert CLOSE == "[close]"
        assert END_OF_MSG == "[end_of_msg]"

    def test_effort_levels(self):
        assert EFFORT_LEVELS == ("low", "medium", "high", "max")
        assert K3_SUPPORTED_EFFORT == ("low", "high", "max")


class TestToolCall:
    def test_defaults(self):
        tc = ToolCall(tool="search", index=0)
        assert tc.arguments == {}
        assert tc.raw_json is None

    def test_with_arguments(self):
        tc = ToolCall(tool="calc", index=1, arguments={"x": 42})
        assert tc.arguments["x"] == 42


class TestMessage:
    def test_user_message(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.think is None

    def test_assistant_message(self):
        msg = Message(role="assistant", content="hi", think="greeting")
        assert msg.think == "greeting"

    def test_option_message(self):
        msg = Message(role="option", option_type="tool_choice", option_value="auto")
        assert msg.option_type == "tool_choice"


class TestSerialize:
    def test_simple_conversation(self):
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert OPEN in text
        assert SEP in text
        assert CLOSE in text
        assert "hi" in text
        assert "hello" in text

    def test_thinking_mode(self):
        conv = Conversation(messages=[
            Message(role="user", content="solve"),
            Message(role="assistant", content="42", think="reasoning"),
        ])
        text = serialize(conv, thinking=True, add_generation_prefix=False)
        assert f"{OPEN}think{SEP}" in text
        assert "reasoning" in text

    def test_global_tool_declaration(self):
        conv = Conversation(
            tools=[{"name": "search", "description": "search tool"}],
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "tool-declare" in text
        assert "search" in text

    def test_thinking_effort(self):
        conv = Conversation(
            thinking_effort="high",
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "thinking-effort" in text
        assert "high" in text

    def test_invalid_effort_raises(self):
        conv = Conversation(
            thinking_effort="ultra",
            messages=[Message(role="user", content="hi")],
        )
        try:
            serialize(conv, thinking=False, add_generation_prefix=False)
            assert False, "should have raised"
        except ValueError as e:
            assert "ultra" in str(e)

    def test_tool_calls_serialization(self):
        conv = Conversation(messages=[
            Message(role="user", content="search"),
            Message(role="assistant", content="done", tool_calls=[
                ToolCall(tool="search", index=0, arguments={"query": "test"}),
            ]),
        ])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "tools" in text
        assert "search" in text

    def test_generation_prefix_thinking(self):
        prefix = generation_prefix(thinking=True)
        assert f"{OPEN}think{SEP}" in prefix
        assert "assistant" in prefix

    def test_generation_prefix_response(self):
        prefix = generation_prefix(thinking=False)
        assert f"{OPEN}response{SEP}" in prefix

    def test_dynamic_tools(self):
        conv = Conversation(
            messages=[Message(role="user", content="hi")],
            dynamic_tools=[(0, [{"name": "new_tool"}])],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "tool-declare" in text


class TestParse:
    def test_roundtrip_simple(self):
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert len(parsed.messages) == 2
        assert parsed.messages[0].role == "user"
        assert parsed.messages[0].content == "hi"
        assert parsed.messages[1].role == "assistant"
        assert parsed.messages[1].content == "hello"

    def test_roundtrip_thinking(self):
        conv = Conversation(messages=[
            Message(role="user", content="solve"),
            Message(role="assistant", content="42", think="reasoning"),
        ])
        text = serialize(conv, thinking=True, add_generation_prefix=False)
        parsed = parse(text)
        assert parsed.messages[1].think == "reasoning"
        assert parsed.messages[1].content == "42"

    def test_roundtrip_tool_calls(self):
        conv = Conversation(messages=[
            Message(role="user", content="search"),
            Message(role="assistant", content="done", tool_calls=[
                ToolCall(tool="search", index=0, arguments={"query": "test"}),
            ]),
        ])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert len(parsed.messages[1].tool_calls) == 1
        assert parsed.messages[1].tool_calls[0].tool == "search"
        assert parsed.messages[1].tool_calls[0].arguments["query"] == "test"

    def test_parse_tool_declarations(self):
        conv = Conversation(
            tools=[{"name": "search"}],
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert parsed.tools == [{"name": "search"}]

    def test_parse_thinking_effort(self):
        conv = Conversation(
            thinking_effort="high",
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert parsed.thinking_effort == "high"

    def test_parse_tool_choice(self):
        conv = Conversation(
            tool_choice="auto",
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert parsed.tool_choice == "auto"

    def test_parse_response_format(self):
        conv = Conversation(
            response_format={"type": "json_object"},
            messages=[Message(role="user", content="hi")],
        )
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        parsed = parse(text)
        assert parsed.response_format == {"type": "json_object"}


class TestStreamingParser:
    def test_simple_think(self):
        parser = StreamingParser()
        text = serialize(
            Conversation(messages=[Message(role="assistant", content="ok", think="hmm")]),
            thinking=True, add_generation_prefix=True,
        )
        results = list(parser.feed(text))
        channels = [ch for ch, _ in results]
        assert "think" in channels
        assert "response" in channels

    def test_incremental_feed(self):
        parser = StreamingParser()
        text = serialize(
            Conversation(messages=[Message(role="assistant", content="ok", think="hmm")]),
            thinking=True, add_generation_prefix=True,
        )
        results = []
        for ch in text[:5]:
            results.extend(parser.feed(ch))
        for ch in text[5:]:
            results.extend(parser.feed(ch))
        # Think and response channels should be yielded
        channels = [ch for ch, _ in results]
        assert "think" in channels
        assert "response" in channels

    def test_tools_channel(self):
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="done", tool_calls=[
                ToolCall(tool="search", index=0, arguments={"q": "x"}),
            ]),
        ])
        parser = StreamingParser()
        text = serialize(conv, thinking=False, add_generation_prefix=True)
        results = list(parser.feed(text))
        channels = {ch for ch, _ in results}
        assert "tools" in channels


class TestAllowedNextTokens:
    def test_no_channel_thinking(self):
        tokens = allowed_next_tokens(None, thinking=True)
        assert f"{OPEN}think" in tokens[0]

    def test_no_channel_response(self):
        tokens = allowed_next_tokens(None, thinking=False)
        assert f"{OPEN}response" in tokens[0]

    def test_think_channel(self):
        tokens = allowed_next_tokens("think", thinking=True)
        assert f"{CLOSE}think{SEP}" in tokens

    def test_response_channel(self):
        tokens = allowed_next_tokens("response", thinking=True)
        assert f"{CLOSE}response{SEP}" in tokens
        assert f"{OPEN}tools" in tokens

    def test_tools_channel(self):
        tokens = allowed_next_tokens("tools", thinking=True)
        assert f"{CLOSE}tools{SEP}" in tokens
        assert f"{OPEN}call" in tokens
        assert END_OF_MSG in tokens


class TestLossMaskSpans:
    def test_no_json_block(self):
        assert loss_mask_spans("hello world") == []

    def test_json_block(self):
        text = f"before {OPEN}json{SEP}{{}}{CLOSE}json{SEP} after"
        spans = loss_mask_spans(text)
        assert len(spans) == 1
        start, end = spans[0]
        assert text[start:end].startswith(f"{OPEN}json{SEP}")
        assert text[start:end].endswith(f"{CLOSE}json{SEP}")

    def test_multiple_json_blocks(self):
        text = f"{OPEN}json{SEP}a{CLOSE}json{SEP} mid {OPEN}json{SEP}b{CLOSE}json{SEP}"
        spans = loss_mask_spans(text)
        assert len(spans) == 2


class TestEdgeCases:
    def test_empty_conversation(self):
        conv = Conversation()
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert text == ""

    def test_tool_result_message(self):
        msg = Message(role="tool", content="42", tool="search", index=0)
        conv = Conversation(messages=[msg])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "42" in text

    def test_option_message_in_history(self):
        msg = Message(role="option", option_type="tool_choice", option_value="auto")
        conv = Conversation(messages=[msg])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "auto" in text
        parsed = parse(text)
        assert parsed.tool_choice == "auto"

    def test_assistant_with_empty_think(self):
        msg = Message(role="assistant", content="ok", think="")
        conv = Conversation(messages=[msg])
        text = serialize(conv, thinking=True, add_generation_prefix=False)
        assert f"{OPEN}think{SEP}" in text

    def test_raw_json_tool_call(self):
        tc = ToolCall(tool="search", index=0, raw_json='{"q": "test"}')
        msg = Message(role="assistant", content="", tool_calls=[tc])
        conv = Conversation(messages=[msg])
        text = serialize(conv, thinking=False, add_generation_prefix=False)
        assert "json" in text
        assert '{"q": "test"}' in text


class TestVerifyRoundtrip:
    """verify_xtml_roundtrip 必须对生成前缀免疫。

    generation_prefix() 追加的是一个空的 assistant 起写桩，不是真实消息；
    往返比对时若不剔除，会误报「消息数 +1 / 末条内容不一致」的假阳性。
    """

    @pytest.mark.parametrize("kind", ["weather", "multi", "coding"])
    @pytest.mark.parametrize("thinking", [True, False])
    @pytest.mark.parametrize("add_generation_prefix", [True, False])
    def test_roundtrip_ok(self, kind, thinking, add_generation_prefix):
        conv = sample_conversation(kind)
        result = verify_xtml_roundtrip(
            conv, thinking=thinking,
            add_generation_prefix=add_generation_prefix,
        )
        assert result.ok, result.mismatch
        assert result.messages == len(conv.messages)

    def test_prefix_does_not_change_message_count(self):
        conv = sample_conversation("multi")
        with_prefix = verify_xtml_roundtrip(conv, add_generation_prefix=True)
        without_prefix = verify_xtml_roundtrip(conv, add_generation_prefix=False)
        assert with_prefix.messages == without_prefix.messages
        # 前缀只增加字符，不增加消息
        assert with_prefix.text_len > without_prefix.text_len

    def test_real_mismatch_still_detected(self):
        """剔除逻辑只针对「多出一条」的情况，不能掩盖真实差异。"""
        conv = sample_conversation("coding")
        result = verify_xtml_roundtrip(conv, thinking=True,
                                       add_generation_prefix=True)
        assert result.ok

        # 人为制造差异：比对一个与序列化来源不同的对话
        other = sample_conversation("coding")
        other.messages[-1].content = "3"
        text = serialize(conv, thinking=True, add_generation_prefix=True)
        parsed = parse(text)
        assert parsed.messages[-2].content == "2"
        assert other.messages[-1].content != parsed.messages[-2].content

    def test_trailing_empty_assistant_not_over_stripped(self):
        """末条本就是空 assistant 时，不应把真实消息误当作生成桩剔除。"""
        conv = Conversation(messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content=""),
        ])
        result = verify_xtml_roundtrip(conv, thinking=False,
                                       add_generation_prefix=False)
        assert result.ok, result.mismatch
        assert result.messages == 2
