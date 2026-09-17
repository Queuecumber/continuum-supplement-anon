"""Recovering a tool call the server failed to parse.

qwen3.5-0.8b emits a well-formed XML tool call that its deployment returns as
reasoning_content with tool_calls empty, because the reasoning parser claims
the whole response. The call is there; the harness reads it rather than
treating the turn as "the model chose not to act".
"""
import json

from continuum.harnesses.agent_loop import salvage_xml_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "generate_image", "parameters": {
        "type": "object", "properties": {
            "prompt": {"type": "string"},
            "width": {"type": "integer"},
            "hires": {"type": "boolean"}}}}},
    {"type": "function", "function": {"name": "mark_best", "parameters": {
        "type": "object", "properties": {"image": {"type": "string"}}}}},
]

REAL = (  # verbatim shape of what qwen3.5-0.8b returns in reasoning_content
    "<tool_call>\n<function=generate_image>\n<parameter=prompt>\n"
    "A close-up of a shiny brass diving helmet\n</parameter>\n"
    "<parameter=width>\n1024\n</parameter>\n</function>\n</tool_call>"
)


def args_of(call):
    return json.loads(call.function.arguments)


def test_recovers_the_call_and_types_it() -> None:
    calls = salvage_xml_tool_calls(REAL, TOOLS)
    assert len(calls) == 1
    assert calls[0].function.name == "generate_image"
    assert calls[0].type == "function"
    assert calls[0].id.startswith("call_")
    # width must be an int: the MCP tool schema rejects "1024"
    assert args_of(calls[0]) == {
        "prompt": "A close-up of a shiny brass diving helmet", "width": 1024}


def test_quiet_when_there_is_nothing_to_salvage() -> None:
    assert salvage_xml_tool_calls("I have finished, the image looks correct.", TOOLS) == []
    assert salvage_xml_tool_calls("", TOOLS) == []
    assert salvage_xml_tool_calls(None, TOOLS) == []


def test_two_calls_in_one_response() -> None:
    text = REAL + "\n<tool_call>\n<function=mark_best>\n<parameter=image>img_0</parameter>\n</function>\n</tool_call>"
    calls = salvage_xml_tool_calls(text, TOOLS)
    assert [c.function.name for c in calls] == ["generate_image", "mark_best"]
    assert args_of(calls[1]) == {"image": "img_0"}


def test_tolerates_a_missing_close_tag() -> None:
    text = ("<tool_call>\n<function=generate_image>\n"
            "<parameter=width>\n768\n"
            "<parameter=hires>\ntrue\n</parameter>\n</function>\n</tool_call>")
    assert args_of(salvage_xml_tool_calls(text, TOOLS)[0]) == {"width": 768, "hires": True}


def test_interior_newlines_survive() -> None:
    text = ("<tool_call>\n<function=generate_image>\n<parameter=prompt>\n"
            "line one\n  indented two\n\n</parameter>\n</function>\n</tool_call>")
    assert args_of(salvage_xml_tool_calls(text, TOOLS)[0])["prompt"] == "line one\n  indented two\n"


def test_uncastable_value_reaches_the_tool_unchanged() -> None:
    text = ("<tool_call>\n<function=generate_image>\n"
            "<parameter=width>wide</parameter>\n</function>\n</tool_call>")
    assert args_of(salvage_xml_tool_calls(text, TOOLS)[0]) == {"width": "wide"}


def test_unknown_tool_still_parses() -> None:
    text = ("<tool_call>\n<function=some_other_tool>\n"
            "<parameter=n>42</parameter>\n</function>\n</tool_call>")
    calls = salvage_xml_tool_calls(text, TOOLS)
    assert calls[0].function.name == "some_other_tool"
    assert args_of(calls[0]) == {"n": 42}
