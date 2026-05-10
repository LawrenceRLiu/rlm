"""Unit tests for the action parser (tag-pair scanner + per-tool schema validation)."""

import pytest

from rlm.utils.action_parser import parse, parse_edit_file_body, parse_final_body
from rlm.utils.exceptions import ActionParseError

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_self_closing_action():
    text = '<action tool="list_directory" />'
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"
    assert actions[0].body is None


def test_self_closing_with_attrs():
    text = '<action tool="read_file" path="src/foo.py" start_line="1" end_line="50" />'
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].args == {"path": "src/foo.py", "start_line": "1", "end_line": "50"}


def test_paired_with_body_preserves_whitespace():
    body = "def foo():\n    return 42\n\n  # trailing"
    text = f'<action tool="write_file" path="x.py">{body}</action>'
    actions = parse(text)
    assert actions[0].body == body


def test_prose_around_actions_is_ignored():
    text = (
        "Let me think about this.\n\n"
        '<action tool="list_directory" />\n\n'
        "Now let me read the file.\n"
        '<action tool="read_file" path="a.txt" />'
    )
    actions = parse(text)
    assert len(actions) == 2
    assert actions[0].tool == "list_directory"
    assert actions[1].args["path"] == "a.txt"


def test_raw_lt_in_body_is_fine():
    """Raw `<` in code body must not break the parser."""
    body = 'if (a < b) { return "<x>"; }\nfor (i = 0; i < 10; i++) { print(i); }'
    text = f'<action tool="shell">{body}</action>'
    actions = parse(text)
    assert actions[0].body == body


def test_raw_ampersand_in_body_is_fine():
    body = 'curl "https://example.com/?a=1&b=2&c=3"'
    text = f'<action tool="shell">{body}</action>'
    actions = parse(text)
    assert actions[0].body == body


def test_actionable_does_not_match():
    """Words like `<actionable>` must not be treated as <action>."""
    text = '<actionable>nope</actionable>\n<action tool="list_directory" />'
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


def test_nested_action_in_body_handled():
    """A literal `<action>` inside a body should be paired correctly via depth-counting."""
    inner = '<action tool="rlm_query">child task body</action>'
    body = f"Here is a nested example: {inner}"
    text = f'<action tool="rlm_query">{body}</action>'
    actions = parse(text)
    assert len(actions) == 1
    # body contains the nested element verbatim
    assert "child task body" in actions[0].body


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_no_actions_raises():
    with pytest.raises(ActionParseError):
        parse("just some prose, no actions here")


def test_missing_tool_attr_raises():
    with pytest.raises(ActionParseError, match="tool"):
        parse("<action />")


def test_unknown_tool_raises():
    with pytest.raises(ActionParseError, match="Unknown tool"):
        parse('<action tool="not_a_real_tool" />')


def test_self_closing_required_body_raises():
    with pytest.raises(ActionParseError, match="self-closing"):
        parse('<action tool="write_file" path="x" />')


def test_empty_body_for_required_body_raises():
    with pytest.raises(ActionParseError, match="non-empty body"):
        parse('<action tool="shell">   </action>')


def test_unknown_attribute_raises():
    with pytest.raises(ActionParseError, match="Unknown attribute"):
        parse('<action tool="read_file" path="x" bogus="y" />')


def test_missing_required_attr_raises():
    with pytest.raises(ActionParseError, match="Missing required"):
        parse('<action tool="write_file">body</action>')


def test_unterminated_action_raises():
    with pytest.raises(ActionParseError, match="Unterminated"):
        parse('<action tool="shell">body without close')


# ---------------------------------------------------------------------------
# edit_file body extraction
# ---------------------------------------------------------------------------


def test_parse_edit_file_body_basic():
    body = "<search>foo</search><replace>bar</replace>"
    s, r = parse_edit_file_body(body)
    assert s == "foo"
    assert r == "bar"


def test_parse_edit_file_body_with_code_inside():
    body = "<search>def foo():\n    return 1</search><replace>def foo():\n    return 2</replace>"
    s, r = parse_edit_file_body(body)
    assert "return 1" in s
    assert "return 2" in r


def test_parse_edit_file_body_missing_search_raises():
    with pytest.raises(ActionParseError):
        parse_edit_file_body("<replace>x</replace>")


def test_parse_edit_file_body_missing_replace_raises():
    with pytest.raises(ActionParseError):
        parse_edit_file_body("<search>x</search>")


# ---------------------------------------------------------------------------
# final body extraction
# ---------------------------------------------------------------------------


def test_parse_final_body_no_artifacts():
    answer, artifacts = parse_final_body("<answer>The answer is 42.</answer>")
    assert answer == "The answer is 42."
    assert artifacts == []


def test_parse_final_body_with_artifacts():
    body = (
        "<answer>See attached files.</answer>"
        '<artifact path="results/summary.md" />'
        '<artifact path="results/data.csv" />'
    )
    answer, artifacts = parse_final_body(body)
    assert answer == "See attached files."
    assert artifacts == ["results/summary.md", "results/data.csv"]


def test_parse_final_body_missing_answer_raises():
    with pytest.raises(ActionParseError):
        parse_final_body('<artifact path="x.txt" />')


def test_parse_final_body_artifact_missing_path_raises():
    with pytest.raises(ActionParseError):
        parse_final_body("<answer>ok</answer><artifact />")


# ---------------------------------------------------------------------------
# Additional edge cases
# ---------------------------------------------------------------------------


def test_mixed_self_closing_and_paired_preserve_order():
    """A response with a self-closing then paired then self-closing keeps order."""
    text = (
        '<action tool="list_directory" />\n'
        '<action tool="write_file" path="a.txt">hello</action>\n'
        '<action tool="read_file" path="a.txt" />'
    )
    actions = parse(text)
    assert [a.tool for a in actions] == ["list_directory", "write_file", "read_file"]
    assert actions[1].body == "hello"


def test_adjacent_actions_with_no_whitespace_between():
    """Two actions back-to-back with no separator must both parse."""
    text = '<action tool="list_directory" /><action tool="read_file" path="x.txt" />'
    actions = parse(text)
    assert len(actions) == 2
    assert actions[0].tool == "list_directory"
    assert actions[1].args["path"] == "x.txt"


def test_stray_close_tag_inside_body_closes_outer():
    """The parser is a tag-pair scanner that depth-counts nested *openings*,
    not closings. A stray ``</action>`` inside a body (e.g. inside a code
    string literal) will therefore close the outer action prematurely; this
    is the documented behavior. Lock it in so a future "fix" doesn't silently
    change semantics for already-trained agents.
    """
    # Body literally contains "</action>" — outer should close at the first
    # one, leaving "leftover</action>" as a parse-failure tail.
    text = '<action tool="shell">echo "</action>" leftover</action>'
    actions = parse(text)
    # First action's body ends at the first </action>; the rest is unparsed
    # text after it, which contains no further valid <action> openings, so
    # parse() returns the single first action. The trailing "leftover</action>"
    # is harmless prose to the scanner.
    assert len(actions) == 1
    assert actions[0].body == 'echo "'


def test_edit_file_allow_multiple_attribute_parses():
    """The optional ``allow_multiple`` attribute on edit_file is recognized."""
    body = "<search>x</search><replace>y</replace>"
    text = f'<action tool="edit_file" path="a.py" allow_multiple="true">{body}</action>'
    actions = parse(text)
    assert actions[0].args == {"path": "a.py", "allow_multiple": "true"}
    s, r = parse_edit_file_body(actions[0].body)
    assert s == "x" and r == "y"


def test_single_quoted_attribute_values_not_supported():
    """The attribute regex only matches double-quoted values. Single-quoted
    values yield a missing-required-attribute error. This locks in current
    behavior; if support is added later, change this test deliberately."""
    text = "<action tool='list_directory' />"
    # Whole tag is unrecognized because the `tool` attr did not parse.
    with pytest.raises(ActionParseError):
        parse(text)


def test_action_tag_is_case_insensitive():
    """``<ACTION>``/``<Action>`` are accepted (the open regex is IGNORECASE)."""
    text = '<ACTION tool="list_directory" />'
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"
