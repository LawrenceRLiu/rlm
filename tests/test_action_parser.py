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
    """Prose before and after an action does not crash the parser. (Prose
    *between* two actions is a cluster boundary under the last-contiguous-block
    rule — see test_multiple_clusters_returns_last_only.)"""
    text = (
        "Let me think about this.\n\n"
        '<action tool="read_file" path="a.txt" />\n\n'
        "Now I have the file contents."
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].args["path"] == "a.txt"


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


# ---------------------------------------------------------------------------
# Think-block stripping (Bug 1 — Qwen3 self-corrective monologue poisoning)
# ---------------------------------------------------------------------------


def test_think_block_with_malformed_actions_is_stripped():
    """Stale ``<action>`` attempts inside ``<think>`` must not crash the
    scanner — the real well-formed action after the think wins."""
    text = (
        '<think>\nLet me try <action tool="read_file">_rlm_query_0.txt</action>'
        " — no wait, that's missing the path attribute. Let me try"
        ' <action tool="read_file" path="_rlm_query_0.txt">.\n</think>\n'
        '<action tool="read_file" path="_rlm_query_0.txt" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "read_file"
    assert actions[0].args["path"] == "_rlm_query_0.txt"


def test_unterminated_think_block_drops_tail():
    """A ``<think>`` with no closing tag is treated as runaway monologue; the
    parser drops everything from there to end-of-text. The pre-think action
    still parses; nothing else can leak through."""
    text = (
        '<action tool="list_directory" />\n'
        "<think>\nrunaway reasoning <action tool='bogus'>never closed"
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


def test_think_block_case_insensitive_and_with_attrs():
    text = (
        '<THINK id="1">malformed <action tool="x"></action> here</think>\n'
        '<action tool="list_directory" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


# ---------------------------------------------------------------------------
# Channel-block stripping (Gemma 4 — <|channel>thought ... <channel|>)
# ---------------------------------------------------------------------------


def test_channel_block_with_malformed_actions_is_stripped():
    """Gemma 4 emits reasoning inside ``<|channel>thought ... <channel|>``
    blocks (note the asymmetric special tokens: ``<|channel>`` opens,
    ``<channel|>`` closes). When served without ``--reasoning-parser gemma4``
    (or with a buggy parser), they appear as literal text. Stale
    ``<action>`` attempts inside the channel must not poison the scanner."""
    text = (
        '<|channel>thought\nLet me try <action tool="read_file">_rlm_query_0.txt</action>'
        " — that's missing the path. I should use"
        ' <action tool="read_file" path="_rlm_query_0.txt"/>.\n<channel|>\n'
        '<action tool="read_file" path="_rlm_query_0.txt" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "read_file"
    assert actions[0].args["path"] == "_rlm_query_0.txt"


def test_unterminated_channel_block_drops_tail():
    """A ``<|channel>thought`` with no closing ``<channel|>`` is treated as
    runaway monologue; everything from there to end-of-text is dropped."""
    text = (
        '<action tool="list_directory" />\n'
        "<|channel>thought\nrunaway reasoning <action tool='bogus'>never closed"
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


def test_channel_and_think_mixed():
    """Both reasoning formats can appear in a single response if a model is
    transitioning between substrates; both must be stripped before scanning."""
    text = (
        "<think>qwen-style monologue</think>\n"
        "<|channel>thought\ngemma-style monologue\n<channel|>\n"
        '<action tool="list_directory" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


# ---------------------------------------------------------------------------
# Last-contiguous-block selection (Bug 2 — prose/code-fence quotations)
# ---------------------------------------------------------------------------


def test_last_cluster_wins_over_quoted_prose_example():
    """A backticked example of ``<action>`` in prose must not pre-empt the
    real action emitted afterward."""
    text = (
        'Here\'s what the format looks like: `<action tool="read_file" path="a.txt" />`.\n'
        "Now the real call:\n"
        '<action tool="list_directory" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].tool == "list_directory"


def test_last_cluster_wins_over_system_prompt_quotation():
    """The Qwen3.5-9B 3b failure pattern: model quotes the system prompt's
    format example (`<action tool="...">...</action>`) before emitting its real
    action. The quoted example has tool='...' which would be Unknown tool;
    must be skipped in favor of the real action below."""
    text = (
        "The system prompt says: Each turn, emit one or more "
        '``<action tool="...">...</action>`` elements. So I should use that '
        "format. Here:\n"
        '<action tool="read_file" path="x.txt" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].args["path"] == "x.txt"


def test_multiple_clusters_returns_last_only():
    """Two clusters separated by prose: only the last is returned."""
    text = (
        '<action tool="list_directory" />\n'
        '<action tool="read_file" path="a.txt" />\n'
        "Hmm wait, I need to do something else first.\n"
        '<action tool="write_file" path="b.txt">hello</action>\n'
        '<action tool="read_file" path="b.txt" />'
    )
    actions = parse(text)
    assert [a.tool for a in actions] == ["write_file", "read_file"]
    assert actions[0].args["path"] == "b.txt"
    assert actions[1].args["path"] == "b.txt"


def test_single_cluster_unchanged():
    """Legacy single-cluster response: behavior identical to pre-cluster impl."""
    text = '<action tool="list_directory" />\n\n<action tool="read_file" path="a.txt" />'
    actions = parse(text)
    assert len(actions) == 2
    assert actions[0].tool == "list_directory"
    assert actions[1].tool == "read_file"


def test_sole_malformed_action_still_raises_validation_error():
    """When the response contains only a malformed action (no later well-formed
    cluster to fall back on), the original validation error is surfaced so
    the model gets actionable feedback — not silently swallowed."""
    with pytest.raises(ActionParseError, match="Missing required attribute 'tool'"):
        parse("<action />")
    with pytest.raises(ActionParseError, match="Unknown tool"):
        parse('<action tool="not_a_real_tool" />')


def test_malformed_example_in_prose_does_not_mask_real_action():
    """Earlier malformed examples should be skipped, and the real well-formed
    action after them should be returned (not a validation error)."""
    text = (
        'Bad attempt: <action tool="read_file">forgot path here</action>\n'
        "Real call:\n"
        '<action tool="read_file" path="x.txt" />'
    )
    actions = parse(text)
    assert len(actions) == 1
    assert actions[0].args["path"] == "x.txt"


def test_structural_error_still_propagates_even_with_good_action_after():
    """An unterminated ``<action>`` is a structural error and cannot be
    skipped (its body would swallow whatever comes after). The scanner
    must surface this rather than silently consume the rest of the text."""
    text = '<action tool="shell">forgot to close\n<action tool="list_directory" />'
    # The first <action>'s body consumption walks until it finds </action>,
    # which it never does — Unterminated raised.
    with pytest.raises(ActionParseError, match="Unterminated"):
        parse(text)
