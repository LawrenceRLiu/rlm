"""
Action parser for the workspace substrate.

Extracts ``<action ...>...</action>`` and self-closing ``<action ... />`` blocks
from an LM response. Implemented as a tolerant tag-pair scanner (depth-counted),
not strict XML — bodies routinely contain raw ``<``, ``&``, code, and HTML, all
of which would crash ``xml.etree``.

Key properties:
- Prose around action tags is allowed and ignored (preserved by the caller).
- Inside an ``<action>``, nested ``<action>`` openings increment depth so the
  matching ``</action>`` is found correctly.
- For ``edit_file``, the body is rescanned with the same depth-counting trick
  to extract ``<search>...</search>`` / ``<replace>...</replace>``.

The parser does not call out to the network, the filesystem, or any backend.
It is pure: ``parse(text) -> list[WorkspaceAction]``. Schema validation rejects
malformed actions with ``ActionParseError`` for the caller's retry loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rlm.core.types import WorkspaceAction
from rlm.utils.exceptions import ActionParseError

# ---------------------------------------------------------------------------
# Per-tool argument schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSchema:
    """Per-tool schema for action validation."""

    name: str
    self_closing_allowed: bool
    body_required: bool
    required_attrs: tuple[str, ...] = ()
    optional_attrs: tuple[str, ...] = ()


# Schemas for the 10 v0.1 tools. Web tools (web_search, fetch_url) deferred.
_DEFAULT_SCHEMAS: dict[str, ActionSchema] = {
    "list_directory": ActionSchema(
        name="list_directory",
        self_closing_allowed=True,
        body_required=False,
        optional_attrs=("path",),
    ),
    "read_file": ActionSchema(
        name="read_file",
        self_closing_allowed=True,
        body_required=False,
        required_attrs=("path",),
        optional_attrs=("start_line", "end_line"),
    ),
    "write_file": ActionSchema(
        name="write_file",
        self_closing_allowed=False,
        body_required=True,
        required_attrs=("path",),
    ),
    "append_file": ActionSchema(
        name="append_file",
        self_closing_allowed=False,
        body_required=True,
        required_attrs=("path",),
    ),
    "edit_file": ActionSchema(
        name="edit_file",
        self_closing_allowed=False,
        body_required=True,
        required_attrs=("path",),
        optional_attrs=("allow_multiple",),
    ),
    "shell": ActionSchema(
        name="shell",
        self_closing_allowed=False,
        body_required=True,
    ),
    "python": ActionSchema(
        name="python",
        self_closing_allowed=False,
        body_required=True,
    ),
    "llm_query": ActionSchema(
        name="llm_query",
        self_closing_allowed=False,
        body_required=True,
    ),
    "rlm_query": ActionSchema(
        name="rlm_query",
        self_closing_allowed=False,
        body_required=True,
    ),
    "final": ActionSchema(
        name="final",
        self_closing_allowed=False,
        body_required=True,
    ),
}


def default_schemas() -> dict[str, ActionSchema]:
    """Return a fresh copy of the default schema registry."""
    return dict(_DEFAULT_SCHEMAS)


# ---------------------------------------------------------------------------
# Tag scanner
# ---------------------------------------------------------------------------

# Matches the opening of a `<action ...>` or `<action />` element. The element
# name must be exactly `action`. Attributes are not parsed here; we just locate
# the start tag and capture its full opening fragment.
_OPEN_RE = re.compile(r"<\s*action\b", re.IGNORECASE)
_CLOSE_RE = re.compile(r"</\s*action\s*>", re.IGNORECASE)
_ATTR_RE = re.compile(r"(\w+)\s*=\s*\"([^\"]*)\"")

# Reasoning-block tags emitted by Qwen3 / Qwen3.5 / Qwen3.6 / R1-style models
# when `enable_thinking=True` (the default for the Qwen3 family). Anything
# inside <think>...</think> is the model's private monologue and must not
# feed the <action> scanner — see 2026-05-10 Qwen sweep reports.
_THINK_PAIR_RE = re.compile(
    r"<\s*think\b[^>]*>.*?<\s*/\s*think\s*>",
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(r"<\s*think\b[^>]*>", flags=re.IGNORECASE)

# Gemma 4 emits reasoning blocks with **asymmetric** channel special tokens:
# start ``<|channel>`` (no trailing pipe), end ``<channel|>`` (no leading pipe).
# When vLLM is run without ``--reasoning-parser gemma4`` (or when the parser
# is buggy — see vllm#38855 in 0.19.1), the tokens detokenize as literal text
# into ``content``. The pattern matches the inner ``thought`` role label so
# an unrelated future channel (e.g. ``<|channel>output``) is not accidentally
# consumed. Token literals confirmed by reading
# ``vllm/reasoning/gemma4_reasoning_parser.py:start_token`` /
# ``end_token`` and by a raw ``/v1/completions`` probe against the served
# replica on 2026-05-11.
_CHANNEL_PAIR_RE = re.compile(
    r"<\|channel>\s*thought\b.*?<channel\|>",
    flags=re.DOTALL,
)
_CHANNEL_OPEN_RE = re.compile(r"<\|channel>\s*thought\b")


@dataclass(frozen=True)
class _OpenTag:
    """A single `<action ...>` opening tag, located in the response."""

    start: int  # index of `<` in the source
    end: int  # index just past `>`
    inner: str  # text between `<action` (exclusive) and `>` (exclusive)
    self_closing: bool


def _find_open_tag(text: str, pos: int) -> _OpenTag | None:
    """Find the next ``<action ...>`` opening at or after ``pos``."""
    while True:
        m = _OPEN_RE.search(text, pos)
        if not m:
            return None

        # `<action` matched; ensure the next char is whitespace, `>`, or `/`.
        # Otherwise it is something like `<actionable>` — keep scanning.
        next_idx = m.end()
        if next_idx >= len(text):
            return None
        ch = text[next_idx]
        if ch not in (" ", "\t", "\n", "\r", ">", "/"):
            pos = next_idx
            continue

        # Walk forward to find the closing `>`, respecting quoted attribute
        # values so that a `>` inside `"..."` does not close the tag.
        i = next_idx
        in_quote = False
        while i < len(text):
            c = text[i]
            if c == '"':
                in_quote = not in_quote
            elif c == ">" and not in_quote:
                break
            i += 1
        if i >= len(text):
            # Truncated tag; treat as a parse miss and stop.
            return None

        inner = text[next_idx:i].strip()
        self_closing = inner.endswith("/")
        if self_closing:
            inner = inner[:-1].rstrip()

        return _OpenTag(start=m.start(), end=i + 1, inner=inner, self_closing=self_closing)


def _find_matching_close(text: str, body_start: int) -> int:
    """Find the index of `</action>` matching an opener whose body starts at
    ``body_start``. Counts nested `<action>` openings as depth +1 each.

    Returns the index of `<` in the matched `</action>`. Raises
    ``ActionParseError`` if no matching close exists.
    """
    pos = body_start
    depth = 0
    while pos < len(text):
        # Find the next opener and the next closer; whichever comes first wins.
        next_open = _find_open_tag(text, pos)
        m_close = _CLOSE_RE.search(text, pos)
        if m_close is None:
            raise ActionParseError(
                "Unterminated <action> element: no matching </action> found.",
                fragment=text[max(0, body_start - 40) : body_start + 200],
            )
        if next_open is not None and next_open.start < m_close.start():
            if next_open.self_closing:
                # Self-closing siblings do not affect depth.
                pos = next_open.end
                continue
            depth += 1
            pos = next_open.end
            continue
        if depth == 0:
            return m_close.start()
        depth -= 1
        pos = m_close.end()
    raise ActionParseError(
        "Unterminated <action> element: no matching </action> found.",
        fragment=text[max(0, body_start - 40) : body_start + 200],
    )


def _parse_attrs(inner: str) -> dict[str, str]:
    """Parse ``key="value"`` attributes from the opening tag's inner text."""
    return {m.group(1): m.group(2) for m in _ATTR_RE.finditer(inner)}


def _extract_tool_attr(attrs: dict[str, str]) -> str | None:
    """Return the value of the ``tool`` attribute, or None if absent."""
    return attrs.get("tool")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def strip_reasoning_blocks(text: str) -> str:
    """Remove model-emitted reasoning spans before action scanning or replay.

    Covers two formats:
    - Qwen3-family ``<think>...</think>`` (plain text tags).
    - Gemma 4 ``<|channel|>thought ... <|channel|>`` (special-token channel
      block; appears as literal text when vLLM's ``--reasoning-parser`` is
      not configured).

    Why: these models emit long self-corrective monologue inside the block.
    The monologue routinely contains prior malformed ``<action>`` attempts
    ("I tried X, that didn't work, let me try Y"). The action scanner is a
    forward regex walk and would dispatch the first stale attempt before
    reaching the model's intended action below. Stripping is independent of
    vLLM's ``--reasoning-parser`` so the substrate is robust regardless of
    serving config. An unterminated open tag (no matching close) drops the
    rest of the text — better than letting half the monologue feed the
    scanner or be replayed verbatim into the next turn's input.
    """
    text = _THINK_PAIR_RE.sub("", text)
    text = _CHANNEL_PAIR_RE.sub("", text)
    earliest_open: int | None = None
    for pattern in (_THINK_OPEN_RE, _CHANNEL_OPEN_RE):
        m = pattern.search(text)
        if m is None:
            continue
        if earliest_open is None or m.start() < earliest_open:
            earliest_open = m.start()
    if earliest_open is not None:
        text = text[:earliest_open]
    return text


# Backwards-compatible alias for any external callers / tests that referenced
# the old name. Internal callers use ``strip_reasoning_blocks`` directly.
_strip_think_blocks = strip_reasoning_blocks


def parse(text: str, schemas: dict[str, ActionSchema] | None = None) -> list[WorkspaceAction]:
    """Parse the **last contiguous block** of ``<action>`` elements from ``text``.

    Strips ``<think>`` reasoning blocks first, then walks the rest tolerating
    per-action schema failures: malformed ``<action>`` elements earlier in the
    response (e.g. backticked examples in prose, system-prompt quotations) are
    skipped rather than aborting the whole turn. The returned list is the last
    cluster of well-formed actions separated only by whitespace — models'
    self-correcting reasoning concludes with the intended action *at the end*.

    Raises ``ActionParseError`` only if zero well-formed actions are recovered.
    In that case the first per-action schema failure is re-raised so the model
    still gets meaningful feedback for its retry; otherwise the standard
    "no <action> elements" error fires. Structural malformation (unterminated
    ``<action>``) always bubbles immediately.
    """
    schemas = schemas if schemas is not None else _DEFAULT_SCHEMAS
    text = strip_reasoning_blocks(text)

    parsed: list[tuple[int, int, WorkspaceAction]] = []
    first_validation_error: ActionParseError | None = None
    pos = 0
    while pos < len(text):
        opener = _find_open_tag(text, pos)
        if opener is None:
            break

        # Locate structural extent first; an unterminated <action> is a real
        # structural error and propagates rather than getting silently skipped.
        if opener.self_closing:
            close_end = opener.end
            body: str | None = None
        else:
            close_start = _find_matching_close(text, opener.end)
            close_match = _CLOSE_RE.match(text, close_start)
            if close_match is None:  # pragma: no cover — defensive
                raise ActionParseError(
                    "Internal parser error: lost close tag for <action>.",
                    fragment=text[opener.start : close_start + 20],
                )
            close_end = close_match.end()
            body = text[opener.end : close_start]

        try:
            action = _validate_action(text, opener, body, schemas)
        except ActionParseError as exc:
            if first_validation_error is None:
                first_validation_error = exc
            pos = close_end
            continue

        parsed.append((opener.start, close_end, action))
        pos = close_end

    if not parsed:
        if first_validation_error is not None:
            raise first_validation_error
        raise ActionParseError(
            "No <action> elements found in response. The model must emit at "
            "least one well-formed <action> per turn."
        )

    # Group into clusters: consecutive actions separated only by whitespace.
    # Return the last cluster. With a single action or all-adjacent actions
    # the result is identical to the legacy "return all" behavior.
    last_cluster_start = 0
    for i in range(1, len(parsed)):
        between = text[parsed[i - 1][1] : parsed[i][0]]
        if between.strip():
            last_cluster_start = i
    return [a for (_, _, a) in parsed[last_cluster_start:]]


def _validate_action(
    text: str,
    opener: _OpenTag,
    body: str | None,
    schemas: dict[str, ActionSchema],
) -> WorkspaceAction:
    """Validate a single ``<action>`` extent against its tool's schema.

    Raises ``ActionParseError`` on any per-action schema failure; the caller in
    ``parse()`` decides whether to skip or surface the error.
    """
    attrs = _parse_attrs(opener.inner)
    tool = _extract_tool_attr(attrs)
    if not tool:
        raise ActionParseError(
            "Missing required attribute 'tool' on <action> element.",
            fragment=text[opener.start : opener.end + 200],
        )
    if tool not in schemas:
        raise ActionParseError(
            f"Unknown tool '{tool}' on <action>. Known tools: {sorted(schemas.keys())}.",
            fragment=text[opener.start : opener.end + 200],
        )

    schema = schemas[tool]
    action_args = {k: v for k, v in attrs.items() if k != "tool"}

    allowed = set(schema.required_attrs) | set(schema.optional_attrs)
    unknown = set(action_args.keys()) - allowed
    if unknown:
        raise ActionParseError(
            f"Unknown attribute(s) {sorted(unknown)} on <action tool='{tool}'>. "
            f"Allowed: {sorted(allowed)}.",
            fragment=text[opener.start : opener.end + 200],
        )
    missing = [a for a in schema.required_attrs if a not in action_args]
    if missing:
        raise ActionParseError(
            f"Missing required attribute(s) {missing} on <action tool='{tool}'>.",
            fragment=text[opener.start : opener.end + 200],
        )

    if opener.self_closing:
        if not schema.self_closing_allowed:
            raise ActionParseError(
                f"<action tool='{tool}'> may not be self-closing; this tool requires a body.",
                fragment=text[opener.start : opener.end],
            )
        return WorkspaceAction(
            tool=tool,
            args=action_args,
            body=None,
            raw=text[opener.start : opener.end],
        )

    assert body is not None  # paired tag always has a body extent
    if schema.body_required and not body.strip():
        raise ActionParseError(
            f"<action tool='{tool}'> requires a non-empty body.",
            fragment=text[opener.start : opener.end + len(body) + 20],
        )
    return WorkspaceAction(
        tool=tool,
        args=action_args,
        body=body,
        raw=text[opener.start : opener.end + len(body) + len("</action>")],
    )


# ---------------------------------------------------------------------------
# edit_file body extraction (nested <search>/<replace> pairs)
# ---------------------------------------------------------------------------


def _generic_tag_pair(text: str, tag: str) -> tuple[int, int, int, int] | None:
    """Locate a single ``<tag>...</tag>`` pair using depth-counted scanning.

    Returns ``(open_start, open_end, close_start, close_end)`` or None if no
    well-formed pair exists. Nested ``<tag>`` openings increment depth.
    """
    open_re = re.compile(rf"<\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    close_re = re.compile(rf"</\s*{re.escape(tag)}\s*>", re.IGNORECASE)

    m = open_re.search(text)
    if m is None:
        return None
    open_start = m.start()
    open_end = m.end()

    pos = open_end
    depth = 0
    while pos < len(text):
        n_open = open_re.search(text, pos)
        n_close = close_re.search(text, pos)
        if n_close is None:
            return None
        if n_open is not None and n_open.start() < n_close.start():
            depth += 1
            pos = n_open.end()
            continue
        if depth == 0:
            return (open_start, open_end, n_close.start(), n_close.end())
        depth -= 1
        pos = n_close.end()
    return None


def parse_edit_file_body(body: str) -> tuple[str, str]:
    """Extract ``<search>...</search>`` and ``<replace>...</replace>`` from an
    ``edit_file`` action body.

    Raises ``ActionParseError`` if either tag is missing or unbalanced.
    """
    s = _generic_tag_pair(body, "search")
    if s is None:
        raise ActionParseError(
            "edit_file body must contain a <search>...</search> block.",
            fragment=body[:300],
        )
    r = _generic_tag_pair(body, "replace")
    if r is None:
        raise ActionParseError(
            "edit_file body must contain a <replace>...</replace> block.",
            fragment=body[:300],
        )
    search_text = body[s[1] : s[2]]
    replace_text = body[r[1] : r[2]]
    return search_text, replace_text


# ---------------------------------------------------------------------------
# final body extraction (<answer> + zero or more <artifact path="..." />)
# ---------------------------------------------------------------------------

_ARTIFACT_OPEN_RE = re.compile(r"<\s*artifact\b", re.IGNORECASE)


def _find_self_closing_artifacts(text: str) -> list[dict[str, str]]:
    """Find all ``<artifact ... />`` self-closing tags in ``text``.

    Walks each opening, then scans for the terminating ``/>`` while honouring
    quoted attribute values so that a ``/`` inside ``path="a/b"`` is not
    mistaken for the close.
    """
    out: list[dict[str, str]] = []
    pos = 0
    while True:
        m = _ARTIFACT_OPEN_RE.search(text, pos)
        if not m:
            return out
        i = m.end()
        in_quote = False
        attr_end = -1
        while i < len(text):
            c = text[i]
            if c == '"':
                in_quote = not in_quote
            elif not in_quote and c == "/" and i + 1 < len(text):
                # Skip whitespace between '/' and '>'.
                j = i + 1
                while j < len(text) and text[j] in (" ", "\t", "\n", "\r"):
                    j += 1
                if j < len(text) and text[j] == ">":
                    attr_end = i
                    pos = j + 1
                    break
            i += 1
        if attr_end < 0:
            # No close found; stop scanning.
            return out
        attrs: dict[str, Any] = _parse_attrs(text[m.end() : attr_end])
        out.append(attrs)


def parse_final_body(body: str) -> tuple[str, list[str]]:
    """Extract the ``<answer>`` text and any ``<artifact path="..." />``
    children from a ``final`` action body.

    Raises ``ActionParseError`` if ``<answer>`` is missing.
    """
    a = _generic_tag_pair(body, "answer")
    if a is None:
        raise ActionParseError(
            "final body must contain an <answer>...</answer> block.",
            fragment=body[:300],
        )
    answer = body[a[1] : a[2]].strip()

    artifacts: list[str] = []
    for attrs in _find_self_closing_artifacts(body):
        path = attrs.get("path")
        if not path:
            raise ActionParseError(
                "<artifact> requires a 'path' attribute.",
                fragment=str(attrs),
            )
        artifacts.append(path)

    return answer, artifacts
