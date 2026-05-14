"""Verbose printing for RLM using rich.

Provides console output for the workspace-substrate driver: configuration
header, limit/budget warnings, final answer, and per-run summary. Uses a
"Tokyo Night" inspired color theme.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

from rlm.core.types import RLMMetadata

# Tokyo Night colour palette
COLORS = {
    "primary": "#7AA2F7",
    "secondary": "#BB9AF7",
    "success": "#9ECE6A",
    "warning": "#E0AF68",
    "error": "#F7768E",
    "text": "#A9B1D6",
    "muted": "#565F89",
    "accent": "#7DCFFF",
    "bg_subtle": "#1A1B26",
    "border": "#3B4261",
    "code_bg": "#24283B",
}

STYLE_PRIMARY = Style(color=COLORS["primary"], bold=True)
STYLE_SECONDARY = Style(color=COLORS["secondary"])
STYLE_SUCCESS = Style(color=COLORS["success"])
STYLE_WARNING = Style(color=COLORS["warning"])
STYLE_ERROR = Style(color=COLORS["error"])
STYLE_TEXT = Style(color=COLORS["text"])
STYLE_MUTED = Style(color=COLORS["muted"])
STYLE_ACCENT = Style(color=COLORS["accent"], bold=True)


def _to_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


class VerbosePrinter:
    """Rich console printer used by the workspace-substrate ``RLM`` loop."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.console = Console() if enabled else None

    def print_header(
        self,
        backend: str,
        model: str,
        environment: str,
        max_iterations: int,
        max_depth: int,
        action_format: str,
        other_backends: list[str] | None = None,
    ) -> None:
        if not self.enabled:
            return

        title = Text()
        title.append("◆ ", style=STYLE_ACCENT)
        title.append("RLM", style=Style(color=COLORS["primary"], bold=True))
        title.append(" ━ Recursive Language Model", style=STYLE_MUTED)

        config_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
            expand=True,
        )
        config_table.add_column("key", style=STYLE_MUTED, width=16)
        config_table.add_column("value", style=STYLE_TEXT)
        config_table.add_column("key2", style=STYLE_MUTED, width=16)
        config_table.add_column("value2", style=STYLE_TEXT)

        config_table.add_row(
            "Backend",
            Text(backend, style=STYLE_SECONDARY),
            "Environment",
            Text(environment, style=STYLE_SECONDARY),
        )
        config_table.add_row(
            "Model",
            Text(model, style=STYLE_ACCENT),
            "Action Format",
            Text(action_format, style=STYLE_ACCENT),
        )
        config_table.add_row(
            "Max Iterations",
            Text(str(max_iterations), style=STYLE_WARNING),
            "Max Depth",
            Text(str(max_depth), style=STYLE_WARNING),
        )

        if other_backends:
            backends_text = Text(", ".join(other_backends), style=STYLE_SECONDARY)
            config_table.add_row(
                "Sub-models",
                backends_text,
                "",
                "",
            )

        panel = Panel(
            config_table,
            title=title,
            title_align="left",
            border_style=COLORS["border"],
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_metadata(self, metadata: RLMMetadata) -> None:
        if not self.enabled:
            return
        model = metadata.backend_kwargs.get("model_name", "unknown")
        other = list(metadata.other_backends) if metadata.other_backends else None
        self.print_header(
            backend=metadata.backend,
            model=model,
            environment=metadata.environment_type,
            max_iterations=metadata.max_iterations,
            max_depth=metadata.max_depth,
            action_format=metadata.action_format,
            other_backends=other,
        )

    def print_budget_exceeded(self, spent: float, budget: float) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("⚠ ", style=STYLE_ERROR)
        title.append("Budget Exceeded", style=Style(color=COLORS["error"], bold=True))
        content = Text()
        content.append(f"Spent: ${spent:.6f}\n", style=STYLE_ERROR)
        content.append(f"Budget: ${budget:.6f}", style=STYLE_MUTED)
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=COLORS["error"],
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_limit_exceeded(self, limit_type: str, details: str) -> None:
        if not self.enabled:
            return
        limit_names = {
            "timeout": "Timeout Exceeded",
            "tokens": "Token Limit Exceeded",
            "errors": "Error Threshold Exceeded",
            "cancelled": "Execution Cancelled",
        }
        display_name = limit_names.get(limit_type, f"{limit_type.title()} Limit Exceeded")
        title = Text()
        title.append("⚠ ", style=STYLE_ERROR)
        title.append(display_name, style=Style(color=COLORS["error"], bold=True))
        content = Text(details, style=STYLE_ERROR)
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=COLORS["error"],
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_compaction(self, *, turn: int, tokens_before: int, threshold: int) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("⤓ ", style=STYLE_ACCENT)
        title.append("History compacted", style=Style(color=COLORS["accent"], bold=True))
        content = Text()
        content.append(f"Turn: {turn}\n", style=STYLE_MUTED)
        content.append(f"Tokens before: {tokens_before:,}\n", style=STYLE_TEXT)
        content.append(f"Threshold: {threshold:,}", style=STYLE_MUTED)
        panel = Panel(
            content,
            title=title,
            title_align="left",
            border_style=COLORS["accent"],
            padding=(0, 2),
        )
        self.console.print()
        self.console.print(panel)

    def print_final_answer(self, answer: Any) -> None:
        if not self.enabled:
            return
        title = Text()
        title.append("★ ", style=STYLE_WARNING)
        title.append("Final Answer", style=Style(color=COLORS["warning"], bold=True))
        answer_text = Text(_to_str(answer), style=STYLE_TEXT)
        panel = Panel(
            answer_text,
            title=title,
            title_align="left",
            border_style=COLORS["warning"],
            padding=(1, 2),
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()

    def print_summary(
        self,
        total_iterations: int,
        total_time: float,
        usage_summary: dict[str, Any] | None = None,
        max_iterations: int | None = None,
    ) -> None:
        if not self.enabled:
            return
        summary_table = Table(
            show_header=False,
            show_edge=False,
            box=None,
            padding=(0, 2),
        )
        summary_table.add_column("metric", style=STYLE_MUTED)
        summary_table.add_column("value", style=STYLE_ACCENT)
        iterations = (
            f"{total_iterations}/{max_iterations}"
            if max_iterations is not None
            else str(total_iterations)
        )
        summary_table.add_row("Iterations", iterations)
        summary_table.add_row("Total Time", f"{total_time:.2f}s")

        if usage_summary:
            total_input = sum(
                m.get("total_input_tokens", 0)
                for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            total_output = sum(
                m.get("total_output_tokens", 0)
                for m in usage_summary.get("model_usage_summaries", {}).values()
            )
            total_cost = usage_summary.get("total_cost")
            if total_input or total_output:
                summary_table.add_row("Input Tokens", f"{total_input:,}")
                summary_table.add_row("Output Tokens", f"{total_output:,}")
            if total_cost is not None:
                summary_table.add_row("Total Cost", f"${total_cost:.6f}")

        self.console.print()
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print(summary_table, justify="center")
        self.console.print(Rule(style=COLORS["border"], characters="═"))
        self.console.print()
