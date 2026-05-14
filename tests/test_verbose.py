from __future__ import annotations

from io import StringIO

from rich.console import Console

from rlm.logger.verbose import VerbosePrinter


def test_summary_prints_actual_over_max_iterations() -> None:
    stream = StringIO()
    printer = VerbosePrinter(enabled=True)
    printer.console = Console(file=stream, force_terminal=False, width=100)

    printer.print_summary(26, 1.23, max_iterations=30)

    assert "Iterations" in stream.getvalue()
    assert "26/30" in stream.getvalue()
