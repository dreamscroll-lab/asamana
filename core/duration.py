"""Pure duration-rendering primitive, in ``core`` so both ``engine`` and ``agent`` can use it."""

from __future__ import annotations


def describe_duration(steps: int, seconds_per_step: int) -> str:
    """Render a step count as a natural-language duration ("约3小时"); steps never reach
    narrative text (see CLAUDE.md's time axis).

    The result already carries "约": callers must not add another ("约约2小时" ends up in memory
    text). Guard: `grep -rn "约{" engine/ agent/` should match nothing outside this file.
    """
    return describe_seconds(max(steps, 0) * max(seconds_per_step, 0))


def describe_seconds(seconds: int) -> str:
    """Render a span of seconds as a natural-language duration; carries "约" like
    ``describe_duration``."""
    total = max(seconds, 0)
    if total < 60:
        return f"约{total}秒"
    # Precise to the minute: dropping the remainder of a larger unit would show a 1h59m walk as
    # "约1小时".
    days, rest = divmod(total // 60, 1440)
    hours, minutes = divmod(rest, 60)
    parts = [(days, "天"), (hours, "小时"), (minutes, "分钟")]
    return "约" + "".join(f"{n}{unit}" for n, unit in parts if n)
