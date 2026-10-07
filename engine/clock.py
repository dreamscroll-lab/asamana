"""Global timekeeping for Asamana."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from core.coerce import coerce_enum


# For the spoken 12-hour clock in _clock_label. Don't use the two-hour shichen: at one hour per
# step two consecutive steps would share a label and narrative time would look stuck.
CHINESE_HOUR_LABELS = {
    0: "零", 1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六",
    7: "七", 8: "八", 9: "九", 10: "十", 11: "十一", 12: "十二",
}


def _clock_label(hour: int, minute: int) -> str:
    """Render a 24h (hour, minute) as a spoken 12-hour clock: period + hour (+ minutes).

    E.g. 8→"上午八点", 14→"下午两点", 0→"凌晨零点". Minutes only when non-zero.
    """
    if hour <= 4:
        period = "凌晨"
    elif hour <= 7:
        period = "早上"
    elif hour <= 11:
        period = "上午"
    elif hour == 12:
        period = "中午"
    elif hour <= 17:
        period = "下午"
    else:
        period = "晚上"
    h12 = hour if hour <= 12 else hour - 12  # 0→0, 12→12, 13→1 … 23→11
    label = CHINESE_HOUR_LABELS.get(h12, str(h12))
    return f"{period}{label}点" + (f"{minute}分" if minute else "")

class CalendarStyle(str, Enum):
    """How a world names its dates.

    What a date is CALLED belongs to the setting (「正月初一」 vs 「3月4日」), so it is authored on
    the map (`calendar` property). Only the date differs: the spoken clock is the language's,
    not the era's.
    """

    CLASSICAL_CN = "classical_cn"  # 正月初一 — ordinal month/day naming
    MODERN = "modern"  # 3月4日 — plain numeric dates


def parse_calendar_style(raw: object) -> CalendarStyle:
    """Read an authored calendar name; anything unknown keeps the default.

    Authored map content: an unrecognised value must not fail the build.
    """
    return coerce_enum(CalendarStyle, raw, default=CalendarStyle.CLASSICAL_CN)


CHINESE_MONTH_LABELS = {
    1: "正月", 2: "二月", 3: "三月", 4: "四月",
    5: "五月", 6: "六月", 7: "七月", 8: "八月",
    9: "九月", 10: "十月", 11: "十一月", 12: "十二月",
}

CHINESE_DAY_LABELS = {
    1: "初一", 2: "初二", 3: "初三", 4: "初四", 5: "初五",
    6: "初六", 7: "初七", 8: "初八", 9: "初九", 10: "初十",
    11: "十一", 12: "十二", 13: "十三", 14: "十四", 15: "十五",
    16: "十六", 17: "十七", 18: "十八", 19: "十九", 20: "二十",
    21: "廿一", 22: "廿二", 23: "廿三", 24: "廿四", 25: "廿五",
    26: "廿六", 27: "廿七", 28: "廿八", 29: "廿九", 30: "三十",
}

# Years are unbounded, so unlike months and days they are computed, not looked up.
_CN_DIGITS = "零一二三四五六七八九"
_CN_UNITS = ("", "十", "百", "千")


def _chinese_number(n: int) -> str:
    """Read 1-9999 as Chinese numerals (10→"十", 26→"二十六", 105→"一百零五").
    Out of range returns the digits."""
    if not 1 <= n <= 9999:
        return str(n)
    digits: list[tuple[int, int]] = []
    rest, unit = n, 0
    while rest:
        digits.append((rest % 10, unit))
        rest //= 10
        unit += 1
    out: list[str] = []
    zero_pending = False
    for value, unit_index in reversed(digits):
        if value == 0:
            zero_pending = True
            continue
        if zero_pending and out:
            out.append("零")
        zero_pending = False
        out.append(_CN_DIGITS[value] + _CN_UNITS[unit_index])
    text = "".join(out)
    return text[1:] if text.startswith("一十") else text  # "十一", not "一十一"


# Coarse calendar: 30 days a month, 12 months a year.
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12


# Step length range, in hours only because theme analysis outputs `hours_per_step`; everything
# downstream works in seconds and doesn't assume whole hours. Authored per world, not per
# deployment: an overnight upheaval and a decade of decline need different steps. The 24-hour cap
# keeps a step within one day, so memory recency and need decay stay comprehensible.
MIN_HOURS_PER_STEP = 1
MAX_HOURS_PER_STEP = 24
DEFAULT_HOURS_PER_STEP = 2
SECONDS_PER_HOUR = 3600
DEFAULT_STEP_SECONDS = DEFAULT_HOURS_PER_STEP * SECONDS_PER_HOUR


def resolve_step_seconds(raw: object) -> int:
    """Normalize a step length: invalid or out-of-range falls back to the default. Don't clamp:
    a runaway 0 or 99 hours would become the most extreme setting while looking deliberate.
    ``int()`` accepts type jitter ("6", 6.0) as field-level coercion.
    """
    try:
        seconds = int(raw)  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return DEFAULT_STEP_SECONDS
    if not MIN_HOURS_PER_STEP * SECONDS_PER_HOUR <= seconds <= MAX_HOURS_PER_STEP * SECONDS_PER_HOUR:
        return DEFAULT_STEP_SECONDS
    return seconds


@dataclass(frozen=True)
class WorldTimeConfig:
    # The era LABEL ONLY — 「武德」, not 「武德九年」: the year must be able to advance. Empty for
    # settings without era names.
    era_name: str = ""
    # See CalendarStyle; frozen with the world's config.
    calendar: CalendarStyle = CalendarStyle.CLASSICAL_CN
    # An ordinal within the era (武德九年 → 9) or the absolute year (2026), per CalendarStyle.
    start_year: int = 1
    start_month: int = 1
    start_day: int = 1
    start_hour: int = 6
    seconds_per_step: int = 3600

    def __post_init__(self) -> None:
        if self.start_year < 1:
            raise ValueError("start_year must be >= 1")
        if not 1 <= self.start_month <= 12:
            raise ValueError("start_month must be in the range [1, 12]")
        if not 1 <= self.start_day <= 30:
            raise ValueError("start_day must be in the range [1, 30]")
        if not 0 <= self.start_hour <= 23:
            raise ValueError("start_hour must be in the range [0, 23]")
        if self.seconds_per_step <= 0:
            raise ValueError("seconds_per_step must be positive")


@dataclass(frozen=True)
class WorldTime:
    step: int
    elapsed_seconds: int
    era_name: str = ""
    calendar: CalendarStyle = CalendarStyle.CLASSICAL_CN
    start_year: int = 1
    start_month: int = 1
    start_day: int = 1

    @property
    def total_days(self) -> int:
        return self.elapsed_seconds // 86400

    @property
    def second_of_day(self) -> int:
        return self.elapsed_seconds % 86400

    @property
    def hour_of_day(self) -> int:
        return self.second_of_day // 3600

    @property
    def minute_of_hour(self) -> int:
        return (self.second_of_day % 3600) // 60

    @property
    def hour(self) -> int:
        return self.hour_of_day

    @property
    def _absolute_day(self) -> int:
        return (self.start_day - 1) + self.total_days

    @property
    def _month_index(self) -> int:
        """Months since month 1 of ``start_year``: the one carry chain year and month share."""
        return (self.start_month - 1) + self._absolute_day // DAYS_PER_MONTH

    @property
    def year(self) -> int:
        return self.start_year + self._month_index // MONTHS_PER_YEAR

    @property
    def month_of_year(self) -> int:
        return self._month_index % MONTHS_PER_YEAR + 1

    @property
    def day_of_month(self) -> int:
        return self._absolute_day % DAYS_PER_MONTH + 1

    @property
    def time_label(self) -> str:
        # Narrative layer, never parsed; the machine clock is clock_payload's hour/minute.
        time_str = _clock_label(self.hour_of_day, self.minute_of_hour)
        if self.calendar is CalendarStyle.MODERN:
            # "2026年3月4日". era_name stays out: prefixing a label like "当代" would read as a
            # yearless date.
            return f"{self.year}年{self.month_of_year}月{self.day_of_month}日，{time_str}"
        # "武德九年，六月初一": year one reads "元"; the number advances from start_year.
        year_label = "元" if self.year == 1 else _chinese_number(self.year)
        date_str = f"{self.era_name}{year_label}年"
        return (
            f"{date_str}，{CHINESE_MONTH_LABELS[self.month_of_year]}"
            f"{CHINESE_DAY_LABELS[self.day_of_month]}，{time_str}"
        )

    def iso_label(self) -> str:
        return (
            f"step={self.step:04d} "
            f"time={self.hour_of_day:02d}:{self.minute_of_hour:02d} "
            f"elapsed={self.elapsed_seconds}s"
        )

    def clock_payload(self) -> dict[str, str | int]:
        """This moment on both layers — the ONE shape every consumer of the clock sees.

        ``hour`` / ``minute`` are the machine clock; ``iso`` is a log string nothing parses;
        ``label`` is narrative, passed through whole. One producer, so the snapshot and the live
        payload can't drift apart.
        """
        return {
            "iso": self.iso_label(),
            "hour": self.hour_of_day,
            "minute": self.minute_of_hour,
            "label": self.time_label,
        }

    @classmethod
    def from_step(cls, step: int, config: WorldTimeConfig) -> "WorldTime":
        return cls(
            step=step,
            elapsed_seconds=(config.start_hour * 3600) + (step * config.seconds_per_step),
            era_name=config.era_name,
            calendar=config.calendar,
            start_year=config.start_year,
            start_month=config.start_month,
            start_day=config.start_day,
        )


class GlobalClock:
    def __init__(self, config: WorldTimeConfig, *, start_step: int = 0) -> None:
        self._config = config
        self._current_step = start_step

    @property
    def config(self) -> WorldTimeConfig:
        return self._config

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def current(self) -> WorldTime:
        return WorldTime.from_step(self._current_step, self._config)

    def tick(self) -> WorldTime:
        self._current_step += 1
        return self.current
