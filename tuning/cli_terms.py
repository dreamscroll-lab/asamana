"""Terms the tuning CLI shares with the developer tools' routes: where ``validate`` writes its
reports and the scope aliases ``audit --scope`` accepts.

Standard library only, like ``scenario_store``: the web process imports it, and importing
``tuning.cli`` would load the whole cognition stack into it.
"""

from __future__ import annotations

# Where `python -m tuning build / plan / validate` write their traces and reports.
TUNING_TRACE_DIR = "./data/tuning_traces"

# Short scope aliases → full audit scope keys (full keys are accepted by the CLI too).
AUDIT_SCOPE_ALIAS: dict[str, str] = {
    "sass": "single_agent_single_step",
    "sams": "single_agent_multi_step",
    "mass": "multi_agent_single_step",
    "mams": "multi_agent_multi_step",
    "init": "initialization",
}
