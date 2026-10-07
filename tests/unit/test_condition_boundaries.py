"""Module-boundary guards for condition — CLAUDE.md's grep rules pinned down as executable
assertions.

The same fact has two read channels (the cognition side via perception, the god's-eye side reading
entities), and once someone crosses them you get "one fact, two sources" (see the gender comment
in world_pressure). The guards are tests rather
than docs because docs can't stop the next change.

Written like CLAUDE.md's existing guard (`grep -rn "WorldDirectory" agent/` must print nothing).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _grep(patterns: str | tuple[str, ...], *paths: str) -> list[str]:
    """One ``-e`` per pattern — do NOT write an alternation with ``|`` inside a single pattern.

    BSD grep defaults to BRE, where ``|`` is a literal character, so the whole pattern never
    matches anything and the guard stays green while never actually working. Inject a real
    violation to confirm a guard fires.
    """
    pats = (patterns,) if isinstance(patterns, str) else patterns
    args = ["grep", "-rn", "--include=*.py"]
    for pat in pats:
        args += ["-e", pat]
    proc = subprocess.run([*args, *paths], cwd=REPO, capture_output=True, text=True)
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def test_cognition_never_reads_another_agents_condition_object() -> None:
    """Cognition paths in ``agent/`` may only get someone else's condition via the perception
    packet.

    Reading the ``Agent`` object directly makes the agent omniscient — the same rule as "``agent/``
    must not use WorldDirectory". The two legitimate sites are ``personality.py`` (own state
    access) and ``agent.py`` (its own landing and expiry).
    """
    # "Reading my own" vs "reading someone else's" is told apart by access shape, not a file
    # whitelist:
    #   personality.state.condition        ← the personality in hand is my own (decision reads
    #                                        its own condition)
    #   <some agent>.personality.state.…   ← reaching through another object: that's overreach
    offenders = [
        ln for ln in _grep(r"\.personality\.state\.condition", "agent/")
        if not re.match(r"agent/agent\.py:", ln)   # own landing and expiry
    ]
    assert offenders == [], "认知路径穿过别的 agent 对象读了处境:\n" + "\n".join(offenders)


def test_the_engine_never_mutates_a_condition_itself() -> None:
    """The only write entry is the feedback layer. Executors only declare it in
    ``TargetAgentEffect``, runtime only triggers ``Agent.expire_condition`` — nobody may reach into
    personality."""
    offenders = _grep(
        (r"\.set_condition(", r"\.clear_condition("), "engine/", "world/", "interaction/",
    )
    assert offenders == [], "engine/world/interaction 直接改了 agent 状态:\n" + "\n".join(offenders)


#: Injection sites allowed to emit the condition label. This is a whitelist meant to grow
#: deliberately, not config: each new entry is another place an agent reads its own or others'
#: condition, and should be a decision, not a side effect. The label itself is defined once in
#: core/prompts.py (condition_line), so this list governs who calls it — i.e. which prompts let an
#: agent read condition.
_CONDITION_LABEL_SITES = (
    "engine/executors/physical.py:",  # both parties of the ruling + the recipient's reaction
    "engine/executors/covert.py:",    # covert-action ruling (stealth depends most on bodily state)
    "engine/executors/social.py:",    # both sides of a dialogue + two first-person conversation memories
    "engine/executors/work.py:",      # solo work self-assessment + the interrupted memory
    "engine/event.py:",               # event editor's character briefing (condition scalar alongside "体力")
    "agent/decision.py:",             # decision: first line of 【我所处的现实】
    "agent/agent.py:",                # interrupt weighing + feedback emotion
    "agent/perception_emotion.py:",   # perception emotion
    "agent/need.py:",                 # short-term goal generation
    "agent/memory.py:",               # first-person experiential memory (gets embedded)
    "core/prompts.py:",               # where the label itself is defined
)


def test_the_prompt_label_is_confined_to_declared_sites() -> None:
    """The "处境：" label line may only appear in _CONDITION_LABEL_SITES.

    This doesn't guard "only one site" — injection sites do grow; it guards that each new site is
    deliberate. Which prompts let an agent read condition directly shapes the world in its
    cognition, and shouldn't be decided by a casual copy-paste.
    """
    # Only find real interpolation points ("处境：{...}"), not comments and docstrings that mention
    # the word in prose.
    offenders = [
        ln for ln in _grep(("处境：{", "condition_line("),
                           "agent/", "engine/", "world/", "interaction/", "core/")
        if not ln.startswith(_CONDITION_LABEL_SITES)
    ]
    assert offenders == [], "在白名单之外拼了处境标签:\n" + "\n".join(offenders)


def test_no_module_builds_a_condition_line_without_the_renderer() -> None:
    """Anywhere condition is put into text must go through one of core.prompts' two renderers:

    - ``condition_line`` — a whole "{人称}此刻的处境：X" line, owning the label/person/whole-line
      omission when empty;
    - ``render_condition`` — value only, for list tags (``person_referent``'s marks).

    They're the only places that know "no condition omits the whole line" and "durations must not
    be written as steps". Hand-built f-strings will sooner or later diverge on label, person, or
    surrounding blank lines.
    """
    users = _grep(r"condition", "engine/scene.py", "engine/world_pressure.py",
                  "engine/director.py", "engine/presence.py", "agent/decision.py",
                  "agent/agent.py", "agent/perception_emotion.py", "agent/need.py", "agent/memory.py")
    assert users, "守卫本身失效了:这些文件本应都在用处境"
    for path in ("engine/scene.py", "engine/world_pressure.py",
                 "engine/director.py", "engine/presence.py", "agent/decision.py",
                 "agent/agent.py", "agent/perception_emotion.py", "agent/need.py", "agent/memory.py",
                 "engine/executors/covert.py", "engine/executors/social.py",
                 "engine/executors/work.py", "engine/event.py"):
        assert _grep(("condition_line", "render_condition"), path), \
            f"{path} 用了处境却没走 condition_line / render_condition"


def test_condition_text_never_carries_a_step_coordinate() -> None:
    """step is a code-layer coordinate; the narrative layer only has time points and durations.
    Inside the renderer only describe_duration may appear."""
    source = (REPO / "core" / "prompts.py").read_text(encoding="utf-8")
    body = source.split("def render_condition", 1)[1].split("\ndef ", 1)[0]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assert "describe_duration" in code
    assert "since_step}" not in code and "步" not in code.split('"""', 2)[-1]


def test_presence_is_assembled_in_exactly_one_place() -> None:
    """``visible_agents`` / ``visible_conditions`` / ``visible_npcs`` may only be assigned by
    ``engine/presence.py``.

    These tables and ``visible_agent_ids`` are parallel structures keyed by the same ids, and
    scattered assignment drifts silently (tuning could see everyone's condition as empty while
    production doesn't). The scene-override site rewrites the ids, and the other tables must be
    recomputed after it.

    Before adding another assembly site, ask whether it can call ``attach_presence`` instead — it
    almost always can.
    """
    # Look for attribute assignment, not same-named locals or comparisons — "visible_agents =" is a
    # substring of "visible_agents ==", and not distinguishing them would hit a `== 0` check in
    # need.py.
    assign = re.compile(r"\.(visible_agents|visible_conditions|visible_npcs)\s*=(?!=)")
    offenders = [
        ln for ln in _grep((r"\.visible_agents", r"\.visible_conditions", r"\.visible_npcs"),
                           "agent/", "engine/", "world/", "interaction/", "tuning/")
        if assign.search(ln) and not ln.startswith("engine/presence.py:")
    ]
    assert offenders == [], "在 attach_presence 之外装配感知包:\n" + "\n".join(offenders)
