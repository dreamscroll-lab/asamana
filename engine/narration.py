"""How an executed act is worded — the shared skeletons and conventions of outcome and memory text.

Every executor phrases its third-person outcomes and first-person memories through these, so each
convention (the place prefix, a named actor, how an interrupt's reason is attributed) is stated
once. Outside ``engine.executors`` for the same reason as ``engine.scene``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.interfaces.action import Observed
from core.prompts import strip_end_punct

if TYPE_CHECKING:
    from engine.environment import EnvironmentSystem


def scene_line(location: str, body: str) -> str:
    """The shared third-person observable skeleton: "在{location}，{body}" (at {location}, {body}).

    Forces every outcome/tick/opening, LLM-adjudicated ones included, to carry a "where"
    prefix; ``body`` must contain who + what [+ result].
    """
    return f"在{location}，{body}"


#: The third-party judge's same-place invariant; it guards the consistency of ``scene_line``'s
#: location prefix. PHYSICAL / COVERT never call ``environment.move_body``, so a ruling that
#: someone "got somewhere else" contradicts world state. It must be an invariant, not "weigh it
#: against the current location": the judge still weighs its way to moving them, and the memory
#: then contradicts its own prefix. It also lets the judge rule "started but didn't get away"
#: instead of all-or-nothing; the second clause exempts sending someone else on an errand.
SAME_PLACE_VERDICT_RULE: str = """\
- 这一行动只发生在【现场】写明的那个地点，**不会改变任何人所在的位置**。行动描述若声称行动者或
  目标已经到了别的地点，那一部分不成立：只裁到不离开此地就能完成的那一部分（动作已经起手，
  人还在原地），不得写成谁已经到了那里。
- 上一条说的是**这一行动本身**的位移。让**另一个人**去某处不属于此列——那是他的行程，不是这一行动。"""


def observed_here(
    environment: "EnvironmentSystem | None", agent_id: str, text: str, *,
    strength: float | None = None,
) -> list[Observed]:
    """One bystander observation "right where he is now"; the single executor-side entry point.

    ``Observed.location_id`` needs a code-layer id, but an executor holds the narrative name from
    ``observe_location()``; passing the name raises nothing and the ambient silently lands
    nowhere. Resolving the id here removes that mistake.

    Empty ``text`` or no environment (``interrupt()`` takes it as optional) → empty list:
    bystanders perceive nothing, the fail-safe side.
    """
    if not text or environment is None:
        return []
    return [Observed(location_id=environment.get_body_location(agent_id), text=text, strength=strength)]


def ensure_actor_named(body: str, actor_name: str) -> str:
    """Prepend the actor's name when the judge's third-person narration doesn't mention it;
    "who" is guaranteed by code, not by prompt compliance.

    A prompt instruction alone breaks: a passer-by framing yields "二人低语匆匆" (the two whisper
    hurriedly) with no name. A present name passes the sentence through unchanged; an extra name
    is merely wordy, a missing one leaves no subject.

    A colon, not concatenation: body may carry another subject ("对方早有防备"), and
    concatenation would produce "李世民对方早有防备".
    """
    name = (actor_name or "").strip()
    if not name or name in body:
        return body
    return f"{name}：{body}"


def format_intent_clause(expected_outcome: str) -> str:
    """The actor's own "what I'm after", as the opening sentence of factual_memory.

    Marked as intent ("我打算", I intend), not "我原本期望" (I had expected): the latter implies it
    failed, and success hasn't been ruled on yet. Omitted when empty.

    First-person channels only (``factual_memory``, ``participant_intent``); never ``outcome`` /
    ``observation``: bystanders see him set off but not what he's after.
    """
    t = strip_end_punct(expected_outcome)
    return f"我打算「{t}」。" if t else ""


def format_interrupt_thought(thought: str) -> str:
    """Format the acting agent's first-person interrupt reaction as a trailing clause.

    The em-dash keeps the actor's subjective ``thought`` (from Agent.evaluate_interrupt) apart
    from the objective record. Empty when there is no thought."""
    t = (thought or "").strip()
    return f"——心想：{t}" if t else ""


def format_interrupt_reason_3p(actor_name: str, thought: str) -> str:
    """The interrupter's reason, ATTRIBUTED, for the third-person authoritative ``outcome``.

    ``outcome`` is the god-view full record (``observation`` is the bystander one), and every
    other outcome states why an action ended; an interrupt must not report an effect with no
    cause. Naming whose reason it is keeps the record third-person. Withheld from ``observation``.

    Attach it only to the interrupter's own outcome: other participants' outcomes feed their own
    cognition, and they would adopt the interrupter's thoughts as their own business.
    """
    t = (thought or "").strip()
    return f"\n{actor_name}当时的心思：{t}" if t else ""
