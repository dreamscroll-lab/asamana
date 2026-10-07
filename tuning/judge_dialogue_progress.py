"""Cross-scene progress metric: does a pair's Nth conversation bring anything new compared with the earlier ones?

Per-scene judging can't see this problem. Each scene passes on its own (the disagreement plays
out, nobody oversteps, persona and emotion are right), yet by the thirteenth talk between the same
two people the scenes stacked together are still going in circles. The defect is in the
sequence, not in any single scene.

So the unit judged here is "a pair's Nth scene", and the input is everything that pair has said
before:
- ``advance``: this scene brings something the earlier ones didn't (a new fact, a shifted
  stance, an arrangement not made before, a changed relationship).
- ``restate``: it repeats what's already there in other words (presses again, refuses again,
  angrier but the same content).

The first scene has no "before" and isn't judged; the progress-rate denominator counts only
scenes with N≥2.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.interfaces.llm import LLMMessage, LLMProvider, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger

from tuning.judge_dialogue_leak import DialogueCase, extract_dialogues
from tuning.judge_llm import create_judge
from tuning.trace import InMemoryTraceSink, write_json

logger = get_logger(__name__)

_VERDICTS = ("advance", "restate")


_SYSTEM = """\
【角色】
你是叙事仿真的推进审查员。给你同一对人此前谈过的全部交谈，再给你他们**最新的一场**。
你只回答一件事：**这最新一场，带来了前面没有的东西吗？**

【算「带来了新东西」(advance)】
- 说出了前面几场里没出现过的事实、消息、来历、意图。
- 有人的**立场变了**：改了主意、松了口、翻了脸、认了错、下了决心。
- 定下了前面没有的**具体安排**：时间、地点、谁去做什么。
- 两人之间的**关系或处境**实质变了：结盟、决裂、有了把柄、欠了人情。

【算「原地重复」(restate) —— 这几样看着热闹，都不是新东西】
1. 同一个诉求换一种说法再提一遍；同一个拒绝换一种说法再拒一遍。
2. 又催了一次、又劝了一次、又追问了一次，对方的答复和前面一样。
3. 情绪更激烈、话更难听、场面更僵——**烈度不是内容**。
4. 把前面已经说过的事复述、总结、确认一遍。
5. 提到了新的人名或物件，但没有带出关于它的任何新内容。
6. 双方各自重申自己那套说法，谁也没动。

【怎么判】
先通读前面几场，记住已经出现过什么；再逐句读最新这场，找**头一次出现**的东西。找得到就
advance，并写清它是什么；通篇都能在前面找到对应，就是 restate。

拿不准时判 restate——重复被误记成推进，会把原地打转粉饰成叙事进展，那比反过来危险。

【输出】
严格输出以下 JSON，不要任何多余内容。先写 reason（你比对下来的判断），再写 new（新东西是
什么），再据此给 verdict。
{"reason": "≤80字", "new": "这一场头一次出现的东西，没有就留空，≤40字", "verdict": "advance|restate"}
"""


def _transcript(case: DialogueCase) -> str:
    return "\n".join(f"{who}：{text}" for who, text in case.lines)


def _user_block(prior: list[DialogueCase], current: DialogueCase) -> str:
    past = "\n\n".join(
        f"【此前第 {i} 场】\n{_transcript(c)}" for i, c in enumerate(prior, start=1))
    return f"""\
{current.name_a} 与 {current.name_b} 此前谈过 {len(prior)} 场：

{past}

【最新这一场（第 {len(prior) + 1} 场）】
{_transcript(current)}

按上面说定的 JSON 格式输出，只输出 JSON。再提醒一次：烈度不是内容，重申不是推进，拿不准判 restate。"""


async def judge_one(router: LLMRouter, prior: list[DialogueCase], current: DialogueCase,
                    *, judge_scene: LLMScene) -> dict[str, Any]:
    """Judge one scene. On failure, return judged=False per Rule 1 without inventing a verdict; the scene is excluded from aggregation."""
    base = {"step": current.step, "call_id": current.call_id,
            "pair": [current.name_a, current.name_b], "nth": len(prior) + 1}
    try:
        response = await router.complete(
            judge_scene,
            [LLMMessage(role="system", content=_SYSTEM),
             LLMMessage(role="user", content=_user_block(prior, current))],
            temperature=0.0,
            # reason ≤80 chars ≈ 120 tok + new ≤40 chars ≈ 60 tok + verdict 3 + structure 15; estimate ~200 tok.
            max_tokens=output_budget(200),
            json_mode=True,
        )
        data = extract_json(response.content)
    except Exception as exc:  # noqa: BLE001 — Rule 1: the judgment didn't happen, so the scene doesn't count
        logger.warning("dialogue_progress_judge_failed",
                       extra={"call_id": current.call_id, "step": current.step, "error": str(exc)})
        return {**base, "judged": False}

    verdict = str(data.get("verdict", "")).strip().lower() if isinstance(data, dict) else ""
    return {
        **base, "judged": True,
        "reason": str(data.get("reason", "")).strip() if isinstance(data, dict) else "",
        "new": str(data.get("new", "")).strip() if isinstance(data, dict) else "",
        # Unrecognized verdicts go to unknown, not restate by default: that would turn a judge failure into "really going in circles".
        "verdict": verdict if verdict in _VERDICTS else "unknown",
    }


async def run_dialogue_progress(
    container: Any,
    config: Any,
    world_id: str,
    *,
    judge_model: str | None = None,
    judge_provider: LLMProvider | None = None,
    judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
    trace_dir: str = "data/traces",
) -> dict[str, Any]:
    """Judge every follow-up conversation of every pair in a world, write the results to disk and return the summary."""
    cases = extract_dialogues(world_id, trace_dir=trace_dir)
    if not cases:
        raise ValueError(f"no TALK dialogues in traces for world {world_id!r} under {trace_dir}")

    # Chain each pair's talks in order. A seeking B and B seeking A are the same thread, so the key is unordered.
    chains: dict[tuple[str, str], list[DialogueCase]] = defaultdict(list)
    for c in sorted(cases, key=lambda x: x.step):
        chains[tuple(sorted((c.name_a, c.name_b)))].append(c)

    work = [(chain[:i], chain[i]) for chain in chains.values() for i in range(1, len(chain))]
    if not work:
        raise ValueError(f"world {world_id!r}: no pair talked twice — nothing to compare")

    judge_llm = judge_provider or create_judge(config, judge_model)
    router = LLMRouter(
        {scene: judge_llm for scene in LLMScene},
        trace_sink=InMemoryTraceSink(),
        max_concurrent=getattr(getattr(config, "engine", None), "max_concurrent_llm", 4))

    results = await asyncio.gather(
        *(judge_one(router, prior, cur, judge_scene=judge_scene) for prior, cur in work),
        return_exceptions=True,
    )
    scenes: list[dict[str, Any]] = []
    for (prior, cur), res in zip(work, results):
        if isinstance(res, BaseException):      # Rule 4: one slot's exception doesn't affect the others
            logger.warning("dialogue_progress_slot_failed",
                           extra={"call_id": cur.call_id, "error": str(res)})
            scenes.append({"step": cur.step, "call_id": cur.call_id, "judged": False,
                           "pair": [cur.name_a, cur.name_b], "nth": len(prior) + 1})
        else:
            scenes.append(res)

    judged = [s for s in scenes if s.get("judged")]
    counts = {v: sum(1 for s in judged if s["verdict"] == v) for v in _VERDICTS + ("unknown",)}
    decided = counts["advance"] + counts["restate"]
    pair_sizes = {"/".join(k): len(v) for k, v in sorted(chains.items(), key=lambda kv: -len(kv[1]))}
    summary = {
        "world_id": world_id,
        "dialogues": len(cases),
        "pairs": len(chains),
        "followups": len(work),          # scenes with N≥2: the progress-rate denominator
        "judged": len(judged),
        "unjudged": len(work) - len(judged),
        "counts": counts,
        "advance_rate": round(counts["advance"] / decided, 4) if decided else None,
        "pair_sizes": pair_sizes,
    }
    write_json(Path(trace_dir, world_id, "audit", "dialogue_progress.json"),
               {"summary": summary, "scenes": scenes})
    return summary
