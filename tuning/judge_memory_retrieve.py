"""LLM-as-judge for the memory **retrieval** stage — SEMANTIC recall quality.

Structural correctness of retrieval (top_k / dedup / kind classification / no leaks /
query-sensitivity) is covered by deterministic checks; the judge scores only semantic recall
quality, given the situation (query) + candidate pool + the top-k actually recalled:

1. relevance (is what was recalled right) — do the recalled memories fit the situation? Did any
   obviously irrelevant ones (noise) get pushed in?
2. coverage (was what should be recalled recalled) — did the memories in the pool that are
   clearly the ones this situation should bring to mind make the top-k, or were obviously
   central ones crowded out by peripheral ones?

Functional third-party verdict, reason first. Deduction-based (full marks by default, only
penalize citable problems), reusing the existing judges' negative-constraint /
don't-bind-the-answer safeguards: no "correct recall set" is fed in; plausibility is judged from
situation + pool + recall alone. The judge model comes from ``llm.judge``.
"""

from __future__ import annotations

import json
from typing import Any

from core.interfaces.llm import LLMMessage, LLMRouter, LLMScene, extract_json, output_budget
from core.logging import get_logger
from tuning.judge_result import empty_judgement

logger = get_logger(__name__)

_CRITERIA = ("relevance", "coverage")

_SYSTEM = (
    "你是叙事仿真的质量评审。被评估的是**记忆检索**:agent 在某个情境下,从自己的记忆库里召回最该想起的若干条。\n"
    "检索按 R/I/R(相关性/近时性/重要性)排序——既要切题(relevance),也允许近期或极重要的事自然浮现。\n"
    "评分以**扣分制**为主:满分(5)默认,只在发现**可指证的具体问题**时下调。**不要因为它没召回你设想的那条就扣分**"
    "——只要召回的在该情境下站得住、没顶出明显无关的、没漏掉明显中心的,就视为合理。情境平淡/无强相关记忆时,"
    "召回近期或重要记忆是合理的(不是噪声)。只输出 JSON。"
)


def _query_str(q: dict[str, Any]) -> str:
    if q.get("kind") == "structured":
        parts = [f"在场/环境:{q.get('spatial')}", f"消息:{q.get('message')}",
                 f"广播:{q.get('broadcast')}", f"当前目标:{q.get('goal')}"]
        return "；".join(p for p in parts if p.split("：", 1)[-1].strip())
    return q.get("text", "")


def _pool_block(pool: list[dict]) -> str:
    return "\n".join(
        f"  - [{m.get('stream')}/{m.get('kind')}|imp{m.get('importance')}|decay{m.get('decay')}|{m.get('created_step')}步] "
        f"{m.get('content', '')[:40]}" for m in pool) or "  （空）"


def _recalled_block(recalled: dict) -> str:
    lines: list[str] = []
    for label, key in (("events", "events"), ("insights", "insights"), ("summaries", "summaries")):
        for m in recalled.get(key, []) or []:
            lines.append(f"  - [{m.get('stream')}/{m.get('kind')}] {m.get('content', '')[:50]}")
    return "\n".join(lines) or "  （未召回）"


def _prompt(meta: dict, inputs: dict, outputs: dict, det: dict, query: dict, recalled: dict, tag: str = "") -> str:
    suffix = f"（{tag}）" if tag else ""
    return (
        f"## 场景:{meta.get('name')}{suffix}\n"
        f"说明:{meta.get('description', '')}\n预期:{meta.get('expect', '')}\n"
        "(「说明/预期」是设计意图,只说何为合理,**不喂答案**(不喂正确召回集);只依据下面情境+池子+召回判断。)\n\n"
        f"## 情境(query)\n  {_query_str(query)}\n\n"
        f"## 记忆库候选池(共 {len(inputs.get('pool', []))} 条,agent 当时拥有的全部)\n"
        f"{_pool_block(inputs.get('pool', []))}\n\n"
        f"## 实际召回的 top-k\n{_recalled_block(recalled)}\n\n"
        "## 确定性预检(供参考)\n"
        f"{json.dumps(det, ensure_ascii=False)}\n\n"
        "## 扣分项(各 1-5;默认 5,命中才下调)\n"
        "1. relevance(召回的对不对)——出现以下即扣:召回里有与情境**明显不相关**的噪声(占了 top-k 名额);"
        "召回整体跑题。**近期/高重要记忆的自然浮现不算噪声**(检索本就含 recency/importance)。\n"
        "2. coverage(该召回的有没有召回)——出现以下即扣:候选池里对该情境**明显最该想起**的中心记忆没进 top-k,"
        "反被边缘记忆挤占。**池子里本就没有强相关记忆时,不因'没召回到完美的'而扣**。\n\n"
        "## 严格输出 JSON(reason 在前):\n"
        '{"relevance": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "coverage": {"rationale": "一句话,≤80字", "score": 1-5, "issues": ["扣分点,≤40字,最多3条", "..."]},'
        ' "overall": "一句话总评,≤60字"}'
    )


async def _one_judge(router, judge_scene, prompt: str, scenario_name: str) -> dict[str, Any]:
    try:
        response = await router.complete(
            judge_scene, [LLMMessage(role="system", content=_SYSTEM), LLMMessage(role="user", content=prompt)],
            # Per dimension = rationale ≤80 chars (120) + score (3) + issues ≤3 × (≤40 chars 60 + structure 5) (195)
            # + structure (15) ≈ 333 tok; 2 dimensions + overall (≤60 chars, 90) ≈ 766.
            temperature=0.2, max_tokens=output_budget(766))
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge_call_failed", extra={"scenario": scenario_name, "error": str(exc)})
        return empty_judgement(_CRITERIA, f"judge 调用失败: {exc}")
    try:
        data = extract_json(response.content)
    except (json.JSONDecodeError, ValueError):
        return empty_judgement(_CRITERIA, "judge 输出非合法 JSON")
    if not isinstance(data, dict):
        return empty_judgement(_CRITERIA, "judge 输出非对象")
    result: dict[str, Any] = {}
    for c in _CRITERIA:
        entry = data.get(c) if isinstance(data.get(c), dict) else {}
        try:
            score = int(entry.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        result[c] = {"score": max(0, min(5, score)), "rationale": str(entry.get("rationale", "")),
                     "issues": [str(x) for x in entry.get("issues", []) if str(x).strip()]}
    result["overall"] = str(data.get("overall", ""))
    return result


async def judge_memory_retrieve(
    router: LLMRouter, *, scenario_meta: dict, inputs: dict, outputs: dict,
    det_findings: dict, judge_scene: LLMScene = LLMScene.WORLD_BUILDING,
) -> dict[str, Any]:
    """Score the semantic quality of recall. Contrast scenarios are scored on the main query (query_b only feeds the deterministic query-sensitivity check). Never raises."""
    name = scenario_meta.get("name", "")
    prompt = _prompt(scenario_meta, inputs, outputs, det_findings,
                     inputs.get("query", {}), outputs.get("recalled", {}))
    return await _one_judge(router, judge_scene, prompt, name)
