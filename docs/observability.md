# Observability and audit

English | [简体中文](observability.zh-CN.md)

**Trace** records every LLM call; **Audit** evaluates completed narratives offline. Together they make narrative quality **visible, measurable and tunable**. Both are available at `#/dev` after enabling the [developer tools](development.md#developer-tools).

## LLM call tracing (Trace)

Trace records each LLM call's prompt, response, token counts, latency and cognition stage, plus the total duration of each step. Use it to locate narrative problems by character, step and stage.

- **Toggle**: `observability.enabled` in the config; on by default (off in the test config).
- **Storage**: `data/traces/<world_id>/` — one file for the world build, then one JSONL file per run step.
- **Viewing**: `#/dev` → **Trace**. Group by step, stage or character, and open any call to see its full prompt and response.

## World quality audit (Audit)

An LLM reviews the narrative quality of a finished world. The core principles:

1. **Retrospective and read-only**: it reads existing traces without re-running the simulation, so it evaluates the narrative that actually occurred.
2. **A director's perspective**: it scores seven dimensions in two tiers: baseline quality (consistency and coherence) and narrative development.
3. **Scopes match the evidence**: memory lapses need evidence across steps; ineffective social interaction needs evidence across characters; narrative convergence needs evidence across the whole world. Each of the five scopes scores only the dimensions it can assess.
4. **Penalty-based scoring**: each dimension starts at 100, with points deducted for specific errors (see below).
5. **Explicit evidence coverage**: narrative quality (the world score) and engineering fidelity (`fidelity_score`) are scored separately. Faithfulness to inputs does not guarantee an engaging story. Reports show incomplete evidence coverage; audits with different coverage are not directly comparable.

### Why scores use penalties

Every dimension starts at **100**. The judge deducts points for specified errors, such as memory lapses, fabricated facts, repetitive behavior and clichéd plot turns. This design has three goals:

1. **Assess specific errors.** Checking whether an action contradicts an earlier step is more concrete than assigning an overall score to a story.
2. **Measure a quality floor.** A high score means few identifiable problems were found; it does not establish artistic merit. The audit targets contradictions, fabrication, memory lapses and other logical problems.
3. **Leave creative quality to emergence.** The judge penalizes demonstrable errors, rather than imposing its own plot or deducting points merely because a story is not exciting enough.

Two consequences follow, and both are visible in the report:

- **Insufficient evidence yields `null`, not zero.** When the judge lacks evidence, it abstains; that dimension is excluded from the weighted score.
- **The criteria list errors only, never a model answer.** The judge is explicitly told not to deduct points because the output differs from what it would have written.

### Dimensions and scopes

| Dimension | What it checks | Weight |
|---|---|---|
| Persona and world consistency | Whether words and actions match the character's personality, identity and the world's established norms | 20% |
| Causal coherence | Whether actions, emotions and needs have identifiable causes, without memory lapses or fabricated facts | 20% |
| Action effectiveness | Whether characters spin their wheels or repeat themselves; whether social interaction actually changes the world or relationships | 20% |
| Narrative convergence | Whether the story moves toward the core conflict, the theme and long-term goals | 15% |
| Dramatic arc | Whether emotion, pressure and conflict rise and fall | 15% |
| Emergent interest | Whether information gaps produce misunderstandings, reversals and strategic play | 5% |
| Character growth | Whether characters change in substance after major upheavals | 5% |

| Scope | Range | What it checks |
|---|---|---|
| `mams` | All characters × all steps | The narrative as a whole: convergence, arc and emergence, whether actions are effective, consistency with the world |
| `sams` | One character × all steps | Persona consistency, causal coherence, action effectiveness and character growth |
| `init` | World initialization | The opening setup: whether personas and relationships are consistent and the backstory timeline makes sense |
| `mass` | All characters × one step | Interaction at a single moment: who knows what, who is affected by each reaction, and whether actions at the same location contradict each other |
| `sass` | One character × one step | Whether each thought stays faithful to its inputs without inventing facts (counts toward fidelity only); requires the most LLM calls and is the most expensive scope |

For routine reviews, use `mams`, `sams` and `init`.

### Running

Start an audit from `#/dev` → **Audit**, or from the command line (requires keys; the judge model is set by `llm.judge` in the config):

```bash
python -m tuning audit <world-id>                     # all scopes
python -m tuning audit <world-id> --scope mams,sams   # only some scopes
```

Running a subset of scopes doesn't overwrite existing results for the others; the world score is always recomputed from all results. Results are saved under `data/traces/<world_id>/audit/`.

### Tuning the criteria

The criteria for each dimension in each scope live in `tuning/audit_criteria.yaml`. Edit them and re-run — no code changes needed. Dimension definitions and weights are in `tuning/audit_metrics.py`.
