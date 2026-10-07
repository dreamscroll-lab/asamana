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

1. **After the fact, read-only**: it reads only the existing traces and never re-runs the simulation, so it judges the narrative that actually happened.
2. **A director's view**: it scores seven dimensions in two tiers: baseline (consistency and coherence) and elevation (narrative development).
3. **Scopes match the evidence**: memory lapses require evidence across steps, ineffective social interaction across characters, and narrative convergence across the whole world. Five scopes each score only the dimensions they can assess.
4. **Deduction-based scoring**: see the next section; this is the core idea behind the whole audit.
5. **Explicit evidence coverage**: narrative quality (the world score) and engineering fidelity (`fidelity_score`) are scored separately. Faithfulness to inputs does not guarantee an engaging story. Reports show incomplete evidence coverage; audits with different coverage are not directly comparable.

### Why everything is a deduction

Every dimension starts at **100**. The judge deducts points for specified errors, such as memory lapses, fabricated facts, repetitive behavior and clichéd plot turns. This design has three goals:

1. **Assess specific errors.** Checking whether an action contradicts an earlier step is more concrete than assigning an overall score to a story.
2. **Measure a quality floor.** A high score means few identifiable problems were found; it does not establish artistic merit. The audit targets contradictions, fabrication, memory lapses and other logical problems.
3. **Leave creative quality to emergence.** The judge penalizes demonstrable errors, rather than imposing its own plot or deducting points merely because a story is not exciting enough.

Two consequences follow, and both are visible in the report:

- **"Can't tell" is a null, not a zero.** When the judge has no evidence, it abstains; that dimension drops out of the weighting instead of dragging the score down.
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
| `sams` | One character × all steps | Whether this character stays in persona, is causally coherent and acts effectively, and grows and changes |
| `init` | World initialization | The opening setup: whether personas and relationships are consistent and the backstory timeline makes sense |
| `mass` | All characters × one step | Interaction at a single moment: who knows what, who is affected by each reaction, and whether actions at the same location contradict each other |
| `sass` | One character × one step | Whether each thought stays faithful to its inputs without adding facts (counts toward fidelity only); the most calls and the most expensive |

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
