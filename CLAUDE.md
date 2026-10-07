# Asamana — Project Guidelines

## Project Overview

Asamana is an AI-driven emergent narrative simulation. Users select a world theme; the system builds a world populated by autonomous AI agents whose decisions and interactions drive the narrative, with no scripts or preset plots.

**Current status**: Active development. The codebase is the source of truth; when code and any document conflict, trust the code.

**Language**: Python 3.11+, 4-space indentation, strict type hints throughout.

---

## Architecture Layers

```
interaction/    — Replayer, WorldManager, api/ (read-only user-facing; REST + WebSocket)
world/          — WorldBuilder: one-shot theme→world construction
engine/         — Runtime loop: Clock, Scheduler, Environment, Message, Event
agent/          — Per-agent cognition: Personality, Memory, Need, Relation, Decision
providers/      — Pluggable infrastructure: LLM, Embedding, VectorStore, AgentStore, Message, Snapshot
core/           — Interfaces, DI Container, Factory, Logging, Context
```

Alongside them: `worlds/` (map templates), `examples/` (example worlds seeded on first start), `frontend/` (React + Phaser observer SPA), `tuning/` (tuning and audit harness), `config/`, `deploy/`, `tests/`.

Dependency direction is strictly downward: `interaction → world/engine/agent → core ← providers`. Business logic never imports concrete provider classes.

---

## Design Principles

Every code change must follow these five principles:

### 1. Holistic Impact
Before changing anything, check whether it affects `core/`, `agent/`, `engine/`, `world/`, `interaction/`, `providers/`, config, snapshots, events, or tests. Cross-boundary changes require identifying all affected sites first.

### 2. Systematic Approach
- Trace the call path in the code before editing.
- Form one clear hypothesis, verify it, then implement the smallest correct change.
- If blocked, step back and re-analyze. Do not iterate on wrong assumptions.

### 3. Extensibility at Known Boundaries
Be extensible at existing seams: `Provider` interfaces, `Container`, config schemas, world definitions, subsystem contracts. Do **not** add abstraction layers for speculative future needs.

### 4. YAGNI
Only implement what is explicitly required right now. No feature flags, no backward-compatibility shims, no speculative helpers. Three similar lines are better than a premature abstraction.

### 5. Emergent Narrative First
Asamana exists so that agents drive emergent narrative on their own. Every design decision serves that goal:

- **Every cognition gate uses the LLM, for every agent. Background agents don't get a rule-based path, and there is no switch for one.** Decision emotion, perception emotion, importance, interrupts, goal progress, reflection, relation-label evolution and executor adjudication all run through the LLM for everyone. Quality comes before cost: the quality of cognition is the quality of the emergent narrative. If background agents think by different rules, the world splits into two sets of physics, where the same event plays out differently depending on whether it happens to a main character or a background one. That costs far more than the tokens it saves.
- **LLM vs. rules** (for deciding whether a **new** cognition step uses the LLM): if the output depends on personality, situational understanding or subjective interpretation → LLM; if it is an objective threshold over structured signals (`urgency >= X`) → rule. The question is whether the computation itself is subjective, **not** how important the agent is. Never split by agent.
- **`is_main_character` is a narrative tier, not a cognition mode.** It controls scheduling priority, perception fidelity, memory-write thresholds and failure retry (`complete_with_retry`). It must **never** switch between LLM and rules — don't write `if is_main_character: LLM else: rule` for a new cognition gate.
- **The only lever for cutting cost is the model tier.** To save money, give background agents a cheaper model through a scene of their own, **not** rule-based cognition. A cheaper model makes the same cognition less precise; rules create a second set of world physics. **Never** introduce a per-agent LLM/rules switch in any form (`tiered_cognition`, `use_llm_cognition` and the like).
- **Any rule-based cognition code that remains is a fallback for LLM failure, never a designed path for any agent** (e.g. the rule fallback in `ImportanceEvaluator`). See the fallback tiers under Rule 1 below.

---

## Workflow

- **Plan AND implement in the same session.** Do not stop at the plan stage unless the user explicitly says "plan only".
- After producing a plan, immediately proceed to implement all changes.
- If scope is too large, implement the highest-priority items and clearly state what was deferred.

---

## Implementation Rules

- Implement interfaces and data models before concrete providers.
- Prefer `InMemory` paths before adding external dependencies.
- Do not bypass `Container` or subsystem contracts for convenience.
- Keep implementation order: `core/` first → `providers/` → subsystem logic → integration.
- Prefer the smallest complete vertical slice over broad scaffolding.

---

## Testing

Run the relevant test suite before reporting done:

```bash
pip install -r requirements-dev.txt
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/ -x -q    # full suite, stop on first failure
ruff check .
```

Run the service with `deploy/asamana.sh` (`start` / `stop` / `restart` / `rebuild`); `CONFIG=<file> python main.py web` runs the web server directly.

- If tests fail, fix them in the same session. Do NOT report completion while tests are broken.
- Write unit tests with `InMemory` or mock providers — keep them fast and deterministic.
- Add/update tests in the same change as implementation. Never treat test updates as a follow-up step.
- Before any multi-file change, identify affected test files first. List which mocks or fixtures need updating.

---

## Debugging Protocol

Mandatory for infrastructure and integration issues:

1. List the top 3 possible root causes with evidence. Rank by likelihood.
2. Create a git checkpoint before each fix attempt.
3. Apply the **minimal fix** for the most likely cause only — no simultaneous changes.
4. Verify with a test command. If it fails, revert completely before trying the next hypothesis.
5. Never retry a disproven hypothesis. If 3 approaches fail, stop and summarize findings.
6. Never make architectural changes while debugging.

---

## Python Conventions

- Type hints on all function signatures and class attributes.
- Naming: `*Provider`, `*System`, `*Engine`, `*Layer`, `*Config` — prefer names from the docs.
- Small, focused modules. One clear owner per file.
- No `print()` calls in production code.
- No `logging.basicConfig()` inside feature modules.

---

## Comments

A comment explains **what the code can't say**. Everything else goes uncommented.

**Where comments belong** (only these): concepts, how core concepts fit together, core algorithms,
core ideas and design philosophy, **pitfalls already hit**, and **trade-offs and compromises** (why
a cost is accepted).

**Where they don't**: anything the code already makes clear. `# find the root` over a while loop,
`# memory prefixes` over `_PREFIX = {...}` — delete restatements like these.

**Five hard rules:**

1. **Describe the present, not the history.** A comment says what the code is **now** and why it
   has to be that way, not how it got there. No "used to / previously / originally / the old implementation /
   changed to / before the refactor". To block a dead end, write a **present-tense prohibition** —
   "**don't** X: that would Y" — not "it was X, then we changed it to Y". The first helps the next
   reader; the second is just history.
2. **No development or tuning process.** Parameter sweep tables, eval-set descriptions, hit-rate/MRR
   numbers, milestone labels (`F18` / `T5.2` / `M6-A`), world ids and step numbers from debugging
   sessions don't belong in code. **Keep the conclusion and the constraint, drop the process**:
   "don't raise this; from 0.4 up it's a net loss" is useful, the sweep table isn't.
3. **Concise and direct.** Say the point in concrete words, in a line or two; no build-up, no stacked
   metaphors or parallelisms, no restating it another way. A comment that can't be said in one short
   paragraph usually means the code should be split.
4. **A wrong comment is worse than none.** Identifiers named in a comment must exist (update them on
   rename or move), and the behavior described must match the code. Fix or delete wrong or redundant
   comments when you find them; don't route around them.
5. **Comments are always in English**, docstrings and test docstrings included. Runtime strings such
   as prompt text and narrative text are not comments and are exempt.

---

## Logging

Use the project's unified logging stack exclusively.

```python
from core.logging import get_logger
from core.context import set_log_context

logger = get_logger(__name__)

# Initialize once at startup
from core.logging import configure_logging
configure_logging()

# Attach context where relevant
set_log_context(world_id=world_id, agent_id=agent_id, step=step)

# Message = snake_case event name; structured fields go in extra=, never as keyword arguments
logger.info("decision_made", extra={"action": action_type, "need": dominant_need})
logger.error("provider_call_failed", extra={"provider": provider_name, "error": str(e)})
```

**Log levels:**

| Level | When |
|-------|------|
| `error` | Failures requiring investigation |
| `warning` | Unusual but recoverable situations |
| `info` | Important runtime events (world init, agent step, snapshot write) |
| `debug` | Detailed troubleshooting |

**Rules:**
- **The message is an English `snake_case` event name, not a sentence**: `"world_run_started"`, not
  `"world run started"` or `"Failed to load relation"`. Event names are identifiers you can grep and
  aggregate; details go in `extra`.
- **Structured fields go only in `extra={...}`.** `get_logger` returns a stdlib logger, so keyword
  arguments (`logger.info("evt", agent_id=x)`) raise `TypeError` — and **only when that level is
  actually enabled**, so a `logger.debug(..., foo=1)` lies dormant until someone turns on DEBUG.
- Never scatter `print()` or ad-hoc `logging.getLogger()` across modules.
- Include `world_id`, `agent_id`, `step`, `event_id` as structured fields when available.
- Do not log full prompts, large memory payloads, or full snapshots unless explicitly needed for debugging.
- Avoid noisy per-step debug logs inside the engine loop — they flood output.

---

## Constants & Configuration

- Avoid hard-coded business values and magic strings.
- Centralize constants near the subsystem that owns them. Do not create a global dump file for unrelated constants.
- Environment-dependent values (API keys, DSNs, model names) go in config, not code.
- Configuration is loaded from the `config/` directory; select a file with `CONFIG=config/config.test.yaml python main.py`.
- **Emotion types in LLM prompts must use `EmotionType.prompt_list()`** — never write a slash-separated emotion list by hand. Any value returned by an LLM at an emotion boundary must go through `parse_emotion_type()`. To support a new informal label (e.g. a synonym the LLM may produce), add it to the synonym table in `parse_emotion_type()` — do not expand the prompt with non-canonical values or widen the enum without a deliberate decision.

### max_tokens needs headroom over the estimated output

Every LLM call sets `max_tokens=output_budget(estimate)` (`core/interfaces/llm.py`). The call site
specifies the **estimated answer token count**; `MAX_TOKENS_MULTIPLIER` supplies the shared headroom.
Never hard-code the final token budget or a multiplier at a call site.

**How to estimate the answer**:
1. **Text fields**: start from the length limit declared in the prompt — Chinese `N chars × 1.5`
   tokens/char (conservatively covers qwen/deepseek/claude CJK tokenization); English
   `N characters ÷ 3`.
2. **Numbers / enums / booleans**: ~3 tokens each.
3. **JSON structure overhead**: ~5 tokens per field (key + quotes + braces + commas).
4. **Array outputs** (goals / dialogue / updates / insights…): `per-element estimate × realistic max
   element count` (the prompt's explicit cap, or the realistic upper bound for that scenario).

**Why the multiplier is large**: thinking and the answer share the budget, and on endpoints that
can't turn thinking off ("always thinking" models such as GLM-5.3) thinking now and then runs to
~3× the answer: an interrupt check whose answer is ~105 tokens spent 318 of 320 thinking and
returned nothing. If calls still truncate, raise `MAX_TOKENS_MULTIPLIER`; don't pad one call's
estimate.

**Why**: `max_tokens` is a **ceiling**, not what you're billed for (billing is on actual output), so
setting it generously only prevents truncation and **costs nothing**. Truncation, on the other hand,
produces **unparseable JSON**: a lost or polluted step at runtime, a retry or raise at build time.

**Discipline**:
- The estimate starts from the prompt's length limits, so **add the limits first, then compute
  the estimate**; don't guess a number. The limits are themselves mandatory; see
  "LLM Output Format · Every text output declares a length limit".
- Whenever you add or change an LLM call (including adding/removing fields, loosening a length
  limit, or adding a lite-CoT `reason`/`thought` field), **recompute** the estimate — a prompt whose
  output grew without raising it is the classic truncation trap.
- Show how the estimate was reached in a comment at the call site (e.g. `# inner_monologue ≤60 chars
  + …; estimate ~290 tok`) so it can be checked.

---

## Provider Rules

- All external integrations live in `providers/`. Business subsystems (`agent/`, `engine/`, `world/`) depend on `core/interfaces/`, never on concrete provider classes.
- Every provider category should have an `InMemory` implementation for tests.
- Provider config is environment-driven and selection-based (not hard-coded model names or DSNs).
- Providers handle retries, timeouts, and error normalization internally — do not scatter retry logic in callers.
- Providers retry transient transport errors (HTTP 429 rate-limit, 503/529 overload) with automatic backoff internally. Callers must not handle or retry these — they are invisible above the provider boundary.

---

## Fallback & Error Handling

The project has two distinct operational phases with completely different fallback strategies:

| Phase | Characteristics | Cost of failure |
|---|---|---|
| **World building** (one-shot init) | User-facing, synchronous, output quality sets the simulation foundation | High — a world built from generic placeholders is worse than a clean failure |
| **Simulation runtime** (loop) | Continuous, frequent LLM calls, transient failures are expected | Low — a single-step failure must never kill the simulation |

### Rule 1: Runtime LLM calls — always fall back, never crash

**Applies to:** All LLM calls inside `agent/decision.py`, `agent/agent.py`, `engine/message_system.py`, `agent/memory.py`, and any other per-step cognition path.

```python
try:
    response = await llm.complete(...)
except Exception as exc:
    logger.warning("llm_call_failed", extra={"context": "...", "error": str(exc)})
    return <most conservative valid result>
```

- The fallback value must be directly usable by the caller — not `None` unless the caller explicitly handles `None`.
- Tag the source where a metadata channel already exists: `llm_model="fallback"` or `metadata={"source": "fallback"}`.
- Log at `warning`, not `error`.

**Fallback tiers.** Whatever a fallback produces ends up in persistent narrative state that
embeddings will recall later, so on failure write as little as possible. Pick the tier with the
smallest blast radius:

1. **No-op / empty / skip (preferred)** — if the output would be persisted (memory prose, emotion,
   relation delta, goal, reflection), a failure **writes nothing**:
   return a structurally valid empty/unknown value (empty list, empty delta, `None` that the caller
   handles explicitly) so downstream knows nothing was obtained, rather than receiving forged
   cognition. A fabricated memory is far worse than a step with no memory. Both cognition paths use
   this tier:
   - **Decision**: `DecisionEngine.decide` returns `None` on failure (`AgentStepPlan.action=None`)
     and the runtime skips that agent for the step — no execution, no commit, no memory. Never fall
     back to a placeholder action that gets executed and embedded.
   - **Executor adjudication**: when adjudication **never happened** (the judge LLM failed, or the
     actor's `Agent` object is missing), return a structurally valid
     `ActionResult(success=False, adjudication_failed=True)` and **don't invent** `factual_memory`,
     relations, entities or detections. The feedback layer reads `adjudication_failed` and treats the
     step as **null**: it skips all writeback (memory, emotion, need feedback, goal progress) and
     only resets the action state to idle so the next step re-plans. **The executor returns
     normally and feedback filters it out**, so an infrastructure failure never reaches the narrative.
2. **Most inert output (when the loop must have an output and even a null step won't do)** — fall
   back to the **most inert, non-advancing** result (observe/wait, etc.) and tag it through the
   existing marker channel. Current cognition paths all sit on tier 1; use this only for a new path
   whose loop strictly needs an output.
3. **Conservative default (last resort)** — return a concrete conservative value only when neither
   of the above fits.

**`success=False` has two meanings that must stay distinct**: the LLM **ruling** the action failed
(missed, the talk broke down) is a **real world event** — `adjudication_failed=False`, remembered
normally ("I tried X and failed" is narrative). Only when the LLM **couldn't rule** (infrastructure
failure) is `adjudication_failed=True`, filtered into a null step by the feedback layer. **Never**
set this flag on an invented "success".

**Never** persist or embed fabricated narrative after a failure, including invented dialogue,
successful action results or relation changes. Retrieval can repeatedly surface this false state,
and downstream systems cannot distinguish it from genuine simulation events. An empty result is
safer than fabricated cognition.

**Failure markers live only in code-layer metadata and never leak into the narrative layer**:
`source="fallback"` or any failure state must not enter memory prose, embedded text or an
in-character prompt (that just creates a new source of pollution, such as a "[cognition failed]"
memory recalled over and over). If a later LLM genuinely needs to sense the failure, translate it through
the narrative layer into a natural in-character description ("你一时心绪纷乱，没能理清"), not a
code-layer signal. See "Narrative layer / code layer reference boundary".

**Rule-based cognition code exists only as a failure floor.** There is no deliberate rule
path (some class of agents using rule adjudication, rule emotion or rule importance) — everyone goes
through the LLM (see §5). So any rule-based cognitive output in the code can **only** be a fallback
for an LLM failure and must obey the tiers above. **Never** reintroduce a rule adjudication path
that by design returns `success=True` plus a template fact: it looks identical to the failure floor,
and once both exist the failure path inherits "assume success and fabricate a fact", laundering
infrastructure faults into narrative facts. **The code structure guarantees this; don't weaken it
into a convention people have to remember.**

**Main-character exception:** For `is_main_character=True` decision / interrupt / emotion paths, retry once before falling back — main-character cognition quality directly determines narrative emergence. Implement this with `LLMRouter.complete_with_retry()` — do not write inline retry loops at call sites. Background agents take the **same LLM cognition path** but call `LLMRouter.complete()` directly (single attempt, same scene); the decision runs `complete_with_retry` for every agent. The difference is retry, never LLM-vs-rules.

### Rule 2: World-building LLM calls — retry once, then raise

**Applies to:** `ThemeAnalyzer`, `TemplateSelector`, `AgentGenerator`, and `CastDesigner` in `world/builders/`.

World building is a one-shot operation. A world assembled from generic placeholders ("The Protagonist", "The Rival") is not a usable result — the user should retry rather than receive a silently degraded world. Failures here are operational failures and must propagate upward.

Field-level coercion helpers (`core/coerce.py`) are still correct and should be kept — when the LLM returns a structurally valid payload with minor type mismatches, coercing is better than rejecting.

The semantic retry loops (validating JSON structure completeness) live inside each builder class and call `LLMRouter.complete()` directly. Transport-level errors within those loops are handled automatically by the provider — callers do not need to handle them.

### Rule 3: Config and init parameters — raise immediately, fail fast

**Applies to:** API key checks, numeric parameter validation, container assembly, location resolution, and all other startup-time validation.

Configuration errors are deterministic — retrying will not change the outcome. Providing a fallback only hides the real problem.

### Rule 4: asyncio.gather — always use return_exceptions=True, handle per slot

```python
results = await asyncio.gather(*tasks, return_exceptions=True)
for ctx, result in zip(contexts, results):
    if isinstance(result, BaseException):
        logger.warning("task_failed", extra={"error": str(result)})
        # apply Rule 1 or Rule 2 for this slot
```

One task's exception must not cancel other in-flight tasks. `return_exceptions=True` converts exceptions into return values; each slot is then handled by its own applicable rule.

### Rule 5: Executor failures — return a failed ActionResult, never swallow silently

```python
except Exception as exc:
    logger.warning("executor_failed", extra={"executor": ..., "error": str(exc)})
    return ActionResult(success=False, outcome="The action could not be completed.")
```

`except Exception: pass` is forbidden in executor code. Callers depend on `ActionResult` to know whether execution succeeded.

An executor's **adjudication that never happened** — the judge LLM call raising, or the actor `Agent` object being missing — is the same case: return `ActionResult(success=False, adjudication_failed=True)` with **no fabricated `factual_memory`/relation/entity** — it must not assert `success=True` or invent an outcome. The feedback layer treats `adjudication_failed=True` as a **null step**: it skips all writeback (memory, emotion, need-feedback, goal-progress) and only resets the agent to idle — so the infra failure never reaches narrative. A genuine in-world failure (the judge ruled "you missed / the talk broke down") keeps `adjudication_failed=False` and is remembered normally. There is no rule-verdict path to fall back to — every actor is LLM-judged (see §5).

### Rule 6: Runtime provider read vs. write failures

- **Read failures** (`embedding.embed`, `vector_store.search`): return an empty result, log `warning`, simulation continues.
- **Write failures** (`agent_store.save_agent_state`, `snapshot.save`): log `error`, do not swallow silently. A missed state write means data loss on the next restore; repeated write failures should surface to the runtime layer.

### Rule 7: No theme-specific logic in generic code paths

Keyword matching, character-name checks, and hardcoded presets for a specific theme belong in `worlds/` as part of the relevant `WorldConfig` — not as `if "some_keyword" in theme` branches inside `world/builders/`. Every such branch in a generic pipeline is a maintenance trap that violates YAGNI.

### Quick reference

```
Config / parameter validation?            → raise, no fallback
World-building LLM call?                  → retry once, then raise
Runtime LLM call (background agent)?      → catch, log warning; prefer no-op/empty > inert+marker > conservative value; never persist/embed fabricated narrative
Runtime LLM call (main character)?        → retry once, then fallback (same LLM cognition path, plus one retry and a higher model tier)
Should a cognition gate use the LLM?      → always, for every agent; there is no "background uses rules" path (§5)
asyncio.gather?                           → return_exceptions=True, handle per slot
Executor exception / actor missing?       → log warning, ActionResult(success=False, adjudication_failed=True), invent nothing
Provider read failure?                    → return empty result, log warning
Provider write failure?                   → log error, do not swallow
Field-level LLM output parsing?           → core.coerce silent correction, no log needed
```

---

## LLM Output Format

When an LLM call must return structured data (multiple fields, typed values), always use JSON output format — never custom line formats like `KEY: value`.

```python
# Prompt instructs strict JSON output
prompt = (
    "...\n"
    "严格输出以下 JSON，不要任何多余内容：\n"
    '{"success": true或false, "fact": "...", "feel": "...", "relation": "positive或negative或neutral"}'
)

# Parse with the shared helper in core/interfaces/llm.py
from core.interfaces.llm import extract_json
data = extract_json(response.content)
succeeded = bool(data.get("success", True))
```

- Use `extract_json()` from `core/interfaces/llm.py` — it strips markdown fences and finds the first `{...}` block.
- Always provide `.get(key, default)` fallbacks so a missing field degrades gracefully rather than raising.
- Migrate any remaining line-format parsers (custom `KEY: value` parsers) to JSON when the surrounding code is touched — do not do a standalone mass migration.

### Every text output declares a length limit

**Every piece of LLM-produced text — string fields in JSON, array elements, and whole free-text
outputs — must have an explicit length limit in the prompt.** Arrays also need an element-count cap
(`最多 N 条`). A field without a limit is a defect; add the limit whenever a prompt is added or changed.

**The unit follows the output language**: Chinese output in characters (`≤N 字`), English output in
characters (`≤N characters`) — never English words, whose length varies too much to convert into a
stable token count.

**Why**: without a limit the output is unbounded, so `max_tokens` can't be estimated (step 1 of the
estimate needs this N), and sooner or later the output gets truncated into **unparseable JSON** — a
lost or polluted step at runtime, a retry or raise at build time. There's a second cost too:
unbounded narrative fields drift longer and looser, and once persisted they make recall and
readability steadily worse.

**How to pick N**: case by case — **how much room does this field need to say what it's for?** There
are no standard tiers; fields do very different jobs, and copying another field's number or picking
something generously large is skipping the work.

**Discipline:**
- Put the limit **in the field's own declaration** (the Output block; see Prompt Design
  Principles §3, "output format goes last"), one limit per field. **A global "please be concise" is
  not a limit** — it can't be converted into tokens, so it doesn't count.
- Free text (non-JSON) needs a limit too; "it isn't parsed anyway" is no excuse.
- After adding a field, loosening a limit, or adding a lite-CoT `reason`/`thought` field,
  **recompute the `output_budget` estimate**.
- **The limit lives only in the prompt; code must not truncate as well**: `text[:N]` leaves broken
  sentences and creates a second source of truth. What separates this from structural coercion
  (`core.coerce`) is whether the content gets damaged.
  - **Truncating with a buffer doesn't help either**: models overshoot rarely and only slightly, so a
    buffer only makes broken sentences rare — and a rare, silent, permanent defect is harder to find
    than a frequent visible one, while the second source of truth is still there. Runaway output is
    `max_tokens`'s job, and its failure mode (unparseable JSON → Rule 1 fallback → retry next step)
    is the better one.
  - **Text that only goes to logs or traces can be truncated** (`extra={"reason": text[:120]}`):
    cutting there is harmless, and not cutting floods the log. The test is the same — anything
    persisted, embedded, fed back into a prompt or shown as narrative is never truncated.

### Prompt string style

Write LLM prompt text as **multi-line strings (triple quotes `"""..."""`) where possible**, not
adjacent string literals concatenated (`"...\n" "...\n" ...`). A multi-line string looks like what the
model sees and keeps the whole prompt easy to read and edit; concatenation is hard to read and
easily drops a space or newline.

```python
# Preferred: triple-quoted multi-line string
_SYSTEM_PROMPT = """\
你是……
【原则】……
只输出 JSON：{"goals": [...]}
"""

# Avoid: adjacent literal concatenation (unless interpolation mid-string truly can't be a multi-line f-string)
_SYSTEM_PROMPT = (
    "你是……\n"
    "【原则】……\n"
)
```

- Use a triple-quoted **f-string** when interpolating. Dynamic parts that can't be interpolated in
  one piece (signal lists built in a loop, etc.) may be built separately and joined, but every
  **static paragraph** is still a multi-line string.
- Prompt templates **must not bind to a specific theme, tension or scene** (Rule 7): proper names and
  specific conflict structures come only from data injected at runtime, never from template literals.

---

## Prompt Design Principles

The previous section covers prompt string syntax; this section covers content and layout.
Every new or modified prompt must follow these principles.

### 1. Organize prompts into sections by responsibility

A prompt must be split into single-responsibility blocks, not one run-on paragraph. Standard sections:

| Block | Question it answers |
|---|---|
| **Role** | Who you are, whose viewpoint you take (see below: in-character vs functional) |
| **Inputs** | What context/data is provided (relations, memory, scene signals…) |
| **Task** | What to do |
| **Constraints** | What not to do, where the boundaries are |
| **Output** | Format, fields, length |

Rationale: sectioning sharply reduces the chance the LLM skips a key constraint, and keeps the
prompt maintainable — changing a constraint won't disturb the task description. Separate blocks
with clear headers/delimiters (`【Role】` `【Task】` `【Output】`, etc.).

**Role has two distinct forms that must not be mixed:**

- **In-character (first-person role-play)**: the LLM **is the agent itself**, experiencing and
  reacting as "I". Used on the main-character cognition path — decision, emotion, dialogue, goal
  generation. The viewpoint must stay first-person throughout: **even the constraints and output
  requirements must be phrased in the "you/I" voice** — never cut back to "as an AI assistant,
  please evaluate…" and break the immersion.
- **Functional (out-of-character)**: the LLM is a **processor performing a specific task** (judge,
  extractor, generator). Used for judging, information extraction, structured output — it should
  not, and need not, stay in character.

The choice follows the LLM-vs-rules principle: output that depends on **personality / situation /
subjective interpretation** → in-character; output that is **objective processing of structured
signals** → functional.

### 2. Negative examples spark creativity, positive examples enforce rigor

Example choice depends on the task's nature:

- **Creative tasks** (emotion, dialogue, goal generation, narrative advancement, etc.):
  **lead with negative constraints** — list the failure modes to avoid (clichés, preachy tone,
  breaking first-person viewpoint, labeling…) and **give as few positive examples as possible**.
  Positive examples get imitated/anchored to, collapsing output diversity — the enemy of emergent
  narrative.
- **Rigorous / structured output** (JSON schema, judging, information extraction, etc.):
  **give both** — positive examples pin down format and fields, negative examples rule out common
  errors.
- Even when a creative task must show an example, **show only the format, never the content**, and
  label it explicitly as "format illustration only".

This generalizes two existing rules that must not be violated: an LLM-judge prompt lists only the
"failure modes to penalize", not the "correct answer"; and never patch a prompt against the
positive example of some specific validation scenario (overfit).

### 3. Put important content at the start or end

The LLM's attention is weakest in the middle of a prompt (lost in the middle); lay out accordingly:

- **Role + the most critical constraints** → at the **start**
- **Output format requirements** → at the **end** (adjacent to the generation point, most likely
  to be obeyed)
- **Large dynamic context / candidate lists** → in the **middle**

### 4. Carry discrete items as lists, not prose

The LLM understands and obeys **discrete structured items** far better than dense prose: each item
is independently addressable, lowering the chance of skipping or merging items.

- Constraints, steps, candidates → use a list, **don't bury them in a paragraph**.
- When there is order, or you need back-references or counting, use **numbering** (1. 2. 3.);
  when order is meaningless, use bullets. A number itself conveys "ordered / referenceable"
  semantics — **don't number for the sake of numbering**; numbering an unordered set is misleading.
- When the list is "a candidate set for the LLM to pick from", numbering is also the **output
  contract**: have the LLM return integer indices, not opaque ids. See the next section,
  LLM Indexed Reference Pattern.

### 5. Think before acting: reasoning fields come before conclusions and actions (global)

**Whenever an LLM output thinks before it acts — one JSON holds both a reasoning field and the
conclusion / decision / action / goal fields derived from it — the reasoning field comes before the
fields it drives.** The model generates in output order, so reasoning written first works as lite
chain-of-thought: the conclusion follows from the analysis instead of a guess. Reasoning placed after
the conclusion achieves nothing: the model has already guessed and is just justifying it. **This is
global, regardless of role** — both forms below apply:

- **Functional third-party rulings** (judges, extractors, objective verdicts — out-of-character tasks
  that produce a verdict, state, score or direction): the JSON opens with a short `reason` /
  `rationale` field, conclusion fields right after. Examples: the covert/physical judges, goal
  progress evaluation (each goal's `reason` before `status`), relation evolution (`rationale` before
  `labels`/`summary`).
- **In-character lite-CoT** (first-person decisions that produce a choice, an action, whether to
  interrupt, a goal revision): the JSON opens with a first-person thinking field (`thought` /
  `inner_monologue`), then the choice/action fields. The thinking stays first-person ("我此刻在
  想……") and never breaks immersion. Examples: interrupt evaluation (`thought` before `interrupt`),
  `DecisionEngine` (`inner_monologue` before `selected_index` and the other action fields).

**Boundaries (don't overcorrect):**
- **Parsing never depends on the reasoning field**: code reads only the conclusion/action keys
  (status / score / selected_index…); the reasoning field exists only to anchor the reasoning and
  can be ignored (if it ends up in a summary, log or memory, that's a bonus parsing doesn't rely
  on). So putting it first carries **no structural risk** and only makes conclusions steadier.
- **"First" means literally the first key — even an IndexedRef selector yields**: per-item array
  rulings (each element shaped `{selector index, reasoning, conclusion}`, e.g. the `index` in goal
  progress or `target_index` in relation evolution) tend to put the back-reference in
  front of the reasoning. `index` only says which item and isn't a conclusion, but once it comes
  first, the reasoning no longer does, and it does less work. Parsing reads by key,
  independent of order, so moving the selector after the reasoning is **free**. Correct shape:
  `{"reason"/"rationale" first, "index"/"target_index", conclusion fields}`.
- **The rule only orders reasoning fields that exist; it doesn't demand one for purely subjective
  output**: free in-character expression that isn't derived from specific evidence (dialogue,
  short/long-term goal text) is direct subjective output with no think-then-act structure. **Don't**
  bolt a justification field onto it — that breaks first person and immersion.
- **But in-character doesn't mean no reason field ever**: if a first-person output must be **derived
  from given evidence or signals**, it is think-then-act, and its grounding/reason field still comes
  before the conclusion — **just phrased in first person** ("我是被…触动的", not a detached "assessment
  follows"). Typical case: the perception-emotion prompt — the emotion must rest on what is perceived
  right now, with no invention, so `reason` (which signal moved me) comes before
  `emotion`/`intensity`/`valence`; grounding-first directly serves anti-fabrication (the emotion
  follows from real signals, rather than picking a dramatic emotion and justifying it afterwards).
  The test is the same as §1's role split and §2's LLM-vs-rules principle.

Any new prompt with a think-then-act structure follows this: reasoning first, conclusion/action after.

### 5b. Order output fields by dependency (the general form of §5)

§5 covers "reasoning vs conclusion", but that is a special case. **The real rule: in one JSON
output, if the correct value of field C depends on fields A and B, then A and B come before C.**
A reasoning field is just the extreme case that every conclusion depends on.

**Why:**
1. **Externalized working memory**: autoregressively, if A and B come first, C is sampled under
   `P(C | A,B)` — A and B are already in context and the model can actually read them. If C comes
   first, it has to be computed internally in one forward pass, with no chance to write down the
   intermediate values.
2. **No backtracking ⇒ the grounds get polluted backwards (worse)**: a token, once written, is in
   context for good. If C comes first, A and B become `P(A,B | C)` — the model **makes up grounds
   for the C it already wrote**. So the wrong order doesn't just make C worse; it spoils A and B
   too: what should have been verifiable grounds becomes post-hoc justification, and nothing
   downstream can tell the difference.
3. **Free**: parsing always uses `.get(key)`, independent of field order, so reordering is pure gain
   with no structural risk.

**Two kinds of selector (IndexedRef) point in opposite directions and are the easiest to get wrong
— §5 covers only the first:**
- **Back-reference coordinate** ("which item am I ruling on"): the `index` in goal progress,
  `target_index` in relation evolution. Nothing depends on it; it's just a coordinate, so it
  **yields to the reasoning**, placed after `reason`/`rationale` (§5, second boundary).
- **Generation premise** ("whom/where does this land on"): `DecisionEngine`'s `person_indices` /
  `destination_index` / `physical_entity_index`, `EventSystem`'s `recipients` / `location_scope`, dialogue's
  `speaker`, reflection's `source_indices`. **It shapes how the content after it gets written**, so
  **it must come before that content**. The exact words said to a recipient can't be
  written before the recipient is chosen; write the words first and then look for an index, and the
  model just grabs one off the list — which is why decisions sometimes cite an index that doesn't
  exist.

**Work out who depends on whom from what the fields mean; don't assume (the same fields can go either way):**
- Third-party rulings (physical/covert): `reason` (grounds) → `success`/`achieved` (verdict) →
  `outcome`/`fact` (narrating the verdict) → `damage` (quantifying the narrated harm). **Narration
  follows the verdict**, so the boolean before the narrative is right.
- First-person self-assessment (TALK under work/social): no separate `reason` field; `fact` is the
  only grounding → `fact` (what actually happened) → `success` (read off from it). **The verdict
  follows the narration** — the opposite direction.
- Qualitative vs quantitative (`initial_relations`): `labels` (what these two are to each other) →
  `trust`/`affection` (numbers to match). Write the numbers first and the labels just bend to fit them.

**Boundaries:** ① Order only where there is a **real one-way derivation**; forcing an order between
independent fields is noise. ② If the dependency is two-way (A and C constrain each other), no
topological order exists — the answer is a free `thought` field first, not a forced order.
③ **This rule covers output only**; input order follows §3 (important content at the start or end)
and the prompt prefix cache guidelines.

### 6. Flat slots in the prompt, structured types in code

**The two ends of one contract should look different. Don't hand the code's structured model to
the LLM as-is just because it looks tidier.**

- **Prompt / JSON schema side: keep fields flat, one named slot per purpose** (`destination_index`,
  `physical_person_index`, `physical_entity_index`, `physical_recipient_index`…), each slot bound to
  its own candidate list. **No `kind` + `id` pair** — that makes the model answer a
  classification question before it can bind a target. Every extra question is another chance to
  hallucinate, and a wrong answer gets the **category** wrong, sending the target into a different
  namespace entirely.
- **Code side: collapse into a structured reference** (`ActionTarget`'s `Ref(kind, id)`). `kind` is
  derived from **which slot was filled**; the LLM's self-reported category is **never trusted**.

**The reason is type safety, not style**: a slot bound to a list means the model has no chance to
name the wrong category — it outputs only an index and code infers the category from the channel.
A single shared slot invites cross-namespace fallback: an out-of-range person index falls back into
the item list, and "按住某人" turns into "攥住某物" — wrong object and wrong category, while
`action_description` still names the person.

**Never make the model classify**: any schema that requires answering a category question before a
target can be bound is wrong — it replaces a fact the slot already implies with a guess that can be
wrong, and a wrong guess sends the whole target into another namespace. A category nothing routes on
is pure overhead: delete it and let the binding code infer it.

**Boundary**: this is about **slots**, not "more fields is better". A slot earns its place only if it
has its own candidate list or is a genuinely separate parsing path; a field with no list to bind to
is just noise. The translation happens in one place (decision parsing in `agent/decision.py`), and
each side keeps its natural shape — the same boundary as IndexedRef's "speak with names, select
with indices".

---

## Prompt prefix cache

**What it is**: providers cache prompts automatically by token prefix — the prefix that is
**byte-for-byte identical from the start** across calls is reused, up to the first differing byte;
everything after that misses.

**Why it matters**: a cached prefix isn't recomputed, cutting latency and cost directly. The
simulation calls the LLM heavily every step, so the savings compound.

**Principle: stable content first, volatile content last.** Put what is invariant across calls
(role setup, principles, constraints, legend/scale constants, output schema) first, forming a
stable reusable prefix; put what changes per call (situation, persona, emotion, memory, candidates)
after it. Anything that varies per call or per agent must stay out of the prefix, or the prefix
changes every time and the cache never hits. Every new or modified prompt is laid out this way.

**Implement it with system / user messages**: the stable prefix goes in the **`system`** message,
volatile content in the **`user`** message. `system` comes first, so the combined prompt is stable
first and volatile last, and `system` naturally becomes the cached prefix. Prompt builders return
`tuple[str, str]` (system, user), and the call site passes
`[LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)]`.
- `system` holds only content that is byte-identical across calls; world-level constants (such as
  durations converted from `seconds_per_step`) may go in, but per-agent variables must first be
  replaced with a generic referent (e.g. write `{name}` as 「被评判者」), with real values left to `user`.
- Order `user` the same way: context shared across calls first, call-specific content last, so the
  cached prefix runs longer.
- References across the two messages must point the right way: `system` refers to `user` content as
  「下面/所给」, and the end of `user` refers back to `system` as 「上面说定的」.
- See `agent/decision.py` and `engine/world_pressure.py`.

---

## Narrative layer vs code layer (ids never cross the LLM boundary)

**Ids belong to the code layer; the narrative layer has none.** A character in a novel doesn't know
what page they're on: an id is the simulation's internal coordinate, not part of the story. Each
layer refers to things in its own way:

- **Code layer**: the id is the only trustworthy reference (precise, stable, joinable). Structured
  fields (`related_agents`, `from_id`/`to_id`, `entity_id`), storage, snapshots and event routing all
  use ids.
- **Narrative layer** (memory prose, LLM prompts, dialogue, ambient/outcome text, anything embedded):
  people and things are referred to **as the character knows them** — a name (if known), a form of
  address or relation ("父亲"), a description ("一个陌生人"). This is inherently subjective (the same id
  can go by different names in different agents' memories); that's part of what makes the narrative
  true, not a defect.

The layers meet only through a **translation boundary**, at fixed crossing points:

1. **Down (code → LLM)**: when building a prompt or writing memory prose, ids become names or
   descriptions before entering text. The god's-eye side uses `WorldDirectory`; the cognition side
   uses names the agent has accumulated through perception: someone in view is known by name and
   gender, a letter gives its sender's name (bound by information asymmetry — someone never
   perceived is described; never consult the global table).
2. **Up (LLM → code)**: the LLM never produces ids; it uses IndexedRef (next sections). Going up,
   **names aren't used** as references either — names are ambiguous and easy to misspell: "speak
   with names, select with indices".

**Test**: an id in any text the LLM reads or that gets embedded is a layer leak — the same kind of
boundary violation as an executor mutating agent state directly, not a matter of style.
**Why**: embedding recall is semantic, so a memory that only contains an id can never be recalled
(it's lost the moment it's written), and an id in an in-character prompt breaks first-person
immersion and reasoning.

- Any `name or id` fallback chain is a leak by this test: fall back to a **descriptive referent**
  ("某人", "那个地方"), not the id.
- **Known, accepted compromise**: ambient text lets bystanders read the names of actors they don't
  know. A name is a legitimate narrative reference, just shown to the wrong audience — a compromise
  within the narrative layer, milder than an id leak, and accepted for now. Don't "fix" it in passing
  under this rule.
- **Memories recalled by embedding should mention known names (recommended, not mandatory)**: a
  first-person memory that uses **only** a form of address ("殿下", "那位大人") recalls poorly when
  later queried by name — a milder version of the id-only problem. So in in-character prompts whose output **gets written to memory**, **suggest**
  that the LLM include the name when mentioning someone **it knows**; where it habitually uses a form
  of address, the name can follow it (「称谓（名字）」). Three boundaries: ① only for people the agent
  **knows** (strangers are still described, no invented names; the name comes from perception or the
  relation's `to_name`, never the god's-eye directory, preserving information asymmetry); ② it's a
  **recall suggestion, not a hard format** — the prompt must say plainly not to apply it
  mechanically or bracket every mention, or memories turn into piles of parentheses; ③ it applies
  only to **memory**, not **dialogue** (in dialogue a form of address is natural speech and must not
  become 「称谓（名字）」).

**Time works the same way: steps belong to the code layer; the narrative layer has only points in
time and durations.** A step is the integer coordinate for scheduling, snapshots and decay; people
in the story experience dates in the world's calendar and natural durations ("次日清晨", "过了约半日"),
not "step 47".

- **Down**: points in time use `WorldTime.time_label` (a calendar label rendered from the world's
  time config); durations are converted from `seconds_per_step` into natural language ("约3小时") —
  see the step-duration hint in `agent/decision.py`. `iso_label()` (which contains `step=NNNN`) is
  only for code-layer logs and snapshots and must never enter a prompt or memory prose.
- **Up**: the LLM outputs a step count (e.g. `estimated_steps`) only when the prompt has explicitly
  taught the conversion unit ("每步约1小时") — a controlled upward channel like IndexedRef;
  otherwise the LLM uses natural durations and code converts them into steps.
- **Test**: "第N步" in in-character text, memory prose or dialogue is a leak (e.g.
  `"我在第{step}步消亡了"`, `"对话进行中（第{elapsed}步）"`). Structured fields (`created_step`,
  `updated_step`, `started_step`) belong to the code layer and are fine.

---

## Executor narrative principle (core)

> **Every executor outcome or tick must identify who is acting, where, and what they are doing. A terminal outcome must also describe the result.**

`outcome` is third-person and observable (the bystander perception channel); `factual_memory` is first-person (the actor's own memory channel). They must not be mixed; first-person cognition paths read only `factual_memory`.

---

## Referring to people: always include gender

Chinese narrative **can't avoid gender** — pronouns (他/她), forms of address (公子/娘娘/兄妹) and
kinship labels (父子 vs 父女) all encode it. An LLM that isn't told will **guess**, and a
wrong guess lands in persistent narrative state that embeddings recall (memory prose, outcome,
dialogue, relation labels), where it never heals and keeps spreading. So gender isn't optional
context; it's as essential as the name.

**Two canonical renderers. Every new or changed rendering of a person goes through one of them;
don't assemble your own:**

| Renderer | Shape | Used for |
|---|---|---|
| `SoulLayer.identity_text()` | `名字，N岁，性别` | **Profile form** — a heading followed by personality/background: the self persona (`to_prompt_context`), director menu, event briefing, the target character in pressure assessment |
| `core.prompts.person_referent(name, gender, *marks)` | `名字（性别，在场/已故/身份…）` | **Roster form** — one line per person: the decision's reachable roster, co-present people in perception, relation lines, the event recipient menu, people present at an adjudication |

Choose by **output shape**, not by whether a `soul` object is available. Event recipient menus and
lists of people present in pressure assessments both use roster lines.

`marks` go in **the same brackets** as gender; separate groups render as 「李世民（男）（在场）」.

**When to include it:** ask whether the prompt produces or rules on text in which this person's
gender shows. If any of these applies, it **must** be included:

1. **It produces third-person narrative** — outcome / observation / ambient / dialogue / event text;
   that's where pronouns and forms of address come from.
2. **It rules on gender-related possibility or plausibility** — physical confrontation, social norms,
   **kinship labels** (blood-kin labels written by relation evolution are never rewritten, so a wrong
   guess is **permanent**).
3. **Telling people apart** — in a roster, gender is often the cheapest way to distinguish two
   people of the same generation and surname.

**Leave it out:**

- For pure rulings **with no people in the output** (importance scoring, goal progress status, the
  event-injection gate's boolean, severity tiers) — their `reason` is a short justification that
  never becomes narrative.
- When the person is only a **routing key** in the prompt (location/item menus, or an id list
  used just to align indices).
- When gender is **already** present via `to_prompt_context()` (the executor's actor, target and both
  dialogue parties go through it) — don't add it twice.
- **No bracketed annotations in memory prose or dialogue**: that's natural narration, and
  `person_referent` is for rosters only (which is why names resolved for memory text are bare).

**How to include it:**

1. **Gender is part of identity and goes with the name** — no separate "gender" section, no JSON field.
2. **Never upward**: the LLM never outputs gender — it is immutable identity fixed at build time
   (generated by `ThemeAnalyzer`, consistent with the gendered forms of address in
   `initial_relations`). If it's missing, **leave it out**; never make one up
   (just as a missing name falls back to 「某人」).
3. **Information asymmetry**: gender is visible **at a glance** — more visible than a name — so any
   channel that already reveals the name can reveal gender too. It must still come through
   **perception** — `WorldDirectory.agent_identity_map` → `SpatialPerception.visible_agents`
   → the agent's known-agents map → `AgentRelation.to_gender`. `agent/` still never uses `WorldDirectory`;
   don't route around that to get gender.

---

## LLM Indexed Reference Pattern

**The LLM must not output long opaque ids.** When the LLM picks items from a known set (memories,
agents, targets, goals…), it uses 1-based indices, not id strings:

1. The prompt lists candidates as `#1 ... #N`; the caller decides how each is displayed.
2. The LLM returns **integer indices** (e.g. `"source_indices": [1, 3]`), never ids.
3. Code maps indices back to real ids with `core.interfaces.llm.IndexedRef.resolve()`.

**Why**:
- LLMs often hallucinate opaque ids. In one observed case the LLM returned the whole
  `[event|m_xuan_002]` label as the id, every reference was filtered out as a hallucination, and
  `reflect()` always came back empty.
- Indices are numeric, bounded, easy to validate and cheap in tokens.
- It decouples the prompt text from the id format, so how the prompt displays items can't drift
  from what the parser expects.

**Anti-pattern**: a schema like `{"source_memory_ids": ["m_xuan_001", "m_xuan_002"]}` that has the LLM
emit ids directly.

**Tool**: `core.interfaces.llm.IndexedRef` — takes `Iterable[str]`, provides
`resolve(Iterable[Any]) -> list[str]`, and filters non-integers, out-of-range values and duplicates.

**Exception**: when the LLM creates a **brand-new** entity (e.g. `ThemeAnalyzer` generating a new
agent), this pattern doesn't apply — code slugifies the id afterwards; the LLM still shouldn't
produce ids directly.

---

## Action Type Rules

Every action type exists to do at least one of:

- **Change state**: affect persistent simulation variables, on two levels: agent-internal state (vitality, emotion, need intensity, relation) and world state (agent position, item ownership and use state). The effect must persist across steps and influence later perception and decisions.
- **Move information**: change who knows what. This covers two-way sharing (TALK), one-way delivery (SEND_MESSAGE), one-way acquisition (COVERT) and passive spread (overheard). The point is changing who knows what, not just passing messages along.
- **Advance goals**: move a `GoalEntity` toward completion and drive the next cognition cycle. After every action, goal progress evaluation decides which goals advanced or completed, triggering new goal generation.

**A new action type must** meet at least one of these, with an effect no existing type can produce. If two candidates serve the same purpose with the same effect, merge them.

---

## Executor / Feedback Boundary Rule

Executors only execute and return results (`ActionResult` / `ActionExecutionState`). They must never directly mutate agent-internal state.

**Agent state** — personality, goals, emotion, needs, relations, memory, vitality, trait drifts — is exclusively updated by the feedback layer on `Agent`:
- the action-completion feedback path for the acting agent
- `apply_target_effect` for the target agent

**Two canonical update paths:**
- `ActionResult.relation_updates` → the acting agent's feedback applies relation deltas.
- `ActionResult.target_effects` → `apply_target_effect` applies effects on the target agent.

Executors must never call `relation_system`, `memory_system`, `personality`, or `need_engine` for mutation. If an executor needs to express a state change, it declares the delta in `ActionResult` fields and leaves application to the feedback layer.

**World infrastructure** (environment mutations such as `move_body`, and message dispatch) is not agent-internal state — executors may handle these directly as part of execution.

---

## WorldDirectory Usage Rule

`WorldDirectory` (contract in `core/interfaces/directory.py`, implementation in `engine/directory.py`)
is a per-world **read-only identity facade**: id → immutable display information (name /
description / kind / role). In short: **when the god's-eye layer holds only an id and needs
immutable identity information, ask the directory. Otherwise, don't reach for it.**

**Use it:**
- To render human-readable text in id-only contexts (action records, message senders, log/snapshot
  assembly) → `agent_name(id)` / `location_name(id)` / `entity_name(id)`.
- When you're about to add an `agents: dict` dependency just to get names → inject `WorldDirectory`
  instead. A `... .soul.name if obj else id` defensive pattern in new code is a violation (guard:
  `grep -rn "soul.name if" engine/` must print nothing).
- For an id of unknown kind (`action.target`, etc.) → `describe(any_id)`; for a bulk id→name map →
  `all_agent_names()`. Don't iterate over agents and assemble it yourself.

**Don't use it (four boundaries):**
1. **`agent/` cognition is strictly off-limits** — a name in an agent's prompt may come only from
   what perception has accumulated (`SpatialPerception.visible_agents` → the agent's known-agents map)
   and the relation's `to_name`. Resolving arbitrary ids through the directory makes the agent
   omniscient and breaks information asymmetry. Guard: `grep -rn "WorldDirectory" agent/` must print
   nothing.
2. **When you legitimately hold the `Agent` object** — read `personality.soul` directly; going through
   the directory is pointless indirection.
3. **Mutable-state questions** — item ownership/location/state come from `EnvironmentSystem`, agent
   location/emotion from `Agent`; never add a mutable-state query to the directory (a second source
   of truth).
4. **Offline snapshot paths** (web server / replayer) — keep using snapshot metadata; don't build a
   directory for offline use.

**How to use it:**
- Always inject it; never construct one ad hoc. Instances are created only in `initialize()` /
  `restore()` in `world/initializer`, attached to `World.directory`, and passed downstream through
  constructors. No module-level singleton (several worlds share a process).
- Annotate with the core contract `WorldDirectory`, not the engine implementation class.
- Tests: `LiveWorldDirectory.from_agents({}, EnvironmentSystem())` — an empty table gives the
  descriptive fallbacks (`agent_name` → 「某人」, `entity_name` → 「某物」, `location_name` → 「某地」),
  **never a bare id**.
- Extending it: a new method needs a consumer now (YAGNI) and exposes only immutable
  identity fields; a miss falls back to a **descriptive referent** (「某人/某物/某地」 for names;
  `describe` returns `None` only on a complete miss) plus `logger.debug("directory_miss")`, never
  raises, and **never falls back to a bare id** — these values enter narrative text, where an id is a
  layer leak (consistent with the narrative/code layer boundary).

---

## Frontend / backend split: the backend defines the API

The backend is independent of client technology and rendering (React / Phaser / CLI / third-party clients).
Clients communicate with it exclusively through the API (REST + WebSocket).

**The backend defines it**: an API field's meaning comes from what state the backend owns and what
the thing is in the world, **never** from what some frontend component wants to show (define a
contract by what the thing is, not by how a consumer uses it).

**What to expose:**
- **Expose** simulation facts: ids, states, values, points in time, text, relations, event order, and
  identity attributes the world already owns (a character's `color`, map asset URLs — data fixed at
  build time).
- **Don't expose** rendering decisions: CSS/class/component names, pixels and layout,
  highlighted/collapsed flags, display-order preferences, strings pre-assembled for display.

**When the frontend really lacks a field**, first ask whether it is **a world fact the backend
already owns**:
- Yes → add it in the backend's own vocabulary: generic name, generic meaning, usable by any client,
  with no frontend purpose in the field.
- No → the frontend computes it. Never add a field that only one page or component can use.

**Red flags** (fix on sight): fields named for rendering like `*_css` / `is_highlighted` in
API models; backend comments like "because that frontend drawer needs…"; a variant field of the same
data added to suit one component.

---

## Frontend layout (`frontend/src`)

Each directory has one job.

| Directory | Holds |
|---|---|
| `views/` | Screens mounted by a route. Page-level state (immersion, fullscreen) is the view's; a child component never uses `fixed inset-0`. |
| `components/` | UI pieces the views compose (`narrative/` is the feed). The two pre-run screens, `SplashScreen` and `BuildingScreen`, also live here. |
| `hooks/` | Stateful React hooks with no app-wide context. |
| `lib/` | Framework-free logic and shared tables (color, contract enums, entity and agent state), plus app-wide context and the hooks bound to it (`worldsContext`, `deployment`). |
| `phaser/` | The 2D map renderer **and the pure data it consumes** (`skins`, `mapSource`, `weatherPlan`, `pathfinding`). Phaser itself is imported only here and by its React host `components/MapStage`, both loaded lazily. |
| `api/`, `types.ts` | The only way to the backend (REST + WebSocket; map artifacts via `phaser/mapSource`) and the wire types, field for field. |
| `dev/` | Developer tools, reached only through `DevView`; UI text in English. |
| `lab/` | The render test bench (the map workbench). Production code never imports it. UI text in English; scene fixtures and their notes are content and stay as written. |

Dependencies: `phaser/` may use `lib/`; `lib/` and `hooks/` may use `api/`; none of the four imports `components/` or `views/`. Only `LabView` imports `lab/`.

---

## Agent Cognition Rules

The cognition loop is `perception → motivation → decision → action → feedback`. Each subsystem in `agent/` is distinct and cooperates through explicit state — do not collapse them into one convenience object.

- Do not mix scheduler, environment, or event logic into `agent/`.
- Do not call concrete storage or LLM implementations directly from cognition code.
- State updates must be explicit so runtime, snapshot, and replay can trust them.
- Preserve information asymmetry: each agent maintains its own world model, no implicit sharing.

---

## Sub-agents

Specialized subsystem agents live in `.claude/agents/` (Claude Code) and `.codex/agents/` (Codex); both hold the same roster:

| Agent | Use when |
|---|---|
| `architecture-governor` | Cross-module decisions, boundary questions, design-doc alignment |
| `code-review-governor` | Milestone review of a completed implementation slice |
| `cognition-systems-engineer` | Work in `agent/` — personality, memory, need, relation, decision |
| `core-foundation-builder` | Work in `core/` — interfaces, container, factory, logging |
| `infrastructure-provider-engineer` | Work in `providers/` — LLM, embedding, vector store, snapshot |
| `runtime-engine-builder` | Work in `engine/` — clock, scheduler, environment, messaging, events |
| `world-builder-engineer` | Work in `world/` — builder, initializer, models |
| `interaction-observer-engineer` | Work in `interaction/` — observer, replayer, CLI |
| `platform-deployment-engineer` | Deployment, Docker, environment setup, health checks |
| `qa-simulation-guardian` | Test strategy, fixtures, regression coverage |
| `agent-operations-governor` | When recurring coordination failures or ownership gaps appear |

---

## Documentation & Commits

- `CLAUDE.md`, `AGENTS.md`, and `.claude/agents/*.md` are written in English. Quoted runtime strings (prompt text, narrative samples) stay in their original language.
- These files reference code sparingly: name public contracts, interfaces and module paths, which are stable; name a private helper only when it is the one canonical example. A rule should still read correctly after a rename.
- Short imperative commits: `engine: add clock skeleton`, `agent: implement need competition`, `core: define LLM provider interface`.
- PRs must state: purpose, affected subsystems, and test evidence.
- Safety rule: if uncertain whether a change violates any rule above, stop and ask — do not guess or invent new conventions.
