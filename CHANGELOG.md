# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-11-21

### Added

- Sub-places: world building gives each map place a few named spots (a courtyard in a mansion, a shop on a street), editable on the review page before confirmation. This cuts down on characters inventing places in their decisions and wandering back and forth.
- Momentary acts: in one step a character can do at most one act that occupies its body plus any number of momentary acts. The test is whether the act still needs the doer's own body once it is issued; if not, it is momentary. Sending a message (`SEND_MESSAGE`) and running an errand (`ERRAND`) are momentary, so sending a letter no longer takes up a character's whole step.
- New example world 玄武门前（李世民最精彩）, on the same theme as 宫门血色.
- The world clock shows the length of a step, plus other display improvements.

### Changed

- Breaking loops, so characters don't circle in one place while nothing in the world changes:
  - A character that keeps repeating the same type of action (especially talking) is held back from it for a while, pushing it to try another.
  - The narrative editor guides characters toward places few people visit that have things lying on the ground.
  - Repeated short-term goal failures bring the long-term goal revision forward.
- Action durations are estimated in hours and converted to steps in code, and each act is sized to the length of a step. Estimating steps directly makes the LLM convert first and often get it wrong; hours match common sense and reduce duration hallucinations.
- Scene rosters and the situation header mark when each person arrived, so characters can tell who just came in from who was already there, and no longer assume a newcomer heard the earlier conversation.
- The offline review (tuning) judge runs on a DashScope model.
- API: world routes and the WebSocket carry a new `epoch` field, renewed when a reset or the actor's exit discards a world's history; a client whose epoch differs should drop the history it holds.
- Other improvements and bug fixes.

### Compatibility

- **Saved worlds**: worlds saved by 0.2.0 can be opened, replayed and continued as they are.
- **Example worlds**: an existing deployment doesn't import the new example 玄武门前, because examples are imported only on a data directory's first start. To get it, stop the service and delete `data/` (`./deploy/asamana.sh stop && rm -rf data/`), then start again. This deletes the worlds you built. `./deploy/asamana.sh reset` also clears `data/`, but it deletes `deploy/.env` with your API keys too.
- **Config**: the `deepseek` and `zhipu` presets run the offline judge on DashScope and need `DASHSCOPE_API_KEY`. A fresh setup with `asamana.sh` asks for it. A deployment upgraded from 0.2.0 with either preset must add `DASHSCOPE_API_KEY` to `deploy/.env` (the same DashScope key as `EMBEDDING_API_KEY`), or the backend fails at startup with `Model keys must be all set or all unset`.

## [0.2.0] - 2026-11-04

### Added

- Actor mode: become an explorer and play any one character in the AI world, exploring it alongside the AI characters.
- World setup editing: before you confirm a built world, edit its characters, NPCs, entity seeds, historical events, relations and more.
- Anonymous messages: characters can send messages and announcements without revealing who sent them.
- Trace records beats that land without an LLM call (actor decisions, injections), and the audit covers actor steps.

### Changed

- Relation labels flip direction: the role in a label (「关系类型:角色」) now names who the other party is to the holder, not who the holder is to them (「父子:父亲」 means the other is the holder's father).
- Snapshots and event records are restructured: each step commits as one complete snapshot, with agent state and relations inside it (the separate agent store is gone). Each step's actions, messages, broadcasts and injections are recorded once, in order and with who perceived each; live observation, replay and the actor's view all read these same records.
- Config renames, rejected at startup if left unchanged: `event_system_enabled` → `narrative_editor_enabled`, `max_events_per_window` → `max_injections_per_window`, `event_quota_window` → `injection_quota_window`, `event_check_interval` → `injection_check_interval`. Remove the `agent_store` section.
- Example worlds replaced with 宫门血色 and 第七把椅子.
- Other improvements and bug fixes.

### Compatibility

**This release is not backward compatible, in code or in data.**

- **Saved worlds**: worlds saved by 0.1.0 can't be opened, run or replayed; the snapshot format changed. Even where old data could be read, its relation labels mean the opposite of what they mean now.
- **Config**: a 0.1.0 config fails at startup until the keys above are renamed and `agent_store` is removed.
- **API**: step payloads (`StepView`) and graphs reference characters, NPCs and places by id only; clients resolve names from `/api/worlds/{id}/agents`, `/npcs` and `/places`. A client written against 0.1.0 has to be updated.
- **Upgrading**: stop the service, delete `data/` (`./deploy/asamana.sh stop && rm -rf data/`), update your config, then start again. Without clearing `data/`, old worlds stay in the list but won't open, and the new example worlds are never imported, because the first-start import has already run for that data directory. `./deploy/asamana.sh reset` also clears `data/`, but it deletes `deploy/.env` with your API keys too.

## [0.1.0] - 2026-10-08

First public release.

### Added

- Theme-driven world building: a single theme produces locations, characters, personas and initial relationships.
- Character cognition: perception, emotions, needs, goals, memory (an objective event stream and a subjective experience stream), relationships, reflection and decision-making, orchestrated in stages.
- Runtime engine: clock, scheduling, environment, messaging, action execution and arbitration, world event injection, NPCs.
- Pluggable infrastructure: any OpenAI-compatible LLM endpoint, with per-scene model routing; swappable embeddings, vector store and snapshots.
- Web UI: world management, live observation, replay, 2D map, relationship graph, director console.
- Developer tools: LLM call tracing, narrative quality audits, prompt replay.
- One-command Docker Compose deployment with a guided first-run setup.
- Two completed example worlds, imported on first start and available for browsing and replay without API keys.

[Unreleased]: https://github.com/dreamscroll-lab/asamana/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/dreamscroll-lab/asamana/releases/tag/v0.2.1
[0.2.0]: https://github.com/dreamscroll-lab/asamana/releases/tag/v0.2.0
[0.1.0]: https://github.com/dreamscroll-lab/asamana/releases/tag/v0.1.0
