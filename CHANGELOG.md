# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

First public release.

### Added

- Theme-driven world building: a single theme produces locations, characters, personas and initial relationships.
- Character cognition: perception, emotions, needs, goals, memory (an objective event stream and a subjective experience stream), relationships, reflection and decision-making, orchestrated in stages.
- Runtime engine: clock, scheduling, environment, messaging, action execution and arbitration, world event injection, NPCs.
- Pluggable infrastructure: any OpenAI-compatible LLM endpoint, with per-scene model routing; swappable embeddings, vector store and snapshots.
- Web UI: world management, live observation, replay, 2D map, relationship graph, director console.
- Developer tools: LLM call tracing, narrative quality audits, prompt replay.
- One-command Docker Compose deployment with a guided first-run setup.
- Two finished example worlds, imported on first start, that can be browsed and replayed without API keys.
