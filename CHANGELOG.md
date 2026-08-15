# Changelog

## [0.17.3a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.17.3a1) (2026-08-15)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.17.2a1...0.17.3a1)

**Merged pull requests:**

- fix: no-answer persona returns 422 with a clear message, not a 500 traceback [\#87](https://github.com/OpenVoiceOS/ovos-persona-server/pull/87) ([JarbasAl](https://github.com/JarbasAl))

## [0.17.2a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.17.2a1) (2026-08-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.17.1a1...0.17.2a1)

**Merged pull requests:**

- fix: migrate A2A adapter to a2a-sdk\>=1.1.2 so it actually loads [\#88](https://github.com/OpenVoiceOS/ovos-persona-server/pull/88) ([JarbasAl](https://github.com/JarbasAl))

## [0.17.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.17.1a1) (2026-08-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.17.0a1...0.17.1a1)

**Merged pull requests:**

- fix: key transparent memory on the caller identity these surfaces already receive [\#86](https://github.com/OpenVoiceOS/ovos-persona-server/pull/86) ([JarbasAl](https://github.com/JarbasAl))

## [0.17.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.17.0a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.16.0a1...0.17.0a1)

**Merged pull requests:**

- feat: make MCP mounting an explicit --mcp opt-in flag [\#83](https://github.com/OpenVoiceOS/ovos-persona-server/pull/83) ([JarbasAl](https://github.com/JarbasAl))

## [0.16.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.16.0a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.15.1a1...0.16.0a1)

**Merged pull requests:**

- feat\(openai\): configurable client system-prompt strategy \(ignore/replace/append\) [\#74](https://github.com/OpenVoiceOS/ovos-persona-server/pull/74) ([JarbasAl](https://github.com/JarbasAl))

## [0.15.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.15.1a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.15.0a1...0.15.1a1)

**Merged pull requests:**

- fix: hand ToolBox plugins their own config section [\#80](https://github.com/OpenVoiceOS/ovos-persona-server/pull/80) ([JarbasAl](https://github.com/JarbasAl))

## [0.15.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.15.0a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.14.0a2...0.15.0a1)

**Merged pull requests:**

- feat\(openai\): function calling through the OVOS tool model \(client + persona ToolBox tools\) [\#75](https://github.com/OpenVoiceOS/ovos-persona-server/pull/75) ([JarbasAl](https://github.com/JarbasAl))

## [0.14.0a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.14.0a2) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.14.0a1...0.14.0a2)

**Merged pull requests:**

- docs: cross-link the technical manual [\#66](https://github.com/OpenVoiceOS/ovos-persona-server/pull/66) ([JarbasAl](https://github.com/JarbasAl))

## [0.14.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.14.0a1) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.4a2...0.14.0a1)

**Merged pull requests:**

- feat: serve multiple personas from one process, selected by `model` [\#72](https://github.com/OpenVoiceOS/ovos-persona-server/pull/72) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.4a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.4a2) (2026-08-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.4a1...0.13.4a2)

**Merged pull requests:**

- refactor: migrate MCP server to the fastmcp package [\#71](https://github.com/OpenVoiceOS/ovos-persona-server/pull/71) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.4a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.4a1) (2026-08-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.3a2...0.13.4a1)

**Merged pull requests:**

- fix: loader never passes toolbox\_id; plugins own their id [\#60](https://github.com/OpenVoiceOS/ovos-persona-server/pull/60) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.3a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.3a2) (2026-08-03)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.3a1...0.13.3a2)

**Merged pull requests:**

- refactor: unify OpenAI-dict → AgentMessage conversion, fix stale passthrough tests [\#68](https://github.com/OpenVoiceOS/ovos-persona-server/pull/68) ([JarbasAl](https://github.com/JarbasAl))
- fix: convert message dicts to AgentMessage in run\_chat/run\_stream [\#67](https://github.com/OpenVoiceOS/ovos-persona-server/pull/67) ([andlo](https://github.com/andlo))

## [0.13.3a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.3a1) (2026-07-26)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.2a1...0.13.3a1)

**Merged pull requests:**

- fix: env-var persona fallback when no --persona \(crash on bare launch\) [\#63](https://github.com/OpenVoiceOS/ovos-persona-server/pull/63) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.2a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.2a1) (2026-07-23)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.1a1...0.13.2a1)

**Merged pull requests:**

- fix\(deps\): pin mcp\<2.0.0 [\#61](https://github.com/OpenVoiceOS/ovos-persona-server/pull/61) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.1a1) (2026-06-20)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.13.0a1...0.13.1a1)

**Merged pull requests:**

- fix\(deps\): pin released ovos-chromadb-embeddings-plugin\>=0.3.0a4; drop last git-ref [\#58](https://github.com/OpenVoiceOS/ovos-persona-server/pull/58) ([JarbasAl](https://github.com/JarbasAl))

## [0.13.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.13.0a1) (2026-06-20)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.12.0a5...0.13.0a1)

**Merged pull requests:**

- feat: embeddings [\#11](https://github.com/OpenVoiceOS/ovos-persona-server/pull/11) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a5](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.12.0a5) (2026-06-14)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.12.0a4...0.12.0a5)

**Merged pull requests:**

- docs: add NGI0 Commons Fund attribution [\#55](https://github.com/OpenVoiceOS/ovos-persona-server/pull/55) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a4](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.12.0a4) (2026-06-13)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.12.0a3...0.12.0a4)

**Merged pull requests:**

- chore: consolidate test/ into tests/ \(single test directory\) [\#53](https://github.com/OpenVoiceOS/ovos-persona-server/pull/53) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a3](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.12.0a3) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.12.0a2...0.12.0a3)

**Merged pull requests:**

- test: real-SDK e2e for merged OpenAI + Ollama endpoints \(+ /generate compat fix\) [\#51](https://github.com/OpenVoiceOS/ovos-persona-server/pull/51) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.12.0a2) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.12.0a1...0.12.0a2)

**Merged pull requests:**

- docs: user-facing documentation for all API surfaces [\#36](https://github.com/OpenVoiceOS/ovos-persona-server/pull/36) ([JarbasAl](https://github.com/JarbasAl))

## [0.12.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.12.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.11.0a1...0.12.0a1)

**Merged pull requests:**

- feat: expose OPM tool plugins via MCP + UTCP [\#37](https://github.com/OpenVoiceOS/ovos-persona-server/pull/37) ([JarbasAl](https://github.com/JarbasAl))

## [0.11.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.11.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.10.0a1...0.11.0a1)

**Merged pull requests:**

- feat\(compat\): Cohere-compatible endpoints \(/cohere/v1\) [\#32](https://github.com/OpenVoiceOS/ovos-persona-server/pull/32) ([JarbasAl](https://github.com/JarbasAl))

## [0.10.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.10.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.9.0a1...0.10.0a1)

**Merged pull requests:**

- feat\(compat\): HuggingFace TGI-compatible endpoints \(/tgi\) [\#33](https://github.com/OpenVoiceOS/ovos-persona-server/pull/33) ([JarbasAl](https://github.com/JarbasAl))

## [0.9.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.9.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.8.0a1...0.9.0a1)

**Merged pull requests:**

- feat\(compat\): AWS Bedrock-compatible endpoints \(/bedrock/model\) [\#34](https://github.com/OpenVoiceOS/ovos-persona-server/pull/34) ([JarbasAl](https://github.com/JarbasAl))

## [0.8.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.8.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.7.0a1...0.8.0a1)

**Merged pull requests:**

- feat\(compat\): Google Gemini-compatible endpoints \(/gemini/v1beta\) [\#31](https://github.com/OpenVoiceOS/ovos-persona-server/pull/31) ([JarbasAl](https://github.com/JarbasAl))

## [0.7.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.7.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.6.0a1...0.7.0a1)

**Merged pull requests:**

- feat\(compat\): Anthropic Claude-compatible endpoints \(/anthropic/v1\) [\#30](https://github.com/OpenVoiceOS/ovos-persona-server/pull/30) ([JarbasAl](https://github.com/JarbasAl))

## [0.6.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.6.0a1) (2026-06-12)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.5.2a2...0.6.0a1)

**Merged pull requests:**

- feat: A2A server endpoint \(/a2a\) with agent card + executor [\#35](https://github.com/OpenVoiceOS/ovos-persona-server/pull/35) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.2a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.5.2a2) (2026-06-11)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.5.2a1...0.5.2a2)

**Merged pull requests:**

- refactor: vendor-prefixed OpenAI/Ollama routers + deprecated legacy paths [\#29](https://github.com/OpenVoiceOS/ovos-persona-server/pull/29) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.2a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.5.2a1) (2026-06-11)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.5.1a2...0.5.2a1)

**Closed issues:**

- PyPI release broken: wheel omits schemas subpackage + undeclared deps + wrong python floor [\#39](https://github.com/OpenVoiceOS/ovos-persona-server/issues/39)

**Merged pull requests:**

- fix: declare uvicorn+ovos-workshop deps; restore py3.9 f-string compat in chat.py [\#40](https://github.com/OpenVoiceOS/ovos-persona-server/pull/40) ([JarbasAl](https://github.com/JarbasAl))
- chore: migrate setup.py→pyproject.toml, consolidate CI workflows [\#28](https://github.com/OpenVoiceOS/ovos-persona-server/pull/28) ([JarbasAl](https://github.com/JarbasAl))

## [0.5.1a2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.5.1a2) (2025-12-19)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.5.1a1...0.5.1a2)

**Merged pull requests:**

- chore\(deps\): update dependency python to 3.14 [\#21](https://github.com/OpenVoiceOS/ovos-persona-server/pull/21) ([renovate[bot]](https://github.com/apps/renovate))

## [0.5.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.5.1a1) (2025-12-18)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.5.0...0.5.1a1)

**Merged pull requests:**

- chore: Configure Renovate [\#19](https://github.com/OpenVoiceOS/ovos-persona-server/pull/19) ([renovate[bot]](https://github.com/apps/renovate))

## [0.5.0](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.5.0) (2025-11-08)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.4.1a1...0.5.0)

**Merged pull requests:**

- Release 0.4.1a1 [\#16](https://github.com/OpenVoiceOS/ovos-persona-server/pull/16) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.4.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.4.1a1) (2025-10-17)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.4.0...0.4.1a1)

**Merged pull requests:**

- refactor: migrate to fastapi [\#14](https://github.com/OpenVoiceOS/ovos-persona-server/pull/14) ([JarbasAl](https://github.com/JarbasAl))

## [0.4.0](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.4.0) (2025-04-10)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.4.0a1...0.4.0)

**Merged pull requests:**

- Release 0.4.0a1 [\#10](https://github.com/OpenVoiceOS/ovos-persona-server/pull/10) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.4.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.4.0a1) (2025-04-10)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.2...0.4.0a1)

**Merged pull requests:**

- feat:ollama\_api\_support [\#9](https://github.com/OpenVoiceOS/ovos-persona-server/pull/9) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.2) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.2a1...0.3.2)

**Merged pull requests:**

- Release 0.3.2a1 [\#7](https://github.com/OpenVoiceOS/ovos-persona-server/pull/7) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.3.2a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.2a1) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.1...0.3.2a1)

**Merged pull requests:**

- Fix/streaming [\#6](https://github.com/OpenVoiceOS/ovos-persona-server/pull/6) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.1) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.1a1...0.3.1)

**Merged pull requests:**

- Release 0.3.1a1 [\#5](https://github.com/OpenVoiceOS/ovos-persona-server/pull/5) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.3.1a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.1a1) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.0...0.3.1a1)

**Merged pull requests:**

- fix: persona name , add models [\#4](https://github.com/OpenVoiceOS/ovos-persona-server/pull/4) ([JarbasAl](https://github.com/JarbasAl))

## [0.3.0](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.0) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.3.0a1...0.3.0)

**Merged pull requests:**

- Release 0.3.0a1 [\#3](https://github.com/OpenVoiceOS/ovos-persona-server/pull/3) ([github-actions[bot]](https://github.com/apps/github-actions))

## [0.3.0a1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.3.0a1) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.2.1...0.3.0a1)

**Merged pull requests:**

- feat: add status endpoint for healthcheck and info [\#2](https://github.com/OpenVoiceOS/ovos-persona-server/pull/2) ([JarbasAl](https://github.com/JarbasAl))

## [0.2.1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.2.1) (2025-03-05)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.2.0...0.2.1)

## [0.2.0](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.2.0) (2025-01-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.1.0...0.2.0)

## [0.1.0](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.1.0) (2025-01-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.0.2...0.1.0)

## [0.0.2](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.0.2) (2025-01-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/0.0.1...0.0.2)

## [0.0.1](https://github.com/OpenVoiceOS/ovos-persona-server/tree/0.0.1) (2025-01-28)

[Full Changelog](https://github.com/OpenVoiceOS/ovos-persona-server/compare/de5796ee4125e077cedb1e1dbf4f43f91fe8e6c0...0.0.1)

**Merged pull requests:**

- added setup.py [\#1](https://github.com/OpenVoiceOS/ovos-persona-server/pull/1) ([builderjer](https://github.com/builderjer))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
