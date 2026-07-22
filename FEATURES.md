# Fairy features

Fairy is an LLM-based software-engineering assistant.
She currently focuses on code review and bug analysis, with support
for additional development workflows being added over time.

- 100% free, 100% open source
- self-hostable
- designed using real FFmpeg-scale review and security workloads
- supports selectable models/providers
- analyzes bugs in addition to patches
- can be adapted to project-specific review policies
- preserves project control over prompts, administration, credentials and model selection
- does not grant an external SaaS broad control over the repository


## Patch and pull-request review

- Reviews complete patches and pull requests
- Identifies correctness, security and regression risks
- Uses repository context rather than reviewing isolated diffs
- Posts structured findings directly to Forgejo
- Supports multiple review models
- Re-reviews updated patches
- ...

## Bug analysis

- Analyzes bug reports and associated source code
- Attempts to identify likely root causes
- Suggests relevant files, functions and debugging directions
- Correlates reports with existing changes or known issues
- ...

## Repository integration

- Native Forgejo integration
- GitHub integration — planned/in development
- GitLab integration — planned/in development
- Configurable per repository
- ...

## User interface

- can be run as a cronjob
- blessings based Text mode UI
- Can be run by a single developer on his development box or on your server fully automated
- multiple repositories in the same UI

## LLMs supported

- openAI codex (subscription)
- opanAI respones API (API credits)
- Anthropic API
- Z.ai
- Anything compatible with the APIs above
- optional OpenAI vector store support

## testing

- extensive selftests
- can simulate past to test/compare different prompts, configuration

## containers

- optional OpenAI container support (not recommanded due to limitation of these containers)
- arbitrary number of podman containers
- trivial to deploy, you need a ssh login (no root, no sudo) and podman, fairy sets up her own containers
- containers can be behind NAT, you just need a single ssh connection to reach them
- codex and the reviewer can be on seperate hw, seperate architecture, seperate accounts, seperate networks, they never have a direct connection between each other
- can run on the same machiene as fairy, or in the cloud, a VM, or on seperate physical hw, x86, arm, riscv5, anything
- sub second full text search (including non text documents like pdfs)

## data storage

- simple json files, each representing a ticket
- edit any time, the UI is just an app modifying these json files
- 100% under your control

## security

- codex runs in its own podman container
- each reviewer runs in its own podman container

