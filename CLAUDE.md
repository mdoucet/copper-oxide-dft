# Claude Code Instructions

This project is configured for AI-assisted development with both **Claude Code** and **GitHub Copilot**. The core workflow (assess → plan → implement → test → review), code quality standards, technology preferences, and review process are shared between the two assistants. They are defined in the imported document below.

@.github/copilot-instructions.md

## Claude-Specific Notes

The instructions above were originally written for GitHub Copilot but apply equally to Claude Code. Where they reference "Copilot," read it as "the AI assistant" — the workflow is identical.

### Subagents

Claude Code subagent definitions live in [`.claude/agents/`](.claude/agents/) and mirror the agents in [`.github/agents/`](.github/agents/) used by Copilot:

- **`design-reviewer`** — reviews architecture, code duplication, hard-coded values, file size, and organic-growth smells
- **`security-reviewer`** — audits for OWASP Top 10, secrets leaks, code injection, path traversal, and unsafe patterns
- **`test-reviewer`** — evaluates test quality, flagging mock-heavy unit tests and missing integration coverage

Invoke them with the Task tool at the "Review" step of the standard workflow, or whenever the user asks for a focused review of design, security, or tests.

### Ground Truths

[`docs/ground_truths.md`](docs/ground_truths.md) is the canonical place for non-derivable project knowledge that should persist across sessions (API quirks, configuration requirements, design decisions, performance constraints). Append to it when you discover something important — don't keep that knowledge in conversation only.

### Project Context

If [`docs/project.md`](docs/project.md) has been filled out, treat it as the source of truth for project goals and scope when planning implementations.

### Package Name

The package directory is `src/package_name/` as a placeholder. Once the user renames it, update references in `pyproject.toml`, tests, and docs as needed.
