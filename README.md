# AI-Assisted Python Package Template

A minimal, streamlined template for scientists who want to build Python packages using AI assistance—no software engineering experience required.

## 🎯 What is this?

This template gives you a clean starting structure for building Python packages with AI tools like **GitHub Copilot** and **Claude Code**. It includes:
- Basic Python package structure 
- Example code showing best practices
- Test setup with pytest
- **Comprehensive AI instructions** to guide development (works with both Copilot and Claude Code)

It's designed to be **simple enough to understand** but **complete enough to build on**.

## 🚀 Quick Start (3 steps)

### 1. Create your repository
Click "Use this template" on GitHub, or clone and rename this repo.

### 2. Set up your environment
```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

### 3. Customize and start building
```bash
# Rename the package (example)
mv src/package_name src/my_package

# Update pyproject.toml with your details
# Then start coding!
```

**Read [docs/getting-started.md](docs/getting-started.md) for detailed setup instructions.**

## 🤖 AI-Assisted Development

This template is optimized for working with **GitHub Copilot** and **Claude Code** (and other AI assistants). The key workflow is:

1. **Assess** - Ask the assistant to examine your current code
2. **Plan** - Get an itemized plan before implementation
3. **Implement** - Build features step by step
4. **Test** - Verify each change works
5. **Review** - Get code review from a specialized sub-agent after major changes
6. **Record** - Key findings are saved to [docs/ground_truths.md](docs/ground_truths.md) so context persists across sessions

**Configuration files:**
- **[.github/copilot-instructions.md](.github/copilot-instructions.md)** — the shared workflow and code-quality standards (used by Copilot; also imported by `CLAUDE.md` so Claude Code follows the same rules).
- **[CLAUDE.md](CLAUDE.md)** — Claude Code entry point. Imports the shared instructions and adds Claude-specific notes.
- **[.github/agents/](.github/agents/)** — sub-agent prompts for Copilot (design, security, test reviews).
- **[.claude/agents/](.claude/agents/)** — the same sub-agents in Claude Code's native format.

## 📦 What's Included

```
├── .github/
│   ├── copilot-instructions.md    # Shared AI workflow + standards
│   ├── agents/                    # Copilot sub-agent prompts
│   └── workflows/                 # CI/CD (tests, linting)
├── .claude/
│   └── agents/                    # Claude Code sub-agent definitions
├── CLAUDE.md                      # Claude Code entry point (imports the shared instructions)
├── src/package_name/              # Your Python package
│   ├── __init__.py                # Package initialization
│   └── cli.py                     # Simple "Hello AI" CLI example
├── tests/
│   └── test_cli.py                # Example CLI tests
├── docs/
│   ├── getting-started.md         # Setup guide
│   └── ground_truths.md           # Key findings & decisions log
├── pyproject.toml                 # Package configuration
└── README.md                      # This file
```

## 🧪 Running Tests

```bash
pytest                              # Run all tests
pytest --cov=src/package_name      # With coverage report
pytest -v                           # Verbose output
```

## 💡 Tips for Getting Started

1. **Start simple** - Replace the example code with one function you need
2. **Describe your project** - Fill out [docs/project.md](docs/project.md) to help the AI understand what you're building
3. **Let AI help** - Ask Copilot or Claude to assess and plan before implementing
4. **Test as you go** - Run pytest after each feature
5. **Keep ground truths** - Key findings are logged in [docs/ground_truths.md](docs/ground_truths.md) so AI remembers context between sessions
6. **Commit often** - Small commits are easier to track
7. **Read the AI instructions** - [.github/copilot-instructions.md](.github/copilot-instructions.md) (also imported by [CLAUDE.md](CLAUDE.md)) explains the recommended workflow

## 🆘 Getting Help

- **Ask your AI assistant directly** (Copilot or Claude Code): "Assess my code and suggest next steps"
- **Read the getting started guide**: [docs/getting-started.md](docs/getting-started.md)
- **Check the example code**: See [src/package_name/cli.py](src/package_name/cli.py) for a simple CLI pattern

## 📄 License

BSD 3-Clause License - see [LICENSE](LICENSE) file for details.

---

**Ready?** Open [docs/getting-started.md](docs/getting-started.md) to begin, or ask Copilot / Claude Code: "Help me customize this template for my project."
