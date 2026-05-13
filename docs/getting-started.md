# Getting Started Guide

This guide will help you set up the template and start building your Python package with AI assistance.

## 📋 Prerequisites

- Python 3.9 or higher
- Git installed
- An AI coding assistant — **GitHub Copilot**, **Claude Code**, or another
- A code editor (VS Code recommended)

Check Python version: `python --version`

## 🚀 Setup (3 Main Steps)

### Step 1: Create Your Repository

**Option A: Use GitHub Template**
1. Click "Use this template" button on GitHub
2. Name your new repository
3. Clone it to your computer

**Option B: Manual Clone**
```bash
git clone https://github.com/yourusername/template-repo.git my-project
cd my-project
rm -rf .git
git init
```

### Step 2: Set Up Your Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e ".[dev]"
```

**What does `pip install -e ".[dev]"` do?**
- `-e` = editable mode (your code changes take effect immediately)
- `.` = install from current directory
- `[dev]` = include development tools (pytest, ruff, black, mypy)

**Alternative: Using Pixi**

If you prefer [pixi](https://pixi.sh) for reproducible environments:
```bash
pixi init --import pyproject.toml
pixi install
pixi shell
```

### Step 3: Customize the Package

**Rename the package:**
```bash
# Replace 'package_name' with your actual package name
mv src/package_name src/my_package
```

**Update `pyproject.toml`:**
```toml
[project]
name = "my-package"  # Your package name
description = "What your package does"
authors = [{name = "Your Name", email = "you@example.com"}]
```

**Verify it works:**
```bash
pytest                    # Run tests
ruff check src/          # Check code style
```

If tests pass, you're ready! ✅

## 📂 Understanding the Structure

```
your-project/
├── src/
│   └── package_name/        # Your Python package
│       ├── __init__.py      # Package initialization
│       └── cli.py           # Simple CLI example ("Hello AI")
├── tests/
│   └── test_cli.py          # Example CLI tests
├── .github/
│   ├── copilot-instructions.md   # ⭐ Shared AI workflow + standards
│   ├── agents/                   # Copilot sub-agent prompts
│   └── workflows/                # CI/CD automation
├── .claude/
│   └── agents/                   # Claude Code sub-agent definitions
├── CLAUDE.md                     # ⭐ Claude Code entry point (imports the shared instructions)
├── docs/
│   ├── getting-started.md        # This file
│   └── ground_truths.md          # Key findings & decisions log
└── pyproject.toml                # Package configuration
```

**Key files:**
- **`pyproject.toml`** - Defines your package metadata and dependencies
- **`.github/copilot-instructions.md`** - Shared workflow and standards (used by both Copilot and Claude Code)
- **`CLAUDE.md`** - Entry point Claude Code reads; imports the shared instructions above
- **`docs/ground_truths.md`** - Persistent log of key findings and decisions (see below)
- **`src/package_name/cli.py`** - Simple example showing Click CLI pattern
- **`tests/test_cli.py`** - Shows how to test CLI commands with CliRunner

## 🤖 Working with AI Assistance

This template is designed for AI-assisted development. The workflow is:

### 1. Assess → 2. Plan → 3. Implement → 4. Test → 5. Review

### Ground Truths: Persistent Context Across Sessions

AI assistants like Copilot and Claude Code don't remember previous conversations. To solve this, the template includes a **ground truths** file at [docs/ground_truths.md](ground_truths.md) where key findings and decisions are recorded during development.

Both Copilot and Claude Code are instructed to automatically update this file when they discover important facts — such as API quirks, design decisions, or configuration requirements. This means the next time you start a session, the assistant can read the file and pick up where you left off.

You can also add entries manually. Use the format:
```markdown
### YYYY-MM-DD: Brief title
Description of the finding and why it matters.
```

**Example: Adding a new feature**

Ask your AI assistant in chat:
```
I want to add a function to load CSV files and return a pandas DataFrame.
Please assess the current code and create an itemized plan.
```

The assistant will:
1. Check what exists in your codebase
2. Provide a clear plan with numbered steps
3. Implement each step with proper documentation and tests
4. Run tests to verify everything works

**Read [.github/copilot-instructions.md](../.github/copilot-instructions.md)** (or [CLAUDE.md](../CLAUDE.md), which imports it) to understand the full workflow and how to get the most from AI assistance.

## 🎯 Your First Feature

**Pro Tip:** Fill out [docs/project.md](project.md) to describe your project, then ask Copilot or Claude Code to create a plan based on it!

Here's how to add your own functionality:

### Option 1: Start simple - add a new module

```
"I need to add a module for [describe functionality].
Please assess the current structure and create a plan."
```

### Option 2: Extend the CLI

Modify `src/package_name/cli.py` to add your own commands:

```
"Help me add a new command to the CLI that [describe what you need].
Please assess the current CLI and create a plan."
```

### Option 3: Build your package from scratch

Replace the example and start fresh:

```
"I want to build [describe your package purpose].
Help me design the structure. Assess what's here and plan the changes."
```

## 🧪 Testing Your Code

Run tests frequently as you develop:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_core.py

# Run with coverage report
pytest --cov=src/package_name

# Run and show which lines aren't tested
pytest --cov=src/package_name --cov-report=html
# Open htmlcov/index.html in browser
```

**Writing your own tests:**

Use the Arrange-Act-Assert pattern (see `tests/test_cli.py` for an example):

```python
def test_my_function():
    # Arrange: Set up test data
    input_data = [1, 2, 3]
    
    # Act: Call your function
    result = my_function(input_data)
    
    # Assert: Check the result
    assert result == expected_value
```

**Testing CLI commands:**

Use Click's `CliRunner` (see `tests/test_cli.py`):

```python
from click.testing import CliRunner
from package_name.cli import main

def test_my_command():
    runner = CliRunner()
    result = runner.invoke(main, ["--option", "value"])
    assert result.exit_code == 0
    assert "expected output" in result.output
```

## 🐛 Common Issues

### "Module not found" error
```bash
pip install -e .         # Reinstall package
```

### Tests can't find your package
```bash
pip install -e ".[dev]"  # Install with dev dependencies
```

### After renaming package
1. Update imports in all test files
2. Update `pyproject.toml` configuration
3. Reinstall: `pip install -e .`

### Virtual environment issues
```bash
deactivate               # Deactivate environment
source venv/bin/activate # Reactivate it
```

## 💡 Tips for Success

1. **Start with one function** - Don't build everything at once
2. **Ask AI to assess first** - Get a plan before implementing
3. **Test immediately** - Run `pytest` after each feature
4. **Commit often** - Small commits are easier to manage
5. **Read error messages** - They usually tell you exactly what's wrong
6. **Use the AI workflow** - Assess → Plan → Implement → Test → Review

## 📚 Additional Resources

- **AI Instructions**: [.github/copilot-instructions.md](../.github/copilot-instructions.md) — shared workflow used by Copilot
- **Claude Entry Point**: [CLAUDE.md](../CLAUDE.md) — imports the shared instructions for Claude Code, plus subagent notes
- **Sub-agents**: [.github/agents/](../.github/agents/) (Copilot) and [.claude/agents/](../.claude/agents/) (Claude Code) — design, security, and test reviewers
- **Ground Truths**: [docs/ground_truths.md](ground_truths.md) - Key findings log that persists across AI sessions
- **CLI Example**: Check `src/package_name/cli.py` for a simple Click pattern
- **Test Example**: See `tests/test_cli.py` for CLI testing with CliRunner
- **Package Config**: Review `pyproject.toml` for dependencies and settings

## 🆘 Getting Help

**From AI:**
- "Assess the current state of [feature/file]"
- "Create an itemized plan for [task]"
- "Review my code for [quality/bugs/improvements]"
- "Help me debug this error: [paste error]"
- "Explain why [something] works this way"

**From the template:**
- CLI example: `src/package_name/cli.py`
- Test example: `tests/test_cli.py`

---

## ✨ You're Ready!

You now have:
- ✅ A working Python package structure
- ✅ Development environment set up
- ✅ AI assistance configured
- ✅ Example code to learn from

**Next step:** Ask Copilot to help you build your first feature!

```
"I want to [describe your goal]. Please assess the current code and create an itemized plan."
```
