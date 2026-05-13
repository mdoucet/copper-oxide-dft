"""
Simple example test for the CLI.

Shows how to test Click commands using the CliRunner.
"""

from click.testing import CliRunner

from package_name.cli import main


def test_cli_default_greeting():
    """Test CLI with default name (AI)."""
    runner = CliRunner()
    result = runner.invoke(main)

    assert result.exit_code == 0
    assert "Hello, AI!" in result.output


def test_cli_custom_name():
    """Test CLI with custom name."""
    runner = CliRunner()
    result = runner.invoke(main, ["--name", "World"])

    assert result.exit_code == 0
    assert "Hello, World!" in result.output
