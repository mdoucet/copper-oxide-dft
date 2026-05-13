"""
Command-line interface for package_name.

A simple example CLI to demonstrate how to create command-line tools.
Replace this with your own commands as you build your package.

Example usage:
    $ python -m package_name.cli
    $ python -m package_name.cli --name "World"
"""

import click


@click.command()
@click.option("--name", default="AI", help="Name to greet (default: AI)")
@click.version_option(version="0.1.0")
def main(name: str):
    """
    A simple greeting CLI example.

    This demonstrates the basic structure of a Click CLI.
    Replace this with your actual functionality!

    Example:
        python -m package_name.cli
        python -m package_name.cli --name "Copilot"
    """
    click.echo(f"Hello, {name}! 👋")
    click.echo("\nThis is a template CLI. Replace it with your own commands!")
    click.echo("Tip: Ask Copilot to help you add functionality here.")


if __name__ == "__main__":
    main()
