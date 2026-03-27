"""Stellaris AI Advisor CLI entry point. Implemented in Plan 7."""

from __future__ import annotations

import typer

app = typer.Typer(help="Stellaris AI Advisor — quantitative strategy recommendations.")


@app.command()
def version() -> None:
    """Show tool version and supported Stellaris versions."""
    from stellaris_advisor import __version__

    typer.echo(f"stellaris-advisor {__version__}")


if __name__ == "__main__":
    app()
