import typer

app = typer.Typer(name="valar", help="VALAR: Value Annotation with LLMs on Russian datasets.")


@app.command()
def annotate(
    experiment_config: str = typer.Option(..., "--experiment-config", help="Path to experiment YAML"),
    run_config: str = typer.Option(..., "--run-config", help="Path to run YAML"),
    output_dir: str = typer.Option(..., "--output-dir", help="Results output directory"),
    limit_items: int = typer.Option(0, "--limit-items", help="Cap items (0 = all)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Build plan only, no API calls"),
) -> None:
    typer.echo(f"[valar] experiment={experiment_config} run={run_config} out={output_dir}")
    if dry_run:
        typer.echo("[valar] DRY RUN — no API calls made.")


@app.command()
def build_banks(
    experiment_config: str = typer.Option(..., "--experiment-config"),
    output_dir: str = typer.Option("data/item_banks/valar", "--output-dir"),
) -> None:
    typer.echo(f"[valar] Building item banks → {output_dir}")


if __name__ == "__main__":
    app()
