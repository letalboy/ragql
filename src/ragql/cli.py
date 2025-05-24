import typer
from pathlib import Path
from .config import Settings
from .core import RagQL

app = typer.Typer(add_completion=False)

@app.command()
def build(path: Path = typer.Argument(..., exists=True)):
    rq = RagQL(path, Settings())
    rq.build()
    typer.echo("Index built ✅")

@app.command()
def chat(path: Path, question: str):
    rq = RagQL(path, Settings())
    answer = rq.query(question)
    typer.echo(answer)

if __name__ == "__main__":
    app()
