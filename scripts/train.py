import typer
from credit_risk.config import load_cfg
from credit_risk.pipeline import Trainer

app = typer.Typer()

@app.command()
def run(params_path: str = "params.yaml"):
    cfg = load_cfg(params_path)
    Trainer(cfg).run()

if __name__ == "__main__":
    app()
