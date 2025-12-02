import typer
from llmreval.runner import generate

app = typer.Typer()

@app.command()
def run(
    pred_in_path: str = typer.Option(..., help="Path to JSONL with prompts or perturbations"),
    variant: str = typer.Option("clean", help="Which variant to run: clean or perturbed"),
    out_path: str = typer.Option(..., help="Where to write model outputs"),
    model_id: str = typer.Option(..., help="Backend id (e.g. openai, transformers, together)"),
    model_name: str = typer.Option(..., help="Model name (e.g. gpt-4o, deepseek-ai/deepseek-llm-7b-chat)"),
):
    """
    CLI entry point for running evaluations.
    """
    model_spec = {"id": model_id, "model": model_name}
    generate(pred_in_path=pred_in_path, variant=variant, out_path=out_path, model_spec=model_spec)

if __name__ == "__main__":
    app()
