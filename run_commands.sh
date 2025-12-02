pip install typer[all] pydantic pyyaml datasets rouge-score
python -m src.llmreval.cli build_data
python -m src.llmreval.cli perturb
python -m src.llmreval.cli run --model-id gpt4o_mini
python -m src.llmreval.cli score
