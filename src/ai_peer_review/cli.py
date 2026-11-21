import click
import json
import re
from pathlib import Path

from .review import process_paper, generate_meta_review, get_meta_reviewer_client
from .utils.config import (
    get_prompt,
    load_config,
    get_local_models,
    set_local_server,
    set_local_model,
    remove_local_model,
    set_meta_reviewer,
)
from .version import __version__


@click.group()
@click.version_option(version=__version__)
@click.option(
    "--config-file",
    type=click.Path(exists=False),
    help="Path to a custom configuration file",
    envvar="AI_PEER_REVIEW_CONFIG_FILE",
)
@click.pass_context
def cli(ctx, config_file):
    """AI-based peer review of academic papers using locally-served LLMs.

    No third-party LLM API is contacted: every reviewer and the meta-reviewer
    are served by an OpenAI-compatible local inference server you configure
    with the ``config`` subcommands.
    """
    # Store config path in context for subcommands to access
    ctx.ensure_object(dict)
    ctx.obj["config_file"] = config_file


@cli.group()
@click.pass_context
def config(ctx):
    """Configure the local inference server and reviewer models."""


@config.command("show")
@click.pass_context
def config_show(ctx):
    """Print the current local-model configuration."""
    config_file = ctx.obj.get("config_file")
    cfg = load_config(config_file)
    click.echo(json.dumps(
        {
            "local_server": cfg.get("local_server", {}),
            "local_models": cfg.get("local_models", {}),
            "meta_reviewer": cfg.get("meta_reviewer", {}),
        },
        indent=2,
    ))


@config.command("server")
@click.option("--base-url", required=True,
              help="OpenAI-compatible base URL of the local server "
                   "(e.g. http://localhost:11434/v1 for Ollama).")
@click.option("--api-key", default=None,
              help="Placeholder API key. Local servers ignore it; the OpenAI "
                   "SDK just requires a non-empty string.")
@click.pass_context
def config_server(ctx, base_url, api_key):
    """Set the default local inference server (inherited by all reviewers)."""
    config_file = ctx.obj.get("config_file")
    set_local_server(base_url=base_url, api_key=api_key, config_file=config_file)
    click.echo(f"Default local server set to {base_url}")


@config.command("add")
@click.argument("name")
@click.option("--model", required=True,
              help="Model tag as recognized by the local server "
                   "(e.g. qwen3:32b, deepseek-r1:70b).")
@click.option("--base-url", default=None,
              help="Override the default server for this reviewer only.")
@click.option("--api-key", default=None, help="Override the default api key.")
@click.option("--temperature", type=float, default=None,
              help="Sampling temperature (default 0.1).")
@click.option("--max-tokens", type=int, default=None,
              help="Max tokens to generate (default 8000).")
@click.pass_context
def config_add(ctx, name, model, base_url, api_key, temperature, max_tokens):
    """Add or update a local reviewer model named NAME."""
    config_file = ctx.obj.get("config_file")
    set_local_model(
        name, model,
        base_url=base_url, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens,
        config_file=config_file,
    )
    click.echo(f"Reviewer '{name}' -> model '{model}' configured.")


@config.command("remove")
@click.argument("name")
@click.pass_context
def config_remove(ctx, name):
    """Remove a local reviewer model named NAME."""
    config_file = ctx.obj.get("config_file")
    if remove_local_model(name, config_file=config_file):
        click.echo(f"Reviewer '{name}' removed.")
    else:
        click.echo(f"Reviewer '{name}' not found.")


@config.command("meta")
@click.option("--model", required=True,
              help="Model tag for the meta-reviewer (use your strongest / "
                   "longest-context model).")
@click.option("--base-url", default=None)
@click.option("--api-key", default=None)
@click.option("--temperature", type=float, default=None)
@click.option("--max-tokens", type=int, default=None)
@click.pass_context
def config_meta(ctx, model, base_url, api_key, temperature, max_tokens):
    """Set the meta-reviewer model."""
    config_file = ctx.obj.get("config_file")
    set_meta_reviewer(
        model,
        base_url=base_url, api_key=api_key,
        temperature=temperature, max_tokens=max_tokens,
        config_file=config_file,
    )
    click.echo(f"Meta-reviewer set to model '{model}'.")


@cli.command("list-models")
@click.pass_context
def list_models(ctx):
    """List all configured local reviewer models."""
    config_file = ctx.obj.get("config_file")
    available_models = get_available_models(config_file)

    click.echo("Configured local reviewer models:")
    for model_name in available_models:
        click.echo(f"- {model_name}")

    click.echo("\nUse these names with the --models option when running "
               "'ai-peer-review review'. Add more with 'ai-peer-review config add'.")


def get_available_models(config_file=None):
    """Return the list of configured reviewer names (keys of local_models)."""
    return list(get_local_models(config_file).keys())


@cli.command("review")
@click.argument("pdf_path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="./papers",
    help="Base directory to store reviews and meta-review",
)
@click.option(
    "--meta-review/--no-meta-review",
    default=True,
    help="Generate meta-review after individual reviews",
)
@click.option(
    "--models",
    type=str,
    help="Comma-separated list of reviewer names to use. Default is all "
         "configured reviewers.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Overwrite existing reviews",
)
@click.pass_context
def review(ctx, pdf_path, output_dir, meta_review, models, overwrite):
    """Process a paper and generate peer reviews using local LLMs."""
    # Use the input file's stem (filename without extension) as the output directory
    pdf_path = Path(pdf_path)
    file_stem = pdf_path.stem
    output_dir = Path(output_dir) / file_stem
    output_dir.mkdir(exist_ok=True, parents=True)

    click.echo(f"Output will be saved to: {output_dir}")

    config_file = ctx.obj.get("config_file")
    available_models = get_available_models(config_file)

    selected_models = available_models
    if models:
        model_list = [m.strip() for m in models.split(",")]
        selected_models = [m for m in model_list if m in available_models]
        if not selected_models:
            click.echo(f"No valid models specified. Available models: {', '.join(available_models)}")
            return

    click.echo(f"Processing paper: {pdf_path}")
    click.echo(f"Selected reviewers: {', '.join(selected_models)}")

    # Check for existing reviews and only process missing ones
    reviews = {}
    review_data = {}
    models_to_process = []

    for model_name in selected_models:
        review_file = output_dir / f"review_{model_name}.md"

        if review_file.exists() and not overwrite:
            # Read existing review
            with open(review_file, "r") as f:
                review_text = f.read()
            reviews[model_name] = review_text
            review_data[model_name] = review_text
            click.echo(f"Using existing review for {model_name} from {review_file}")
        else:
            models_to_process.append(model_name)

    # Process papers with LLMs for missing or to-be-overwritten reviews
    if models_to_process:
        click.echo(f"Processing reviewers: {', '.join(models_to_process)}")
        new_reviews = process_paper(pdf_path, models_to_process, config_file)

        # Save new reviews
        for model, review_text in new_reviews.items():
            review_file = output_dir / f"review_{model}.md"
            with open(review_file, "w") as f:
                f.write(review_text)
            reviews[model] = review_text
            review_data[model] = review_text
            click.echo(f"Review from {model} saved to {review_file}")

    if not reviews:
        click.echo("No reviews to process. Use --overwrite to regenerate existing reviews.")

    # Generate meta-review if requested
    if meta_review:
        click.echo("Generating meta-review...")
        meta_review_text, nato_to_model = generate_meta_review(reviews, config_file)

        meta_review_file = output_dir / "meta_review.md"
        with open(meta_review_file, "w") as f:
            f.write(meta_review_text)
        click.echo(f"Meta-review saved to {meta_review_file}")

        # Use the same configured local meta-reviewer to extract a clean
        # concerns JSON. (Replaces the previous hardcoded Google Gemini call.)
        meta_reviewer = get_meta_reviewer_client(config_file)

        # Create reverse mapping from NATO code to reviewer name
        model_names = list(reviews.keys())
        reviewer_codes = list(nato_to_model.keys())

        # Get concerns extraction prompt from config
        prompt_template = get_prompt("concerns_extraction", config_file)
        if not prompt_template:
            # Fallback to hardcoded prompt if not found in config
            prompt_template = (
                "Based on the meta-review, extract all major concerns identified by reviewers.\n\n"
                "Create a JSON object with a 'concerns' array. Each concern object should have:\n"
                "1. A 'concern' field with a brief description\n"
                "2. One field for each model: {model_names}\n"
                "3. Each model field should be true if that model identified the concern, false otherwise\n\n"
                "Example structure:\n"
                "{{\n"
                "  \"concerns\": [\n"
                "    {{\n"
                "      \"concern\": \"Brief description of concern 1\",\n"
                "      \"{first_model}\": true,\n"
                "      \"{second_model}\": false,\n"
                "      ...\n"
                "    }},\n"
                "    ...\n"
                "  ]\n"
                "}}\n\n"
                "Return only valid JSON without any explanation.\n\n"
                "Meta-review:\n{meta_review_text}\n\n"
                "Model name mapping (for reference):\n{model_mapping}"
            )

        # Format the model names and mapping
        model_names_str = ', '.join(model_names)
        first_model = model_names[0] if model_names else ""
        second_model = model_names[1] if len(model_names) > 1 else ""
        model_mapping = ""
        for nato_code, model_name in nato_to_model.items():
            model_mapping += f"{nato_code}: {model_name}\n"

        prompt = prompt_template.format(
            model_names=model_names_str,
            first_model=first_model,
            second_model=second_model,
            meta_review_text=meta_review_text,
            model_mapping=model_mapping
        )

        json_text = meta_reviewer.generate(prompt)
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', json_text, re.DOTALL)
        if json_match:
            try:
                # Convert to DataFrame
                import pandas as pd
                concerns_data = json.loads(json_match.group(0))
                if "concerns" in concerns_data:
                    df = pd.DataFrame(concerns_data["concerns"])

                    # Ensure all model columns exist
                    for model_name in model_names:
                        if model_name not in df.columns:
                            df[model_name] = False

                    # Save CSV
                    csv_path = output_dir / "concerns_table.csv"
                    df.to_csv(csv_path, index=False)
                    click.echo(f"Concerns table saved to {csv_path}")
            except Exception as e:
                click.echo(f"Error processing concerns table: {e}")

        # Save everything to JSON
        result = {
            "individual_reviews": review_data,
            "meta_review": meta_review_text,
            "reviewer_to_model": nato_to_model
        }

        json_file = output_dir / "results.json"
        with open(json_file, "w") as f:
            json.dump(result, f, indent=2)
        click.echo(f"All results saved to {json_file}")

    return reviews


def main():
    """Entry point for the application."""
    cli()


if __name__ == "__main__":
    main()
