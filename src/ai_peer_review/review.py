import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
import pandas as pd

from .utils.pdf import extract_text_from_pdf
from .utils.config import (
    get_prompt,
    get_local_models,
    get_local_server_config,
    get_meta_reviewer_config,
)
from .llm_clients.local_client import LocalLLMClient


def _build_local_client(cfg: Dict, config_file: Optional[str] = None) -> LocalLLMClient:
    """Build a LocalLLMClient from a config entry, inheriting local_server
    defaults for any field the entry leaves as ``None`` (or omits)."""
    server = get_local_server_config(config_file)
    base_url = cfg.get("base_url") or server.get("base_url") or "http://localhost:11434/v1"
    api_key = cfg.get("api_key") or server.get("api_key") or "local"
    return LocalLLMClient(
        model=cfg["model"],
        base_url=base_url,
        api_key=api_key,
        temperature=cfg.get("temperature", 0.1),
        max_tokens=cfg.get("max_tokens", 8000),
    )


def get_review_prompt(paper_text: str, config_file: Optional[str] = None) -> str:
    """
    Generate the prompt for peer review.

    Args:
        paper_text: The text of the paper to review
        config_file: Optional path to a custom config file

    Returns:
        The formatted prompt string
    """
    prompt_template = get_prompt("review", config_file)
    if not prompt_template:
        # Fallback to hardcoded prompt if not found in config
        prompt_template = (
            "You are a neuroscientist and expert in brain imaging who has been asked to provide "
            "a peer review for a submitted research paper, which is attached here. "
            "Please provide a thorough and critical review of the paper. "
            "First provide a summary of the study and its results, and then provide "
            "a detailed point-by-point analysis of any flaws in the study.\n\n"
            "Here is the paper to review:\n\n{paper_text}"
        )

    return prompt_template.format(paper_text=paper_text)


def get_metareview_prompt(reviews: List[str], config_file: Optional[str] = None) -> str:
    """
    Generate the prompt for meta-review.

    Args:
        reviews: List of review texts
        config_file: Optional path to a custom config file

    Returns:
        The formatted meta-review prompt
    """
    # NATO phonetic alphabet for reviewers
    nato_names = ["alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]

    # Build reviews text with NATO phonetic names
    reviews_text = ""
    for i, review in enumerate(reviews):
        reviewer_name = nato_names[i] if i < len(nato_names) else f"reviewer_{i+1}"
        reviews_text += f"Review from {reviewer_name}:\n\n{review}\n\n"

    # Get prompt template from config
    prompt_template = get_prompt("metareview", config_file)
    if not prompt_template:
        # Fallback to hardcoded prompt if not found in config
        prompt_template = (
            "The attached files contain peer reviews of a research article. "
            "Please summarize these into a meta-review, highlighting both the common points "
            "raised across reviewers as well as any specific concerns that were only raised "
            "by some reviewers. In your meta-review, identify all major concerns raised by any reviewer. "
            "After your meta-review, include a section titled 'CONCERNS_TABLE_DATA' where you provide a JSON object "
            "representing a table of concerns. Each row should be a distinct concern, with columns for each reviewer. "
            "Use the following format: \n\n"
            "```json\n"
            "{{\n"
            "  \"concerns\": [\n"
            "    {{\n"
            "      \"concern\": \"Brief description of concern 1\",\n"
            "      \"alfa\": true/false,\n"
            "      \"bravo\": true/false,\n"
            "      ...\n"
            "    }},\n"
            "    ...\n"
            "  ]\n"
            "}}\n"
            "```\n\n"
            "Refer to each of the reviewers using their assigned NATO phonetic alphabet name "
            "(e.g., alfa, bravo, charlie) throughout your meta-review.\n\n"
            "{reviews_text}"
        )

    return prompt_template.format(reviews_text=reviews_text)


def get_client_factories(config_file: Optional[str] = None) -> Dict[str, callable]:
    """Build a {reviewer_name: client_factory} mapping from the configured
    local models. Each factory is only instantiated on demand (so unused
    reviewers never open a connection)."""
    models = get_local_models(config_file)
    return {
        name: (lambda cfg=cfg: _build_local_client(cfg, config_file))
        for name, cfg in models.items()
    }


def get_meta_reviewer_client(config_file: Optional[str] = None) -> LocalLLMClient:
    """Build the LocalLLMClient used for the meta-review and the
    concerns-extraction JSON call. Driven by the ``meta_reviewer`` config
    section so the meta-reviewer can be a different (stronger / longer-context)
    model than the individual reviewers."""
    return _build_local_client(get_meta_reviewer_config(config_file), config_file)


def process_paper(pdf_path: str, models: List[str], config_file: Optional[str] = None) -> Dict[str, str]:
    """
    Process a paper and generate reviews using multiple local LLMs.

    Args:
        pdf_path: Path to the PDF file to process
        models: List of reviewer names (keys of ``local_models`` in the config)
            to use for reviews
        config_file: Optional path to a custom config file

    Returns:
        Dictionary mapping reviewer names to their review texts
    """
    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)

    # Generate the review prompt
    prompt = get_review_prompt(paper_text, config_file)

    # Build client factories from the configured local models
    client_factories = get_client_factories(config_file)

    # Get reviews from each requested local LLM
    reviews = {}
    for model_name in models:
        if model_name in client_factories:
            # Only initialize the client if the model is requested
            client = client_factories[model_name]()
            reviews[model_name] = client.generate(prompt)

    return reviews


def extract_concerns_table(meta_review_text: str) -> Optional[Dict]:
    """Extract the JSON concerns table from the meta-review text."""
    # Look for JSON data between CONCERNS_TABLE_DATA section and the end of a JSON block
    pattern = r'CONCERNS_TABLE_DATA.*?```json\s*(.*?)\s*```'
    match = re.search(pattern, meta_review_text, re.DOTALL)

    if not match:
        # Try without the json tag in case the model formatted it differently
        pattern = r'CONCERNS_TABLE_DATA.*?```\s*(.*?)\s*```'
        match = re.search(pattern, meta_review_text, re.DOTALL)

    if not match:
        # Try finding a JSON object with "concerns" key anywhere in the text
        pattern = r'\{\s*"concerns"\s*:\s*\[.*?\]\s*\}'
        match = re.search(pattern, meta_review_text, re.DOTALL)

    if match:
        try:
            json_str = match.group(1)
            # Remove the CONCERNS_TABLE_DATA header if it's still there
            json_str = re.sub(r'CONCERNS_TABLE_DATA.*?{', '{', json_str, flags=re.DOTALL)
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Error parsing JSON data from meta-review")
            return None
    return None


def generate_meta_review(reviews: Dict[str, str], config_file: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
    """
    Generate a meta-review of all reviews using the configured local
    meta-reviewer model.

    Args:
        reviews: Dictionary mapping reviewer names to their review texts
        config_file: Optional path to a custom config file

    Returns:
        Tuple of (meta-review text, NATO-to-reviewer mapping)
    """
    # NATO phonetic alphabet for reviewers
    nato_names = ["alfa", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]

    # Get list of reviewers in a fixed order
    models_list = list(reviews.keys())

    # Remove reviewer identifiers but maintain order for mapping
    anonymized_reviews = [reviews[model] for model in models_list]

    # Create mapping of NATO name to reviewer name
    nato_to_model = {nato_names[i]: model for i, model in enumerate(models_list) if i < len(nato_names)}
    # Handle any extras if we have more than 10 reviewers
    for i, model in enumerate(models_list[len(nato_names):], len(nato_names)):
        nato_to_model[f"reviewer_{i+1}"] = model

    # Generate meta-review prompt
    prompt = get_metareview_prompt(anonymized_reviews, config_file)

    # Use the configured local meta-reviewer (replaces the previous hardcoded
    # Google Gemini client). No third-party API is contacted.
    meta_reviewer = get_meta_reviewer_client(config_file)
    meta_review_text = meta_reviewer.generate(prompt)

    # Filter out the CONCERNS_TABLE_DATA section from the meta-review
    clean_meta_review = re.sub(r'CONCERNS_TABLE_DATA.*', '', meta_review_text, flags=re.DOTALL)

    return clean_meta_review, nato_to_model


def save_concerns_as_csv(meta_review_text: str, output_path: Path) -> bool:
    """Extract concerns from meta-review and save as CSV."""
    concerns_data = extract_concerns_table(meta_review_text)

    if not concerns_data or "concerns" not in concerns_data:
        print("No valid concerns table found in meta-review")
        return False

    # Convert to DataFrame
    df = pd.DataFrame(concerns_data["concerns"])

    # Save to CSV
    csv_path = output_path / "concerns_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Concerns table saved to {csv_path}")
    return True
