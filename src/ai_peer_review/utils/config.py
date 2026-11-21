import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv


def load_dotenv_file() -> None:
    """Load environment variables from .env file."""
    # Load from .env file in the current directory
    load_dotenv()
    
    # Also try loading from config directory
    config_dir = Path.home() / ".ai-peer-review"
    env_path = config_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def get_config_path(custom_path: Optional[str] = None) -> Path:
    """
    Get the path to the configuration file.
    
    Args:
        custom_path: Optional custom path to the config file
        
    Returns:
        Path to the configuration file
    """
    if custom_path:
        config_path = Path(custom_path)
        # Create parent directory if it doesn't exist
        config_path.parent.mkdir(exist_ok=True, parents=True)
        return config_path
    
    # Default path
    config_dir = Path.home() / ".ai-peer-review"
    config_dir.mkdir(exist_ok=True)
    return config_dir / "config.json"


def get_default_config() -> Dict[str, Any]:
    """Return the default configuration.

    The default ships a single local reviewer plus a meta-reviewer, both
    pointing at a local Ollama instance. Add more entries under
    ``local_models`` (each can point at a different model or a different
    server) to get genuine multi-perspective review.
    """
    return {
        "api_keys": {},
        "local_server": {
            # Default OpenAI-compatible local server. Ollama's port is 11434.
            # LM Studio uses http://localhost:1234/v1, llama.cpp uses 8080,
            # vLLM uses 8000. Change this once and every reviewer inherits it
            # unless the reviewer overrides "base_url".
            "base_url": "http://localhost:11434/v1",
            # Local servers ignore this value, but the OpenAI SDK requires a
            # non-empty string.
            "api_key": "local",
        },
        "local_models": {
            # Each entry is one "reviewer". The key is the friendly name shown
            # by `ai-peer-review list-models` and accepted by --models.
            # Default tag is qwen3:32b (strong reasoning, fits a single 24GB
            # GPU in 4-bit). See README for stronger alternatives.
            "reviewer-1": {
                "model": "qwen3:32b",
                "base_url": None,   # None => inherit local_server.base_url
                "api_key": None,    # None => inherit local_server.api_key
                "temperature": 0.1,
                "max_tokens": 8000,
            }
        },
        "meta_reviewer": {
            # Used for the meta-review and the concerns-extraction JSON call.
            # Point this at your strongest available model with the longest
            # context (the meta-review input is all individual reviews
            # concatenated).
            "model": "qwen3:32b",
            "base_url": None,
            "api_key": None,
            "temperature": 0.1,
            "max_tokens": 8000,
        },
        "prompts": {
            "system": "You are a neuroscientist and expert in brain imaging.",
            "review": (
                "You are a neuroscientist and expert in brain imaging who has been asked to provide "
                "a peer review for a submitted research paper, which is attached here. "
                "Please provide a thorough and critical review of the paper. "
                "First provide a summary of the study and its results, and then provide "
                "a detailed point-by-point analysis of any flaws in the study.\n\n"
                "Here is the paper to review:\n\n{paper_text}"
            ),
            "metareview": (
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
            ),
            "concerns_extraction": (
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
        }
    }

def load_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file, or create default if it doesn't exist.
    
    Args:
        config_file: Optional path to a custom config file
        
    Returns:
        The loaded configuration dictionary
    """
    config_path = get_config_path(config_file)
    
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Update with any missing default sections
            default_config = get_default_config()
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
            
            return config
        except json.JSONDecodeError:
            # If config file is invalid, return default config
            return get_default_config()
    else:
        # Create default config
        default_config = get_default_config()
        save_config(default_config, config_path)
        return default_config


def save_config(config: Dict[str, Any], config_path: Optional[Path] = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: The configuration dictionary to save
        config_path: Optional explicit path to save to (otherwise uses default path)
    """
    if config_path is None:
        config_path = get_config_path()
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def get_api_key(key_name: str, config_file: Optional[str] = None) -> Optional[str]:
    """
    Get API key from .env file, environment variable, or configuration file.
    
    Args:
        key_name: Name of the API key (e.g., 'openai', 'anthropic')
        config_file: Optional path to a custom config file
        
    Returns:
        API key if found, None otherwise
    """
    # Make sure .env variables are loaded
    load_dotenv_file()
    
    # Try environment variable
    env_var_name = f"{key_name.upper()}_API_KEY"
    if env_var_name in os.environ:
        return os.environ[env_var_name]
    
    # Then try config file
    config = load_config(config_file)
    return config.get("api_keys", {}).get(key_name)


def set_api_key(key_name: str, api_key: str, config_file: Optional[str] = None) -> None:
    """
    Set API key in configuration file.
    
    Args:
        key_name: Name of the API key (e.g., 'openai', 'anthropic')
        api_key: API key value
        config_file: Optional path to a custom config file
    """
    config = load_config(config_file)
    config_path = get_config_path(config_file)
    
    if "api_keys" not in config:
        config["api_keys"] = {}
    
    config["api_keys"][key_name] = api_key
    save_config(config, config_path)


def get_prompt(prompt_name: str, config_file: Optional[str] = None) -> str:
    """
    Get a prompt from the configuration file.

    Args:
        prompt_name: Name of the prompt (e.g., 'system', 'review', 'metareview')
        config_file: Optional path to the config file

    Returns:
        Prompt template string if found, empty string otherwise
    """
    config = load_config(config_file)
    return config.get("prompts", {}).get(prompt_name, "")


def get_local_server_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Return the default local-server settings (base_url, api_key).

    Individual reviewers and the meta-reviewer inherit these unless they
    override ``base_url`` / ``api_key`` themselves.
    """
    config = load_config(config_file)
    return config.get("local_server", {}) or {}


def set_local_server(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    config_file: Optional[str] = None,
) -> None:
    """Update the default local-server settings. Only non-None fields are changed."""
    config = load_config(config_file)
    config_path = get_config_path(config_file)
    server = config.setdefault("local_server", {})
    if base_url is not None:
        server["base_url"] = base_url
    if api_key is not None:
        server["api_key"] = api_key
    save_config(config, config_path)


def get_local_models(config_file: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return the configured local reviewer models as ``{name: cfg}``."""
    config = load_config(config_file)
    models = config.get("local_models", {})
    if not isinstance(models, dict) or not models:
        # Fall back to the default reviewer so the tool always has something
        # to run even on a fresh / hand-edited config.
        return get_default_config()["local_models"]
    return models


def set_local_model(
    name: str,
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    config_file: Optional[str] = None,
) -> None:
    """Add or update a local reviewer model entry."""
    config = load_config(config_file)
    config_path = get_config_path(config_file)
    models = config.setdefault("local_models", {})
    entry = models.get(name, {})
    entry["model"] = model
    if base_url is not None:
        entry["base_url"] = base_url
    if api_key is not None:
        entry["api_key"] = api_key
    if temperature is not None:
        entry["temperature"] = temperature
    if max_tokens is not None:
        entry["max_tokens"] = max_tokens
    models[name] = entry
    save_config(config, config_path)


def remove_local_model(name: str, config_file: Optional[str] = None) -> bool:
    """Remove a local reviewer model entry. Returns True if it existed."""
    config = load_config(config_file)
    config_path = get_config_path(config_file)
    models = config.get("local_models", {})
    if name in models:
        del models[name]
        save_config(config, config_path)
        return True
    return False


def get_meta_reviewer_config(config_file: Optional[str] = None) -> Dict[str, Any]:
    """Return the meta-reviewer model config."""
    config = load_config(config_file)
    meta = config.get("meta_reviewer", {})
    if not isinstance(meta, dict) or not meta:
        return get_default_config()["meta_reviewer"]
    return meta


def set_meta_reviewer(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    config_file: Optional[str] = None,
) -> None:
    """Update the meta-reviewer model entry."""
    config = load_config(config_file)
    config_path = get_config_path(config_file)
    meta = config.setdefault("meta_reviewer", {})
    meta["model"] = model
    if base_url is not None:
        meta["base_url"] = base_url
    if api_key is not None:
        meta["api_key"] = api_key
    if temperature is not None:
        meta["temperature"] = temperature
    if max_tokens is not None:
        meta["max_tokens"] = max_tokens
    save_config(config, config_path)