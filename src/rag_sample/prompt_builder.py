"""Prompt builder utility for constructing prompts from YAML configuration."""

import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


def lowercase_first_char(text: str) -> str:
    """Convert first character of text to lowercase."""
    if not text:
        return text
    return text[0].lower() + text[1:] if len(text) > 1 else text.lower()


def format_prompt_section(header: str, content: Any) -> str:
    """Format a prompt section with header and content."""
    if isinstance(content, list):
        content_str = "\n".join(f"- {item}" for item in content)
    else:
        content_str = str(content)
    
    return f"{header}\n{content_str}"


def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
    app_config: Optional[Dict[str, Any]] = None,
) -> str:
    """Builds a complete prompt string based on a config dictionary.

    Args:
        config: Dictionary specifying prompt components.
        input_data: Content to be summarized or processed.
        app_config: Optional app-wide configuration (e.g., reasoning strategies).

    Returns:
        A fully constructed prompt as a string.

    Raises:
        ValueError: If the required 'instruction' field is missing.
    """
    prompt_parts = []

    if role := config.get("role"):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(format_prompt_section("Your task is as follows:", instruction))

    if context := config.get("context"):
        prompt_parts.append(f"Here's some background that may help you:\n{context}")

    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Ensure your response follows these rules:", constraints
            )
        )

    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Follow these style and tone guidelines in your response:", tone
            )
        )

    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Structure your response as follows:", format_)
        )

    if examples := config.get("examples"):
        prompt_parts.append("Here are some examples to guide your response:")
        if isinstance(examples, list):
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:\n{example}")
        else:
            prompt_parts.append(str(examples))

    if goal := config.get("goal"):
        prompt_parts.append(f"Your goal is to achieve the following outcome:\n{goal}")

    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    reasoning_strategy = config.get("reasoning_strategy")
    if reasoning_strategy and reasoning_strategy != "None" and app_config:
        strategies = app_config.get("reasoning_strategies", {})
        if strategy_text := strategies.get(reasoning_strategy):
            prompt_parts.append(strategy_text.strip())

    prompt_parts.append("Now perform the task as instructed above.")
    return "\n\n".join(prompt_parts)


class PromptManager:
    """Manages prompt configurations and building."""
    
    def __init__(self, config_path: str = "config/prompts.yaml"):
        """Initialize prompt manager.
        
        Args:
            config_path: Path to the prompts YAML configuration file.
        """
        self.config_path = Path(config_path)
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, Any]:
        """Load prompts from YAML configuration file."""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Prompt configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading prompt configuration: {e}")
            return {}
    
    def get_prompt_config(self, prompt_name: str) -> Dict[str, Any]:
        """Get configuration for a specific prompt.
        
        Args:
            prompt_name: Name of the prompt configuration.
            
        Returns:
            Dictionary containing prompt configuration.
            
        Raises:
            KeyError: If prompt configuration not found.
        """
        if prompt_name not in self.prompts:
            available = list(self.prompts.keys())
            raise KeyError(f"Prompt '{prompt_name}' not found. Available prompts: {available}")
        
        return self.prompts[prompt_name]
    
    def build_prompt(self, prompt_name: str, input_data: str = "", app_config: Optional[Dict[str, Any]] = None) -> str:
        """Build a prompt from configuration.
        
        Args:
            prompt_name: Name of the prompt configuration.
            input_data: Content to be processed.
            app_config: Optional app-wide configuration.
            
        Returns:
            Built prompt string.
        """
        config = self.get_prompt_config(prompt_name)
        return build_prompt_from_config(config, input_data, app_config)
    
    def list_prompts(self) -> List[str]:
        """List available prompt configurations.
        
        Returns:
            List of available prompt names.
        """
        return list(self.prompts.keys())
    
    def get_prompt_info(self, prompt_name: str) -> Dict[str, Any]:
        """Get information about a prompt configuration.
        
        Args:
            prompt_name: Name of the prompt configuration.
            
        Returns:
            Dictionary containing prompt information.
        """
        config = self.get_prompt_config(prompt_name)
        return {
            "name": prompt_name,
            "description": config.get("description", "No description"),
            "has_role": "role" in config,
            "has_instruction": "instruction" in config,
            "has_constraints": "output_constraints" in config,
            "has_examples": "examples" in config,
        }
    
    def reload_prompts(self):
        """Reload prompts from configuration file."""
        self.prompts = self._load_prompts()
