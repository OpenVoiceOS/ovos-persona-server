"""
Configuration settings for the OVOS Persona Server.

This module defines the `Settings` dataclass to manage various
configuration parameters for the server, including persona file paths.
It also handles loading environment variables.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any

from dotenv import load_dotenv

# Load .env file automatically
load_dotenv()


@dataclass
class Settings:
    """
    Manages configuration settings for the Persona Server.

    If a persona is not provided, it defaults to a configuration using
    `https://llama.smartgic.io` as the OpenAI compatible API endpoint.

    Attributes:
        persona (str): Path to the persona JSON file. Defaults to an empty string,
                       which will be overridden by the PERSONA_PATH environment variable
                       or a command-line argument.
    """
    persona: str = field(default_factory=lambda: os.environ.get('PERSONA_PATH', ""))

    @property
    def persona_config(self) -> Dict[str, Any]:
        """
        Loads and returns the persona configuration from the specified JSON file,
        or constructs a default configuration if no persona file is provided.

        Returns:
            Dict[str, Any]: A dictionary containing the persona configuration.
        """
        with open(self.persona, "r", encoding="utf-8") as f:
            persona = json.load(f)
        persona["name"] = persona.get("name") or os.path.basename(self.persona)
        return persona


settings = Settings()
