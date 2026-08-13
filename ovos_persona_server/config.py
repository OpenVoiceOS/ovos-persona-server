"""
Configuration settings for the OVOS Persona Server.

This module defines the `Settings` dataclass to manage various
configuration parameters for the server, including persona file paths,
and settings for text and image embeddings. It also handles loading
environment variables.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

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
                       or a command-line argument. If not provided,
                       LLM settings from environment variables are used.
        chat_memory (str): Whether the chat endpoints transparently apply the persona's
                           memory/RAG plugin to incoming requests. ``"off"`` (default) is
                           the *backend* mode: requests are a stateless passthrough, the
                           client owns conversation state and drives the Files / Vector-Stores
                           endpoints itself. ``"transparent"`` is the single-user *hosted
                           agent* mode: the server keys history by session (the OpenAI
                           ``user`` field when present, else a default session) and folds the
                           persona's ``memory_module`` into every turn. Loaded from the
                           CHAT_MEMORY environment variable. Leave ``"off"`` for multi-user /
                           drop-in-OpenAI-replacement deployments (shared server memory would
                           leak across users).
        file_storage_path (str): The directory path where uploaded files will be stored.
                                 Defaults to "~/.cache/ovos-persona-server/files".
        file_storage_strategy (str): Defines how files are stored.
                                     "disk": Store only on disk.
                                     "database": Store only in the database (as binary).
                                     "both": Store both on disk and in the database.
                                     Defaults to "disk".
        openai_key (str): API key for OpenAI compatible services. Defaults to "sk-xxx",
                          loaded from OPENAI_KEY environment variable.
        openai_url (str): URL for OpenAI compatible services. Defaults to
                          "https://llama.smartgic.io/v1", loaded from OPENAI_URL
                          environment variable.
        openai_model (Optional[str]): Default model name for OpenAI compatible services.
                                      Defaults to "llama3.1:8b", loaded from OPENAI_MODEL
                                      environment variable.
        llm_solver (Optional[str]): The solver plugin to use for the LLM. Loaded from
                                    SOLVER_PLUGIN environment variable.
        llm_name (Optional[str]): The name of the LLM persona. Loaded from PERSONA_NAME
                                  environment variable.
        llm_model (Optional[str]): The model name for the LLM. Loaded from LLM_MODEL
                                   environment variable.
        llm_system_prompt (Optional[str]): The system prompt for the LLM. Loaded from
                                           LLM_SYSTEM_PROMPT environment variable.
        llm_url (Optional[str]): URL for a remote LLM service. Loaded from LLM_URL
                                 environment variable.
        llm_key (Optional[str]): API key for a remote LLM service. Loaded from LLM_KEY
                                 environment variable.
        text_embeddings_plugin (str): The name of the text embeddings plugin to load.
                                      Defaults to "ovos-gguf-embeddings-plugin", loaded
                                      from TEXT_EMBEDDINGS_PLUGIN environment variable.
        image_embeddings_plugin (str): The name of the image embeddings plugin to load.
                                       Defaults to None, loaded from IMAGE_EMBEDDINGS_PLUGIN
                                       environment variable.
        embeddings_db_plugin (str): The name of the embeddings database plugin to load.
                                    Defaults to "ovos-chromadb-embeddings-plugin", loaded
                                    from EMBEDDINGS_DB_PLUGIN environment variable.
        embeddings_verbose (Optional[str]): Verbosity setting for embeddings plugins.
                                            Loaded from EMBEDDINGS_VERBOSE environment variable.
        embeddings_model (Optional[str]): Model name for embeddings. Loaded from
                                          EMBEDDINGS_MODEL environment variable.
        llama_n_gpu_layers (Optional[int]): Number of GPU layers for llama.cpp based
                                            embeddings. Loaded from
                                            EMBEDDINGS_LLAMACPP_N_GPU_LAYERS environment variable.
        embeddings_url (Optional[str]): URL for remote embeddings service. Loaded from
                                        EMBEDDINGS_URL environment variable.
        embeddings_key (Optional[str]): API key for remote embeddings service. Loaded from
                                        EMBEDDINGS_KEY environment variable.
        summarizer_plugin (str): The name of the summarizer plugin to load. Defaults to
                                 "ovos-summarizer-openai-plugin", loaded from
                                 SUMMARIZER_PLUGIN environment variable.
        summarizer_url (Optional[str]): URL for a remote summarizer service. Loaded from
                                        SUMMARIZER_URL environment variable.
        summarizer_key (Optional[str]): API key for a remote summarizer service. Loaded from
                                        SUMMARIZER_KEY environment variable.
        summarizer_model (Optional[str]): Model name for the summarizer. Loaded from
                                          SUMMARIZER_MODEL environment variable.
        summarizer_prompt (Optional[str]): Custom prompt for the summarizer. Loaded from
                                           SUMMARIZER_PROMPT environment variable.
        reranker_plugin (str): The name of the reranker plugin to load. Defaults to
                               "ovos-flashrank-reranker-plugin", loaded from
                               RERANKER_PLUGIN environment variable.
        reranker_model (str): The model name for the reranker. Defaults to
                              "ms-marco-MultiBERT-L-12", loaded from RERANKER_MODEL
                              environment variable.
        WtP_cuda (bool): Whether to use CUDA for the "What-to-Process" (WtP) text splitter.
                         Defaults to False.
        WtP_model (str): The model name for the "What-to-Process" (WtP) text splitter.
                         Defaults to "wtp-bert-mini".
    """
    persona: str = field(default_factory=lambda: os.environ.get('PERSONA_PATH', ""))
    chat_memory: str = field(default_factory=lambda: os.environ.get('CHAT_MEMORY', 'off'))
    file_storage_path: str = field(default_factory=lambda: os.environ.get('FILE_STORAGE_PATH', os.path.expanduser(
        "~/.cache/ovos-persona-server/files")))
    file_storage_strategy: str = field(default_factory=lambda: os.environ.get('FILE_STORAGE_STRATEGY', 'disk'))
    openai_key: str = field(default_factory=lambda: os.environ.get('OPENAI_KEY', "sk-xxx"))
    openai_url: str = field(default_factory=lambda: os.environ.get('OPENAI_URL', "https://llama.smartgic.io/v1"))
    openai_model: Optional[str] = os.environ.get('OPENAI_MODEL', "llama3.1:8b")
    llm_solver: Optional[str] = os.environ.get('SOLVER_PLUGIN')
    llm_name: Optional[str] = os.environ.get('PERSONA_NAME')
    llm_model: Optional[str] = os.environ.get('LLM_MODEL')
    llm_system_prompt: Optional[str] = os.environ.get('LLM_SYSTEM_PROMPT')
    llm_url: Optional[str] = os.environ.get('LLM_URL')
    llm_key: Optional[str] = os.environ.get('LLM_KEY')
    text_embeddings_plugin: str = field(
        default_factory=lambda: os.environ.get('TEXT_EMBEDDINGS_PLUGIN', "ovos-gguf-embeddings-plugin"))
    image_embeddings_plugin: Optional[str] = field(default_factory=lambda: os.environ.get('IMAGE_EMBEDDINGS_PLUGIN'))
    embeddings_db_plugin: str = field(
        default_factory=lambda: os.environ.get('EMBEDDINGS_DB_PLUGIN', "ovos-chromadb-embeddings-plugin"))
    embeddings_verbose: Optional[str] = os.environ.get('EMBEDDINGS_VERBOSE')
    embeddings_model: Optional[str] = os.environ.get('EMBEDDINGS_MODEL')
    llama_n_gpu_layers: Optional[int] = field(default_factory=lambda: int(os.environ['EMBEDDINGS_LLAMACPP_N_GPU_LAYERS']) if os.environ.get(
        'EMBEDDINGS_LLAMACPP_N_GPU_LAYERS') else None)
    embeddings_url: Optional[str] = os.environ.get('EMBEDDINGS_URL')
    embeddings_key: Optional[str] = os.environ.get('EMBEDDINGS_KEY')
    summarizer_plugin: str = field(
        default_factory=lambda: os.environ.get('SUMMARIZER_PLUGIN', "ovos-summarizer-openai-plugin"))
    summarizer_url: Optional[str] = os.environ.get('SUMMARIZER_URL')
    summarizer_key: Optional[str] = os.environ.get('SUMMARIZER_KEY')
    summarizer_model: Optional[str] = os.environ.get('SUMMARIZER_MODEL')
    summarizer_prompt: Optional[str] = os.environ.get('SUMMARIZER_PROMPT')
    reranker_plugin: str = field(
        default_factory=lambda: os.environ.get('RERANKER_PLUGIN', "ovos-flashrank-reranker-plugin"))
    reranker_model: str = field(
        default_factory=lambda: os.environ.get('RERANKER_MODEL', "ms-marco-MultiBERT-L-12"))
    WtP_cuda: bool = field(default=False)
    WtP_model: str = field(default="wtp-bert-mini")

    @property
    def persona_config(self) -> Dict[str, Any]:
        """
        Loads and returns the persona configuration from the specified JSON file,
        or constructs a default configuration if no persona file is provided.

        Returns:
            Dict[str, Any]: A dictionary containing the persona configuration.
        """
        if self.persona:
            with open(self.persona, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "name": self.llm_name,
            "solvers": [
                self.llm_solver
            ],
            self.llm_solver: {
                "api_url": self.llm_url or self.openai_url,
                "key": self.llm_key or self.openai_key,
                "model": self.llm_model or self.openai_model,
                "system_prompt": self.llm_system_prompt
            }
        }

    @property
    def reranker_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of reranker configuration parameters
        suitable for passing to reranker plugin constructors.

        Returns:
            Dict[str, Any]: A dictionary containing reranker configuration.
        """
        return {
            "model": self.reranker_model
        }

    @property
    def summarizer_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of summarizer configuration parameters
        suitable for passing to summarizer plugin constructors.

        Returns:
            Dict[str, Any]: A dictionary containing summarizer configuration.
        """
        # assume ovos-summarizer-openai-plugin compatible summarizer plugin
        if self.summarizer_plugin == "ovos-summarizer-openai-plugin":
            default_prompt = """Your task is to summarize text.
Always answer in plaintext with no formatting.
Focus on the most important information.
-------
{content}
"""
            return {
                "key": self.summarizer_key or self.openai_key,
                "api_url": self.summarizer_url or self.openai_url,
                "model": self.summarizer_model or self.openai_model,
                "system_prompt": self.summarizer_prompt or default_prompt
            }
        return {}

    @property
    def embeddings_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of embedding configuration parameters
        suitable for passing to embedding plugin constructors.

        Returns:
            Dict[str, Any]: A dictionary containing embedding configuration.
        """
        # assume either openai or ovos-gguf-embeddings-plugin compatible plugin
        if self.text_embeddings_plugin == "ovos-gguf-embeddings-plugin":
            cfg: Dict[str, Any] = {
                "n_gpu_layers": self.llama_n_gpu_layers or 0,
                "verbose": self.embeddings_verbose is not None and self.embeddings_verbose.lower() == 'true',
                "model": self.embeddings_model or "all-MiniLM-L6-v2",
            }
            return cfg
        return {
            "key": self.embeddings_key or self.openai_key,
            "api_url": self.embeddings_url or self.openai_url,
            "model": self.embeddings_model or self.openai_model
        }

    @property
    def embeddings_db_config(self) -> Dict[str, Any]:
        """
        Returns a dictionary of embeddings database configuration parameters.
        Currently, this primarily includes the 'path' for ChromaDB.

        Returns:
            Dict[str, Any]: A dictionary containing embeddings database configuration.
        """
        config: Dict[str, Any] = {}
        # For most dbs the primary config is the path
        # TODO: Potentially load more config from .env for specific DBs
        config["path"] = os.path.join(self.file_storage_path, "embeddings_db")
        return config

    def __post_init__(self) -> None:
        """
        Post-initialization hook to set default LLM persona parameters
        from environment variables if no persona file is explicitly set.
        """
        if not self.persona:
            self.llm_name = self.llm_name or os.environ.get('PERSONA_NAME', "OpenVoiceOS")
            self.llm_solver = self.llm_solver or os.environ.get('SOLVER_PLUGIN', "ovos-solver-openai-plugin")
            self.llm_url = self.llm_url or os.environ.get('LLM_URL', self.openai_url)
            self.llm_model = self.llm_model or os.environ.get('LLM_MODEL', self.openai_model)
            self.llm_key = self.llm_key or os.environ.get('LLM_KEY', self.openai_key)
            self.llm_system_prompt = self.llm_system_prompt or os.environ.get('LLM_SYSTEM_PROMPT',
                                                                              "you are a voice assistant")


settings = Settings()
