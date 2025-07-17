import json
from typing import Any, List, Dict, Optional, Union
from ovos_persona_server.config import settings
from pydantic import BaseModel, Field, field_serializer


# --- Ollama API Models ---

class OllamaChatResponse(BaseModel):
    """
    Represents a chat response from the Ollama API.

    Attributes:
        model (str): The name of the model that generated the response.
        created_at (str): The timestamp when the response was created (ISO 8601 format).
        message (Dict[str, str]): The message object containing the role and content of the assistant's reply.
        done (bool): Indicates if the response is the final one in a stream.
        context (Optional[List[int]]): An encoding of the conversation used, can be sent in the next request
                                       to keep conversational memory (list of token IDs).
        total_duration (Optional[int]): Time spent generating the response in nanoseconds.
        load_duration (Optional[int]): Time spent loading the model into memory in nanoseconds.
        prompt_eval_count (Optional[int]): Number of tokens in the prompt.
        prompt_eval_duration (Optional[int]): Time spent evaluating the prompt in nanoseconds.
        eval_count (Optional[int]): Number of tokens in the response.
        eval_duration (Optional[int]): Time spent generating the response in nanoseconds.
        done_reason (Optional[str]): The reason the model stopped generating (e.g., "stop", "unload").
    """
    model: Optional[str] = Field(settings.openai_model, description="The name of the model that generated the response.")
    created_at: str = Field(..., description="The timestamp when the response was created (ISO 8601 format).")
    message: Dict[str, str] = Field(..., description="The message object containing the role and content of the assistant's reply.")
    done: bool = Field(..., description="Indicates if the response is the final one in a stream.")
    context: Optional[List[int]] = Field(None, description="An encoding of the conversation used, can be sent in the next request to keep conversational memory (list of token IDs).")
    total_duration: Optional[int] = Field(None, description="Time spent generating the response in nanoseconds.")
    load_duration: Optional[int] = Field(None, description="Time spent loading the model into memory in nanoseconds.")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in the prompt.")
    prompt_eval_duration: Optional[int] = Field(None, description="Time spent evaluating the prompt in nanoseconds.")
    eval_count: Optional[int] = Field(None, description="Number of tokens in the response.")
    eval_duration: Optional[int] = Field(None, description="Time spent generating the response in nanoseconds.")
    done_reason: Optional[str] = Field(None, description="The reason the model stopped generating (e.g., 'stop', 'unload').")


class FunctionCall(BaseModel):
    """
    Represents a function call within a tool call.

    Attributes:
        name (str): The name of the function to be called.
        arguments (Dict[str, Any]): A dictionary of arguments to pass to the function.
    """
    name: str = Field(..., description="The name of the function to be called.")
    arguments: Dict[str, Any] = Field(..., description="A dictionary of arguments to pass to the function.")

    @field_serializer('arguments')
    def serialize_arguments(self, arguments: Dict[str, Any]) -> str:
        """
        Serializes the arguments dictionary into a JSON string.
        This is needed because the Ollama API expects the 'arguments' field
        within 'tool_calls.function' to be a stringified JSON object.
        """
        return json.dumps(arguments)


class ToolCall(BaseModel):
    """
    Represents a tool call made by the model.

    Attributes:
        function (FunctionCall): The details of the function to be called.
    """
    function: FunctionCall = Field(..., description="The details of the function to be called.")


class OllamaChatMessage(BaseModel):
    """
    A single message in an Ollama chat conversation.

    Attributes:
        role (str): The role of the message, either 'system', 'user', 'assistant', or 'tool'.
        content (str): The content of the message.
        thinking (Optional[str]): (For thinking models) The model's thinking process.
        images (Optional[List[str]]): A list of base64-encoded images to include in the message
                                      (for multimodal models such as llava).
        tool_calls (Optional[List[ToolCall]]): A list of tools in JSON that the model wants to use.
        tool_name (Optional[str]): The name of the tool that was executed (for tool role messages).
    """
    role: str = Field(..., description="The role of the message, either 'system', 'user', 'assistant', or 'tool'.")
    content: str = Field(..., description="The content of the message.")
    thinking: Optional[str] = Field(None, description="(For thinking models) The model's thinking process.")
    images: Optional[List[str]] = Field(None, description="A list of base64-encoded images to include in the message (for multimodal models such as llava).")
    tool_calls: Optional[List[ToolCall]] = Field(None, description="A list of tools in JSON that the model wants to use.")
    tool_name: Optional[str] = Field(None, description="The name of the tool that was executed (for tool role messages).")


class OllamaChatRequest(BaseModel):
    """
    Represents a chat request to the Ollama API.

    Attributes:
        model (Optional[str]): The model to use for the chat. Defaults to "server-default".
        messages (List[OllamaChatMessage]): The messages of the chat, used to keep a chat memory.
        tools (Optional[List[Dict[str, Any]]]): List of tools in JSON for the model to use if supported.
        think (Optional[bool]): (For thinking models) Should the model think before responding?
        format (Optional[Union[str, Dict[str, Any]]]): The format to return a response in. Can be "json" or a JSON schema object.
        options (Optional[Dict[str, Any]]): Additional model parameters (e.g., temperature, top_k)
                                           listed in the Modelfile documentation.
        stream (Optional[bool]): If true, the response will be streamed as a series of JSON objects.
        keep_alive (Optional[str]): Controls how long the model will stay loaded into memory following the request (default: "5m").
    """
    model: Optional[str] = Field(settings.openai_model, description="The model to use for the chat. Currently, only the persona's default model is supported.")
    messages: List[OllamaChatMessage] = Field(..., description="The messages to generate a response for.")
    tools: Optional[List[Dict[str, Any]]] = Field(None, description="List of tools in JSON for the model to use if supported.")
    think: Optional[bool] = Field(False, description="(For thinking models) Should the model think before responding?")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="The format to return a response in. Can be 'json' or a JSON schema object.")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters (e.g., temperature, top_k) listed in the Modelfile documentation.")
    stream: Optional[bool] = Field(False, description="If true, the response will be streamed as a series of JSON objects.")
    keep_alive: Optional[str] = Field("5m", description="Controls how long the model will stay loaded into memory following the request (default: '5m').")


class OllamaGenerateRequest(BaseModel):
    """
    Represents a generation request to the Ollama API.

    Attributes:
        model (Optional[str]): The model name to use for generation. Defaults to "server-default".
        prompt (str): The prompt to generate a response for.
        suffix (Optional[str]): The text after the model response.
        images (Optional[List[str]]): A list of base64-encoded images (for multimodal models such as llava).
        think (Optional[bool]): (For thinking models) Should the model think before responding?
        format (Optional[Union[str, Dict[str, Any]]]): The format to return a response in. Can be "json" or a JSON schema object.
        options (Optional[Dict[str, Any]]): Additional model parameters (e.g., temperature, top_k)
                                           listed in the Modelfile documentation.
        system (Optional[str]): System message to use (overrides what is defined in the Modelfile).
        template (Optional[str]): The prompt template to use (overrides what is defined in the Modelfile).
        stream (Optional[bool]): If false, the response will be returned as a single response object.
        raw (Optional[bool]): If true, no formatting will be applied to the prompt.
        keep_alive (Optional[str]): Controls how long the model will stay loaded into memory following the request (default: "5m").
        context (Optional[List[int]]): The context parameter returned from a previous request to /generate,
                                       this can be used to keep a short conversational memory (deprecated).
    """
    model: Optional[str] = Field(settings.llm_model, description="The model name to use for generation.")
    prompt: str = Field(..., description="The prompt to generate a response for.")
    suffix: Optional[str] = Field(None, description="The text after the model response.")
    images: Optional[List[str]] = Field(None, description="A list of base64-encoded images (for multimodal models such as llava).")
    think: Optional[bool] = Field(False, description="(For thinking models) Should the model think before responding?")
    format: Optional[Union[str, Dict[str, Any]]] = Field(None, description="The format to return a response in. Can be 'json' or a JSON schema object.")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters (e.g., temperature, top_k) listed in the Modelfile documentation.")
    system: Optional[str] = Field(None, description="System message to use (overrides what is defined in the Modelfile).")
    template: Optional[str] = Field(None, description="The prompt template to use (overrides what is defined in the Modelfile).")
    stream: Optional[bool] = Field(False, description="If false, the response will be returned as a single response object.")
    raw: Optional[bool] = Field(False, description="If true, no formatting will be applied to the prompt.")
    keep_alive: Optional[str] = Field("5m", description="Controls how long the model will stay loaded into memory following the request (default: '5m').")
    context: Optional[List[int]] = Field(None, description="The context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory (deprecated).")


class OllamaModelDetails(BaseModel):
    """
    Details about an Ollama model.

    Attributes:
        parent_model (Optional[str]): The parent model name if this is a derived model.
        format (Optional[str]): The format of the model (e.g., "json").
        family (Optional[str]): The family of the model (e.g., "llama").
        families (List[str]): A list of families the model belongs to.
        parameter_size (Optional[str]): The size of the model's parameters (e.g., "7B", "3B").
        quantization_level (Optional[str]): The quantization level of the model (e.g., "Q4_0").
    """
    parent_model: Optional[str] = Field(None, description="The parent model name if this is a derived model.")
    format: Optional[str] = Field(None, description="The format of the model (e.g., 'json').")
    family: Optional[str] = Field(None, description="The family of the model (e.g., 'llama').")
    families: List[str] = Field(default_factory=list, description="A list of families the model belongs to.")
    parameter_size: Optional[str] = Field(None, description="The size of the model's parameters (e.g., '7B', '3B').")
    quantization_level: Optional[str] = Field(None, description="The quantization level of the model (e.g., 'Q4_0').")


class OllamaModel(BaseModel):
    """
    Represents an Ollama model.

    Attributes:
        name (Optional[str]): The full name of the model (e.g., "llama3:latest").
        model (Optional[str]): The base model name.
        digest (Optional[str]): The SHA256 digest of the model.
        size (Optional[int]): The size of the model in bytes.
        modified_at (Optional[str]): The timestamp when the model was last modified (ISO 8601 format).
        details (OllamaModelDetails): Detailed information about the model.
    """
    name: Optional[str] = Field(None, description="The full name of the model (e.g., 'llama3:latest').")
    model: Optional[str] = Field(settings.llm_model, description="The base model name.")
    digest: Optional[str] = Field("sha256:placeholder_digest", description="The SHA256 digest of the model.")
    size: Optional[int] = Field(0, description="The size of the model in bytes.")
    modified_at: Optional[str] = Field("0", description="The timestamp when the model was last modified (ISO 8601 format).")
    details: OllamaModelDetails = Field(..., description="Detailed information about the model.")


class OllamaTagsResponse(BaseModel):
    """
    Represents the response for listing Ollama models (tags).

    Attributes:
        models (List[OllamaModel]): A list of available Ollama models.
    """
    models: List[OllamaModel] = Field(..., description="A list of available Ollama models.")


class OllamaEmbedRequest(BaseModel):
    """
    Represents an embedding generation request to the Ollama API.

    Attributes:
        model (str): The model name to use for generating embeddings.
        input (Union[str, List[str]]): Text or list of texts to generate embeddings for.
        truncate (Optional[bool]): Truncates the end of each input to fit within context length.
                                   Returns error if `false` and context length is exceeded. Defaults to `true`.
        options (Optional[Dict[str, Any]]): Additional model parameters listed in the documentation
                                            for the Modelfile such as `temperature`.
        keep_alive (Optional[str]): Controls how long the model will stay loaded into memory
                                    following the request (default: `5m`).
    """
    model: Optional[str] = Field(settings.openai_model, description="The model name to use for generating embeddings.")
    input: Union[str, List[str]] = Field(..., description="Text or list of texts to generate embeddings for.")
    truncate: Optional[bool] = Field(True, description="Truncates the end of each input to fit within context length. Returns error if `false` and context length is exceeded. Defaults to `true`.")
    options: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters listed in the documentation for the Modelfile such as `temperature`.")
    keep_alive: Optional[str] = Field("5m", description="Controls how long the model will stay loaded into memory following the request (default: `5m`).")


class OllamaEmbedResponse(BaseModel):
    """
    Represents an embedding generation response from the Ollama API.

    Attributes:
        model (str): The name of the model that generated the embeddings.
        embeddings (List[List[float]]): A list of embedding vectors. Each inner list is an embedding for a single input.
        total_duration (Optional[int]): Time spent generating the response in nanoseconds.
        load_duration (Optional[int]): Time spent loading the model into memory in nanoseconds.
        prompt_eval_count (Optional[int]): Number of tokens in the prompt (or total tokens for multiple inputs).
    """
    model: Optional[str] = Field(settings.embeddings_model, description="The name of the model that generated the embeddings.")
    embeddings: List[List[float]] = Field(..., description="A list of embedding vectors. Each inner list is an embedding for a single input.")
    total_duration: Optional[int] = Field(None, description="Time spent generating the response in nanoseconds.")
    load_duration: Optional[int] = Field(None, description="Time spent loading the model into memory in nanoseconds.")
    prompt_eval_count: Optional[int] = Field(None, description="Number of tokens in the prompt (or total tokens for multiple inputs).")

