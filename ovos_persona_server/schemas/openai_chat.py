"""
Pydantic models for OpenAI-compatible Chat and Completions endpoints.

This module defines the data structures for requests and responses
according to the OpenAI API specification, ensuring compatibility.
It includes models for chat completions, legacy completions, and their
streaming variants.
"""
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Literal, Annotated
from ovos_persona_server.config import settings
from pydantic import BaseModel, Field, RootModel, ConfigDict

# --- Shared Types ---

# For easy typing of metadata, which is a dictionary with string keys and string values
Metadata = Dict[str, str]


# --- Enums ---

class FinishReason(str, Enum):
    """The reason the model stopped generating tokens."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"
    FUNCTION_CALL = "function_call"  # Deprecated


class Role(str, Enum):
    """The role of the author of a message."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated
    DEVELOPER = "developer"  # Added based on OpenAPI spec


class ChatCompletionResponseFormatType(str, Enum):
    """The type of response format."""
    TEXT = "text"
    JSON_OBJECT = "json_object"


class ChatCompletionToolType(str, Enum):
    """The type of tool being used."""
    FUNCTION = "function"
    FILE_SEARCH = "file_search"


class ChatCompletionToolChoiceType(str, Enum):
    """The type of tool choice."""
    FUNCTION = "function"


# --- Chat Completion Models ---

class ChatCompletionFunction(BaseModel):
    """
    Deprecated.
    A chat completion function.
    """
    model_config = ConfigDict(deprecated='This model is deprecated.') # Pydantic V2 way to mark deprecated

    description: Optional[str] = Field("", description="The description of the function.")
    name: str = Field(..., description="The name of the function to call.")
    parameters: Optional[Dict[str, Any]] = Field({}, description="The parameters of the function, described as a JSON Schema object.")


class ChatCompletionToolFunction(BaseModel):
    """
    The definition of the function that the model can call.
    """
    description: Optional[str] = Field("", description="The description of the function.")
    name: str = Field(..., description="The name of the function to be called. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64.")
    parameters: Optional[Dict[str, Any]] = Field({}, description="The parameters the functions accepts, described as a JSON Schema object. See the [guide](/docs/guides/function-calling) for examples, and the [JSON Schema reference](https://json-schema.org/understanding-json-schema/) for documentation about the format. OpenAI models can only call functions with basic types, i.e. non-nested objects.")


class ChatCompletionTool(BaseModel):
    """
    A tool the model may call.
    """
    type: ChatCompletionToolType = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ChatCompletionToolFunction = Field(..., description="The function definition.")


class ChatCompletionResponseFormat(BaseModel):
    """
    An object specifying the format that the model must output.
    Used to enable JSON mode.
    """
    type: ChatCompletionResponseFormatType = Field(ChatCompletionResponseFormatType.TEXT, description="Must be one of `text` or `json_object`.")


class ChatCompletionStreamOptions(BaseModel):
    """
    Options for streaming responses.
    """
    include_usage: Optional[bool] = Field(False, description="If set, an additional chunk will be appended to the stream with the `usage` field set to the completion usage statistics.")


class ChatCompletionNamedToolChoiceFunction(BaseModel):
    """
    Specifies a function that the model should call.
    """
    name: str = Field(..., description="The name of the function to call.")


class ChatCompletionNamedToolChoice(BaseModel):
    """
    Controls which (if any) function is called by the model.
    `none` means the model will not call a function and instead generates a message.
    `auto` means the model can pick between generating a message or calling a function.
    `tool` means the model will call a tool.
    Specifying a particular function via `{"type": "function", "function": {"name": "my_function"}}`
    forces the model to call that function.
    """
    type: ChatCompletionToolChoiceType = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ChatCompletionNamedToolChoiceFunction = Field(..., description="Specifies the function that the model should call.")


class ChatCompletionToolChoiceOption(str, Enum):
    """
    Controls which (if any) function is called by the model.
    `none` means the model will not call a function and instead generates a message.
    `auto` means the model can pick between generating a message or calling a function.
    `tool` means the model will call a tool.
    """
    NONE = "none"
    AUTO = "auto"
    TOOL = "tool"


class ChatCompletionMessageToolCallFunction(BaseModel):
    """
    The function that the model called.
    """
    name: str = Field(..., description="The name of the function to call.")
    arguments: str = Field(..., description="The arguments to call the function with, as generated by the model in JSON format. Note that the model may generate invalid JSON.")


class ChatCompletionMessageToolCall(BaseModel):
    """
    A tool call generated by the model, which can be a function call.
    """
    id: str = Field(..., description="The ID of the tool call.")
    type: Literal["function"] = Field(..., description="The type of the tool. Currently, only `function` is supported.")
    function: ChatCompletionMessageToolCallFunction = Field(..., description="The function that the model called.")


class ChatCompletionMessageFunctionCall(BaseModel):
    """
    Deprecated.
    The name and arguments of a function that should be called, as generated by the model.
    """
    model_config = ConfigDict(deprecated='This model is deprecated.') # Pydantic V2 way to mark deprecated

    arguments: str = Field(..., description="The arguments to call the function with, as generated by the model in JSON format. Note that the model may generate invalid JSON.")
    name: str = Field(..., description="The name of the function to call.")


class ChatCompletionLogprobsContent(BaseModel):
    """
    Log probability information for a single token generated by the model.
    """
    token: str = Field(..., description="The token.")
    logprob: float = Field(..., description="The log probability of the token.")
    bytes: Optional[List[int]] = Field([], description="A list of integers representing the UTF-8 bytes representation of the token. Useful for visualising a token correctly in tools that do not handle UTF-8 well.")
    top_logprobs: List[Dict[str, float]] = Field(..., description="A list of the most likely tokens and their log probabilities at this token position.")


class ChatCompletionLogprobs(BaseModel):
    """
    Log probability information for the choice.
    """
    content: Optional[List[ChatCompletionLogprobsContent]] = Field([], description="A list of message content tokens with their log probabilities.")


class ChatCompletionMessageContentPartText(BaseModel):
    """
    A text content part.
    """
    type: Literal["text"] = Field(..., description="The type of the content part.")
    text: str = Field(..., description="The text content.")


class ChatCompletionMessageContentPartImageUrl(BaseModel):
    """
    The URL of the image.
    """
    url: str = Field(..., description="The URL of the image.")
    detail: Optional[Literal["auto", "low", "high"]] = Field("auto", description="Specifies the detail level of the image. `low` will generate a low-resolution image, `high` will generate a high-resolution image, and `auto` will automatically determine the detail level. Defaults to `auto`.")


class ChatCompletionMessageContentPartImage(BaseModel):
    """
    An image content part.
    """
    type: Literal["image_url"] = Field(..., description="The type of the content part.")
    image_url: ChatCompletionMessageContentPartImageUrl = Field(..., description="The image URL content.")


class ChatCompletionMessageContentPartImageFile(BaseModel):
    """
    An image file content part.
    """
    type: Literal["image_file"] = Field(..., description="The type of the content part.")
    image_file: Dict[str, str] = Field(..., description="The image file content. Currently only `file_id` is supported.")


class ChatCompletionRequestMessageContent(RootModel[Union[
    str,
    List[
        Union[
            ChatCompletionMessageContentPartText,
            ChatCompletionMessageContentPartImage,
            ChatCompletionMessageContentPartImageFile
        ]
    ]
]]):
    """
    The contents of the message.
    """


class ChatCompletionRequestMessage(BaseModel):
    """
    A message part of a chat conversation.
    """
    role: Role = Field(..., description="The role of the message's author.")
    content: Optional[ChatCompletionRequestMessageContent] = Field(None, description="The contents of the message.")
    name: Optional[str] = Field("", description="An optional name for the participant. Provides the model with a name for the participant in the conversation. Can be used to differentiate between multiple users or assistants.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field([], description="The tool calls generated by the model, if any.")
    function_call: Optional[ChatCompletionMessageFunctionCall] = Field(None, description="Deprecated. The name and arguments of a function that should be called, as generated by the model.")


class ChatCompletionRequestSystemMessage(BaseModel):
    """
    A system message.
    """
    role: Literal["system"] = Field(..., description="The role of the message's author, in this case `system`.")
    content: str = Field(..., description="The contents of the system message.")
    name: Optional[str] = Field("", description="An optional name for the participant. Provides the model with a name for the participant in the conversation. Can be used to differentiate between multiple users or assistants.")


class ChatCompletionRequestUserMessage(BaseModel):
    """
    A user message.
    """
    role: Literal["user"] = Field(..., description="The role of the message's author, in this case `user`.")
    content: ChatCompletionRequestMessageContent = Field(..., description="The contents of the user message.")
    name: Optional[str] = Field("", description="An optional name for the participant. Provides the model with a name for the participant in the conversation. Can be used to differentiate between multiple users or assistants.")


class ChatCompletionRequestToolMessage(BaseModel):
    """
    A tool message.
    """
    role: Literal["tool"] = Field(..., description="The role of the message's author, in this case `tool`.")
    content: str = Field(..., description="The contents of the tool message.")
    tool_call_id: str = Field(..., description="Tool call that this message is responding to.")


class ChatCompletionRequestFunctionMessage(BaseModel):
    """
    Deprecated.
    A function message.
    """
    model_config = ConfigDict(deprecated='This model is deprecated.') # Pydantic V2 way to mark deprecated

    role: Literal["function"] = Field(..., description="The role of the message's author, in this case `function`.")
    content: Optional[str] = Field("", description="The contents of the function message.")
    name: str = Field(..., description="The name of the function to call.")


class ChatCompletionRequestDeveloperMessage(BaseModel):
    """
    A developer message.
    """
    role: Literal["developer"] = Field(..., description="The role of the message's author, in this case `developer`.")
    content: ChatCompletionRequestMessageContent = Field(..., description="The contents of the developer message.")
    name: Optional[str] = Field("", description="An optional name for the participant. Provides the model with a name for the participant in the conversation. Can be used to differentiate between multiple users or assistants.")


class ChatCompletionRequestAssistantMessage(BaseModel):
    """
    An assistant message.
    """
    role: Literal["assistant"] = Field(..., description="The role of the message's author, in this case `assistant`.")
    content: Optional[str] = Field("", description="The contents of the assistant message.")
    name: Optional[str] = Field("", description="An optional name for the participant. Provides the model with a name for the participant in the conversation. Can be used to differentiate between multiple users or assistants.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field([], description="The tool calls generated by the model, if any.")
    function_call: Optional[ChatCompletionMessageFunctionCall] = Field(None, description="Deprecated. The name and arguments of a function that should be called, as generated by the model.")


class ChatCompletionResponseMessage(BaseModel):
    """
    A message generated by the model.
    """
    role: Role = Field(..., description="The role of the author of this message.")
    content: Optional[str] = Field("", description="The contents of the message.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field([], description="The tool calls generated by the model, if any.")
    function_call: Optional[ChatCompletionMessageFunctionCall] = Field(None, description="Deprecated. The name and arguments of a function that should be called, as generated by the model.")


class ChatCompletionChoice(BaseModel):
    """
    A choice in the chat completion response.
    """
    finish_reason: FinishReason = Field(..., description="The reason the model stopped generating tokens.")
    index: int = Field(..., description="The index of the choice in the list of choices.")
    message: ChatCompletionResponseMessage = Field(..., description="A message generated by the model.")
    logprobs: Optional[ChatCompletionLogprobs] = Field(None, description="Log probability information for the choice.")


class ChatCompletionStreamChoice(BaseModel):
    """
    A choice in the chat completion stream response.
    """
    finish_reason: Optional[FinishReason] = Field(None, description="The reason the model stopped generating tokens.")
    index: int = Field(..., description="The index of the choice in the list of choices.")
    delta: ChatCompletionResponseMessage = Field(..., description="A chat completion message delta generated by the model.")
    logprobs: Optional[ChatCompletionLogprobs] = Field(None, description="Log probability information for the choice.")


class CompletionUsage(BaseModel):
    """
    Usage statistics for the completion request.
    """
    prompt_tokens: int = Field(..., description="The number of tokens in the prompt.")
    completion_tokens: Optional[int] = Field(None, description="The number of tokens in the completion. This is only present for completion endpoints.")
    total_tokens: int = Field(..., description="The total number of tokens used in the request (prompt + completion).")


class CreateChatCompletionRequest(BaseModel):
    """
    Request body for creating a chat completion.
    """
    messages: List[
        Union[
            ChatCompletionRequestSystemMessage,
            ChatCompletionRequestUserMessage,
            ChatCompletionRequestAssistantMessage,  # This one is not explicitly defined in the OpenAPI, but used in examples. Adding for completeness.
            ChatCompletionRequestToolMessage,
            ChatCompletionRequestFunctionMessage,  # Deprecated
            ChatCompletionRequestDeveloperMessage
        ]
    ] = Field([ChatCompletionRequestMessage(role="user", content="hello world")], description="A list of messages comprising the conversation so far. [Example Python code](https://cookbook.openai.com/examples/how_to_format_messages_for_chat_completions).")
    model: Optional[str] = Field(settings.llm_model, description="ID of the model to use. See the [model endpoint compatibility](/docs/models/model-endpoint-compatibility) table for details on which models work with the Chat API.")
    frequency_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    logit_bias: Optional[Dict[str, int]] = Field({}, description="Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase the likelihood of selection by a noticeable amount.")
    logprobs: Optional[bool] = Field(False, description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`.")
    top_logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = Field(None, description="An integer between 0 and 5 specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.")
    max_tokens: Optional[int] = Field(None, description="The maximum number of [tokens](/tokenizer) that can be generated in the chat completion. The token count of your prompt plus `max_tokens` cannot exceed the model's context length.")
    n: Optional[Annotated[int, Field(ge=1, le=128)]] = Field(1, description="How many chat completion choices to generate for each input message. Note that you will be charged per token for the messages sent and received. If you want to generate more than 1 choice, we recommend using `n`=1 and making multiple requests to parallelize the beam search over the API instead.")
    presence_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    response_format: Optional[ChatCompletionResponseFormat] = Field(None, description="An object specifying the format that the model must output. Used to enable JSON mode.")
    seed: Optional[Annotated[int, Field(ge=-9223372036854775808, le=9223372036854775807)]] = Field(42, description="This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result. Determinism is not guaranteed.")
    stop: Optional[Union[str, List[str]]] = Field("", description="Up to 4 sequences where the API will stop generating further tokens.")
    stream: Optional[bool] = Field(False, description="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.")
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = Field(1.0, description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.")
    top_p: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(1.0, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.")
    tools: Optional[List[ChatCompletionTool]] = Field([], description="A list of tools the model may call. Currently, only functions are supported. Use this to provide a list of functions the model may generate JSON inputs for.")
    tool_choice: Optional[Union[ChatCompletionToolChoiceOption, ChatCompletionNamedToolChoice]] = Field(None, description="Controls which (if any) function is called by the model. `none` means the model will not call a function and instead generates a message. `auto` means the model can pick between generating a message or calling a function. Specifying a particular function via `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}` forces the model to call that function.")
    user: Optional[str] = Field("", description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.")
    parallel_tool_calls: Optional[bool] = Field(True, description="If `true`, the model may make multiple tool calls in a single response. Defaults to `true`.")
    stream_options: Optional[ChatCompletionStreamOptions] = Field(None, description="Options for streaming responses.")
    response_metadata: Optional[Metadata] = Field(None, alias="metadata", description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values will be truncated to 512 characters. ")


class CreateChatCompletionResponse(BaseModel):
    """
    Response body for a chat completion.
    """
    id: str = Field(..., description="A unique identifier for the chat completion.")
    choices: List[ChatCompletionChoice] = Field(..., description="A list of chat completion choices. Can be more than one if `n` is greater than 1.")
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the chat completion was created.")
    model: Optional[str] = Field(settings.llm_model, description="The model used for the chat completion.")
    system_fingerprint: Optional[str] = Field("", description="This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.")
    object: Literal["chat.completion"] = Field("chat.completion", description="The object type, which is always `chat.completion`.")
    usage: Optional[CompletionUsage] = Field(None, description="Usage statistics for the completion request.")
    request_id: Optional[str] = Field("", description="The ID of the request.")
    tool_choice: Optional[Union[ChatCompletionToolChoiceOption, ChatCompletionNamedToolChoice]] = Field(None, description="Controls which (if any) function is called by the model. `none` means the model will not call a function and instead generates a message. `auto` means the model can pick between generating a message or calling a function. Specifying a particular function via `{\"type\": \"function\", \"function\": {\"name\": \"my_function\"}}` forces the model to call that function.")
    seed: Optional[int] = Field(42, description="The seed used for the completion.")
    top_p: Optional[float] = Field(None, description="The top_p value used for the completion.")
    temperature: Optional[float] = Field(None, description="The temperature value used for the completion.")
    presence_penalty: Optional[float] = Field(None, description="The presence penalty value used for the completion.")
    frequency_penalty: Optional[float] = Field(None, description="The frequency penalty value used for the completion.")
    input_user: Optional[str] = Field("", description="The user ID provided in the request.")
    service_tier: Optional[str] = Field("", description="The service tier used for the completion.")
    tools: Optional[List[ChatCompletionTool]] = Field([], description="A list of tools the model may call. Currently, only functions are supported. Use this to provide a list of functions the model may generate JSON inputs for.")
    metadata: Optional[Metadata] = Field(None, description="Set of 16 key-value pairs that can be attached to an object. This can be useful for storing additional information about the object in a structured format. Keys can be a maximum of 64 characters long and values will be truncated to 512 characters. ")
    response_format: Optional[ChatCompletionResponseFormat] = Field(None, description="An object specifying the format that the model must output. Used to enable JSON mode.")
    parallel_tool_calls: Optional[bool] = Field(False, description="Whether the model allowed parallel tool calls.")


class CreateChatCompletionStreamResponse(BaseModel):
    """
    Response body for a streamed chat completion.
    """
    id: str = Field(..., description="A unique identifier for the chat completion.")
    choices: List[ChatCompletionStreamChoice] = Field(..., description="A list of chat completion choices. Can be more than one if `n` is greater than 1.")
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the chat completion was created.")
    model: Optional[str] = Field(settings.llm_model, description="The model used for the chat completion.")
    system_fingerprint: Optional[str] = Field("", description="This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.")
    object: Literal["chat.completion.chunk"] = Field("chat.completion.chunk", description="The object type, which is always `chat.completion.chunk`.")
    usage: Optional[CompletionUsage] = Field(None, description="Usage statistics for the completion request. This field is only present when `stream_options.include_usage` is set to `true` and it is the last chunk in the stream.")


class ChatCompletionDeleted(BaseModel):
    """
    Confirmation of a chat completion deletion operation.
    """
    id: str = Field(..., description="The ID of the deleted chat completion.")
    object: Literal["chat.completion.deleted"] = Field("chat.completion.deleted", description="The object type, which is always `chat.completion.deleted`.")
    deleted: bool = Field(..., description="A flag indicating if the chat completion was successfully deleted.")


class ChatCompletionMessageObject(BaseModel):
    """
    Represents a message in a chat completion that has been stored.
    """
    id: str = Field(..., description="The message identifier, which can be referenced in the API endpoints.")
    object: Literal["chat.completion.message"] = Field("chat.completion.message", description="The object type, which is always `chat.completion.message`.")
    created_at: int = Field(..., description="The Unix timestamp (in seconds) for when the message was created.")
    completion_id: str = Field(..., description="The ID of the chat completion this message belongs to.")
    role: Role = Field(..., description="The role of the author of this message.")
    content: Optional[str] = Field("", description="The contents of the message.")
    name: Optional[str] = Field("", description="An optional name for the participant.")
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = Field([], description="The tool calls generated by the model, if any.")
    function_call: Optional[ChatCompletionMessageFunctionCall] = Field(None, description="Deprecated. The name and arguments of a function that should be called, as generated by the model.")
    content_parts: Optional[List[Dict[str, Any]]] = Field([], description="A list of content parts for the message. This field is only present if the message contains image or audio content.")


class ChatCompletionList(BaseModel):
    """
    A list of chat completions.
    """
    object: Literal["list"] = Field("list", description="The object type, which is always `list`.")
    data: List[CreateChatCompletionResponse] = Field(..., description="A list of chat completion objects.")
    first_id: Optional[str] = Field("", description="The ID of the first object in the list.")
    last_id: Optional[str] = Field("", description="The ID of the last object in the list.")
    has_more: bool = Field(False, description="A flag indicating whether there are more objects to retrieve.") # Default to False


class ChatCompletionMessageList(BaseModel):
    """
    A list of messages in a stored chat completion.
    """
    object: Literal["list"] = Field("list", description="The object type, which is always `list`.")
    data: List[ChatCompletionMessageObject] = Field(..., description="A list of message objects.")
    first_id: Optional[str] = Field("", description="The ID of the first object in the list.")
    last_id: Optional[str] = Field("", description="The ID of the last object in the list.")
    has_more: bool = Field(False, description="A flag indicating whether there are more objects to retrieve.") # Default to False


# --- Legacy Completions Models ---

class CreateCompletionRequest(BaseModel):
    """
    Request body for creating a legacy completion.
    """
    model: Optional[str] = Field(settings.llm_model, description="ID of the model to use. You can use the [List models](/docs/api-reference/models/list) API to see all of your available models, or see our [Model overview](/docs/models/overview) for descriptions of them.")
    prompt: Optional[Union[str, List[str], List[int], List[List[int]]]] = Field(None, description="The prompt(s) to generate completions for, encoded as a string, array of strings, array of tokens, or array of token arrays. Note that <|endoftext|> is the default stop token for most models, so your completion will stop there if not otherwise specified. Alternatively, you can use `prompt_file` to upload a file containing prompts.")
    best_of: Optional[Annotated[int, Field(ge=0)]] = Field(1, description="Generates `best_of` completions server-side and returns the \"best\" (the one with the highest log probability per token). Results cannot be streamed. When used with `n`, and `top_logprobs`, the `n` and `top_logprobs` parameters are applied to the `best_of` results, and only then is the \"best\" of them returned. When `best_of` is greater than 1, the `logit_bias` field is not supported.")
    echo: Optional[bool] = Field(False, description="Echo back the prompt in addition to the completion.")
    frequency_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.")
    logit_bias: Optional[Dict[str, int]] = Field({}, description="Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase the likelihood of selection by a noticeable amount.")
    logprobs: Optional[Annotated[int, Field(ge=0, le=5)]] = Field(None, description="Include the log probabilities on the `logprobs` field of the chosen tokens. For example, if `logprobs` is 10, the 10 most likely tokens will be returned, at each step. This parameter is only supported for `gpt-3.5-turbo-instruct` and `babbage-002`.")
    max_tokens: Optional[int] = Field(16, description="The maximum number of [tokens](/tokenizer) to generate in the completion. The token count of your prompt plus `max_tokens` cannot exceed the model's context length.")
    n: Optional[Annotated[int, Field(ge=1)]] = Field(1, description="How many completions to generate for each prompt. Note: Because this parameter generates many completions, it can quickly consume your token quota. Use carefully and ensure that you have reasonable settings for `max_tokens` and `stop`. [Example Python code](https://cookbook.openai.com/examples/how_to_generate_multiple_completions).")
    presence_penalty: Optional[Annotated[float, Field(ge=-2.0, le=2.0)]] = Field(0.0, description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.")
    seed: Optional[Annotated[int, Field(ge=-9223372036854775808, le=9223372036854775807)]] = Field(42, description="If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same `seed` and parameters should return the same result. Determinism is not guaranteed.")
    stop: Optional[Union[str, List[str]]] = Field("", description="Up to 4 sequences where the API will stop generating further tokens. The returned text will not contain the stop sequence.")
    stream: Optional[bool] = Field(False, description="Whether to stream back partial progress. If set, tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]` message.")
    suffix: Optional[str] = Field("", description="The suffix that comes after a completion of inserted text.")
    temperature: Optional[Annotated[float, Field(ge=0.0, le=2.0)]] = Field(1.0, description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. We generally recommend altering this or `top_p` but not both.")
    top_p: Optional[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(1.0, description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.")
    user: Optional[str] = Field("", description="A unique identifier representing your end-user, which can help OpenAI to monitor and detect abuse.")


class CreateCompletionResponse(BaseModel):
    """Response body for a legacy completion."""
    id: str = Field(..., description="A unique identifier for the completion.")
    object: Literal["text_completion"] = Field("text_completion", description="The object type, which is always `text_completion`.")
    created: int = Field(..., description="The Unix timestamp (in seconds) of when the completion was created.")
    model: Optional[str] = Field(settings.llm_model, description="The model used for completion.")
    choices: List[
        Dict[str, Any]
    ] = Field(..., description="The list of completion choices the model generated for the input prompt.")
    usage: Optional[CompletionUsage] = Field(None, description="Usage statistics for the completion request.")
    system_fingerprint: Optional[str] = Field("", description="This fingerprint represents the backend configuration that the model runs with. Can be used in conjunction with the `seed` request parameter to understand when backend changes have been made that might impact determinism.")

