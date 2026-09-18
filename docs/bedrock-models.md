# Bedrock Model Detection

## Overview

The AWS Bedrock router at `/bedrock/model` accepts any `model_id` path parameter. Two functions detect the model family from the ID string to select the correct request parser and response format.

**Source**: `ovos_persona_server/aws_bedrock.py`

## Request Parsing — `_extract_messages(body, model_id)`

`_extract_messages` — `aws_bedrock.py:23`

Converts any Bedrock invoke body to the internal `[{"role": ..., "content": ...}]` format. Detection is done by inspecting the request body fields (not the `model_id`):

| Condition | Format detected | Handling |
| :--- | :--- | :--- |
| `"messages"` key present | Anthropic / Bedrock Converse | Iterates messages; content blocks (list of `{"text":...}`) are joined with spaces; optional `"system"` prepended |
| `"prompt"` key present | Llama / generic | Single user message |
| `"inputText"` key present | Amazon Titan | Single user message |
| `"message"` key present | Cohere Command | Optional `chat_history` + user message |
| None of the above | Unknown | `json.dumps(body)` as user message |

## Response Building — `_build_response(text, model_id, body)`

`_build_response` — `aws_bedrock.py:84`

Selects the response schema based on `model_id.lower()`:

| `model_id` prefix | Response schema |
| :--- | :--- |
| `anthropic.claude` | `{"id":...,"type":"message","role":"assistant","content":[{"type":"text","text":"..."}],"model":...,"stop_reason":"end_turn","usage":{...}}` |
| `meta.llama` | `{"generation":"...","prompt_token_count":0,"generation_token_count":N,"stop_reason":"stop"}` |
| `amazon.titan` | `{"inputTextTokenCount":0,"results":[{"tokenCount":N,"outputText":"...","completionReason":"FINISH"}]}` |
| `cohere.command` | `{"id":...,"generations":[{"id":...,"text":"...","finish_reason":"COMPLETE"}]}` |
| anything else | `{"outputText":"...","generated_text":"..."}` |

## Endpoints

### `POST /bedrock/model/{model_id}/invoke`

Accepts any Bedrock body. Detects format, calls `persona.chat()`, returns model-specific response.

### `POST /bedrock/model/{model_id}/invoke-with-response-stream`

Same detection, calls `persona.stream()`, returns SSE stream with `"outputText"` chunks. See [streaming.md](streaming.md).

### `POST /bedrock/model/{model_id}/converse`

Uses the typed `BedrockConverseRequest` schema (`aws_bedrock.py:201`) — a unified message format that works across all model families. Content is always a list of `BedrockConverseContentBlock` objects with a `text` field.

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": [{"text": "Hello"}]}
  ],
  "system": [{"text": "You are helpful."}]
}
```

**Response**:
```json
{
  "output": {
    "message": {
      "role": "assistant",
      "content": [{"text": "..."}]
    }
  },
  "stopReason": "end_turn",
  "usage": {"inputTokens": N, "outputTokens": M, "totalTokens": N+M}
}
```
