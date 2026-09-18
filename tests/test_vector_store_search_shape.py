"""Vector-store search hits carry ``content`` as OpenAI text parts.

The OpenAI API returns each hit's ``content`` as ``[{"type": "text", "text": ...}]``
and the official SDK types it that way; a bare string breaks
``client.vector_stores.search(...).data[0].content[0].text``.
"""
from ovos_persona_server.schemas.openai_vectorstore import SearchResultChunk, VectorStoreSearchResponse
from ovos_persona_server.vector_stores import _search_hit


def test_search_hit_wraps_text_in_parts():
    hit = _search_hit({"file_id": "f1", "filename": "cats.txt", "content": "cats are fluffy"}, 0.9)
    assert hit.model_dump()["content"] == [{"type": "text", "text": "cats are fluffy"}]
    assert hit.file_id == "f1" and hit.filename == "cats.txt" and hit.score == 0.9 and hit.type == "file_search"


def test_response_matches_openai_sdk_type():
    from openai.types.vector_store_search_response import VectorStoreSearchResponse as SDKHit
    body = VectorStoreSearchResponse(data=[_search_hit({"file_id": "f1", "content": "x"}, 0.5)]).model_dump()
    parsed = SDKHit.model_validate(body["data"][0])
    assert parsed.content[0].text == "x" and parsed.content[0].type == "text"
