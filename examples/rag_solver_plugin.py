"""Drive a full RAG turn through the ovos-openai-plugin OpenAIRAGSolver.

The solver searches a vector store on this server, injects the retrieved context
into the prompt, and calls the server's /chat/completions endpoint.

    uv pip install ovos-openai-plugin        # provides OpenAIRAGSolver
    python examples/rag_solver_plugin.py     # after building a store (see rag_files_vector_stores.py)
"""
from openai import OpenAI
from ovos_solver_openai_persona.rag import OpenAIRAGSolver

BASE_URL = "http://localhost:8337/openai/v1"
DOCS = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "moon.txt": b"the moon orbits the earth and affects ocean tides.",
}


def build_store() -> str:
    client = OpenAI(base_url=BASE_URL, api_key="unused")
    store = client.vector_stores.create(name="solver-demo")
    for name, body in DOCS.items():
        f = client.files.create(file=(name, body), purpose="assistants")
        client.vector_stores.files.create(vector_store_id=store.id, file_id=f.id)
    return store.id


def main() -> None:
    vector_store_id = build_store()

    rag = OpenAIRAGSolver({
        "api_url": BASE_URL,
        "vector_store_id": vector_store_id,
        "llm_model": "kb-assistant",   # persona name served by this server
        "key": "unused",
        "max_num_results": 3,
    })

    answer = rag.continue_chat(
        [{"role": "user", "content": "what fluffy animal sits on the mat?"}],
        lang="en-us",
    )
    print(answer)


if __name__ == "__main__":
    main()
