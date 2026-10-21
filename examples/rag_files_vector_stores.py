"""Upload documents, build a vector store, and search it — using the OpenAI SDK.

Run a RAG-capable server first (see examples/README.md), then:

    python examples/rag_files_vector_stores.py
"""
from openai import OpenAI

BASE_URL = "http://localhost:8337/openai/v1"

DOCS = {
    "cats.txt": b"the cat sat on the mat. cats are fluffy animals that purr.",
    "python.txt": b"python is a programming language for data science and the web.",
    "moon.txt": b"the moon orbits the earth and affects ocean tides.",
}
QUERY = "fluffy animal sitting on a mat"


def main() -> None:
    client = OpenAI(base_url=BASE_URL, api_key="unused")

    # 1. Upload the documents
    file_ids = {}
    for name, body in DOCS.items():
        f = client.files.create(file=(name, body), purpose="assistants")
        file_ids[f.id] = name
        print(f"uploaded {name} -> {f.id}")

    # 2. Create a vector store and attach the files (chunked + embedded server-side)
    store = client.vector_stores.create(name="example-kb")
    print(f"vector store: {store.id}")
    for fid in file_ids:
        client.vector_stores.files.create(vector_store_id=store.id, file_id=fid)

    # 3. Search by similarity
    results = client.vector_stores.search(
        vector_store_id=store.id, query=QUERY, max_num_results=3
    )
    print(f"\nquery: {QUERY!r}")
    for rank, r in enumerate(results.data, 1):
        print(f"  {rank}. {file_ids.get(r.file_id, r.file_id)}  score={r.score:.4f}")


if __name__ == "__main__":
    main()
