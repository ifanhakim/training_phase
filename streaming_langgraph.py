import os
from pathlib import Path
from typing import Annotated

import typesense
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")


class OpenRouterEmbeddings(Embeddings):
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = os.getenv(
            "OPENROUTER_EMBEDDING_MODEL",
            "nvidia/nemotron-3-embed-1b:free",
        )

    def embed_documents(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


embeddings = OpenRouterEmbeddings()

model = ChatOpenAI(
    model="inclusionai/ling-3.0-flash-fin:free",
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.2,
    timeout=60,
    max_retries=1,
)

typesense_client = typesense.Client(
    {
        "nodes": [
            {
                "host": os.getenv("TYPESENSE_HOST", "localhost"),
                "port": os.getenv("TYPESENSE_PORT", "8108"),
                "protocol": os.getenv("TYPESENSE_PROTOCOL", "http"),
            }
        ],
        "api_key": os.getenv("TYPESENSE_API_KEY", "xyz"),
        "connection_timeout_seconds": 10,
    }
)

COLLECTION_NAME = os.getenv("TYPESENSE_COLLECTION", "knowledge_base")
COLLECTION = None


class RAGState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    context: str
    answer: str


def get_question(state):
    messages = state.get("messages", [])
    if not messages:
        return ""

    last_message = messages[-1]
    if isinstance(last_message, BaseMessage):
        return last_message.content
    if isinstance(last_message, dict):
        return last_message.get("content", "")
    return str(last_message)


def get_or_create_collection():
    schema = {
        "name": COLLECTION_NAME,
        "fields": [
            {"name": "content", "type": "string"},
            {"name": "source", "type": "string"},
            {"name": "chunk_id", "type": "int32"},
            {
                "name": "embedding",
                "type": "float[]",
                "num_dim": len(embeddings.embed_query("dimension check")),
            },
        ],
    }

    try:
        typesense_client.collections[COLLECTION_NAME].retrieve()
    except typesense.exceptions.ObjectNotFound:
        typesense_client.collections.create(schema)

    return typesense_client.collections[COLLECTION_NAME]


def load_documents(folder_name="knowledge_base"):
    documents = []
    folder_path = Path(folder_name)

    for file_path in folder_path.glob("*.txt"):
        documents.append(
            {
                "filename": file_path.name,
                "content": file_path.read_text(encoding="utf-8"),
            }
        )

    return documents


def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def add_documents_to_typesense(collection):
    documents = load_documents()
    number_of_chunks = 0

    for document in documents:
        chunks = split_text(document["content"])
        vectors = embeddings.embed_documents(chunks)

        for chunk_id, (chunk, vector) in enumerate(zip(chunks, vectors)):
            collection.documents.upsert(
                {
                    "id": f"{document['filename']}_{chunk_id}",
                    "content": chunk,
                    "source": document["filename"],
                    "chunk_id": chunk_id,
                    "embedding": vector,
                }
            )
            number_of_chunks += 1

    print(f"Berhasil menyimpan {number_of_chunks} chunk ke Typesense.")


def search(collection, question, number_of_results=3):
    question_vector = embeddings.embed_query(question)
    vector_query = f"embedding:({question_vector}, k:{number_of_results})"

    results = typesense_client.multi_search.perform(
        {
            "searches": [
                {
                    "collection": COLLECTION_NAME,
                    "q": "*",
                    "query_by": "content",
                    "vector_query": vector_query,
                    "per_page": number_of_results,
                }
            ]
        }
    )

    return [hit["document"] for hit in results["results"][0]["hits"]]


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Jawab pertanyaan berdasarkan konteks berikut. Jika jawabannya tidak ada di konteks, katakan bahwa informasi perlu dikonfirmasi. Jawab dalam bahasa Indonesia dengan singkat dan jelas.",
        ),
        (
            "human",
            "Konteks:\n{context}\n\nPertanyaan:\n{question}",
        ),
    ]
)

answer_chain = prompt | model


def _normalize_stream_chunk(chunk):
    if hasattr(chunk, "content"):
        content = chunk.content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)

    if isinstance(chunk, str):
        return chunk

    return str(chunk)


def retrieve_context(state):
    question = get_question(state)
    if not COLLECTION:
        raise RuntimeError("Collection Typesense belum diinisialisasi.")

    documents = search(COLLECTION, question)
    if not documents:
        context = "Tidak ada informasi yang relevan di basis pengetahuan."
    else:
        context = "\n\n".join(doc["content"] for doc in documents)

    return {"question": question, "context": context}


def generate_answer(state):
    question = state.get("question") or get_question(state)
    context = state.get("context", "")

    chunks = []
    for chunk in answer_chain.stream({"question": question, "context": context}):
        text = _normalize_stream_chunk(chunk)
        if text:
            chunks.append(text)

    return {"answer": "".join(chunks)}


workflow = StateGraph(RAGState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_answer", generate_answer)
workflow.add_edge(START, "retrieve_context")
workflow.add_edge("retrieve_context", "generate_answer")
workflow.add_edge("generate_answer", END)
app = workflow.compile()


def count_documents(collection):
    result = collection.documents.search({
        "q": "*",
        "query_by": "content",
        "per_page": 0,
    })
    return result["found"]


def run_streaming_workflow(question):
    print("AI:", end="", flush=True)
    final_state = {}

    for update in app.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        if "answer" in update:
            print(update["answer"], end="", flush=True)
            final_state = update

    print()
    return final_state.get("answer", "")


def main():
    global COLLECTION

    try:
        COLLECTION = get_or_create_collection()
    except Exception as error:
        print("Tidak bisa terhubung ke Typesense.")
        print("Pastikan Typesense berjalan di localhost:8108 atau ubah .env.")
        print(f"Detail: {error}")
        return

    number_of_documents = count_documents(COLLECTION)

    if number_of_documents == 0:
        print("Typesense masih kosong. Membaca knowledge_base...")
        add_documents_to_typesense(COLLECTION)
    else:
        print(f"Typesense sudah berisi {number_of_documents} dokumen.")

    print('RAG Typesense + LangGraph Streaming (ketik "exit" untuk keluar)')

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Sampai jumpa!")
            break

        if question:
            run_streaming_workflow(question)


if __name__ == "__main__":
    main()
