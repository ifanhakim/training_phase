import json
import os
from pathlib import Path

import typesense
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from openai import OpenAI


load_dotenv()

FAQ_FILE = Path("faqs_extend_no_split.jsonl")
COLLECTION_NAME = "faq_agent_self_consistency"
SEMANTIC_WEIGHT = 0.3
TEXT_WEIGHT = 0.7

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")


class OpenRouterEmbeddings(Embeddings):
    """Adapter embedding OpenRouter agar bisa dipakai oleh LangChain."""

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
    model=os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-fin:free"),
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.5,
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


def get_or_create_collection():
    schema = {
        "name": COLLECTION_NAME,
        "fields": [
            {"name": "question", "type": "string"},
            {"name": "answer", "type": "string"},
            {"name": "embedding", "type": "float[]", "num_dim": 2048},
        ],
    }

    try:
        typesense_client.collections[COLLECTION_NAME].retrieve()
    except typesense.exceptions.ObjectNotFound:
        typesense_client.collections.create(schema)

    return typesense_client.collections[COLLECTION_NAME]


def load_faqs():
    faqs = []

    with FAQ_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            faqs.append(json.loads(line))

    return faqs


def count_documents(collection):
    result = collection.documents.search(
        {"q": "*", "query_by": "question", "per_page": 0}
    )
    return result["found"]


def index_faqs(collection):
    faqs = load_faqs()
    questions = [faq["prompt"] for faq in faqs]
    vectors = embeddings.embed_documents(questions)

    documents = []
    for index, (faq, vector) in enumerate(zip(faqs, vectors)):
        documents.append(
            {
                "id": f"faq_agent_{index}",
                "question": faq["prompt"],
                "answer": faq["completion"],
                "embedding": vector,
            }
        )

    typesense_client.collections[COLLECTION_NAME].documents.import_(
        documents, {"action": "upsert"}
    )
    print(f"Berhasil mengindeks {len(documents)} FAQ ke {COLLECTION_NAME}.")


query_prompt = ChatPromptTemplate.from_template(
    """Ubah pertanyaan pengguna menjadi query pencarian FAQ yang singkat dan jelas.
Pertahankan maksud asli dan gunakan bahasa Indonesia.

Pertanyaan pengguna: {question}

Kembalikan hanya query hasil perbaikan, tanpa penjelasan tambahan."""
)
query_chain = query_prompt | model | StrOutputParser()


def improve_query(question):
    try:
        improved = query_chain.invoke({"question": question}).strip()
        return improved or question
    except Exception:
        return question


def hybrid_search(collection, question, number_of_results=3):
    improved_question = improve_query(question)
    question_vector = embeddings.embed_query(improved_question)
    vector_query = (
        f"embedding:({question_vector}, k:{number_of_results}, "
        f"alpha:{SEMANTIC_WEIGHT})"
    )

    response = typesense_client.multi_search.perform(
        {
            "searches": [
                {
                    "collection": COLLECTION_NAME,
                    "q": improved_question,
                    "query_by": "question,answer",
                    "vector_query": vector_query,
                    "per_page": number_of_results,
                }
            ]
        }
    )

    hits = response["results"][0]["hits"]
    return improved_question, [hit["document"] for hit in hits]


@tool
def search_faq(question: str) -> str:
    """Cari FAQ memakai hybrid search: semantic 30% dan text match 70%."""
    collection = get_or_create_collection()
    improved_question, documents = hybrid_search(collection, question)

    if not documents:
        return "Tidak ada FAQ yang relevan ditemukan."

    context = "\n\n".join(
        f"Pertanyaan FAQ: {document['question']}\n"
        f"Jawaban FAQ: {document['answer']}"
        for document in documents
    )
    return (
        f"Query yang diperbaiki: {improved_question}\n"
        f"Bobot pencarian: semantic 30%, text match 70%\n\n{context}"
    )


agent = create_agent(
    model=model,
    tools=[search_faq],
    system_prompt=(
        "Anda adalah customer service yang ramah. "
        "Selalu gunakan tool search_faq untuk mencari informasi. "
        "Jawab hanya berdasarkan hasil FAQ. "
        "Jika tidak ada hasil yang relevan, katakan bahwa informasi belum tersedia. "
        "Buat tiga kandidat jawaban secara independen, bandingkan faktanya, "
        "lalu tampilkan hanya jawaban final yang paling konsisten."
    ),
    name="faq_agent_typesense",
)


def ask_agent(question):
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content
    except Exception as error:
        if "free-models-per-day" in str(error):
            return "Kuota model gratis OpenRouter hari ini sudah habis."
        return f"Agent gagal memproses pertanyaan: {error}"


def show_graph():
    graph = agent.get_graph()
    print("Node dalam graph:", list(graph.nodes))
    print("Edge dalam graph:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}")


if __name__ == "__main__":
    try:
        collection = get_or_create_collection()
        number_of_documents = count_documents(collection)

        if number_of_documents == 0:
            print("Collection agent masih kosong. Mulai indexing FAQ...")
            index_faqs(collection)
        else:
            print(
                f"Collection {COLLECTION_NAME} sudah berisi "
                f"{number_of_documents} FAQ."
            )

        print(f"FAQ yang tersedia: {len(load_faqs())}")
        show_graph()
        print('\nFAQ Agent (ketik \"exit\" untuk keluar)')

        while True:
            question = input("You: ").strip()

            if question.lower() == "exit":
                print("Sampai jumpa!")
                break

            if question:
                print("AI:", ask_agent(question))
    except Exception as error:
        print(f"Program gagal dijalankan: {error}")
