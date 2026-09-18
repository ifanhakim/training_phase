import json
import os
from pathlib import Path
from typing import Any, Iterable, Literal

import typesense
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
FAQ_FILE = PROJECT_ROOT / "faqs_extend_no_split.jsonl"
HOSPITAL_FILE = PROJECT_ROOT / "hospitals_prod.json"
MEDICAL_JSON_DIR = PROJECT_ROOT / "data_icd_lab_rad" / "json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")

llm = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-vl:free"),
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.2,
    timeout=60,
    max_retries=1,
)

client = typesense.Client(
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


class RouterState(TypedDict):
    thread_id: str
    turn_id: int
    route: str
    messages: Annotated[list, add_messages]
    summary: str
    recent_messages: list
    response: str
    context: str


class RouteDecision(BaseModel):
    route: Literal["faq", "hospital", "medical"] = Field(
        default="faq",
        description="Pilih sumber pengetahuan yang paling sesuai dengan pertanyaan.",
    )


# ---------------------------------------------------------------------------
# FAQ retrieval
# ---------------------------------------------------------------------------
FAQ_COLLECTION = "faq_router_collection"


class OpenRouterEmbeddings:
    def __init__(self):
        self.client = OpenAI(
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
        )
        self.model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def get_or_create_faq_collection():
    schema = {
        "name": FAQ_COLLECTION,
        "fields": [
            {"name": "question", "type": "string"},
            {"name": "answer", "type": "string"},
            {"name": "embedding", "type": "float[]", "num_dim": 2048},
        ],
    }
    try:
        client.collections[FAQ_COLLECTION].retrieve()
    except typesense.exceptions.ObjectNotFound:
        client.collections.create(schema)
    return client.collections[FAQ_COLLECTION]


faq_embeddings = OpenRouterEmbeddings()


def load_faqs() -> list[dict]:
    if not FAQ_FILE.exists():
        return []
    records: list[dict] = []
    with FAQ_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def count_faqs(collection):
    result = collection.documents.search({"q": "*", "query_by": "question", "per_page": 0})
    return result["found"]


def index_faqs(collection):
    records = load_faqs()
    if not records:
        return

    batch_size = 64
    total_indexed = 0
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        questions = [item.get("prompt", "") for item in batch if item.get("prompt")]
        embeddings = faq_embeddings.embed_documents(questions)
        docs = []
        for idx, item in enumerate(batch):
            if not item.get("prompt"):
                continue
            docs.append(
                {
                    "id": f"faq_{start}_{idx}",
                    "question": item.get("prompt", ""),
                    "answer": item.get("completion", ""),
                    "embedding": embeddings[idx],
                }
            )
        if docs:
            collection.documents.import_(docs, {"action": "upsert"})
            total_indexed += len(docs)
    print(f"[FAQ] Terindeks {total_indexed} FAQ.")


def normalize_for_overlap(text: str) -> set[str]:
    stopwords = {
        "apa", "adalah", "apakah", "bagaimana", "berapa", "bisa", "dan", "di", "dengan",
        "itu", "ke", "yang", "saya", "kami", "anda", "untuk", "atau", "dari", "jika",
        "saat", "pada", "ini", "itu", "ya", "tidak", "jadi", "mohon", "silakan"
    }
    cleaned = []
    for token in text.lower().replace("?", " ").replace("/", " ").replace("-", " ").split():
        token = token.strip(".,:;()[]{}\"'")
        if len(token) <= 2:
            continue
        if token in stopwords:
            continue
        cleaned.append(token)
    return set(cleaned)


def faq_relevance_score(question: str, doc: dict) -> int:
    q_text = (question or "").lower()
    question_tokens = normalize_for_overlap(q_text)
    doc_text = f"{doc.get('question', '')} {doc.get('answer', '')}".lower()
    doc_tokens = normalize_for_overlap(doc_text)
    overlap_count = len(question_tokens & doc_tokens)

    phrase_bonus = 0
    for phrase in ["rumah sakit", "pendaftaran", "daftar", "syarat", "prosedur", "radiologi", "lab", "icd"]:
        if phrase in q_text and phrase in doc_text:
            phrase_bonus += 1

    return overlap_count + phrase_bonus


def search_faq_collection(question: str, top_k: int = 3):
    collection = get_or_create_faq_collection()
    if count_faqs(collection) == 0:
        index_faqs(collection)

    improved_question = question.strip() or "informasi"
    vector = faq_embeddings.embed_query(improved_question)
    response = client.multi_search.perform(
        {
            "searches": [
                {
                    "collection": FAQ_COLLECTION,
                    "q": improved_question,
                    "query_by": "question,answer",
                    "vector_query": f"embedding:({vector}, k:{top_k * 3}, alpha:0.3)",
                    "per_page": top_k * 3,
                }
            ]
        }
    )
    hits = response["results"][0].get("hits", [])
    ranked = []
    for hit in hits:
        doc = hit["document"]
        score = faq_relevance_score(improved_question, doc)
        if score >= 2 or improved_question.lower() in (doc.get("question") or "").lower():
            ranked.append((score, doc))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


@tool
def faq_search_tool(question: str) -> str:
    """Cari jawaban FAQ dari kumpulan data FAQ."""
    docs = search_faq_collection(question)
    if not docs:
        return "Tidak ada FAQ yang relevan ditemukan."
    context = "\n\n".join(
        f"Q: {doc.get('question', '')}\nA: {doc.get('answer', '')}"
        for doc in docs
    )
    return context


# ---------------------------------------------------------------------------
# Hospital retrieval
# ---------------------------------------------------------------------------
HOSPITAL_COLLECTION = "hospital_router_collection"


def get_or_create_hospital_collection():
    schema = {
        "name": HOSPITAL_COLLECTION,
        "fields": [
            {"name": "hospital", "type": "string", "infix": True},
            {"name": "alias", "type": "string", "infix": True},
            {"name": "address", "type": "string", "infix": True},
            {"name": "district", "type": "string", "infix": True},
            {"name": "city", "type": "string", "infix": True},
            {"name": "province", "type": "string", "infix": True},
            {"name": "slug", "type": "string", "infix": True},
            {"name": "latitude", "type": "float"},
            {"name": "longitude", "type": "float"},
        ],
    }
    try:
        client.collections[HOSPITAL_COLLECTION].retrieve()
    except typesense.exceptions.ObjectNotFound:
        client.collections.create(schema)
    return client.collections[HOSPITAL_COLLECTION]


def load_hospitals():
    with HOSPITAL_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def count_hospitals(collection):
    result = collection.documents.search({"q": "*", "query_by": "hospital", "per_page": 0})
    return result["found"]


def index_hospitals(collection):
    hospitals = load_hospitals()
    docs = []
    for hospital in hospitals:
        docs.append(
            {
                "id": hospital.get("Id") or hospital.get("id"),
                "hospital": hospital.get("Hospital") or hospital.get("hospital") or "",
                "alias": hospital.get("Alias") or "",
                "address": hospital.get("Address") or "",
                "district": hospital.get("District") or "",
                "city": hospital.get("City") or "",
                "province": hospital.get("Province") or "",
                "slug": hospital.get("Slug") or "",
                "latitude": float(hospital.get("lat") or 0.0),
                "longitude": float(hospital.get("lng") or 0.0),
            }
        )
    if docs:
        collection.documents.import_(docs, {"action": "upsert"})
    print(f"[HOSPITAL] Terindeks {len(docs)} rumah sakit.")


def search_hospital_collection(question: str, top_k: int = 5):
    collection = get_or_create_hospital_collection()
    if count_hospitals(collection) == 0:
        index_hospitals(collection)

    result = collection.documents.search(
        {
            "q": question,
            "query_by": "hospital,alias,address,district,city,province,slug",
            "prefix": True,
            "infix": "always",
            "num_typos": 2,
            "min_len_1typo": 4,
            "min_len_2typo": 7,
            "typo_tokens_threshold": 1,
            "drop_tokens_threshold": 1,
            "split_join_tokens": "fallback",
            "per_page": top_k,
        }
    )
    hits = result.get("hits", [])
    return [hit["document"] for hit in hits]


@tool
def hospital_search_tool(question: str) -> str:
    """Cari rumah sakit berdasarkan nama, kota, alias, alamat, atau provinsi."""
    docs = search_hospital_collection(question)
    if not docs:
        return "Tidak ada rumah sakit yang sesuai dengan pencarian tersebut."

    rows = []
    for doc in docs:
        rows.append(
            f"- {doc.get('hospital', '')} | {doc.get('city', '')} | {doc.get('province', '')} | "
            f"{doc.get('address', '')}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Medical record retrieval
# ---------------------------------------------------------------------------
MEDICAL_COLLECTION = "medical_router_collection"


def first_non_empty(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                return cleaned
        else:
            return str(value)
    return ""


def normalize_record(item: dict) -> dict:
    record_type = item.get("record_type") or item.get("classificationName") or item.get("salesItemName") or "GENERIC"
    service_name = (
        item.get("service_name")
        or item.get("salesItemName")
        or item.get("classificationName")
        or item.get("name")
        or item.get("title")
        or ""
    )
    category = item.get("category") or item.get("dataset_name") or item.get("record_type") or "MEDIS"
    return {
        "id": item.get("id") or item.get("code") or item.get("serviceCode") or str(abs(hash(str(item))) % 1000000),
        "record_type": record_type,
        "category": category,
        "service_name": service_name,
        "source_file": item.get("source_file") or "",
        "raw_text": " ".join(
            str(value)
            for value in [
                item.get("service_name"),
                item.get("classificationName"),
                item.get("salesItemName"),
                item.get("name"),
                item.get("title"),
                item.get("category"),
                item.get("record_type"),
            ]
            if value is not None
        ),
    }


def fetch_local_medical_records() -> list[dict]:
    if not MEDICAL_JSON_DIR.exists():
        return []

    files = [
        ("ICD_10.json", "ICD-10"),
        ("ICD_9.json", "ICD-9"),
        ("laboratory.json", "Laboratorium"),
        ("radiology.json", "Radiologi"),
    ]

    records: list[dict] = []
    for filename, dataset_name in files:
        file_path = MEDICAL_JSON_DIR / filename
        if not file_path.exists():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            normalized = dict(item)
            normalized["category"] = dataset_name
            normalized["source_file"] = filename
            if "classificationName" in item:
                normalized["record_type"] = "ICD"
                normalized["service_name"] = item.get("classificationName")
            elif "salesItemName" in item:
                normalized["record_type"] = "LAYANAN"
                normalized["service_name"] = item.get("salesItemName")
            else:
                normalized["record_type"] = "GENERIC"
                normalized["service_name"] = item.get("name") or item.get("title") or ""
            records.append(normalize_record(normalized))

    return records


def get_or_create_medical_collection():
    schema = {
        "name": MEDICAL_COLLECTION,
        "fields": [
            {"name": "record_type", "type": "string", "infix": True},
            {"name": "category", "type": "string", "infix": True},
            {"name": "service_name", "type": "string", "infix": True},
            {"name": "source_file", "type": "string", "infix": True},
            {"name": "raw_text", "type": "string", "infix": True},
        ],
    }
    try:
        client.collections[MEDICAL_COLLECTION].retrieve()
    except typesense.exceptions.ObjectNotFound:
        client.collections.create(schema)
    return client.collections[MEDICAL_COLLECTION]


def count_medical_records(collection):
    result = collection.documents.search({"q": "*", "query_by": "service_name,raw_text", "per_page": 0})
    return result["found"]


def index_medical_records(collection):
    records = fetch_local_medical_records()
    docs = []
    for item in records:
        docs.append(
            {
                "id": item.get("id") or str(len(docs)),
                "record_type": item.get("record_type") or "GENERIC",
                "category": item.get("category") or "MEDIS",
                "service_name": item.get("service_name") or "",
                "source_file": item.get("source_file") or "",
                "raw_text": item.get("raw_text") or "",
            }
        )
    if docs:
        collection.documents.import_(docs, {"action": "upsert"})
    print(f"[MEDICAL] Terindeks {len(docs)} data ICD/Lab/Radiologi.")


def search_medical_collection(question: str, top_k: int = 5):
    collection = get_or_create_medical_collection()
    if count_medical_records(collection) == 0:
        index_medical_records(collection)

    result = collection.documents.search(
        {
            "q": question,
            "query_by": "service_name,category,record_type,source_file,raw_text",
            "prefix": True,
            "infix": "always",
            "num_typos": 2,
            "min_len_1typo": 4,
            "min_len_2typo": 7,
            "typo_tokens_threshold": 1,
            "drop_tokens_threshold": 1,
            "split_join_tokens": "fallback",
            "per_page": top_k,
        }
    )
    hits = result.get("hits", [])
    return [hit["document"] for hit in hits]


@tool
def medical_search_tool(question: str) -> str:
    """Cari data medis dari ICD, laboratorium, dan radiologi."""
    docs = search_medical_collection(question)
    if not docs:
        return "Tidak ada data medis yang sesuai dengan pertanyaan tersebut."

    rows = []
    for doc in docs:
        rows.append(
            f"- {doc.get('service_name', '')} | {doc.get('category', '')} | {doc.get('record_type', '')}"
        )
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Router and memory logic
# ---------------------------------------------------------------------------

def message_to_text(message: BaseMessage | dict) -> str:
    if isinstance(message, BaseMessage):
        return message.content if isinstance(message.content, str) else str(message.content)
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)
    return str(message)


def summarize_window(messages: list, max_turns: int = 10) -> str:
    window = messages[-max_turns:]
    text = "\n".join(
        f"{idx + 1}. {message_to_text(msg)}"
        for idx, msg in enumerate(window)
    )
    if not text.strip():
        return "Belum ada riwayat percakapan."

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Anda adalah asisten ringkas yang menulis summary singkat tapi lengkap. "
                    "Ringkas percakapan untuk sesi berikutnya. Fokus pada kebutuhan pengguna dan hasil yang sudah dibahas.",
                ),
                (
                    "user",
                    "Ringkas percakapan berikut dalam bahasa Indonesia. "
                    "Hanya output ringkasan singkat, jelas, dan rapi.\n\n"
                    f"{text}",
                ),
            ]
        )
        result = llm.invoke(prompt.format_messages())
        return result.content.strip() if isinstance(result.content, str) else str(result.content)
    except Exception:
        return "Ringkasan sesi: pengguna menanyakan kebutuhan terkait data yang relevan dan sistem menyimpan konteks percakapan terakhir untuk menjaga konsistensi jawaban."


def classify_route(question: str) -> str:
    q = question.lower()
    faq_signals = [
        "faq", "cara", "tutorial", "biaya", "jadwal", "aturan", "syarat", "berkas", "kebijakan",
        "pendaftaran", "daftar", "prosedur", "bagaimana", "apa itu", "siapa", "ketentuan",
    ]
    hospital_signals = [
        "rumah sakit", "rs", "dokter", "lokasi", "alamat", "kota", "provinsi", "puskesmas",
        "klinik", "jadwal dokter", "rumah sakit terdekat",
    ]
    medical_signals = [
        "icd", "laboratorium", "lab", "radiologi", "radiology", "diagnosis", "penyakit", "gejala",
        "kode penyakit", "hasil lab", "hasil radiologi",
    ]

    procedural_faq = any(token in q for token in ["syarat", "cara", "prosedur", "daftar", "pendaftaran", "biaya", "jadwal", "aturan"])
    if procedural_faq and not any(token in q for token in ["icd", "lab", "radiologi", "penyakit", "diagnosis"]):
        return "faq"

    faq_score = sum(1 for signal in faq_signals if signal in q)
    hospital_score = sum(1 for signal in hospital_signals if signal in q)
    medical_score = sum(1 for signal in medical_signals if signal in q)

    if medical_score > 0 and medical_score >= hospital_score and medical_score >= faq_score:
        return "medical"
    if faq_score > 0 and faq_score >= hospital_score and faq_score >= medical_score:
        return "faq"
    if hospital_score > 0 and hospital_score >= medical_score and hospital_score >= faq_score:
        return "hospital"

    if "lab" in q or "radiologi" in q or "icd" in q:
        return "medical"
    if "rumah sakit" in q or "rs " in q or "dokter" in q:
        return "hospital"
    return "faq"


def llm_route_fallback(question: str) -> str:
    try:
        decision = llm.with_structured_output(RouteDecision).invoke(
            [
                (
                    "system",
                    "Anda adalah router yang memilih satu route terbaik untuk pertanyaan pengguna. "
                    "Pilih route yang paling sesuai: faq, hospital, atau medical. "
                    "Gunakan kata kunci dan konteks, tetapi jangan terlalu banyak menebak. "
                    "Hanya pilih satu route yang paling tepat."
                ),
                (
                    "user",
                    f"Pertanyaan: {question}\n\nPilih salah satu route berikut: faq, hospital, medical"
                ),
            ]
        )
        if decision and getattr(decision, "route", None) in {"faq", "hospital", "medical"}:
            return decision.route
    except Exception:
        pass
    return "faq"


def route_question(state: RouterState) -> dict:
    latest_message = state["messages"][-1]
    question = message_to_text(latest_message)
    route = classify_route(question)

    q = question.lower()
    strong_signals = any(
        keyword in q
        for keyword in [
            "rumah sakit", "rs", "dokter", "alamat", "kota", "provinsi", "puskesmas", "klinik",
            "icd", "laboratorium", "lab", "radiologi", "diagnosis", "penyakit", "gejala",
            "faq", "cara", "tutorial", "biaya", "jadwal", "aturan", "syarat", "berkas",
            "pendaftaran", "daftar", "prosedur",
        ]
    )

    if not strong_signals:
        route = llm_route_fallback(question)

    turn_id = state.get("turn_id", 0) + 1
    return {"route": route, "turn_id": turn_id}


def build_context_from_route(route: str, question: str) -> str:
    if route == "faq":
        return faq_search_tool.invoke({"question": question})
    if route == "hospital":
        return hospital_search_tool.invoke({"question": question})
    return medical_search_tool.invoke({"question": question})


def specialist_answer(state: RouterState) -> dict:
    latest_message = state["messages"][-1]
    question = message_to_text(latest_message)
    route = state["route"]
    context = build_context_from_route(route, question)

    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Anda adalah agen yang menjawab berdasarkan konteks yang diberikan. "
                    "Jawab dengan ringkas, akurat, dan hanya berdasarkan konteks. "
                    "Jika tidak ada data, katakan bahwa informasi tidak tersedia."
                ),
                (
                    "user",
                    "Route: {route}\n\nPertanyaan: {question}\n\nKonteks:\n{context}",
                ),
            ]
        )
        final = llm.invoke(prompt.format_messages(route=route, question=question, context=context))
        answer = final.content if isinstance(final.content, str) else str(final.content)
    except Exception:
        if not context or "Tidak ada" in context:
            answer = "Informasi yang Anda cari belum tersedia pada dataset yang saat ini aktif."
        else:
            lines = [line.strip() for line in str(context).splitlines() if line.strip()]
            preview = "\n".join(lines[:3])
            answer = (
                "Berdasarkan data yang tersedia, saya menemukan informasi berikut:\n\n"
                f"{preview}"
            )

    return {"context": context, "response": answer}


def checkpoint_memory(state: RouterState) -> dict:
    recent_messages = state["messages"][-5:]
    summary = state.get("summary", "")

    if state.get("turn_id", 0) % 2 == 0:
        summary = summarize_window(state["messages"], max_turns=10)

    if len(state["messages"]) > 5:
        messages = state["messages"][-5:]
    else:
        messages = state["messages"]

    return {
        "summary": summary,
        "recent_messages": recent_messages,
        "messages": messages,
    }


workflow = StateGraph(RouterState)
workflow.add_node("route_question", route_question)
workflow.add_node("specialist_answer", specialist_answer)
workflow.add_node("checkpoint_memory", checkpoint_memory)

workflow.add_edge(START, "route_question")
workflow.add_conditional_edges(
    "route_question",
    lambda state: state["route"],
    {
        "faq": "specialist_answer",
        "hospital": "specialist_answer",
        "medical": "specialist_answer",
    },
)
workflow.add_edge("specialist_answer", "checkpoint_memory")
workflow.add_edge("checkpoint_memory", END)

app = workflow.compile(checkpointer=InMemorySaver())


def run_workflow(question: str, thread_id: str = "session-1"):
    config = {"configurable": {"thread_id": thread_id}}
    previous = app.get_state(config)
    existing = previous.values if previous else {}

    messages = list(existing.get("messages", []))
    messages.append({"role": "user", "content": question})

    state = {
        "thread_id": thread_id,
        "turn_id": int(existing.get("turn_id", 0)) + 1,
        "route": existing.get("route", ""),
        "messages": messages,
        "summary": existing.get("summary", ""),
        "recent_messages": existing.get("recent_messages", []),
        "response": existing.get("response", ""),
        "context": existing.get("context", ""),
    }

    result = app.invoke(state, config=config)
    return result


if __name__ == "__main__":
    try:
        # Auto-index if needed
        faq_collection = get_or_create_faq_collection()
        if count_faqs(faq_collection) == 0:
            index_faqs(faq_collection)

        hospital_collection = get_or_create_hospital_collection()
        if count_hospitals(hospital_collection) == 0:
            index_hospitals(hospital_collection)

        medical_collection = get_or_create_medical_collection()
        if count_medical_records(medical_collection) == 0:
            index_medical_records(medical_collection)

        print("Router workflow siap digunakan. Ketik 'exit' untuk keluar.")
        thread_id = "demo-router"
        while True:
            question = input("User: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Sampai jumpa.")
                break
            if not question:
                continue
            result = run_workflow(question, thread_id=thread_id)
            print("\n[Agen]")
            print(result["response"])
            print("\n[Summary]")
            print(result.get("summary", "Belum ada summary."))
            print("\n[Recent]")
            print(result.get("recent_messages", []))
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}")
