import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import typesense
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict


load_dotenv()

COLLECTION_NAME = "doctor_availability"
LOCAL_DATASET_DIR = Path(__file__).resolve().parent / "data_icd_lab_rad" / "json"

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")

base_url = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_MODEL", "stealth/union-alpha"),
    api_key=api_key,
    base_url=base_url,
    temperature=0.0,
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


def first_non_empty(*values: Any) -> str:
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


def extract_nested_value(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] not in (None, ""):
            return data[key]
    return None


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    return str(value)


def flatten_availability(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "; ".join(
            str(item)
            for item in value
            if item is not None and str(item).strip()
        )
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = flatten_availability(item)
            if text:
                parts.append(f"{key}: {text}")
        return "; ".join(parts)
    return str(value)


def fetch_local_fallback_records() -> List[Dict[str, Any]]:
    """Read the bundled ICD/lab/radiology JSON dataset as the project data source."""
    if not LOCAL_DATASET_DIR.exists():
        print(f"[DEBUG] Local dataset not found at {LOCAL_DATASET_DIR}")
        return []

    records: List[Dict[str, Any]] = []
    dataset_files = [
        ("ICD_10.json", "ICD-10"),
        ("ICD_9.json", "ICD-9"),
        ("laboratory.json", "Laboratorium"),
        ("radiology.json", "Radiologi"),
    ]

    for filename, dataset_name in dataset_files:
        file_path = LOCAL_DATASET_DIR / filename
        if not file_path.exists():
            print(f"[DEBUG] Skipping missing local dataset: {file_path}")
            continue

        with file_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            print(f"[DEBUG] Local dataset {filename} is not a list; skipping.")
            continue

        for item in data:
            transformed: Dict[str, Any] = dict(item)
            if "classificationName" in item:
                transformed["record_type"] = "ICD"
                transformed["category"] = dataset_name
                transformed["service_name"] = item.get("classificationName")
                transformed["source_file"] = filename
            elif "salesItemName" in item:
                transformed["record_type"] = "SERVICE"
                transformed["category"] = dataset_name
                transformed["service_name"] = item.get("salesItemName")
                transformed["organization_id"] = item.get("organizationId")
                transformed["source_file"] = filename
            else:
                transformed["record_type"] = "GENERIC"
                transformed["category"] = dataset_name
                transformed["service_name"] = item.get("name") or item.get("title") or ""
                transformed["source_file"] = filename

            records.append(transformed)

    print(f"[DEBUG] Loaded {len(records)} records from local JSON dataset")
    return records


def fetch_doctor_payload() -> List[Dict[str, Any]]:
    """Use only local JSON data as the data source for this project."""
    return fetch_local_fallback_records()


def normalize_doctor_record(item: Dict[str, Any]) -> Dict[str, Any]:
    print(f"[DEBUG] Normalizing record sample: {str(item)[:400]}")

    doctor_name = first_non_empty(
        extract_nested_value(item, "doctorName", "doctor_name", "name", "fullName", "full_name"),
        extract_nested_value(item, "doctor", "doctor_detail"),
        item.get("service_name"),
        item.get("salesItemName"),
        item.get("classificationName"),
    )

    specialty = first_non_empty(
        extract_nested_value(item, "specialty", "specialization", "specialist", "doctorSpecialty"),
        extract_nested_value(item, "specialistName"),
        item.get("category"),
        item.get("record_type"),
    )

    hospital = first_non_empty(
        extract_nested_value(item, "hospital", "hospitalName", "clinicName", "facilityName"),
        extract_nested_value(item, "location", "branch"),
        item.get("organizationId"),
        item.get("organization_id"),
    )

    city = first_non_empty(
        extract_nested_value(item, "city", "cityName", "locationName"),
        extract_nested_value(item, "branchCity"),
        item.get("cityName"),
        item.get("city_name"),
    )

    availability = first_non_empty(
        extract_nested_value(item, "availability", "schedule", "availableSchedule"),
        extract_nested_value(item, "doctorAvailability"),
        item.get("remarks"),
        item.get("source_file"),
    )

    if isinstance(availability, (dict, list)):
        availability = flatten_availability(availability)

    raw_text = " ".join([
        normalize_text(doctor_name),
        normalize_text(specialty),
        normalize_text(hospital),
        normalize_text(city),
        normalize_text(availability),
        normalize_text(item.get("source_file") or ""),
    ])

    normalized = {
        "doctor_name": doctor_name,
        "specialty": specialty,
        "hospital": hospital,
        "city": city,
        "availability": availability,
        "raw_text": raw_text,
    }

    print(f"[DEBUG] Normalized doctor: {normalized}")
    return normalized


class MedicalRecordSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Kode ICD, nama diagnosis, nama layanan laboratorium/radiologi, kategori, atau kata kunci medis yang dicari.",
    )


def get_or_create_collection():
    schema = {
        "name": COLLECTION_NAME,
        "fields": [
            {"name": "doctor_name", "type": "string"},
            {"name": "specialty", "type": "string"},
            {"name": "hospital", "type": "string"},
            {"name": "city", "type": "string"},
            {"name": "availability", "type": "string"},
            {"name": "raw_text", "type": "string"},
        ],
    }

    try:
        typesense_client.collections[COLLECTION_NAME].retrieve()
    except typesense.exceptions.ObjectNotFound:
        typesense_client.collections.create(schema)

    return typesense_client.collections[COLLECTION_NAME]


def count_doctors(collection):
    result = collection.documents.search({"q": "*", "query_by": "doctor_name,specialty,hospital,city,availability,raw_text", "per_page": 0})
    return result["found"]


def index_doctors(collection):
    records = fetch_doctor_payload()
    print(f"[DEBUG] Total records fetched from API: {len(records)}")
    documents = []

    for idx, item in enumerate(records):
        try:
            doctor = normalize_doctor_record(item)
        except Exception as exc:
            print(f"[DEBUG] Failed to normalize item {idx}: {exc}")
            continue

        if not doctor["doctor_name"] and not doctor["specialty"] and not doctor["hospital"]:
            print(f"[DEBUG] Skipping item {idx} because all core fields are empty")
            continue

        documents.append(
            {
                "id": str(idx),
                "doctor_name": doctor["doctor_name"],
                "specialty": doctor["specialty"],
                "hospital": doctor["hospital"],
                "city": doctor["city"],
                "availability": doctor["availability"],
                "raw_text": doctor["raw_text"],
            }
        )

    if not documents:
        print("Tidak ada data dokter yang valid untuk diindex.")
        return

    print(f"[DEBUG] Ready to import {len(documents)} normalized documents to Typesense")
    results = collection.documents.import_(documents, {"action": "upsert"})
    failed = [result for result in results if not result.get("success")]
    if failed:
        raise RuntimeError(f"Gagal mengindeks {len(failed)} dokter.")

    print(f"Berhasil mengindeks {len(documents)} dokter.")


def search_doctors(collection, query: str, number_of_results: int = 5):
    print(f"[DEBUG] Searching Typesense with query: {query!r}")
    result = collection.documents.search(
        {
            "q": query,
            "query_by": "doctor_name,specialty,hospital,city,availability,raw_text",
            "per_page": number_of_results,
        }
    )
    hits = [hit["document"] for hit in result["hits"]]
    print(f"[DEBUG] Search returned {len(hits)} hit(s)")
    for hit in hits:
        print(f"[DEBUG] Hit: {hit}")
    return hits


@tool(args_schema=MedicalRecordSearchInput)
def search_medical_record(query: str) -> Dict[str, Any]:
    """Mencari data ICD, laboratorium, atau radiologi berdasarkan kode, nama, atau kata kunci medis."""
    print(f"[DEBUG] Tool search_medical_record invoked with query: {query}")
    collection = get_or_create_collection()
    doctors = search_doctors(collection, query, number_of_results=5)

    if not doctors:
        print("[DEBUG] No medical records found for the query")
        return {"status": "success", "count": 0, "data": []}

    print(f"[DEBUG] Returning {len(doctors)} medical records to the agent")
    return {"status": "success", "count": len(doctors), "data": doctors}


tools = [search_medical_record]

system_prompt = """Anda adalah agen pencarian data medis lokal untuk dataset ICD, laboratorium, dan radiologi.
Gunakan tool search_medical_record untuk mencari informasi berdasarkan:
- kode ICD atau diagnosis, misalnya A00, E11, J18
- nama pemeriksaan laboratorium, misalnya hemoglobin, gula darah, kultur urine
- nama pemeriksaan radiologi, misalnya CT scan, MRI, ultrasound
- kategori dataset seperti ICD-10, ICD-9, Laboratorium, Radiologi
- kata kunci medis umum yang muncul di data lokal

Aturan penting:
1. Jawab hanya berdasarkan hasil tool.
2. Jika data tidak ditemukan, katakan bahwa informasi tidak tersedia di dataset lokal.
3. Berikan jawaban yang jelas, ringkas, dan relevan untuk kebutuhan pengguna.
4. Hindari asumsi di luar data yang ada.
5. Jika user menanyakan kode diagnosis atau nama layanan, tampilkan hasil yang paling cocok dari dataset."""

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm_with_tools = model.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState):
    messages = prompt_template.invoke({"messages": state["messages"]})
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else "__end__"


tool_node = ToolNode(tools)
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "__end__": END},
)
workflow.add_edge("tools", "agent")
app = workflow.compile()


if __name__ == "__main__":
    try:
        collection = get_or_create_collection()
        number_of_doctors = count_doctors(collection)

        if number_of_doctors == 0:
            print("Collection data medis masih kosong. Mulai indexing dataset ICD, laboratorium, dan radiologi...")
            index_doctors(collection)
        else:
            print(f"Typesense sudah berisi {number_of_doctors} data medis.")

        print('Medical Records Agent (ketik "exit" untuk keluar)')

        while True:
            question = input("You: ").strip()
            if question.lower() == "exit":
                print("Sampai jumpa!")
                break

            if question:
                result = app.invoke({"messages": [{"role": "user", "content": question}]})
                print("AI:", result["messages"][-1].content)
    except Exception as error:
        print(f"Program gagal dijalankan: {error}")
