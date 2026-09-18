import json
import os
from pathlib import Path
from typing import Any, Dict

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

HOSPITAL_FILE = Path("hospitals_prod.json")
COLLECTION_NAME = "hospital_directory"

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

DEFAULT_SEARCH_PARAMS = {
    "query_by": "hospital,alias,address,district,city,province,slug",
    "prefix": True,
    "infix": "always",
    "num_typos": 2,
    "min_len_1typo": 4,
    "min_len_2typo": 7,
    "typo_tokens_threshold": 1,
    "drop_tokens_threshold": 1,
    "split_join_tokens": "fallback",
    "per_page": 5,
}


class HospitalSearchInput(BaseModel):
    query: str = Field(
        ...,
        description="Nama rumah sakit, kota, provinsi, alamat, atau alias yang dicari.",
    )


def get_or_create_collection():
    schema = {
        "name": COLLECTION_NAME,
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
        collection = typesense_client.collections[COLLECTION_NAME].retrieve()
        existing_fields = {field["name"]: field for field in collection["fields"]}
        needs_rebuild = False
        for field_name in ["hospital", "alias", "address", "district", "city", "province", "slug"]:
            field = existing_fields.get(field_name)
            if not field or field.get("infix") is not True:
                needs_rebuild = True
                break

        if needs_rebuild:
            print(f"[DEBUG] Rebuilding collection {COLLECTION_NAME} to enable infix search for typo-tolerance testing.")
            typesense_client.collections[COLLECTION_NAME].delete()
            typesense_client.collections.create(schema)
            return typesense_client.collections[COLLECTION_NAME]

        return collection
    except typesense.exceptions.ObjectNotFound:
        typesense_client.collections.create(schema)
        return typesense_client.collections[COLLECTION_NAME]


def load_hospitals():
    with HOSPITAL_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_hospitals(collection):
    result = collection.documents.search(
        {"q": "*", "query_by": "hospital", "per_page": 0}
    )
    return result["found"]


def index_hospitals(collection):
    hospitals = load_hospitals()
    documents = []

    for hospital in hospitals:
        documents.append(
            {
                "id": hospital["Id"],
                "hospital": hospital["Hospital"],
                "alias": hospital.get("Alias") or "",
                "address": hospital.get("Address") or "",
                "district": hospital.get("District") or "",
                "city": hospital.get("City") or "",
                "province": hospital.get("Province") or "",
                "slug": hospital.get("Slug") or "",
                "latitude": float(hospital["lat"]),
                "longitude": float(hospital["lng"]),
            }
        )

    results = collection.documents.import_(documents, {"action": "upsert"})
    failed = [result for result in results if not result.get("success")]

    if failed:
        raise RuntimeError(f"Gagal mengindeks {len(failed)} data rumah sakit.")

    print(f"Berhasil mengindeks {len(documents)} rumah sakit.")


def build_hospital_search_params(query: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = {
        "q": query,
        **DEFAULT_SEARCH_PARAMS,
    }
    if overrides:
        params.update(overrides)
    return params


def search_hospital_collection(collection, query: str, overrides: Dict[str, Any] | None = None):
    params = build_hospital_search_params(query, overrides)
    print(f"[DEBUG] Search params: {params}")
    result = collection.documents.search(params)
    hits = [hit["document"] for hit in result["hits"]]
    print(f"[DEBUG] Found {len(hits)} hit(s) for query: {query!r}")
    return hits


@tool(args_schema=HospitalSearchInput)
def search_hospital(query: str) -> Dict[str, Any]:
    """Mencari informasi rumah sakit berdasarkan nama atau lokasi dengan typo tolerance yang aman."""
    collection = get_or_create_collection()
    hospitals = search_hospital_collection(collection, query)
    return {"status": "success", "count": len(hospitals), "data": hospitals}


def debug_hospital_search(query: str):
    collection = get_or_create_collection()
    scenarios = {
        "default": {},
        "anti_typo": {
            "num_typos": 2,
            "min_len_1typo": 4,
            "min_len_2typo": 7,
            "prefix": True,
            "infix": "always",
            "split_join_tokens": "always",
        },
        "strict": {
            "num_typos": 0,
            "prefix": False,
            "infix": "off",
            "split_join_tokens": "off",
        },
        "lenient_prefix": {
            "num_typos": 2,
            "min_len_1typo": 3,
            "min_len_2typo": 5,
            "prefix": True,
            "infix": "fallback",
        },
    }

    for label, overrides in scenarios.items():
        hits = search_hospital_collection(collection, query, overrides)
        print(f"\n=== {label.upper()} ===")
        for hit in hits[:3]:
            print(hit)


tools = [search_hospital]

system_prompt = """Anda adalah agen informasi operasional rumah sakit.
Gunakan search_hospital untuk mencari nama atau lokasi rumah sakit.
Jawab hanya berdasarkan hasil tool.
Jangan memberikan diagnosis medis, saran obat, atau informasi rekam medis.
Jika pertanyaan meminta diagnosis atau saran medis, arahkan pengguna untuk berkonsultasi dengan dokter atau IGD.
Jika data rumah sakit tidak ditemukan, katakan bahwa informasi belum tersedia."""

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
        number_of_hospitals = count_hospitals(collection)

        if number_of_hospitals == 0:
            print("Collection rumah sakit masih kosong. Mulai indexing...")
            index_hospitals(collection)
        else:
            print(f"Typesense sudah berisi {number_of_hospitals} rumah sakit.")

        print('Hospital Agent (ketik "exit" untuk keluar)')

        while True:
            question = input("You: ").strip()
            if question.lower() == "exit":
                print("Sampai jumpa!")
                break

            if question:
                result = app.invoke(
                    {"messages": [{"role": "user", "content": question}]}
                )
                print("AI:", result["messages"][-1].content)
    except Exception as error:
        print(f"Program gagal dijalankan: {error}")
