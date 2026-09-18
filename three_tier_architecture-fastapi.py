import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typesense
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATASET_DIR = PROJECT_ROOT / "data_icd_lab_rad" / "json"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "inclusionai/ling-3.0-flash-vl:free",
).strip()

TYPESENSE_COLLECTION = os.getenv("TYPESENSE_COLLECTION", "medical_reference")
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "localhost")
TYPESENSE_PORT = os.getenv("TYPESENSE_PORT", "8108")
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL", "http")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "xyz")

llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
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
                "host": TYPESENSE_HOST,
                "port": TYPESENSE_PORT,
                "protocol": TYPESENSE_PROTOCOL,
            }
        ],
        "api_key": TYPESENSE_API_KEY,
        "connection_timeout_seconds": 10,
    }
)


class MedicalQueryRequest(BaseModel):
    query: str = Field(..., example="gejala demam disertai batuk dan pilek")


class MedicalReferenceItem(BaseModel):
    code: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    source_file: Optional[str] = None


class MedicalQueryResponse(BaseModel):
    query: str
    references: List[MedicalReferenceItem]
    answer: str


class MedicalReferenceRepository:
    def __init__(self, client):
        self.client = client

    def _build_schema(self):
        return {
            "name": TYPESENSE_COLLECTION,
            "fields": [
                {"name": "code", "type": "string"},
                {"name": "name", "type": "string"},
                {"name": "category", "type": "string"},
                {"name": "record_type", "type": "string"},
                {"name": "description", "type": "string"},
                {"name": "raw_text", "type": "string"},
                {"name": "source_file", "type": "string"},
            ],
        }

    def get_or_create_collection(self):
        try:
            return self.client.collections[TYPESENSE_COLLECTION].retrieve()
        except typesense.exceptions.ObjectNotFound:
            self.client.collections.create(self._build_schema())
            return self.client.collections[TYPESENSE_COLLECTION]

    def load_local_json_records(self) -> List[Dict[str, Any]]:
        if not LOCAL_DATASET_DIR.exists():
            return []

        records: List[Dict[str, Any]] = []
        dataset_files = [
            ("ICD_10.json", "ICD-10"),
            ("ICD_9.json", "ICD-9"),
            ("laboratory.json", "Laboratorium"),
            ("radiology.json", "Radiologi"),
        ]

        for file_name, category in dataset_files:
            file_path = LOCAL_DATASET_DIR / file_name
            if not file_path.exists():
                continue

            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                continue

            for item in data:
                if not isinstance(item, dict):
                    continue

                record = {
                    "id": f"{file_name}:{item.get('id') or item.get('code') or item.get('classificationCode') or item.get('salesItemCode') or len(records)}",
                    "code": item.get("code") or item.get("classificationCode") or item.get("ICDCode") or item.get("salesItemCode") or "",
                    "name": item.get("name") or item.get("classificationName") or item.get("salesItemName") or item.get("title") or "",
                    "category": category,
                    "record_type": item.get("record_type") or "diagnostic",
                    "description": item.get("description") or item.get("descriptionText") or item.get("notes") or item.get("summary") or item.get("classificationName") or item.get("salesItemName") or "",
                    "raw_text": " ".join(
                        str(value)
                        for value in [
                            item.get("code"),
                            item.get("classificationCode"),
                            item.get("classificationName"),
                            item.get("salesItemName"),
                            item.get("description"),
                            item.get("descriptionText"),
                            item.get("notes"),
                            item.get("summary"),
                        ]
                        if value not in (None, "")
                    ),
                    "source_file": file_name,
                }

                records.append(record)

        return records

    def index_dataset_if_empty(self):
        collection = self.get_or_create_collection()
        count_result = collection.documents.search({
            "q": "*",
            "query_by": "code,name,category,description,raw_text",
            "per_page": 0,
        })

        if count_result.get("found", 0) > 0:
            return collection

        documents = self.load_local_json_records()
        if not documents:
            return collection

        for document in documents:
            collection.documents.upsert(document)

        return collection

    def search_relevant_records(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        collection = self.get_or_create_collection()
        result = collection.documents.search(
            {
                "q": query,
                "query_by": "code,name,category,description,raw_text",
                "per_page": limit,
            }
        )

        hits = result.get("hits", [])
        extracted = []
        for hit in hits:
            extracted.append(hit.get("document", {}))
        return extracted


class LLMOrchestrationService:
    def __init__(self, medical_repository: MedicalReferenceRepository):
        self.medical_repository = medical_repository

    def answer_from_typesense(self, request: MedicalQueryRequest) -> MedicalQueryResponse:
        relevant_records = self.medical_repository.search_relevant_records(request.query, limit=5)

        references = [
            MedicalReferenceItem(
                code=str(record.get("code") or ""),
                name=str(record.get("name") or ""),
                category=str(record.get("category") or ""),
                description=str(record.get("description") or ""),
                source_file=str(record.get("source_file") or ""),
            )
            for record in relevant_records
        ]

        context_text = "\n".join(
            (
                f"- {item.category} | {item.code} | {item.name} | {item.description}"
                if item.code or item.name or item.description
                else "- Referensi tidak memiliki rincian lengkap."
            )
            for item in references
        ) or "- Tidak ada referensi relevan yang ditemukan di data Typesense."

        prompt = [
            SystemMessage(
                content=(
                    "Anda adalah asisten medis yang menjawab berdasarkan data referensi yang diberikan. "
                    "Jawab hanya menggunakan informasi dari data referensi yang tersedia. "
                    "Jika data tidak tersedia, katakan bahwa informasi belum tersedia. "
                    "Gunakan bahasa Indonesia singkat, jelas, dan tetap profesional."
                )
            ),
            HumanMessage(
                content=(
                    f"Pertanyaan: {request.query}\n\n"
                    f"Referensi dari Typesense:\n{context_text}\n\n"
                    "Buat jawaban yang menjelaskan informasi relevan berdasarkan referensi tersebut."
                )
            ),
        ]

        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        return MedicalQueryResponse(
            query=request.query,
            references=references,
            answer=str(answer).strip(),
        )


# DEPENDENCIES

def get_medical_repository() -> MedicalReferenceRepository:
    repository = MedicalReferenceRepository(typesense_client)
    repository.index_dataset_if_empty()
    return repository


def get_llm_service(
    med_repo: MedicalReferenceRepository = Depends(get_medical_repository),
) -> LLMOrchestrationService:
    return LLMOrchestrationService(medical_repository=med_repo)


triage_router = APIRouter(prefix="/api/v1/triage", tags=["AI Triage"])


@triage_router.post("/search", response_model=MedicalQueryResponse)
async def search_medical_reference(
    payload: MedicalQueryRequest,
    llm_service: LLMOrchestrationService = Depends(get_llm_service),
):
    try:
        return llm_service.answer_from_typesense(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Terjadi kesalahan pada pencarian data medis: {str(exc)}",
        )


app = FastAPI(title="BitHealth AI Orchestration API")
app.include_router(triage_router)