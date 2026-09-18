import json
import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypedDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
FAQ_FILE = PROJECT_ROOT / "faqs_extend_no_split.jsonl"
HOSPITAL_FILE = PROJECT_ROOT / "hospitals_prod.json"
MEDICAL_JSON_DIR = PROJECT_ROOT / "data_icd_lab_rad" / "json"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL).strip()

if GEMINI_MODEL in {"gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-2.5-flash"}:
    GEMINI_MODEL = DEFAULT_GEMINI_MODEL

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY belum ditemukan di file .env")

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    api_key=GEMINI_API_KEY,
    temperature=0.4,
    timeout=60,
    max_retries=2,
    max_output_tokens=1024,
    convert_system_message_to_human=True,
)

# ---------------------------------------------------------------------------
# Main agent character — satu system prompt tunggal, tidak diulang di tiap node
# ---------------------------------------------------------------------------

MAIN_AGENT_SYSTEM_PROMPT = (
    "Anda adalah asisten layanan kesehatan digital yang cerdas, ramah, dan informatif. "
    "Anda memiliki akses ke data FAQ layanan, data rumah sakit, dan data medis (ICD, laboratorium, radiologi). "
    "Tugas Anda adalah membantu pengguna dengan menjawab pertanyaan mereka secara natural, jelas, dan relevan.\n\n"
    "Panduan menjawab:\n"
    "- Gunakan bahasa Indonesia yang hangat, komunikatif, dan mudah dipahami.\n"
    "- Jawablah berdasarkan data yang tersedia; jangan mengarang informasi.\n"
    "- Jika data tidak cukup, akui dengan jujur dan sopan, lalu arahkan ke tindakan yang masuk akal.\n"
    "- Tulis dalam paragraf yang mengalir — hindari daftar nomor yang kaku.\n"
    "- Sesuaikan gaya, panjang, dan nada jawaban dengan konteks pertanyaan.\n"
    "- Jangan menyebut nama tool, route, atau detail teknis sistem kepada pengguna."
)


class AgentState(TypedDict):
    thread_id: str
    turn_id: int
    messages: Annotated[list, add_messages]
    summary: str
    recent_messages: list
    route: str
    context: str
    response: str
    need_more_tool: bool
    tool_rounds: int


class RouteDecision(BaseModel):
    route: Literal["faq", "hospital", "medical"] = Field(
        default="faq",
        description="Pilih route yang paling sesuai untuk pertanyaan pengguna.",
    )
    rationale: str = Field(
        default="",
        description="Alasan ringkas mengapa route ini dipilih.",
    )


class ToolDecision(BaseModel):
    need_more_tool: bool = Field(
        default=False,
        description="Apakah masih perlu alat/route lain untuk mengumpulkan informasi yang cukup sebelum menjawab?",
    )
    rationale: str = Field(
        default="",
        description="Alasan singkat mengapa membutuhkan atau tidak membutuhkan tool lebih lanjut.",
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def message_to_text(message: BaseMessage | dict) -> str:
    if isinstance(message, BaseMessage):
        return message.content if isinstance(message.content, str) else str(message.content)
    if isinstance(message, dict):
        content = message.get("content", "")
        return content if isinstance(content, str) else str(content)
    return str(message)


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "answer", "response"):
                    value = item.get(key)
                    if isinstance(value, str):
                        chunks.append(value)
                        break
                else:
                    for val in item.values():
                        if isinstance(val, str):
                            chunks.append(val)
                            break
        return "\n".join(part.strip() for part in chunks if part and str(part).strip())

    if isinstance(content, dict):
        for key in ("text", "content", "answer", "response"):
            value = content.get(key)
            if isinstance(value, str):
                return value.strip()
        for value in content.values():
            text = extract_text_from_content(value)
            if text:
                return text
        return ""

    text = str(content).strip()
    return text if text != "None" else ""


def pick_candidate_label(item: dict) -> str:
    if not isinstance(item, dict):
        return "item"

    candidate_keys = (
        "hospital", "Hospital", "service_name", "Service_Name", "serviceName",
        "question", "Question", "category", "Category", "name", "Name",
        "title", "Title", "record_type", "Record_Type", "raw_text", "Raw_Text"
    )
    for key in candidate_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if not isinstance(value, str) and value is not None:
            text = str(value).strip()
            if text:
                return text

    for value in item.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list)):
            text = str(value).strip()
            if text:
                return text

    return "item"


def summarize_window(messages: list, max_turns: int = 10) -> str:
    window = messages[-max_turns:]
    text = "\n".join(
        f"{idx + 1}. {message_to_text(msg)}"
        for idx, msg in enumerate(window)
    )
    if not text.strip():
        return "Belum ada riwayat percakapan."

    try:
        sys_msg = SystemMessage(
            content=(
                "Anda adalah asisten ringkas yang menulis summary singkat tapi lengkap. "
                "Ringkas percakapan untuk sesi berikutnya. Fokus pada kebutuhan pengguna dan hasil yang sudah dibahas. "
                "Hasilnya harus natural, singkat, jelas, dan bukan salinan semua kalimat dari percakapan."
            )
        )
        user_msg = HumanMessage(
            content=(
                "Ringkas percakapan berikut dalam bahasa Indonesia. "
                f"Hanya output ringkasan singkat, jelas, dan rapi.\n\n{text}"
            )
        )
        result = llm.invoke([sys_msg, user_msg])
        summary = extract_text_from_content(result.content)
        return summary if summary else str(result.content).strip()
    except Exception:
        return "Ringkasan sesi: pengguna menanyakan kebutuhan terkait data yang relevan dan sistem menyimpan konteks percakapan terakhir untuk menjaga konsistensi jawaban."


def update_conversation_summary(existing_summary: str, user_question: str, bot_response: str) -> str:
    """
    Mengakumulasi dan memperbarui summary percakapan secara natural menggunakan LLM.
    Summary dari turn sebelumnya akan digabungkan/di-append dengan intisari dari turn saat ini.
    """
    existing_clean = (existing_summary or "").strip()
    q_clean = (user_question or "").strip()
    ans_clean = (bot_response or "").strip()

    if not ans_clean:
        return existing_clean

    sys_msg = SystemMessage(
        content=(
            "Anda adalah asisten pencatat ringkasan percakapan (memory summary keeper). "
            "Tugas Anda adalah memperbarui dan mengakumulasi ringkasan percakapan multi-turn secara ringkas, kronologis, dan padat informasi. "
            "Aturan:\n"
            "1. Jika sudah ada ringkasan sesi sebelumnya, pertahankan poin penting dari sesi sebelumnya dan tambahkan/integrasikan topik serta hasil dari turn terbaru ini.\n"
            "2. Jangan menyalin seluruh percakapan. Tuliskan ringkasan perkembangan diskusi dalam 1-3 kalimat yang mengalir dan natural.\n"
            "3. Output HANYA teks ringkasan akumulatif, tanpa pembuka, tanpa bullet points berlebihan, dan tanpa penjelasan tambahan."
        )
    )

    context_prompt = ""
    if existing_clean:
        context_prompt += f"Ringkasan percakapan sebelumnya:\n{existing_clean}\n\n"
    context_prompt += (
        f"Turn terbaru:\n"
        f"- Pengguna menanyakan: {q_clean}\n"
        f"- Asisten menjawab: {ans_clean[:800]}\n\n"
        "Perbarui ringkasan percakapan agar mencakup seluruh riwayat sampai turn terbaru ini:"
    )

    user_msg = HumanMessage(content=context_prompt)

    try:
        result = llm.invoke([sys_msg, user_msg])
        new_summary = extract_text_from_content(result.content)
        if new_summary:
            return new_summary.strip()
    except Exception:
        pass

    # Fallback sederhana jika LLM gagal
    new_point = f"Pengguna menanyakan '{q_clean[:50]}' dan asisten memberikan informasi terkait."
    if existing_clean:
        return f"{existing_clean} Selanjutnya, {new_point}"
    return new_point


def normalize_question(value: str) -> str:
    return " ".join((value or "").lower().replace("-", " ").split())


def keyword_tokens(value: str) -> set[str]:
    tokens = []
    for part in normalize_question(value).split():
        if len(part) <= 2:
            continue
        tokens.append(part)
    return set(tokens)


def score_text_relevance(question: str, text: str) -> int:
    q_tokens = keyword_tokens(question)
    if not q_tokens:
        return 0
    t = normalize_question(text)
    matches = 0
    for token in q_tokens:
        if token in t:
            matches += 1
    return matches


def heuristic_route(question: str) -> str:
    q = normalize_question(question)
    if not q:
        return "faq"

    faq_keywords = [
        "cara", "prosedur", "syarat", "jadwal", "mendaftar", "batas", "biaya",
        "registrasi", "pendaftaran", "persyaratan", "proses", "dokumen", "alur",
        "asuransi", "bpjs", "telekonsultasi", "telechat", "mcu", "medical check up",
        "homecare", "appointment", "janji temu", "mysiloam", "my siloam", "dokter",
        "konsultasi", "booking", "reservasi"
    ]
    hospital_keywords = [
        "rumah sakit", "hospital", "rs", "klinik", "puskesmas", "lokasi", "alamat",
        "kota", "daerah", "terdekat", "poliklinik", "jakarta", "cempaka putih",
        "semanggi", "asri", "kebon jeruk", "mampang", "lippo village"
    ]
    medical_keywords = [
        "icd", "lab", "laboratorium", "radiologi", "diagnosa", "penyakit",
        "hasil pemeriksaan", "hasil lab", "hasil radiologi", "kode penyakit",
        "konsultasi medis", "medical", "rekam medis", "gula darah", "hb1ac",
        "radiology", "laboratory", "hasil", "tes"
    ]

    if any(keyword in q for keyword in faq_keywords):
        return "faq"
    if any(keyword in q for keyword in hospital_keywords):
        return "hospital"
    if any(keyword in q for keyword in medical_keywords):
        return "medical"

    return "faq"


def load_faqs() -> list[dict]:
    if not FAQ_FILE.exists():
        return []
    rows = []
    with FAQ_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_hospitals() -> list[dict]:
    if not HOSPITAL_FILE.exists():
        return []
    with HOSPITAL_FILE.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_medical_records() -> list[dict]:
    if not MEDICAL_JSON_DIR.exists():
        return []

    dataset_files = [
        ("ICD_10.json", "ICD-10"),
        ("ICD_9.json", "ICD-9"),
        ("laboratory.json", "Laboratorium"),
        ("radiology.json", "Radiologi"),
    ]

    records: list[dict] = []
    for filename, category in dataset_files:
        path = MEDICAL_JSON_DIR / filename
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        if not isinstance(payload, list):
            continue

        for item in payload[:60]:
            record = dict(item)
            record["_category"] = category
            record["_source_file"] = filename
            records.append(record)

    return records


# ---------------------------------------------------------------------------
# Route selection by LLM
# ---------------------------------------------------------------------------

def decide_route(question: str) -> str:
    """LLM memilih route berdasarkan pemahaman konteks. Heuristic hanya fallback jika LLM gagal."""
    try:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Anda adalah router untuk sistem layanan kesehatan.\n"
                    "Tentukan satu route yang paling tepat berdasarkan maksud dan konteks pertanyaan pengguna:\n"
                    "- 'faq': pertanyaan prosedur, cara, syarat, jadwal, booking, BPJS, asuransi, telekonsultasi, MCU, atau informasi umum layanan.\n"
                    "- 'hospital': pertanyaan mencari, menemukan, lokasi, alamat, atau informasi spesifik rumah sakit atau klinik.\n"
                    "- 'medical': pertanyaan kode ICD, data laboratorium, radiologi, atau diagnosis medis.\n\n"
                    "Pertimbangkan maksud keseluruhan pertanyaan, bukan hanya kata kunci.",
                ),
                (
                    "user",
                    "Pertanyaan pengguna: {question}",
                ),
            ]
        )
        decision = llm.with_structured_output(RouteDecision).invoke(
            prompt.format_messages(question=question)
        )
        if getattr(decision, "route", None) in {"faq", "hospital", "medical"}:
            return decision.route
    except Exception:
        pass
    # Fallback: heuristic jika LLM tidak tersedia atau gagal
    return heuristic_route(question)


# ---------------------------------------------------------------------------
# LLM-based retrieval tools
# ---------------------------------------------------------------------------

def get_faq_candidates(question: str, limit: int = 20) -> list[dict]:
    faqs = load_faqs()
    if not faqs:
        return []

    scored: list[tuple[int, dict]] = []
    q_tokens = keyword_tokens(question)
    for item in faqs:
        text = f"{item.get('prompt', '')} {item.get('completion', '')}"
        score = score_text_relevance(question, text)
        if q_tokens and score == 0:
            if any(token in normalize_question(text) for token in q_tokens):
                score = 1
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = [item for _, item in scored[:limit]]
    return [{"question": item.get("prompt", ""), "answer": item.get("completion", "")} for item in ordered]


def faq_tool(question: str) -> str:
    """Sub-agent FAQ: pilih data relevan dan sajikan sebagai teks narasi untuk main agent."""
    candidates = get_faq_candidates(question, limit=20)
    if not candidates:
        return "Tidak ada data FAQ yang tersedia."

    sys_msg = SystemMessage(
        content=(
            "Anda adalah sub-agent yang bertugas mengambil informasi dari database FAQ. "
            "Pilih maksimal 3 entri FAQ yang paling relevan dengan pertanyaan pengguna, "
            "lalu sajikan isinya sebagai teks informatif yang ringkas dan mudah dipahami. "
            "Fokus pada isi jawaban, bukan pada format. Jangan mengarang informasi di luar data."
        )
    )
    user_msg = HumanMessage(
        content=(
            f"Pertanyaan pengguna: {question}\n\n"
            f"Daftar FAQ kandidat:\n{json.dumps(candidates, ensure_ascii=False)[:8000]}\n\n"
            "Sajikan informasi yang paling relevan dalam bentuk teks narasi singkat."
        )
    )
    try:
        result = llm.invoke([sys_msg, user_msg])
        text = extract_text_from_content(result.content)
        return text.strip() or "Tidak ditemukan informasi FAQ yang relevan."
    except Exception:
        # Fallback: gabungkan teks kandidat pertama secara langsung
        lines = []
        for item in candidates[:3]:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                lines.append(f"Pertanyaan: {q}\nJawaban: {a}")
        return "\n\n".join(lines) if lines else "Tidak ditemukan informasi FAQ yang relevan."


def get_hospital_candidates(question: str, limit: int = 60) -> list[dict]:
    hospitals = load_hospitals()
    if not hospitals:
        return []

    q = normalize_question(question)
    scored: list[tuple[int, dict]] = []
    for item in hospitals:
        fields = [
            item.get("Hospital") or item.get("hospital") or "",
            item.get("City") or item.get("city") or "",
            item.get("District") or item.get("district") or "",
            item.get("Address") or item.get("address") or "",
            item.get("Province") or item.get("province") or "",
            item.get("Alias") or item.get("alias") or "",
        ]
        text = " ".join(fields)
        score = score_text_relevance(question, text)
        if q:
            for field in fields:
                if not field:
                    continue
                token_set = keyword_tokens(field)
                score += sum(1 for token in keyword_tokens(q) if token in token_set)
        if score > 0:
            scored.append((score, item))

    if not scored:
        scored = [(1, item) for item in hospitals[:limit]]

    scored.sort(key=lambda x: x[0], reverse=True)
    compact = []
    for _, item in scored[:limit]:
        compact.append(
            {
                "Hospital": item.get("Hospital") or item.get("hospital") or "",
                "City": item.get("City") or item.get("city") or "",
                "Province": item.get("Province") or item.get("province") or "",
                "Address": item.get("Address") or item.get("address") or "",
                "Alias": item.get("Alias") or item.get("alias") or "",
                "District": item.get("District") or item.get("district") or "",
            }
        )
    return compact


def hospital_tool(question: str) -> str:
    """Sub-agent Hospital: pilih rumah sakit relevan dan sajikan sebagai teks narasi untuk main agent."""
    candidates = get_hospital_candidates(question)
    if not candidates:
        return "Tidak ada data rumah sakit yang tersedia."

    sys_msg = SystemMessage(
        content=(
            "Anda adalah sub-agent yang bertugas mengambil informasi dari database rumah sakit. "
            "Pilih maksimal 3 rumah sakit yang paling relevan berdasarkan lokasi, nama, atau konteks yang disebutkan pengguna. "
            "Sajikan hasilnya sebagai teks narasi yang informatif: sebutkan nama rumah sakit, kota, dan informasi relevan lainnya. "
            "Jangan menambah informasi yang tidak ada di data. Jangan mengarang alamat atau layanan."
        )
    )
    user_msg = HumanMessage(
        content=(
            f"Pertanyaan pengguna: {question}\n\n"
            f"Daftar rumah sakit kandidat:\n{json.dumps(candidates, ensure_ascii=False)[:12000]}\n\n"
            "Sajikan rumah sakit yang paling relevan dalam bentuk teks narasi ringkas."
        )
    )
    try:
        result = llm.invoke([sys_msg, user_msg])
        text = extract_text_from_content(result.content)
        return text.strip() or "Tidak ditemukan data rumah sakit yang relevan."
    except Exception:
        # Fallback: rangkum kandidat pertama sebagai teks
        lines = []
        for item in candidates[:3]:
            name = item.get("Hospital", "").strip()
            city = item.get("City", "").strip()
            prov = item.get("Province", "").strip()
            addr = item.get("Address", "").strip()
            parts = filter(None, [name, city, prov, addr])
            lines.append(" — ".join(parts))
        return "\n".join(lines) if lines else "Tidak ditemukan data rumah sakit yang relevan."


def get_medical_candidates(question: str, limit: int = 40) -> list[dict]:
    records = load_medical_records()
    if not records:
        return []

    scored: list[tuple[int, dict]] = []
    q_tokens = keyword_tokens(question)
    for item in records:
        text = " ".join(
            str(value)
            for value in [
                item.get("record_type") or item.get("classificationName") or item.get("salesItemName") or "",
                item.get("service_name") or item.get("classificationName") or item.get("salesItemName") or item.get("name") or "",
                item.get("raw_text") or "",
                item.get("_category") or "",
            ]
        )
        score = score_text_relevance(question, text)
        if q_tokens and score == 0:
            if any(token in normalize_question(text) for token in q_tokens):
                score = 1
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    subset = [item for _, item in scored[:limit]]
    compact = []
    for item in subset:
        compact.append(
            {
                "category": item.get("_category", ""),
                "source_file": item.get("_source_file", ""),
                "record_type": item.get("record_type") or item.get("classificationName") or item.get("salesItemName") or "",
                "service_name": item.get("service_name") or item.get("classificationName") or item.get("salesItemName") or item.get("name") or "",
                "raw_text": item.get("raw_text") or "",
            }
        )
    return compact


def medical_tool(question: str) -> str:
    """Sub-agent Medical: pilih data ICD/lab/radiologi relevan dan sajikan sebagai teks narasi untuk main agent."""
    candidates = get_medical_candidates(question)
    if not candidates:
        return "Tidak ada data medis yang tersedia."

    sys_msg = SystemMessage(
        content=(
            "Anda adalah sub-agent yang bertugas mengambil informasi dari database medis (ICD-10, ICD-9, laboratorium, radiologi). "
            "Pilih maksimal 3-5 entri yang paling relevan terhadap pertanyaan pengguna. "
            "Sajikan hasilnya sebagai teks narasi yang ringkas: sebutkan kategori, nama layanan/rekam medis, dan informasi pendukung yang ada. "
            "Jangan membuat asumsi yang tidak didukung data. Jangan mengarang kode atau nama diagnosa."
        )
    )
    user_msg = HumanMessage(
        content=(
            f"Pertanyaan pengguna: {question}\n\n"
            f"Daftar data medis kandidat:\n{json.dumps(candidates, ensure_ascii=False)[:12000]}\n\n"
            "Sajikan data yang paling relevan dalam bentuk teks narasi ringkas."
        )
    )
    try:
        result = llm.invoke([sys_msg, user_msg])
        text = extract_text_from_content(result.content)
        return text.strip() or "Tidak ditemukan data medis yang relevan."
    except Exception:
        # Fallback: rangkum kandidat pertama sebagai teks
        lines = []
        for item in candidates[:3]:
            cat = item.get("category", "").strip()
            svc = item.get("service_name", "").strip()
            rec = item.get("record_type", "").strip()
            parts = filter(None, [cat, rec, svc])
            lines.append(" | ".join(parts))
        return "\n".join(lines) if lines else "Tidak ditemukan data medis yang relevan."


# ---------------------------------------------------------------------------
# Main agent graph and memory state
# ---------------------------------------------------------------------------

def route_and_select(question: str) -> tuple[str, str]:
    route = decide_route(question)
    if route == "faq":
        return route, faq_tool(question)
    if route == "hospital":
        return route, hospital_tool(question)
    return route, medical_tool(question)


def route_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    question = message_to_text(last)
    route = decide_route(question)
    print(f"[DEBUG] route_node :: question={question!r} -> route={route}")
    return {
        "route": route,
        "turn_id": state.get("turn_id", 0) + 1,
    }


def decide_if_more_tool_needed(question: str, route: str, context: str) -> bool:
    try:
        sys_msg = SystemMessage(
            content=(
                "Anda adalah manajer keputusan. Evaluasi apakah informasi yang sudah dikumpulkan sudah cukup untuk menjawab pertanyaan user. "
                "Jika informasi masih kurang, set need_more_tool=true dan kembalikan alasan singkat. "
                "Jika sudah cukup, set need_more_tool=false. "
                "Jawab dalam format JSON dengan field need_more_tool dan rationale."
            )
        )
        user_msg = HumanMessage(
            content=f"Route: {route}\n\nPertanyaan: {question}\n\nKonteks saat ini:\n{context[:3000]}"
        )
        decision = llm.with_structured_output(ToolDecision).invoke([sys_msg, user_msg])
        if isinstance(decision, ToolDecision):
            return bool(getattr(decision, "need_more_tool", False))
    except Exception:
        pass
    return False


def decision_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    question = message_to_text(last)
    route = state.get("route", "faq")
    context = state.get("context", "")

    need_more_tool = decide_if_more_tool_needed(question, route, context)
    tool_rounds = int(state.get("tool_rounds", 0)) + 1
    print(f"[DEBUG] decision_node :: route={route} need_more_tool={need_more_tool} tool_rounds={tool_rounds} context_len={len(context)}")

    return {
        "need_more_tool": need_more_tool,
        "tool_rounds": tool_rounds,
    }


def specialist_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    question = message_to_text(last)
    route = state.get("route", "faq")

    if route == "faq":
        context = faq_tool(question)
    elif route == "hospital":
        context = hospital_tool(question)
    else:
        context = medical_tool(question)

    print(f"[DEBUG] specialist_node :: route={route} context_preview={context[:300]!r}")
    return {"context": context}


def clean_context(raw: str, route: str) -> str:
    """
    Membersihkan output dari sub-agent sebelum dikirim ke main agent.
    Jika output berupa JSON, konversi ke teks readable.
    Tambahkan label sumber agar main agent tahu konteks datanya.
    """
    label_map = {
        "faq": "Informasi dari FAQ Layanan",
        "hospital": "Data Rumah Sakit",
        "medical": "Data Medis (ICD / Laboratorium / Radiologi)",
    }
    label = label_map.get(route, "Data Referensi")

    # Coba parse JSON; jika berhasil, flatten ke teks readable
    if isinstance(raw, str):
        stripped = raw.strip()
        # Coba cek apakah ada JSON di dalam teks (model kadang membungkus dengan markdown)
        json_match = re.search(r"```(?:json)?\s*([\[\{].*?)\s*```", stripped, re.DOTALL)
        if json_match:
            stripped = json_match.group(1)

        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                lines = []
                for item in parsed[:5]:
                    if isinstance(item, dict):
                        parts = [str(v).strip() for v in item.values() if v and str(v).strip()]
                        lines.append(" | ".join(parts))
                    else:
                        lines.append(str(item))
                readable = "\n".join(lines)
            elif isinstance(parsed, dict):
                parts = [f"{k}: {v}" for k, v in parsed.items() if v]
                readable = "\n".join(parts)
            else:
                readable = str(parsed)
            raw = readable
        except (json.JSONDecodeError, ValueError):
            pass  # Bukan JSON, pakai teks apa adanya

    text = (raw or "").strip()
    if not text:
        return f"[{label}]\nTidak ada data yang relevan ditemukan."

    # Batasi panjang agar tidak overflow ke LLM
    if len(text) > 3500:
        text = text[:3500].rsplit("\n", 1)[0] + "\n..."

    return f"[{label}]\n{text}"


def answer_node(state: AgentState) -> dict:
    """
    Main agent yang merumuskan jawaban akhir.
    Menggunakan MAIN_AGENT_SYSTEM_PROMPT tunggal — tidak ada hardcoded template per route.
    LLM yang memutuskan cara terbaik menjawab berdasarkan pertanyaan dan konteks sub-agent.

    Catatan: pesan dibangun langsung dengan HumanMessage/SystemMessage (bukan ChatPromptTemplate)
    agar konten dinamis yang mengandung '{' atau '}' tidak ditafsirkan sebagai template variable
    dan menyebabkan KeyError.
    """
    last = state["messages"][-1]
    question = message_to_text(last)
    route = state.get("route", "faq")
    raw_context = state.get("context", "")
    existing_summary = state.get("summary", "").strip()

    # Bersihkan output sub-agent sebelum dikirim ke main agent
    context = clean_context(raw_context, route)

    summary_block = f"Ringkasan percakapan sebelumnya:\n{existing_summary}\n\n" if existing_summary else ""

    user_content = (
        f"{summary_block}"
        f"Pertanyaan pengguna terkini:\n{question}\n\n"
        f"Informasi yang dikumpulkan oleh sistem:\n{context}\n\n"
        "Berikan jawaban yang natural, relevan, dan membantu berdasarkan informasi di atas."
    )

    answer = ""
    try:
        result = llm.invoke([
            SystemMessage(content=MAIN_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        answer = extract_text_from_content(result.content)
        if not answer:
            answer = str(result.content).strip()
    except Exception as exc:
        print(f"[DEBUG] answer_node :: main call FAILED — {exc!r}")
        # Retry dengan prompt lebih singkat jika panggilan utama gagal
        try:
            retry = llm.invoke([
                SystemMessage(content=MAIN_AGENT_SYSTEM_PROMPT),
                HumanMessage(
                    content=f"Pertanyaan: {question}\n\nData: {context[:2000]}"
                ),
            ])
            answer = extract_text_from_content(retry.content) or str(retry.content).strip()
        except Exception as exc2:
            print(f"[DEBUG] answer_node :: retry FAILED — {exc2!r}")
            answer = ""

    print(f"[DEBUG] answer_node :: route={route} question={question!r} answer={answer[:500]!r}")
    return {"response": answer}


def checkpoint_node(state: AgentState) -> dict:
    recent_messages = state["messages"][-5:]
    last = state["messages"][-1]
    question = message_to_text(last)
    current_response = state.get("response", "").strip()
    previous_summary = state.get("summary", "").strip()

    # Akumulasi ringkasan sesi turn demi turn secara dinamis menggunakan LLM
    if current_response:
        summary = update_conversation_summary(
            existing_summary=previous_summary,
            user_question=question,
            bot_response=current_response
        )
    else:
        summary = previous_summary

    trimmed = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
    print(f"[DEBUG] checkpoint_node :: turn={state.get('turn_id', 0)} summary={summary[:500]!r} recent_count={len(recent_messages)}")
    return {
        "summary": summary,
        "recent_messages": recent_messages,
        "messages": trimmed,
    }


workflow = StateGraph(AgentState)
workflow.add_node("route_node", route_node)
workflow.add_node("specialist_node", specialist_node)
workflow.add_node("decision_node", decision_node)
workflow.add_node("answer_node", answer_node)
workflow.add_node("checkpoint_node", checkpoint_node)

workflow.add_edge(START, "route_node")
workflow.add_edge("route_node", "specialist_node")
workflow.add_edge("specialist_node", "decision_node")
workflow.add_conditional_edges(
    "decision_node",
    lambda state: "route_node" if state.get("need_more_tool") and state.get("tool_rounds", 0) < 2 else "answer_node",
    {
        "route_node": "route_node",
        "answer_node": "answer_node",
    },
)
workflow.add_edge("answer_node", "checkpoint_node")
workflow.add_edge("checkpoint_node", END)

app = workflow.compile(checkpointer=InMemorySaver())


def run_main_agent(question: str, thread_id: str = "main-agent-session"):
    config = {"configurable": {"thread_id": thread_id}}
    previous = app.get_state(config)
    existing = previous.values if previous else {}

    messages = list(existing.get("messages", []))
    messages.append({"role": "user", "content": question})

    state = {
        "thread_id": thread_id,
        "turn_id": int(existing.get("turn_id", 0)) + 1,
        "messages": messages,
        "summary": existing.get("summary", ""),
        "recent_messages": existing.get("recent_messages", []),
        "route": "",
        "context": "",
        "response": "",
        "need_more_tool": True,
        "tool_rounds": 0,
    }

    result = app.invoke(state, config=config)
    return result


if __name__ == "__main__":
    print("Main agent router demo siap. Ketik 'exit' untuk keluar.")
    thread_id = "main-agent-demo"
    while True:
        user_input = input("User: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Sampai jumpa.")
            break
        if not user_input:
            continue
        result = run_main_agent(user_input, thread_id=thread_id)
        print("\n[Route]", result.get("route"))
        print("\n[Answer]", result.get("response", ""))
        print("\n[Summary]", result.get("summary", "Belum ada summary."))
        print("\n[Recent]", result.get("recent_messages", []))
