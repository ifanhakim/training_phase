# Training Project – LangChain, RAG, dan AI Agent

Repositori ini berisi berbagai contoh implementasi AI dan retrieval-augmented generation (RAG) berbasis Python. Fokus utama proyek adalah eksperimen pencarian semantik, pengolahan dokumen, penggunaan model OpenRouter, serta agent berbasis LangChain dan LangGraph.

Proyek ini tidak hanya berisi satu aplikasi saja, melainkan beberapa script dengan tujuan yang berbeda, seperti:

- RAG dengan ChromaDB
- RAG dengan Typesense + LangChain
- Agent pencarian rumah sakit berbasis LangGraph
- Prompt chaining dan self-consistency
- Pembuatan knowledge base otomatis dari dokumen
- Notebook eksperimen LangChain dan reasoning tanpa LangChain

## Struktur direktori

```text
training/
├── .env                     # konfigurasi environment lokal (tidak untuk di-commit)
├── .gitignore
├── LICENSE
├── README.md
├── chaining_langchain-generate_knowledge.py
├── create_agent_langchain-self_consistency_prompt.py
├── create_react_agent_langchain.py
├── faqs_extend_no_split.jsonl
├── hospitals_prod.json
├── invoke_langchain.ipynb
├── knowledge_base/
│   └── PANDUAN_TUGAS_AKHIR_DAN_YUDISIUM_2026.txt
├── no_langchain-cot.ipynb
├── production_rag/          # virtual environment lokal
├── rag_chromadb.py
├── rag_typesense_langchain.py
├── small_project.py
├── typesense-data/          # data local Typesense
├── workflow_agent_langgraph.py
├── chroma_db/               # data local ChromaDB
└── __pycache__/             # cache Python lokal
```

## Script utama yang ada

### 1. `rag_chromadb.py`
Script RAG sederhana dengan:

- ChromaDB sebagai vector database
- OpenRouter untuk embedding dan chat completion
- `knowledge_base/` sebagai sumber dokumen
- Chat interaktif di terminal

### 2. `rag_typesense_langchain.py`
Implementasi RAG dengan:

- Typesense sebagai vector store
- LangChain Core dan ChatOpenAI
- OpenRouter sebagai penyedia model
- `knowledge_base` untuk indexing dan retrieval

### 3. `small_project.py`
Contoh agent berbasis LangGraph yang bekerja dengan data rumah sakit dan Typesense:

- indexing `hospitals_prod.json`
- pencarian rumah sakit melalui tool
- agent memilih tool berdasarkan pertanyaan user

### 4. Script agen dan workflow lainnya

- `create_agent_langchain-self_consistency_prompt.py`
- `create_react_agent_langchain.py`
- `workflow_agent_langgraph.py`
- `chaining_langchain-generate_knowledge.py`

Semua file ini merupakan eksperimen penggunaan LangChain, prompt design, dan workflow agent.

## Library yang dibutuhkan

Dependency yang digunakan oleh proyek ini antara lain:

```bash
python-dotenv
openai
numpy
chromadb
typesense
langchain
langchain-core
langchain-openai
langgraph
pydantic
typing-extensions
```

Jika Anda ingin menjalankan notebook, biasanya juga diperlukan:

```bash
jupyter
ipykernel
```

## Persyaratan sistem

- Python 3.9 atau lebih baru
- Internet untuk mengakses OpenRouter
- Docker (opsional, jika ingin menjalankan Typesense lokal)
- Akses ke API key OpenRouter

## Instalasi

### 1. Buat virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Jika Anda ingin menggunakan environment yang sudah ada di repo, bisa juga:

```powershell
.\production_rag\Scripts\Activate.ps1
```

### 2. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 3. Install library yang dibutuhkan

Instal semua dependency utama yang dipakai oleh project ini:

```bash
python -m pip install python-dotenv openai numpy chromadb typesense langchain langchain-core langchain-openai langgraph pydantic typing-extensions
```

Jika ingin juga menjalankan notebook:

```bash
python -m pip install jupyter ipykernel
```

## Konfigurasi environment

Buat file `.env` di root project dan isi seperti berikut:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_MODEL=inclusionai/ling-3.0-flash-fin:free

TYPESENSE_HOST=localhost
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=http
TYPESENSE_API_KEY=xyz
```

Catatan:

- `OPENROUTER_API_KEY` wajib diisi agar script dapat akses model OpenRouter.
- `OPENROUTER_BASE_URL` biasanya default ke `https://openrouter.ai/api/v1`.
- `TYPESENSE_API_KEY` perlu sesuai dengan server Typesense Anda.

Jangan commit file `.env` ke repository publik. Simpan secara lokal saja.

## Menjalankan Typesense lokal

Beberapa script menggunakan Typesense. Jika Anda ingin menjalankannya secara lokal via Docker, gunakan perintah berikut:

```bash
docker run -d \
  --name typesense \
  -p 8108:8108 \
  -v $(pwd)/typesense-data:/data \
  typesense/typesense:29.0 \
  --data-dir /data \
  --api-key=xyz \
  --enable-cors
```

Periksa apakah service berjalan:

```bash
curl http://localhost:8108/health
```

## Menjalankan project

### RAG dengan ChromaDB

```bash
python rag_chromadb.py
```

### RAG dengan Typesense + LangChain

```bash
python rag_typesense_langchain.py
```

### Hospital agent dengan LangGraph

```bash
python small_project.py
```

### Script eksperimen lainnya

```bash
python chaining_langchain-generate_knowledge.py
python create_agent_langchain-self_consistency_prompt.py
python create_react_agent_langchain.py
python workflow_agent_langgraph.py
```

## Cara kerja umum

1. Dokumen sumber dibaca dari `knowledge_base/` atau file JSON tertentu.
2. Teks dipotong menjadi chunk dengan overlap tertentu.
3. Chunk dikirim ke model embedding OpenRouter.
4. Vektor disimpan di ChromaDB atau Typesense.
5. User query diproses dan dibandingkan dengan konteks yang tersimpan.
6. Model LLM mengambil chunk yang relevan lalu menghasilkan jawaban berdasarkan konteks tersebut.
7. Beberapa script juga menggunakan LangGraph untuk mengelola alur tool-calling dan workflow multi-step.

## Data yang dipakai

- `knowledge_base/PANDUAN_TUGAS_AKHIR_DAN_YUDISIUM_2026.txt`
- `hospitals_prod.json`
- `faqs_extend_no_split.jsonl`

Data ini digunakan untuk eksperimen RAG, indexing, dan real-world search task.

## Catatan penting

- Beberapa script menggunakan OpenRouter model gratis dan bisa berubah sewaktu-waktu.
- Jika model tidak tersedia, mungkin perlu mengganti value `OPENROUTER_MODEL` atau model di script.
- `chroma_db/` dan `typesense-data/` adalah folder data lokal, biasanya dibuat otomatis saat aplikasi berjalan.
- Format `knowledge_base` bersifat plain text, sehingga Anda dapat menambahkan file `.txt` baru sesuai kebutuhan.

## Troubleshooting umum

### `OPENROUTER_API_KEY` tidak ditemukan
Pastikan `.env` sudah dibuat di root project dan nama variabel sesuai.

### `Typesense` tidak terhubung
Periksa apakah server Typesense sudah aktif di localhost:8108 dan API key sesuai.

### `Collection` atau database kosong
Jalankan script yang bersangkutan sekali lagi agar indexing otomatis terjadi.

### `Model` tidak tersedia / 403 / 429
Coba ganti model ke model OpenRouter lain yang masih aktif, atau cek status API OpenRouter.

## Lisensi

Proyek ini menggunakan lisensi Apache License 2.0. Lihat file [LICENSE](LICENSE) untuk detail lengkap.
