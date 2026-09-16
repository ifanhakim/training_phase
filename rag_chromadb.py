from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import chromadb
from chromadb import EmbeddingFunction

load_dotenv()

client = OpenAI(
    base_url=os.getenv('OPENROUTER_BASE_URL'),
    api_key=os.getenv('OPENROUTER_API_KEY')
)


# EmbeddingFunction custom yang memanggil OpenAI SDK dengan base_url OpenRouter
class OpenRouterEmbeddingFunction(EmbeddingFunction):
    """Embedding function khusus OpenRouter."""

    def __init__(self, model_name='nvidia/nemotron-3-embed-1b:free'):
        self.client = OpenAI(
            base_url=os.getenv('OPENROUTER_BASE_URL'),
            api_key=os.getenv('OPENROUTER_API_KEY')
        )
        self.model_name = model_name

    def __call__(self, input):
        # Wajib paksa 'float'.
        response = self.client.embeddings.create(
            model=self.model_name,
            input=input,
            encoding_format='float'
        )
        return [item.embedding for item in response.data]

    def name(self):
        # FIX 2b: wajib di-override supaya chromadb bisa menyimpan nama
        # embedding function ini di konfigurasi collection. Tanpa ini
        # chromadb akan melempar NotImplementedError saat menyimpan config.
        return f'openrouter-{self.model_name}'


chroma_client = chromadb.PersistentClient('./chroma_db')


def get_or_create_collection():
    config = {
        'metadata': {'description': 'Production RAG knowledge base example'},
        'embedding_function': OpenRouterEmbeddingFunction(),
    }
    try:
        return chroma_client.get_or_create_collection(
            name='knowledge_base',
            **config
        )
    except ValueError:
        print('Collection lama memakai embedding function berbeda. Membuat ulang...')
        chroma_client.delete_collection('knowledge_base')
        return chroma_client.create_collection(
            name='knowledge_base',
            **config
        )


collection = get_or_create_collection()

# hidden process -> collection.add -> embedding disimpan ke vectorstore


def load_documents(folder_path):
    documents = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': file_name,
                    'content': content
                })

    return documents


def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # start = 0, end = 0 + 800, chunk = 0 - 800
        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def add_documents_to_db(folder_path):
    docs = load_documents(folder_path)

    all_chunks = []
    all_ids = []
    all_metadatas = []

    chunk_counter = 0

    for doc in docs:
        chunks = chunk_text(doc['content'])  # chunks = [chunk, chunk, chunk ...]

        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f'chunk_{chunk_counter}')
            all_metadatas.append({
                'source': doc['filename'],
                'chunk_id': i
            })

            chunk_counter += 1

    # tambahan dalam batch biar ngga kena rate limit
    batch_size = 20
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            ids=all_ids[i:end],
            metadatas=all_metadatas[i:end]
        )

    print(f'Berhasil menambahkan {chunk_counter} chunk dari {len(docs)} dokumen.')


def search(query, n_result=3):
    total = collection.count()
    if total == 0:
        return []

    # jangan minta lebih banyak hasil dari jumlah chunk yang tersedia
    n_result = min(n_result, total)

    results = collection.query(
        query_texts=[query],
        n_results=n_result
    )

    relevant_chunks = []
    for i in range(n_result):
        relevant_chunks.append({
            'text': results['documents'][0][i],
            'source': results['metadatas'][0][i]['source'],
            'distance': results['distances'][0][i]  # cosine similarity
        })

    return relevant_chunks


def generate_answer(history):
    response = client.chat.completions.create(
        model='nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
        messages=history,
        max_tokens=1000
    )

    if not response.choices:
        raise RuntimeError('OpenRouter tidak mengembalikan pilihan jawaban.')

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError('OpenRouter mengembalikan jawaban kosong.')

    return content.strip()


def get_embeddings(text):
    response = client.embeddings.create(
        model='nvidia/nemotron-3-embed-1b:free',
        input=text,
        encoding_format='float'  # wajib float, bukan base64
    )

    return response.data[0].embedding


def cosine_similarity(vector1, vector2):
    """Fungsi ini digunakan untuk menghitung cosine similarity"""
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)
    return similarity


def main():
    if collection.count() == 0:
        print('Database kosong, menambahkan semua dokumen...')
        add_documents_to_db('knowledge_base')
    else:
        print(f'Database sudah berisi {collection.count()} chunk.')

    print('RAG CHATBOT (ketik "exit" untuk keluar)')

    history = [{
        'role': 'system',
        'content': '''Anda adalah asisten layanan informasi akademik yang profesional dan ramah untuk Direktorat Tata Kelola Akademik Universitas Nahdlatul Ulama (UNU) Yogyakarta. Jawablah menggunakan bahasa yang digunakan oleh pengguna.

CARA MENJAWAB:
1. Selalu awali jawaban dengan sapaan yang sopan dan empatik.
2. Jawablah berdasarkan konteks yang tersedia, yaitu Panduan Tugas Akhir dan Yudisium UNU Yogyakarta Tahun 2026.
3. Apabila informasi yang ditanyakan tidak terdapat dalam konteks, sampaikan dengan sopan bahwa hal tersebut perlu dikonfirmasi lebih lanjut kepada Direktorat Tata Kelola Akademik.
4. Berikan jawaban yang jelas, terstruktur, dan mudah dipahami, terutama untuk prosedur yang terdiri atas beberapa langkah.
5. Hindari penggunaan emotikon secara berlebihan. Gunakan emotikon hanya jika relevan untuk menyampaikan kesan ramah.
6. Selalu tawarkan bantuan tambahan pada akhir jawaban.
7. Sebutkan angka, tanggal, batas waktu, dan prosedur secara spesifik dan akurat sesuai dengan yang tercantum dalam panduan (misalnya persyaratan SKS, ketentuan nilai kelulusan, atau tata cara pengunggahan berkas).
8. PENTING: Jangan mengarang atau menambahkan informasi di luar konteks yang tersedia. Gunakan hanya informasi yang terdapat dalam Panduan Tugas Akhir dan Yudisium.
9. Apabila pertanyaan menyangkut data pribadi mahasiswa (misalnya status kelulusan, hasil verifikasi berkas, atau nilai), arahkan mahasiswa untuk menghubungi Direktorat Tata Kelola Akademik melalui nomor WhatsApp 081393722904 dengan mencantumkan Nama, NIM, Program Studi, dan keperluan.

GAYA BAHASA:
- Formal, sopan, dan komunikatif, sebagaimana lazim digunakan dalam layanan akademik perguruan tinggi.
- Gunakan istilah baku sesuai Panduan (misalnya "Tugas Akhir", "Yudisium", "Direktorat Tata Kelola Akademik").
- Akhiri jawaban dengan pertanyaan lanjutan yang relevan untuk memastikan kebutuhan mahasiswa terpenuhi.

CONTOH:
"Terima kasih atas pertanyaannya. 🙏
Berdasarkan Panduan Tugas Akhir dan Yudisium UNU Yogyakarta, [jawaban spesifik sesuai konteks]...
Apakah ada hal lain yang dapat kami bantu terkait proses Tugas Akhir atau Yudisium Anda?"'''
    }]

    while True:
        try:
            raw_query = input('You: ').strip()
        except (KeyboardInterrupt, EOFError):
            print('\nSampai jumpa!')
            break

        if not raw_query or raw_query.lower() == 'exit':
            print('Sampai jumpa!')
            break

        # Perjelas pertanyaan & terjemahkan ke English supaya hasil embedding/search lebih akurat
        try:
            prompt_enhancement = client.chat.completions.create(
                model='nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
                messages=[
                    {'role': 'system', 'content': 'You are a user question translator. Translate user questions from any language into English and make them clearer and more detailed.'},
                    {'role': 'user', 'content': raw_query},
                ],
                max_tokens=300
            )
            if not prompt_enhancement.choices:
                raise RuntimeError('respons penerjemah tidak memiliki pilihan jawaban')
            query = prompt_enhancement.choices[0].message.content
            if not query or not query.strip():
                raise RuntimeError('respons penerjemah kosong')
            query = query.strip()
        except Exception as e:
            print(f'  (penerjemahan gagal: {e})')
            query = None

        # kalau terjemahan gagal/kosong, pakai pertanyaan asli
        if not query:
            print('  (pertanyaan tidak diterjemahkan, memakai pertanyaan asli)')
            query = raw_query

        # Ambil konteks relevan dari vector DB
        results = search(query, n_result=3)

        if not results:
            print('AI: Maaf, saya tidak menemukan informasi yang relevan di knowledge base.')
            continue

        context = "\n".join([chunk['text'] for chunk in results])

        user_prompt = f"""Customer Question: {query}

Context from knowledge base:
{context}"""

        history.append({'role': 'user', 'content': user_prompt})

        # Generate jawaban
        try:
            answer = generate_answer(history)
        except Exception as e:
            print(f'AI: Maaf, terjadi kesalahan saat generate jawaban: {e}')
            history.pop()
            continue

        # simpan history yang ringkas (query tanpa context) biar hemat token
        history[-1] = {'role': 'user', 'content': query}
        history.append({'role': 'assistant', 'content': answer})

        print(f'AI: {answer}')


if __name__ == '__main__':
    main()
