import os
from pathlib import Path

import typesense
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv(
	"OPENROUTER_BASE_URL",
	"https://openrouter.ai/api/v1",
).rstrip("/")

if not OPENROUTER_API_KEY:
	raise RuntimeError("OPENROUTER_API_KEY belum ditemukan di file .env")


# Adapter ini membuat OpenRouter embedding bisa dipakai oleh LangChain.
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


# Typesense harus berjalan di localhost:8108 atau di server Typesense Cloud.
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
		chunk = text[start:start + chunk_size]
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

	return [
		hit["document"]
		for hit in results["results"][0]["hits"]
	]


prompt = ChatPromptTemplate.from_template(
	"""Jawab pertanyaan berdasarkan konteks berikut.
Jika jawabannya tidak ada di konteks, katakan bahwa informasi perlu dikonfirmasi.
Jawab dalam bahasa Indonesia dengan singkat dan jelas.

Konteks:
{context}

Pertanyaan:
{question}
"""
)

answer_chain = prompt | model | StrOutputParser()


def answer_question(collection, question):
	documents = search(collection, question)

	if not documents:
		return "Maaf, informasi yang relevan tidak ditemukan."

	context = "\n\n".join(document["content"] for document in documents)
	return answer_chain.invoke({"context": context, "question": question})


def count_documents(collection):
	result = collection.documents.search({
		"q": "*",
		"query_by": "content",
		"per_page": 0,
	})
	return result["found"]


def main():
	try:
		collection = get_or_create_collection()
	except Exception as error:
		print("Tidak bisa terhubung ke Typesense.")
		print("Pastikan Typesense berjalan di localhost:8108 atau ubah .env.")
		print(f"Detail: {error}")
		return

	number_of_documents = count_documents(collection)

	if number_of_documents == 0:
		print("Typesense masih kosong. Membaca knowledge_base...")
		add_documents_to_typesense(collection)
	else:
		print(f"Typesense sudah berisi {number_of_documents} dokumen.")

	print('RAG Typesense + LangChain (ketik "exit" untuk keluar)')

	while True:
		question = input("You: ").strip()

		if question.lower() == "exit":
			print("Sampai jumpa!")
			break

		if question:
			print("AI:", answer_question(collection, question))


if __name__ == "__main__":
	main()
