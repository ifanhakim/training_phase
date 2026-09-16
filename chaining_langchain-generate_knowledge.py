import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

client = ChatOpenAI(
    model="inclusionai/ling-3.0-flash-fin:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    timeout=60,
    max_retries=0,
)

prompt = ChatPromptTemplate.from_template(
    "Jelaskan {topik} untuk pemula dalam bahasa Indonesia. "
    "Berikan definisi, tiga fakta penting, dan satu contoh. "
    "Gunakan maksimal 250 kata."
)

parser = StrOutputParser()

chain = prompt | client | parser

hasil = chain.invoke({"topik": "LangChain"})

if hasil:
    print(hasil)
else:
    print("Model tidak mengembalikan jawaban. Coba jalankan lagi.")


