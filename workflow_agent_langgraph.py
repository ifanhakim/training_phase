from typing_extensions import TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated

from small_project import (
    count_hospitals,
    get_or_create_collection,
    index_hospitals,
    model,
    search_hospital,
)


class WorkflowState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    route: str
    search_result: dict
    answer: str


answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Anda adalah agen informasi operasional rumah sakit. "
            "Jawab hanya berdasarkan hasil pencarian. "
            "Jangan memberikan diagnosis medis, saran obat, atau informasi rekam medis. "
            "Jika data tidak cukup, katakan bahwa informasi belum tersedia.",
        ),
        (
            "human",
            "Pertanyaan pengguna:\n{question}\n\n"
            "Hasil pencarian Typesense:\n{search_result}",
        ),
    ]
)
answer_chain = answer_prompt | model | StrOutputParser()


def get_question(state: WorkflowState):
    return state["messages"][-1].content


def classify_question(state: WorkflowState):
    question = get_question(state).lower()
    medical_words = {
        "diagnosis",
        "penyakit",
        "obat",
        "gejala",
        "sakit",
        "nyeri",
        "serangan jantung",
    }

    if any(word in question for word in medical_words):
        route = "medical"
    else:
        route = "hospital_search"

    return {"route": route}


def route_question(state: WorkflowState):
    return state["route"]


def handle_medical_question(state: WorkflowState):
    answer = (
        "Saya tidak dapat memberikan diagnosis atau saran obat. "
        "Silakan berkonsultasi dengan dokter atau segera ke IGD rumah sakit terdekat."
    )
    return {"answer": answer}


def search_hospitals(state: WorkflowState):
    question = get_question(state)
    result = search_hospital.invoke({"query": question})
    return {"search_result": result}


def generate_answer(state: WorkflowState):
    question = get_question(state)
    result = answer_chain.invoke(
        {
            "question": question,
            "search_result": state["search_result"],
        }
    )
    return {"answer": result}


workflow = StateGraph(WorkflowState)
workflow.add_node("classify", classify_question)
workflow.add_node("medical_response", handle_medical_question)
workflow.add_node("search_hospitals", search_hospitals)
workflow.add_node("generate_answer", generate_answer)

workflow.add_edge(START, "classify")
workflow.add_conditional_edges(
    "classify",
    route_question,
    {
        "medical": "medical_response",
        "hospital_search": "search_hospitals",
    },
)
workflow.add_edge("medical_response", END)
workflow.add_edge("search_hospitals", "generate_answer")
workflow.add_edge("generate_answer", END)

app = workflow.compile()


def show_graph():
    graph = app.get_graph()
    print("Node:", list(graph.nodes))
    print("Edge:")
    for edge in graph.edges:
        print(f"  {edge.source} -> {edge.target}")


def run_workflow(question):
    result = app.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    return result["answer"]


if __name__ == "__main__":
    try:
        collection = get_or_create_collection()
        number_of_hospitals = count_hospitals(collection)

        if number_of_hospitals == 0:
            index_hospitals(collection)
        else:
            print(f"Typesense berisi {number_of_hospitals} rumah sakit.")

        show_graph()
        print('Workflow Agent (ketik "exit" untuk keluar)')

        while True:
            question = input("You: ").strip()
            if question.lower() == "exit":
                print("Sampai jumpa!")
                break

            if question:
                print("AI:", run_workflow(question))
    except Exception as error:
        print(f"Program gagal dijalankan: {error}")
