from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama

# =====================================================
# 1. STATE (SHARED MEMORY)
# =====================================================

class ArticleState(TypedDict):
    topic: str
    research_notes: List[str]
    article: str
    revision_count: int
    next_step: str

# =====================================================
# 2. LLM SETUP
# =====================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0.5,
    max_tokens=1000
)

# =====================================================
# 3. AGENTS
# =====================================================
def research_agent(state: ArticleState):
    print("\n[Research Agent] Collecting research notes...")

    prompt = f"""
    Research the topic: {state['topic']}
    Provide exactly 3 short bullet points and talk about Minecraft , Valorant, and any other game.
    """

    response = llm.invoke(prompt)

    return {
        "research_notes": response.content.split("\n"),
        "next_step": "writer"
    }


def writer_agent(state: ArticleState):
    print("\n[Writer Agent] Writing article...")

    notes = "\n".join(state["research_notes"])
    prompt = f"""
    Write an article of at least 120 words using the following notes:

    {notes}
    """

    response = llm.invoke(prompt)
    article = response.content
    word_count = len(article.split())

    print(f"[Writer Agent] Word count: {word_count}")

    if word_count < 120:
        print("[Writer Agent] Too short → rewrite required")
        next_step = "rewrite"
    else:
        print("[Writer Agent] Accepted")
        next_step = "end"

    return {
        "article": article,
        "revision_count": state["revision_count"] + 1,
        "next_step": next_step
    }


# =====================================================
# 4. SUPERVISOR
# =====================================================
def supervisor(state: ArticleState):
    print(f"\n[Supervisor] Next step → {state['next_step']}")
    return state

def route_next_step(state: ArticleState):
    return state["next_step"]


# =====================================================
# 5. BUILD LANGGRAPH
# =====================================================
graph = StateGraph(ArticleState)

graph.add_node("research", research_agent)
graph.add_node("writer", writer_agent)
graph.add_node("supervisor", supervisor)

graph.add_edge("research", "supervisor")
graph.add_edge("writer", "supervisor")

graph.add_conditional_edges(
    "supervisor",
    route_next_step,
    {
        "writer": "writer",
        "rewrite": "writer",  
        "end": END
    }
)

graph.set_entry_point("research")
app = graph.compile()

def save_architecture():
    png = app.get_graph().draw_mermaid_png()
    with open("architecture.png", "wb") as f:
        f.write(png)
    print("\n[System] architecture.png saved")


if __name__ == "__main__":
    save_architecture()

    print("\n======= MULTI-AGENT SYSTEM START =======\n")

    final_state = app.invoke({
        "topic": "Games using AI for NPC behavior",
        "research_notes": [],
        "article": "",
        "revision_count": 0,
        "next_step": ""
    })

    print("\n======= FINAL ARTICLE =======\n")
    print(final_state["article"])
    print("="*30)
    print("Revisions:", final_state["revision_count"])
