import operator
from typing import Annotated, TypedDict
from langgraph.graph import END, START, StateGraph


# 1. State adalah TypedDict yang dibagikan ke seluruh node
# Annotated dengan fungsi reducer (operator.add) menentukan cara menggabungkan update list
class AgentState(TypedDict):
  messages: Annotated[list[str], operator.add]  # Append pesan baru ke list
  step_count: int
  final_answer: str


# 2. Nodes adalah fungsi Python biasa yang menerima State dan mengembalikan dict update
def think_node(state: AgentState) -> dict:
  thought = f"Step {state['step_count'] + 1}: analysing the problem..."
  return {
      "messages": [thought],
      "step_count": state["step_count"] + 1,
  }


def answer_node(state: AgentState) -> dict:
  return {"final_answer": "Based on my analysis, the answer is X."}


# 3. Conditional edge - menentukan rute node selanjutnya berdasarkan kondisi State
def should_continue(state: AgentState) -> str:
  if state["step_count"] < 2:
    return "think"
  return "answer"


# 4. Membangun dan mengompilasi Graph
graph = StateGraph(AgentState)

graph.add_node("think", think_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "think")
graph.add_conditional_edges(
    "think", should_continue, {"think": "think", "answer": "answer"}
)
graph.add_edge("answer", END)

app = graph.compile()

# 5. Eksekusi Graph
print("=== Menjalankan LangGraph Core Concepts ===")
result = app.invoke({"messages": [], "step_count": 0, "final_answer": ""})

print("\nMessages:")
for msg in result["messages"]:
  print(f"  - {msg}")

print(f"\nStep Count : {result['step_count']}")
print(f"Answer     : {result['final_answer']}")