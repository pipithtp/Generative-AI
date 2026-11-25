import operator
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph


class ReviewState(TypedDict):
  plan: str
  approved: bool
  result: str
  messages: Annotated[list[str], operator.add]


def plan_node(state: ReviewState) -> dict:
  plan = (
      "Step 1: Query vector DB. Step 2: Summarise results. Step 3: Format"
      " response."
  )
  return {"plan": plan, "messages": [f"Plan created: {plan}"]}


def execute_node(state: ReviewState) -> dict:
  if not state["approved"]:
    return {
        "result": "Execution skipped - plan not approved.",
        "messages": ["Skipped."],
    }
  return {
      "result": "Pipeline executed successfully.",
      "messages": ["Executed."],
  }


def review_gate(state: ReviewState) -> str:
  """Interrupt here for human review."""
  return "execute"


# MemorySaver mengaktifkan checkpointing (in-memory untuk dev; gunakan SqliteSaver untuk prod)
checkpointer = MemorySaver()

graph = StateGraph(ReviewState)
graph.add_node("plan", plan_node)
graph.add_node("execute", execute_node)

graph.add_edge(START, "plan")
graph.add_edge("plan", "execute")
graph.add_edge("execute", END)

# interrupt_before menjeda graph tepat sebelum node 'execute' dijalankan
app = graph.compile(checkpointer=checkpointer, interrupt_before=["execute"])

config = {"configurable": {"thread_id": "review-session-1"}}

print("=== 1. Menjalankan Graph hingga Titik Interrupt ===")
state = app.invoke(
    {"plan": "", "approved": False, "result": "", "messages": []},
    config=config,
)

print("Plan   :", state["plan"])
print("Status : Waiting for human approval...")

# Cek node berikutnya yang tertunda di antrean checkpoint
snapshot = app.get_state(config)
print("Next node pending:", snapshot.next)

print("\n=== 2. Human Memberikan Approval & Melanjutkan Graph ===")
# Perbarui state di dalam checkpoint sebelum melanjutkan eksekusi
app.update_state(config, {"approved": True})

# Pass None sebagai input agar LangGraph melanjutkan dari checkpoint terakhir
final = app.invoke(None, config=config)

print("Result  :", final["result"])
print("Messages:", final["messages"])