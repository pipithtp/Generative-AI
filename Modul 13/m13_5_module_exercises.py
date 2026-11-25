import operator
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

# Inisialisasi model Ollama lokal (100% gratis & offline)
base_llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0,
)

# =====================================================================
# SOAL 1: Research Agent (web_search, summarise, save_note)
# =====================================================================
print("=" * 75)
print("SOAL 1: LangGraph Research Agent")
print("=" * 75)

saved_notes: list[str] = []


def web_search(query: str) -> str:
  """Search the web for factual information about a topic."""
  mock_db = {
      "langgraph": (
          "LangGraph is a library by LangChain for building stateful,"
          " multi-actor applications with LLMs. It models agent workflows as"
          " cyclic graphs with nodes, edges, and shared state, supporting"
          " persistence and human-in-the-loop checkpoints."
      ),
      "rag": (
          "Retrieval-Augmented Generation (RAG) improves LLM accuracy by"
          " fetching relevant documents from an external vector database before"
          " generating answers."
      ),
  }
  q_lower = query.lower()
  for key, val in mock_db.items():
    if key in q_lower:
      return val
  return (
      "General search result: AI agents combine LLMs with planning, memory,"
      " and tool execution."
  )


def summarise(text: str) -> str:
  """Summarise a given piece of text into a concise bullet-point summary."""
  resp = base_llm.invoke([
      SystemMessage(
          content="Summarise the provided text concisely in 2 bullet points."
      ),
      HumanMessage(content=text),
  ])
  return resp.content


def save_note(content: str) -> str:
  """Save a structured research note to persistent storage."""
  saved_notes.append(content)
  return f"Note saved successfully (total notes: {len(saved_notes)})."


RESEARCH_TOOLS = {
    "web_search": web_search,
    "summarise": summarise,
    "save_note": save_note,
}


class ResearchState(TypedDict):
  messages: Annotated[list, operator.add]


research_llm = base_llm.bind_tools([web_search, summarise, save_note])


def research_agent_node(state: ResearchState) -> dict:
  sys_msg = SystemMessage(
      content=(
          "You are a research assistant. Follow these exact steps:\n"
          "1. Call web_search to gather information on the user's topic.\n"
          "2. Call summarise on the search result.\n"
          "3. Call save_note with a structured note containing Title and"
          " Summary.\n"
          "4. Provide a brief final confirmation to the user."
      )
  )
  resp = research_llm.invoke([sys_msg] + state["messages"])
  return {"messages": [resp]}


def research_tool_node(state: ResearchState) -> dict:
  last_msg = state["messages"][-1]
  outputs = []
  for tc in last_msg.tool_calls:
    print(f"  [Research Tool] {tc['name']}({tc['args']})")
    res = RESEARCH_TOOLS[tc["name"]](**tc["args"])
    outputs.append(ToolMessage(content=str(res), tool_call_id=tc["id"]))
  return {"messages": outputs}


def route_research(state: ResearchState) -> str:
  last = state["messages"][-1]
  return "tools" if getattr(last, "tool_calls", None) else END


g1 = StateGraph(ResearchState)
g1.add_node("agent", research_agent_node)
g1.add_node("tools", research_tool_node)
g1.add_edge(START, "agent")
g1.add_conditional_edges(
    "agent", route_research, {"tools": "tools", END: END}
)
g1.add_edge("tools", "agent")
research_app = g1.compile()

res1 = research_app.invoke(
    {"messages": [HumanMessage(content="Research LangGraph and save a note.")]}
)
print("\nSaved Notes List:")
for idx, note in enumerate(saved_notes, 1):
  print(f"--- Note #{idx} ---\n{note}\n")


# =====================================================================
# SOAL 2: ReAct Agent dengan Maximum Step Limit (Max 8 Tool Calls)
# =====================================================================
print("=" * 75)
print("SOAL 2: ReAct Agent dengan Batas Maksimal 8 Tool Calls")
print("=" * 75)


def endless_search(query: str) -> str:
  """Search for data; returns incomplete info requiring more lookups."""
  return (
      f"Partial clue for '{query}': need deeper search with another keyword."
  )


class BoundedReActState(TypedDict):
  messages: Annotated[list, operator.add]
  tool_call_count: int
  final_status: str


bounded_llm = base_llm.bind_tools([endless_search])


def bounded_agent_node(state: BoundedReActState) -> dict:
  if state["tool_call_count"] > 8:
    return {
        "messages": [AIMessage(content="Reached step limit.")],
        "final_status": "Reached step limit.",
    }
  sys_msg = SystemMessage(
      content=(
          "You must keep calling endless_search repeatedly until you find"
          " the secret code 'XYZ-999'. Never guess without finding it."
      )
  )
  resp = bounded_llm.invoke([sys_msg] + state["messages"])
  return {"messages": [resp]}


def bounded_tool_node(state: BoundedReActState) -> dict:
  last_msg = state["messages"][-1]
  new_count = state["tool_call_count"]
  outputs = []

  for tc in last_msg.tool_calls:
    new_count += 1
    if new_count > 8:
      print(f"  [Limit Triggered] Tool call #{new_count} melebihi batas 8!")
      return {
          "messages": [AIMessage(content="Reached step limit.")],
          "tool_call_count": new_count,
          "final_status": "Reached step limit.",
      }
    print(f"  [Tool Call #{new_count}/8] {tc['name']}({tc['args']})")
    out = endless_search(**tc["args"])
    outputs.append(ToolMessage(content=out, tool_call_id=tc["id"]))

  return {"messages": outputs, "tool_call_count": new_count}


def route_bounded(state: BoundedReActState) -> str:
  if state.get("final_status") == "Reached step limit.":
    return END
  last = state["messages"][-1]
  if getattr(last, "tool_calls", None):
    return "tools"
  return END


g2 = StateGraph(BoundedReActState)
g2.add_node("agent", bounded_agent_node)
g2.add_node("tools", bounded_tool_node)
g2.add_edge(START, "agent")
g2.add_conditional_edges(
    "agent", route_bounded, {"tools": "tools", END: END}
)
g2.add_conditional_edges(
    "tools",
    lambda s: END if s.get("final_status") == "Reached step limit." else "agent",
    {"agent": "agent", END: END},
)
bounded_app = g2.compile()

res2 = bounded_app.invoke({
    "messages": [HumanMessage(content="Find the secret code XYZ-999.")],
    "tool_call_count": 0,
    "final_status": "OK",
})
print("\nFinal Output Soal 2:", res2["messages"][-1].content)
print("Total Tool Calls   :", res2["tool_call_count"])


# =====================================================================
# SOAL 3: PlanAndExecute Agent (Planner & Executor Nodes)
# =====================================================================
print("\n" + "=" * 75)
print("SOAL 3: PlanAndExecute Agent")
print("=" * 75)


class PlanExecuteState(TypedDict):
  objective: str
  plan: list[str]
  current_step: int
  step_results: Annotated[list[str], operator.add]


def mock_step_tool(step_instruction: str) -> str:
  """Simulasi eksekusi tool untuk satu langkah rencana."""
  return f"Executed -> '{step_instruction}' [Status: Completed]"


def planner_node(state: PlanExecuteState) -> dict:
  prompt = (
      f"Create a concise 3-step numbered plan to achieve: {state['objective']}.\n"
      "Format strictly as:\n1. First step\n2. Second step\n3. Third step"
  )
  resp = base_llm.invoke([HumanMessage(content=prompt)])
  lines = [
      re.sub(r"^\d+[\.\)]\s*", "", line.strip())
      for line in resp.content.splitlines()
      if re.match(r"^\d+[\.\)]", line.strip())
  ]
  if not lines:
    lines = [
        "Gather dataset requirements",
        "Configure local vector index",
        "Validate retrieval accuracy",
    ]
  print("Generated Plan:")
  for i, s in enumerate(lines, 1):
    print(f"  {i}. {s}")
  return {"plan": lines, "current_step": 0}


def executor_node(state: PlanExecuteState) -> dict:
  idx = state["current_step"]
  step_text = state["plan"][idx]
  print(f"  [Executor] Running step index {idx} ({idx + 1}/{len(state['plan'])}): {step_text}")
  result = mock_step_tool(step_text)
  return {
      "step_results": [result],
      "current_step": idx + 1,
  }


def has_more_steps(state: PlanExecuteState) -> str:
  if state["current_step"] < len(state["plan"]):
    return "executor"
  return END


g3 = StateGraph(PlanExecuteState)
g3.add_node("planner", planner_node)
g3.add_node("executor", executor_node)
g3.add_edge(START, "planner")
g3.add_edge("planner", "executor")
g3.add_conditional_edges(
    "executor", has_more_steps, {"executor": "executor", END: END}
)
plan_exec_app = g3.compile()

res3 = plan_exec_app.invoke({
    "objective": "Deploy a local RAG pipeline using Ollama and ChromaDB",
    "plan": [],
    "current_step": 0,
    "step_results": [],
})
print("\nStep Execution Logs:")
for r in res3["step_results"]:
  print(" -", r)


# =====================================================================
# SOAL 4: Multi-Turn Agent dengan MemorySaver & thread_id
# =====================================================================
print("\n" + "=" * 75)
print("SOAL 4: Multi-Turn Agent dengan MemorySaver (thread_id)")
print("=" * 75)


class ChatMemoryState(TypedDict):
  messages: Annotated[list, operator.add]


def chat_node(state: ChatMemoryState) -> dict:
  sys_msg = SystemMessage(
      content="You are a helpful assistant with memory of previous turns."
  )
  resp = base_llm.invoke([sys_msg] + state["messages"])
  return {"messages": [resp]}


memory = MemorySaver()
g4 = StateGraph(ChatMemoryState)
g4.add_node("chat", chat_node)
g4.add_edge(START, "chat")
g4.add_edge("chat", END)

chat_app = g4.compile(checkpointer=memory)
thread_config = {"configurable": {"thread_id": "student-session-42"}}

# Turn 1: Memperkenalkan konteks ke agent
print("Turn 1 (User): Hi, my name is Pipit and I am building a Reog Ponorogo app in Unity.")
out_t1 = chat_app.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "Hi, my name is Pipit and I am building a Reog Ponorogo"
                    " app in Unity."
                )
            )
        ]
    },
    config=thread_config,
)
print("Turn 1 (AI)  :", out_t1["messages"][-1].content, "\n")

# Turn 2: Menguji apakah agent mengingat konteks dari Turn 1 pada thread_id yang sama
print("Turn 2 (User): What is my name and what project am I working on?")
out_t2 = chat_app.invoke(
    {
        "messages": [
            HumanMessage(
                content="What is my name and what project am I working on?"
            )
        ]
    },
    config=thread_config,
)
print("Turn 2 (AI)  :", out_t2["messages"][-1].content)