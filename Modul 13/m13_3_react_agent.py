import json
import math
import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# ── Tools ─────────────────────────────────────────────────────────────────────
def calculator(expression: str) -> str:
  """Evaluate a safe mathematical expression. Use Python math syntax."""
  try:
    allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    allowed.update({"abs": abs, "round": round})
    result = eval(expression, {"__builtins__": {}}, allowed)
    return str(result)
  except Exception as e:
    return f"Error: {e}"


def token_cost_lookup(model: str) -> str:
  """Look up the cost per 1M tokens for a given LLM model."""
  costs = {
      "gpt-4o": {"input": 2.50, "output": 10.00},
      "gpt-4o-mini": {"input": 0.15, "output": 0.60},
      "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
      "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
  }
  if model not in costs:
    return f"Unknown model: {model}. Known: {list(costs.keys())}"
  c = costs[model]
  return json.dumps({
      "model": model,
      "input_per_1M": c["input"],
      "output_per_1M": c["output"],
  })


TOOLS = {
    "calculator": calculator,
    "token_cost_lookup": token_cost_lookup,
}


# ── State ─────────────────────────────────────────────────────────────────────
class ReActState(TypedDict):
  messages: Annotated[list, operator.add]


# ── LLM (Ollama Lokal dengan Tool Binding) ────────────────────────────────────
llm = ChatOllama(
    model="qwen2.5:3b",
    base_url="http://localhost:11434",
    temperature=0,
).bind_tools([calculator, token_cost_lookup])

SYSTEM = SystemMessage(
    content="""You are a helpful AI assistant with access to two tools:
1. `token_cost_lookup(model)`: Use this FIRST to look up the pricing per 1M tokens for an LLM model (e.g., model="gpt-4o").
2. `calculator(expression)`: Use this SECOND to evaluate math expressions (e.g., expression="10000 * 800").

Follow these rules strictly:
- Never pass `model` to `calculator`.
- Never pass `expression` to `token_cost_lookup`.
- Think step by step and call the appropriate tool."""
)


# ── Nodes ─────────────────────────────────────────────────────────────────────
def agent_node(state: ReActState) -> dict:
  """Call the LLM - it may request tool use or give a final answer."""
  response = llm.invoke([SYSTEM] + state["messages"])
  return {"messages": [response]}


def tool_node(state: ReActState) -> dict:
  """Execute all requested tool calls safely and return results or error feedback."""
  last_msg = state["messages"][-1]
  results = []
  for tool_call in last_msg.tool_calls:
    fn_name = tool_call["name"]
    fn_args = tool_call["args"]
    print(f"  [Tool Call] {fn_name}({fn_args})")

    if fn_name not in TOOLS:
      output = (
          f"Error: Tool '{fn_name}' does not exist. Available tools:"
          f" {list(TOOLS.keys())}"
      )
    else:
      try:
        output = TOOLS[fn_name](**fn_args)
      except TypeError as e:
        # Jika LLM salah mengirim argumen, kembalikan pesan error agar LLM memperbaiki sendiri
        output = (
            f"Argument Error calling '{fn_name}': {e}. Did you mean to call"
            " 'token_cost_lookup' for pricing or 'calculator' with"
            " 'expression'?"
        )
      except Exception as e:
        output = f"Execution Error: {e}"

    print(f"  [Tool Output] {output}")
    results.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
  return {"messages": results}


def should_use_tools(state: ReActState) -> str:
  last = state["messages"][-1]
  if hasattr(last, "tool_calls") and last.tool_calls:
    return "tools"
  return END


# ── Graph ─────────────────────────────────────────────────────────────────────
graph = StateGraph(ReActState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent", should_use_tools, {"tools": "tools", END: END}
)
graph.add_edge("tools", "agent")

agent = graph.compile()

# ── Run ───────────────────────────────────────────────────────────────────────
question = """I'm running a pipeline that processes 10,000 requests per day.
Each request uses 800 input tokens and 300 output tokens with gpt-4o.
What is my daily cost in USD?"""

print("=== Menjalankan ReAct Agent (Ollama qwen2.5:3b) ===")
print(f"Question:\n{question}\n")

result = agent.invoke({"messages": [HumanMessage(content=question)]})

# Cetak jawaban akhir
for msg in result["messages"]:
  if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
    print("\nFinal answer:\n", msg.content)