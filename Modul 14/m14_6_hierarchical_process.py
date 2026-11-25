import os
from crewai import Agent, Crew, LLM, Process, Task

# Set dummy OPENAI_API_KEY agar CrewAI berjalan 100% lokal dengan Ollama
os.environ.setdefault("OPENAI_API_KEY", "NA")

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
)

# ── 1. Manager Agent (Mengkoordinasikan & Mendelegasikan Tugas) ──────────────
manager = Agent(
    role="AI Content Production Manager",
    goal="Coordinate a team to produce high-quality AI content efficiently.",
    backstory="""You are an experienced content production manager who excels at
breaking down projects, assigning work to the right specialist, and ensuring
quality standards are met before publication.""",
    llm=llm,
    allow_delegation=True,
    verbose=True,
    max_iter=5,
)

# ── 2. Worker Agents ─────────────────────────────────────────────────────────
researcher = Agent(
    role="AI Research Analyst",
    goal="Research AI topics thoroughly.",
    backstory="Senior researcher specialising in LLM systems and RAG.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
    max_iter=4,
)

writer = Agent(
    role="Technical Writer",
    goal="Write developer-focused technical content.",
    backstory="Technical writer with Python and ML experience.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
    max_iter=4,
)

# ── 3. Main Task (Dikelola oleh Manager) ─────────────────────────────────────
main_task = Task(
    description="""Produce a complete technical brief on 'Vector Databases for RAG in Production'.
Include: what they are, how to choose one, and three implementation best practices.""",
    expected_output=(
        "A structured technical brief in Markdown, 400-500 words."
    ),
)

# ── 4. Hierarchical Crew ─────────────────────────────────────────────────────
# Catatan CrewAI v0.80+: manager_agent dipisahkan dari list worker agents
crew = Crew(
    agents=[researcher, writer],
    tasks=[main_task],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=True,
    memory=False,
)

if __name__ == "__main__":
  print("=== Menjalankan Hierarchical CrewAI (Ollama qwen2.5:3b) ===")
  result = crew.kickoff()
  print("\n=== FINAL TECHNICAL BRIEF ===")
  print(result)