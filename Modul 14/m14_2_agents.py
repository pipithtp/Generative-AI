import os
from crewai import Agent, LLM

# CrewAI v0.80+ menggunakan class LLM bawaan dengan prefix 'ollama/'
# Pastikan dummy OPENAI_API_KEY diset agar CrewAI tidak memicu error pengecekan env
os.environ.setdefault("OPENAI_API_KEY", "NA")

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
)

# ── 1. Research Specialist ───────────────────────────────────────────────────
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory="""You are a senior AI research analyst with 10 years of experience
reading and synthesising academic papers, technical blogs, and release notes.
You are known for being precise, citing sources, and never making things up.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,  # Agent ini tidak mendelegasikan tugas ke agent lain
    max_iter=5,  # Maksimal iterasi penalaran sebelum berhenti
)

# ── 2. Writing Specialist ────────────────────────────────────────────────────
writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory="""You are a technical writer with a background in software engineering.
You excel at turning complex AI concepts into practical, well-structured guides.
You always write in active voice, use concrete examples, and avoid jargon.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

# ── 3. Quality Reviewer ──────────────────────────────────────────────────────
reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory="""You are a meticulous editor who reviews technical content for
accuracy, clarity, and completeness. You check facts, flag unsupported claims,
and ensure the writing matches the target audience's level.""",
    llm=llm,
    verbose=True,
    allow_delegation=True,  # Reviewer dapat mendelegasikan revisi kembali ke writer
)

# Verifikasi konfigurasi ketiga agent
if __name__ == "__main__":
  print("=== CrewAI Agents Berhasil Diinisialisasi (Ollama qwen2.5:3b) ===")
  for idx, agent in enumerate([researcher, writer, reviewer], 1):
    print(f"{idx}. Role       : {agent.role}")
    print(f"   Goal       : {agent.goal}")
    print(f"   Delegation : {agent.allow_delegation}\n")