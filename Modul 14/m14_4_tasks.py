import os
from crewai import Agent, LLM, Task

# Set dummy OPENAI_API_KEY agar CrewAI menggunakan endpoint Ollama lokal tanpa error
os.environ.setdefault("OPENAI_API_KEY", "NA")

llm = LLM(
    model="ollama/qwen2.5:3b",
    base_url="http://localhost:11434",
)

# ── Definisi Agents (dari 14.2) ──────────────────────────────────────────────
researcher = Agent(
    role="AI Research Analyst",
    goal="Find accurate and up-to-date information on AI topics.",
    backstory=(
        "You are a senior AI research analyst with 10 years of experience "
        "reading and synthesising academic papers, technical blogs, and release notes. "
        "You are known for being precise, citing sources, and never making things up."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)

writer = Agent(
    role="Technical Content Writer",
    goal="Write clear, accurate, and engaging technical content for developers.",
    backstory=(
        "You are a technical writer with a background in software engineering. "
        "You excel at turning complex AI concepts into practical, well-structured guides. "
        "You always write in active voice, use concrete examples, and avoid jargon."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=False,
)

reviewer = Agent(
    role="Content Quality Reviewer",
    goal="Ensure content is accurate, complete, and publication-ready.",
    backstory=(
        "You are a meticulous editor who reviews technical content for "
        "accuracy, clarity, and completeness. You check facts, flag unsupported claims, "
        "and ensure the writing matches the target audience's level."
    ),
    llm=llm,
    verbose=True,
    allow_delegation=True,
)


# ── Definisi Tasks (14.4) ────────────────────────────────────────────────────
def make_tasks(researcher: Agent, writer: Agent, reviewer: Agent):
  research_task = Task(
      description="""Research the current state of RAG (Retrieval-Augmented Generation) in 2024-2025.
Cover:
1. What RAG is and how it works
2. The main variants (naive RAG, advanced RAG, modular RAG)
3. Common production challenges and how practitioners solve them
4. Three real-world use cases with specific results
Be specific and factual. Avoid vague generalities.""",
      expected_output="""A structured research report with:
- A clear definition and architecture overview
- Descriptions of the three RAG variants with key differences
- Three specific production challenges with concrete solutions
- Three real-world use cases with measurable outcomes
Format: Markdown with headers.""",
      agent=researcher,
  )

  writing_task = Task(
      description="""Using the research report provided, write a developer guide titled
'Building Production RAG Systems'.
The guide should:
- Be written for senior developers with Python experience
- Include a conceptual overview, implementation considerations, and common pitfalls
- Use active voice and concrete examples
- Be 600-800 words""",
      expected_output="A complete, publication-ready developer guide in Markdown format.",
      agent=writer,
      context=[research_task],  # Bergantung pada output research_task
  )

  review_task = Task(
      description="""Review the developer guide for:
1. Technical accuracy - flag any claims that are incorrect or unsupported
2. Completeness - identify important topics that are missing
3. Clarity - suggest any sections that need rewriting
4. Audience fit - check that the level is appropriate for senior developers
Provide a structured review with specific, actionable feedback.""",
      expected_output="""A quality review report containing:
- Overall assessment (Ready / Needs Revision)
- Specific accuracy issues (if any) with corrections
- Missing content recommendations
- Clarity improvement suggestions
- Final verdict""",
      agent=reviewer,
      context=[writing_task],  # Bergantung pada output writing_task
  )

  return research_task, writing_task, review_task


# ── Verifikasi Pembuatan Tasks ───────────────────────────────────────────────
if __name__ == "__main__":
  t_research, t_writing, t_review = make_tasks(researcher, writer, reviewer)
  print("=== 14.4 Verifikasi Konfigurasi Tasks CrewAI ===")
  for idx, t in enumerate([t_research, t_writing, t_review], 1):
    ctx_count = len(t.context) if isinstance(t.context, list) else 0
    print(f"{idx}. Assigned Agent : {t.agent.role}")
    print(f"   Context Deps   : {ctx_count} task(s)")
    print(f"   Expected Output: {t.expected_output.splitlines()[0]}...\n")