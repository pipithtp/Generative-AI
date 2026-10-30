import pandas as pd

data = [
    {"model": "gpt-4o", "provider": "OpenAI", "context_k": 128, "cost_input": 2.50},
    {"model": "claude-sonnet-4-5", "provider": "Anthropic", "context_k": 200, "cost_input": 3.00},
    {"model": "gemini-1.5-pro", "provider": "Google", "context_k": 1000, "cost_input": 1.25},
    {"model": "llama-3.1-70b", "provider": "Meta", "context_k": 128, "cost_input": 0.00},
]

df = pd.DataFrame(data)
print(df.shape)  # (4, 4)
print(df.dtypes)
print(df.head())
print(df.describe())


# Selection and Filtering

df = pd.DataFrame([
    {"model": "gpt-4o", "context_k": 128, "cost_input": 2.50},
    {"model": "claude-sonnet-4-5", "context_k": 200, "cost_input": 3.00},
    {"model": "gemini-1.5-pro", "context_k": 1000, "cost_input": 1.25},
    {"model": "llama-3.1-70b", "context_k": 128, "cost_input": 0.00},
])

# Select column
print(df["model"].tolist())

# Filter rows
affordable = df[df["cost_input"] < 2.0]
print(affordable)

# Multiple conditions
big_and_cheap = df[(df["context_k"] >= 128) & (df["cost_input"] < 2.0)]
print(big_and_cheap[["model", "context_k", "cost_input"]])

# loc (label-based) vs iloc (position-based)
print(df.loc[0, "model"])  # 'gpt-4o'
print(df.iloc[0, 0])  # 'gpt-4o'


# Cleaning and Transforming

import numpy as np

# Simulate a messy evaluation dataset
raw = pd.DataFrame({
    "prompt": ["Q1", "Q2", "Q3", "Q4", "Q5"],
    "response": ["OK", None, "Good", "Bad", "OK"],
    "score": [0.9, None, 0.85, 0.3, 0.88],
    "latency_ms": [410, 520, None, 390, 480],
})

# Inspect missing data
print(raw.isnull().sum())

# Fill missing values
raw["score"] = raw["score"].fillna(raw["score"].mean())
raw["latency_ms"] = raw["latency_ms"].fillna(raw["latency_ms"].median())

# Drop rows still missing critical columns
clean = raw.dropna(subset=["response"]).copy()

# Add computed column
clean["pass"] = clean["score"] >= 0.7
print(clean)
print(f"Pass rate: {clean['pass'].mean():.0%}")


# GroupBy and Aggregation

evals = pd.DataFrame({
    "model": ["claude", "gpt-4o", "claude", "gpt-4o", "claude", "gpt-4o"],
    "task": ["qa", "qa", "summarise", "summarise", "code", "code"],
    "score": [0.91, 0.88, 0.85, 0.82, 0.93, 0.90],
    "latency_ms": [420, 380, 610, 550, 340, 300],
})

# Average score per model
print(evals.groupby("model")["score"].mean())

# Multiple aggregations
summary = evals.groupby("model").agg(
    avg_score=("score", "mean"),
    avg_latency=("latency_ms", "mean"),
    num_tasks=("task", "count"),
)

print(summary)

# Pivot table - model vs task
pivot = evals.pivot_table(
    values="score", index="model", columns="task", aggfunc="mean"
)

print(pivot)

