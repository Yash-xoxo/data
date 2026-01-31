#!/usr/bin/env python3
"""
Reads a debate transcript file and asks another Ollama model
to produce a human-like final conclusion.
"""

from pathlib import Path
import requests
import sys

OLLAMA_BASE = "http://localhost:11434"
SUMMARY_MODEL = "qwen3:1.7b"

OPTIONS = {
    "num_ctx": 1024,
    "num_predict": 256,
    "temperature": 0.7
}

def call_ollama(prompt: str):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": OPTIONS
    }
    r = requests.post(url, json=payload, timeout=240)
    r.raise_for_status()
    return r.json().get("response", "").strip()

def main(debate_file: Path):
    if not debate_file.exists():
        print("Debate file not found")
        sys.exit(1)

    debate_text = debate_file.read_text(encoding="utf-8")

    prompt = f"""
You are a neutral human moderator.

Below is a debate between two AI models.
Your task:
- Read the entire debate carefully
- Extract the strongest arguments from both sides
- Resolve contradictions
- Write a final conclusion in a **human, natural, non-AI tone**
- Do NOT mention models or AI
- Write as if a thoughtful human analyzed the discussion

Debate transcript:
-------------------
{debate_text}
-------------------

Final conclusion:
"""

    print("\n=== Generating human-style conclusion ===\n")
    conclusion = call_ollama(prompt)
    print(conclusion)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 summarize_debate.py debates/debate_xxx.txt")
        sys.exit(1)

    main(Path(sys.argv[1]))
