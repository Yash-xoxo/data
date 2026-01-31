#!/usr/bin/env python3
"""
Python UI on port 3030 that runs:
1) Curious question generator
2) Detailed explainer
3) Human-like summarizer

Uses Ollama /api/generate
"""
from pathlib import Path
import re
from flask import Flask, request, render_template_string
import requests
import time
import os

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_CURIOUS = "granite4:350m"
MODEL_EXPLAINER = "granite4:350m"
MODEL_SUMMARY = "granite4:350m"

NUM_CTX = 512
NUM_PREDICT = 128
ROUNDS = 4
CHAT_FILE = "chat.txt"
SUMMARY_FILE = "summary.txt"
TMP_FILE = "chat.tmp"

# ----------------------------------

app = Flask(__name__)

# ---------- Ollama Call ----------
def ollama_call(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "temperature": 0.7
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return r.json().get("response", "").strip() or "[EMPTY RESPONSE]"


# ---------- Core Logic ----------
def run_conversation(topic):
    # Cleanup
    for f in (CHAT_FILE, SUMMARY_FILE, TMP_FILE):
        if os.path.exists(f):
            os.remove(f)

    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        f.write(f"Topic: {topic}\n")
        f.write("-" * 40 + "\n\n")

    open(TMP_FILE, "w").close()

    # Conversation loop
    for i in range(ROUNDS):
        # ----- Curious Model -----
        prompt_q = f"""
You are a very curious human.
Topic: {topic}

Conversation so far:
{open(TMP_FILE).read()}

Ask ONE thoughtful, curiosity-driven question.
Avoid summaries.
"""
        question = ollama_call(MODEL_CURIOUS, prompt_q)

        with open(CHAT_FILE, "a", encoding="utf-8") as f:
            f.write("[Curious]\n")
            f.write(question + "\n\n")

        with open(TMP_FILE, "a", encoding="utf-8") as f:
            f.write(f"Curious: {question}\n")

        time.sleep(0.8)

        # ----- Explainer Model -----
        prompt_a = f"""
You are an expert explainer.

Topic: {topic}
Question:
{question}

Explain in deep detail.
Be clear and structured.
"""
        answer = ollama_call(MODEL_EXPLAINER, prompt_a)

        with open(CHAT_FILE, "a", encoding="utf-8") as f:
            f.write("[Explainer]\n")
            f.write(answer + "\n\n")

        with open(TMP_FILE, "a", encoding="utf-8") as f:
            f.write(f"Explainer: {answer}\n")

        time.sleep(0.8)

    # ----- Summarizer -----
    summary_prompt = f"""
You are a thoughtful human reader.

Below is a conversation where one person asked curious questions
and another provided detailed explanations.

Your task:
- Read everything
- Extract key insights
- Write a natural, human-like conclusion
- Do NOT mention AI or models

Conversation:
----------------
{open(CHAT_FILE).read()}
----------------

Final conclusion:
"""
    summary = ollama_call(MODEL_SUMMARY, summary_prompt)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary)

    # Cleanup temp file
    if os.path.exists(TMP_FILE):
        os.remove(TMP_FILE)


# ---------- UI ----------
HTML = """
<!doctype html>
<title>Curious AI Conversation</title>
<h2>Curious Conversation Generator</h2>

<form method="post">
  <label>Enter Topic</label><br>
  <input name="topic" style="width:600px" required><br><br>
  <button type="submit">Start</button>
</form>

{% if done %}
<hr>
<h3>Conversation</h3>
<pre>{{ chat }}</pre>

<h3>Final Summary</h3>
<pre>{{ summary }}</pre>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        topic = request.form["topic"].strip()
        run_conversation(topic)

        chat = open(CHAT_FILE).read()
        summary = open(SUMMARY_FILE).read()

        return render_template_string(
            HTML, done=True, chat=chat, summary=summary
        )

    return render_template_string(HTML, done=False)


# ---------- Run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030)
