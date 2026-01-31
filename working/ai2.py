#!/usr/bin/env python3

from flask import Flask, request, render_template_string, jsonify
from pathlib import Path
import requests
import time
import threading
import re

# ================= CONFIG =================
OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_CURIOUS = "llama2-uncensored:7b"
MODEL_EXPLAINER = "llama2-uncensored:7b"
MODEL_SUMMARIZER = "granite4:350m"

NUM_CTX = 512
NUM_PREDICT = 128
ROUNDS = 4

BASE_DIR = Path("debate2")
# =========================================

app = Flask(__name__)
state = {
    "running": False,
    "topic": None,
    "chat": "",
    "summary": "",
    "status": "idle"
}

# ---------- Helpers ----------
def safe_topic_folder(topic: str) -> str:
    topic = topic.strip().lower()
    topic = re.sub(r"[^a-z0-9 _-]", "", topic)
    topic = re.sub(r"\s+", "_", topic)
    return topic[:60] or "untitled_topic"

def ollama_call(model: str, prompt: str) -> str:
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

def append_chat(text: str):
    state["chat"] += text + "\n\n"

# ---------- Core Worker ----------
def run_conversation(topic: str):
    state.update({"running": True, "chat": "", "summary": "", "status": "running"})

    topic_dir = BASE_DIR / safe_topic_folder(topic)
    topic_dir.mkdir(parents=True, exist_ok=True)

    chat_file = topic_dir / "chat.txt"
    summary_file = topic_dir / "summary.txt"
    tmp_file = topic_dir / "chat.tmp"

    for f in (chat_file, summary_file, tmp_file):
        if f.exists():
            f.unlink()

    chat_file.write_text(f"Topic: {topic}\n" + "-" * 50 + "\n\n", encoding="utf-8")
    tmp_file.touch()

    append_chat(f"### Topic: {topic}")

    # ---- Conversation Loop ----
    for round_no in range(1, ROUNDS + 1):
        # Curious
        curious_prompt = f"""
You are a very curious human.

Topic: {topic}

Conversation so far:
{tmp_file.read_text(encoding="utf-8")}

Ask ONE deep, curiosity-driven question.
"""
        question = ollama_call(MODEL_CURIOUS, curious_prompt)

        chat_file.write_text(chat_file.read_text() + "[Curious]\n" + question + "\n\n")
        tmp_file.write_text(tmp_file.read_text() + f"Curious: {question}\n")

        append_chat(f"[Curious]\n{question}")
        time.sleep(0.5)

        # Explainer
        explainer_prompt = f"""
You are an expert explainer.

Topic: {topic}

Question:
{question}

Explain clearly and in detail.
"""
        answer = ollama_call(MODEL_EXPLAINER, explainer_prompt)

        chat_file.write_text(chat_file.read_text() + "[Explainer]\n" + answer + "\n\n")
        tmp_file.write_text(tmp_file.read_text() + f"Explainer: {answer}\n")

        append_chat(f"[Explainer]\n{answer}")
        time.sleep(0.5)

    # ---- Summary ----
    state["status"] = "summarizing"

    summary_prompt = f"""
You are a thoughtful human reader.

Read the conversation below and write a natural,
human-style conclusion. Do not mention AI.

Conversation:
----------------
{chat_file.read_text(encoding="utf-8")}
----------------

Final conclusion:
"""
    summary = ollama_call(MODEL_SUMMARIZER, summary_prompt)

    summary_file.write_text(summary, encoding="utf-8")
    state["summary"] = summary
    state["status"] = "finished"

    if tmp_file.exists():
        tmp_file.unlink()

    state["running"] = False


# ---------- UI ----------
HTML = """
<!doctype html>
<html>
<head>
<title>Real-Time Curious Debate</title>
<style>
body { font-family: Arial; background:#f2f4f7; padding:30px; }
.box { background:white; padding:20px; border-radius:8px; max-width:900px; margin:auto; }
pre { background:#111; color:#0f0; padding:15px; white-space:pre-wrap; }
.status { font-weight:bold; }
</style>
</head>
<body>
<div class="box">
<h2>Curious Real-Time Debate</h2>

<form method="post" action="/start">
<input name="topic" placeholder="Enter topic" style="width:100%" required><br><br>
<button type="submit">Start</button>
</form>

<hr>
<p>Status: <span class="status" id="status">idle</span></p>

<h3>Live Conversation</h3>
<pre id="chat"></pre>

<h3>Final Conclusion</h3>
<pre id="summary"></pre>
</div>

<script>
async function poll() {
  const r = await fetch("/state");
  const d = await r.json();
  document.getElementById("status").innerText = d.status;
  document.getElementById("chat").innerText = d.chat;
  document.getElementById("summary").innerText = d.summary;
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/start", methods=["POST"])
def start():
    if not state["running"]:
        topic = request.form["topic"]
        state["topic"] = topic
        threading.Thread(target=run_conversation, args=(topic,), daemon=True).start()
    return render_template_string(HTML)

@app.route("/state")
def get_state():
    return jsonify(state)

# ---------- Run ----------
if __name__ == "__main__":
    BASE_DIR.mkdir(exist_ok=True)
    app.run(host="0.0.0.0", port=3030)
