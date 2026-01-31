import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, request, render_template_string, jsonify, redirect, url_for

# ---------------- CONFIG ----------------
OLLAMA_BASE = "http://localhost:11434"
SAVE_DIR = Path("./debates")
SAVE_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL_A = "gamma3:4b"
DEFAULT_MODEL_B = "qwen3:4b"   # keep both small on CPU

OLLAMA_OPTIONS = {
    "num_ctx": 512,
    "num_predict": 128,
    "temperature": 0.7
}

# ----------------------------------------
app = Flask(__name__)
debates = {}

# ---------- OLLAMA GENERATE CALL ----------
def call_ollama_generate(model: str, prompt: str, timeout=180):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": OLLAMA_OPTIONS
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return f"[OLLAMA ERROR: {e}]"


# ---------- DEBATE RUNNER ----------
def run_debate(debate_id, topic, turns_a, turns_b, model_a, model_b):
    meta = debates[debate_id]
    transcript = []
    file_path = meta["file"]

    debate_prompt = f"""Debate topic:
{topic}

Rules:
- Be concise
- Respond in under 6 sentences
- Argue clearly
"""

    def log(model, text):
        ts = datetime.utcnow().isoformat() + "Z"
        entry = {"time": ts, "model": model, "text": text}
        transcript.append(entry)
        debates[debate_id]["transcript"] = transcript
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {model}\n{text}\n\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Debate started {datetime.utcnow().isoformat()}Z\nTopic: {topic}\n\n")

    turn = 0
    while turns_a > 0 or turns_b > 0:
        if turn == 0 and turns_a > 0:
            prompt = debate_prompt + "\nModel A, your argument:"
            reply = call_ollama_generate(model_a, prompt)
            log(model_a, reply)
            debate_prompt += f"\nModel A:\n{reply}\n"
            turns_a -= 1

        elif turn == 1 and turns_b > 0:
            prompt = debate_prompt + "\nModel B, your rebuttal:"
            reply = call_ollama_generate(model_b, prompt)
            log(model_b, reply)
            debate_prompt += f"\nModel B:\n{reply}\n"
            turns_b -= 1

        turn = 1 - turn
        time.sleep(0.8)

    debates[debate_id]["status"] = "finished"


# ---------- UI ----------
INDEX_HTML = """
<h2>Ollama Debate (Low-Memory Mode)</h2>
<form method="post" action="/start">
Topic:<br>
<input name="topic" style="width:600px" required><br><br>
Model A turns <input name="turns_a" value="3" type="number">
Model B turns <input name="turns_b" value="3" type="number"><br><br>
<button type="submit">Start Debate</button>
</form>
"""

DEBATE_HTML = """
<h3>Debate</h3>
Status: <span id="status">{{ status }}</span>
<pre id="log" style="height:60vh;overflow:auto;border:1px solid #aaa"></pre>

<script>
async function poll() {
  const r = await fetch("/debate/{{ id }}/json");
  const d = await r.json();
  document.getElementById("status").textContent = d.status;
  let t = "";
  for (const e of d.transcript) {
    t += `[${e.time}] ${e.model}\n${e.text}\n\n`;
  }
  document.getElementById("log").textContent = t;
  if (d.status !== "finished") setTimeout(poll, 1200);
}
poll();
</script>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/start", methods=["POST"])
def start():
    debate_id = uuid.uuid4().hex[:10]
    topic = request.form["topic"]
    turns_a = int(request.form["turns_a"])
    turns_b = int(request.form["turns_b"])

    file = SAVE_DIR / f"debate_{debate_id}.txt"
    debates[debate_id] = {
        "status": "running",
        "transcript": [],
        "file": str(file)
    }

    t = threading.Thread(
        target=run_debate,
        args=(debate_id, topic, turns_a, turns_b, DEFAULT_MODEL_A, DEFAULT_MODEL_B),
        daemon=True
    )
    t.start()

    return redirect(url_for("debate", id=debate_id))

@app.route("/debate/<id>")
def debate(id):
    return render_template_string(DEBATE_HTML, id=id, status=debates[id]["status"])

@app.route("/debate/<id>/json")
def debate_json(id):
    return jsonify(debates[id])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3434)
