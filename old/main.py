"""
ollama_debate_app.py
Flask app to run an automated debate between two local Ollama models,
save the transcript to a text file and serve a live-readable UI on port 3434.
"""

import threading
import time
import uuid
import json
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, request, render_template_string, jsonify, redirect, url_for

# Configuration
OLLAMA_BASE = "http://localhost:11434"   # change if your Ollama server uses different host/port
SAVE_DIR = Path("./debates")
SAVE_DIR.mkdir(exist_ok=True)
DEFAULT_MODEL_A = "gamma3:4b"
DEFAULT_MODEL_B = "qwen3:4b"

app = Flask(__name__)
debates = {}  # in-memory store: debate_id -> {status, transcript(list of dicts), file}

# --- Helper: call Ollama chat endpoint robustly ---
def call_ollama_chat(model: str, messages: list, timeout=120):
    """
    Calls /api/chat on the local Ollama server and returns the assistant text.
    Uses a few fallback strategies to extract text from common response shapes.
    """
    url = f"{OLLAMA_BASE}/api/chat"
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        return f"[ERROR calling Ollama model {model}: {e}]"

    try:
        data = r.json()
    except ValueError:
        return r.text or ""

    # Common possible response shapes:
    if isinstance(data, dict):
        if "response" in data and isinstance(data["response"], str):
            return data["response"]
        # OpenAI-compatible /v1/responses or chat-like returns
        if "output_text" in data and isinstance(data["output_text"], str):
            return data["output_text"]
        if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
            # try to extract message content
            c = data["choices"][0]
            if isinstance(c, dict):
                # openai chat/completions style
                msg = c.get("message") or c.get("delta") or c.get("text") or c.get("content")
                if isinstance(msg, dict) and "content" in msg:
                    return msg["content"]
                if isinstance(msg, str):
                    return msg
        # generic messages field
        if "messages" in data and isinstance(data["messages"], list):
            # take last assistant message content
            for m in reversed(data["messages"]):
                if isinstance(m, dict) and m.get("role") in ("assistant", "system", "bot"):
                    if isinstance(m.get("content"), str):
                        return m["content"]
        # fallback: pretty print
        return json.dumps(data, ensure_ascii=False, indent=2)

    # fallback to string
    return str(data)


# --- Core debate runner (runs in background thread) ---
def run_debate(debate_id: str, topic: str, turns_a: int, turns_b: int, model_a: str, model_b: str):
    """
    Alternates between model_a and model_b until their requested turns are exhausted.
    Writes incremental transcript to debates[debate_id] and to a file.
    """
    meta = debates[debate_id]
    transcript = []
    file_path = meta["file"]
    # Basic system instructions to each model to force debate role
    sys_a = f"You are Model A. Argue for/against the topic below concisely and robustly."
    sys_b = f"You are Model B. Argue for/against the topic below concisely and robustly."

    # initial user prompt (the topic)
    user_prompt = f"Debate topic: {topic}\nStart your argument."

    # We'll maintain a shared message history (cross-model) so each model sees prior content.
    shared_messages = [
        {"role": "system", "content": "You are participating in a debate. Keep answers concise and focused."},
        {"role": "user", "content": user_prompt}
    ]

    remaining_a = int(turns_a)
    remaining_b = int(turns_b)
    turn = 0  # 0 => model A's turn next, 1 => model B

    def append_entry(who, text):
        ts = datetime.utcnow().isoformat() + "Z"
        entry = {"time": ts, "model": who, "text": text}
        transcript.append(entry)
        # append to file
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {who}\n{text}\n\n")

        # update in-memory store so UI can poll
        debates[debate_id]["transcript"] = transcript

    # add header to file
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Debate started: {datetime.utcnow().isoformat()}Z\nTopic: {topic}\nModels: {model_a} vs {model_b}\n\n")

    # alternate until both run out
    while remaining_a > 0 or remaining_b > 0:
        # decide who to run
        if (turn == 0 and remaining_a <= 0) or (turn == 1 and remaining_b <= 0):
            # switch if current model has no remaining
            turn = 1 - turn

        if turn == 0 and remaining_a > 0:
            # model A's turn
            # provide a custom system message at beginning of each call to emphasize role
            messages = shared_messages + [{"role": "system", "content": sys_a}]
            reply = call_ollama_chat(model_a, messages)
            append_entry(model_a, reply)
            # add assistant response to shared history so the other model sees it
            shared_messages.append({"role": "assistant", "content": reply})
            remaining_a -= 1
        elif turn == 1 and remaining_b > 0:
            messages = shared_messages + [{"role": "system", "content": sys_b}]
            reply = call_ollama_chat(model_b, messages)
            append_entry(model_b, reply)
            shared_messages.append({"role": "assistant", "content": reply})
            remaining_b -= 1
        else:
            # no remaining for either (shouldn't happen due to loop condition)
            break

        # small delay to avoid hammering local model server and to make UI readable
        time.sleep(0.6)
        # flip turn for next loop
        turn = 1 - turn

    debates[debate_id]["status"] = "finished"
    debates[debate_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"


# --- Flask routes / UI ---
INDEX_HTML = """
<!doctype html>
<title>Ollama Debate</title>
<h2>Ollama debate — run two local models</h2>
<form method="post" action="/start">
  <label>Topic<br><input name="topic" style="width:600px" required></label><br><br>
  <label>Model A <input name="model_a" value="{{ default_a }}" style="width:260px"></label>
  <label>Turns for Model A <input name="turns_a" value="3" type="number" min="0" style="width:80px"></label><br><br>
  <label>Model B <input name="model_b" value="{{ default_b }}" style="width:260px"></label>
  <label>Turns for Model B <input name="turns_b" value="3" type="number" min="0" style="width:80px"></label><br><br>
  <button type="submit">Start debate</button>
</form>
<hr>
<p>After starting you will be redirected to the live debate page.</p>
"""

DEBATE_HTML = """
<!doctype html>
<title>Debate {{ debate_id }}</title>
<h2>Debate: {{ topic }}</h2>
<p>Status: <span id="status">{{ status }}</span></p>
<pre id="transcript" style="white-space:pre-wrap; border:1px solid #ccc; padding:10px; max-height:60vh; overflow:auto;">
Loading...
</pre>

<script>
const id = "{{ debate_id }}";
async function fetchUpdate() {
    const r = await fetch('/debate/' + id + '/json');
    const data = await r.json();
    document.getElementById('status').textContent = data.status;
    let out = '';
    for (const e of data.transcript) {
        out += `[${e.time}] ${e.model}\\n${e.text}\\n\\n`;
    }
    document.getElementById('transcript').textContent = out;
    if (data.status !== 'finished') {
        setTimeout(fetchUpdate, 1200);
    }
}
fetchUpdate();
</script>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, default_a=DEFAULT_MODEL_A, default_b=DEFAULT_MODEL_B)

@app.route("/start", methods=["POST"])
def start():
    topic = request.form.get("topic", "").strip()
    model_a = request.form.get("model_a", DEFAULT_MODEL_A).strip() or DEFAULT_MODEL_A
    model_b = request.form.get("model_b", DEFAULT_MODEL_B).strip() or DEFAULT_MODEL_B
    try:
        turns_a = max(0, int(request.form.get("turns_a", "3")))
    except:
        turns_a = 3
    try:
        turns_b = max(0, int(request.form.get("turns_b", "3")))
    except:
        turns_b = 3

    debate_id = uuid.uuid4().hex[:12]
    fname = SAVE_DIR / f"debate_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{debate_id}.txt"
    debates[debate_id] = {
        "status": "running",
        "transcript": [],
        "file": str(fname),
        "topic": topic,
        "model_a": model_a,
        "model_b": model_b
    }

    # start background thread to run the debate
    t = threading.Thread(target=run_debate, args=(debate_id, topic, turns_a, turns_b, model_a, model_b), daemon=True)
    t.start()

    return redirect(url_for("debate_page", debate_id=debate_id))

@app.route("/debate/<debate_id>", methods=["GET"])
def debate_page(debate_id):
    meta = debates.get(debate_id)
    if not meta:
        return "Debate not found", 404
    return render_template_string(DEBATE_HTML, debate_id=debate_id, topic=meta.get("topic"), status=meta.get("status"))

@app.route("/debate/<debate_id>/json", methods=["GET"])
def debate_json(debate_id):
    meta = debates.get(debate_id)
    if not meta:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": meta.get("status"),
        "transcript": meta.get("transcript"),
        "file": meta.get("file")
    })

if __name__ == "__main__":
    # Run on port 3434 accessible from the local LAN if needed
    app.run(host="0.0.0.0", port=3434, threaded=True)
