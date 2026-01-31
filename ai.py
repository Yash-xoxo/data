#!/usr/bin/env python3
"""
ollama_debate_app.py

Flask app that runs an automated debate between two local Ollama models using
the reliable /api/generate endpoint. Saves transcripts to text files and
serves a live UI on port 3434.
"""

from pathlib import Path
from datetime import datetime
import threading
import time
import uuid
import requests
from flask import Flask, request, render_template_string, jsonify, redirect, url_for

# ---------- CONFIG ----------
OLLAMA_BASE = "http://localhost:11434"
SAVE_DIR = Path("./debates")
SAVE_DIR.mkdir(exist_ok=True)

# Set default models here (replace with exact names on your system if different)
DEFAULT_MODEL_A = "granite4:350m"
DEFAULT_MODEL_B = "granite4:350m"

# Safe generation options for CPU-bound systems. Adjust down if memory issues occur.
DEFAULT_OPTIONS = {
    "num_ctx": 512,
    "num_predict": 128,
    "temperature": 0.6
}

# Safety caps
MAX_TOTAL_TURNS = 20        # avoid extremely long debates
PER_REQUEST_TIMEOUT = 180   # seconds

app = Flask(__name__)
debates = {}  # debate_id -> metadata


# ---------- Utility: call Ollama /api/generate ----------
def call_ollama_generate(model: str, prompt: str, options: dict = None, timeout: int = PER_REQUEST_TIMEOUT):
    """
    Call Ollama /api/generate. Returns tuple (success:bool, text_or_error:str, status_code:int|None).
    On success: (True, response_text, 200)
    On failure: (False, error_message, status_code_or_None)
    """
    if options is None:
        options = DEFAULT_OPTIONS
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as e:
        return False, f"request error: {e}", None

    status = r.status_code
    try:
        data = r.json()
    except ValueError:
        # non-json response
        return False, f"non-json response (status {status}): {r.text[:400]}", status

    # Preferred response field
    resp_text = ""
    if isinstance(data, dict):
        # Ollama often returns "response"
        if "response" in data and isinstance(data["response"], str):
            resp_text = data["response"].strip()
        # fallback: some versions return different shapes (try to join other fields)
        elif "output" in data and isinstance(data["output"], str):
            resp_text = data["output"].strip()
        else:
            # safe stringify
            resp_text = (data.get("response") or "") if isinstance(data.get("response"), str) else ""
            resp_text = resp_text.strip()

    if status >= 400:
        # capture server-side message if present
        err = data.get("error") if isinstance(data, dict) else r.text
        return False, f"status {status}: {err}", status

    if not resp_text:
        # mark empty
        return True, "", status

    return True, resp_text, status


# ---------- Helper: quick model probe ----------
def probe_model(model: str) -> (bool, str):
    """
    Light probe to check model availability. Tries a minimal generate and interprets response.
    Returns (ok, message). ok==True means model responded (possibly empty string).
    """
    ok, text, status = call_ollama_generate(model, "PING_MODEL_OK", options={"num_ctx": 256, "num_predict": 32, "temperature": 0.0}, timeout=25)
    if not ok:
        return False, f"probe failed: {text} (status {status})"
    # ok but empty response is still a sign the model is reachable (we'll treat empty as reachable)
    return True, "model reachable"


# ---------- Debate runner ----------
def run_debate(debate_id: str, topic: str, turns_a: int, turns_b: int, model_a: str, model_b: str, options: dict):
    meta = debates[debate_id]
    file_path = meta["file"]
    transcript = []
    debates[debate_id]["transcript"] = transcript

    # Ensure max total turns
    turns_a = min(turns_a, MAX_TOTAL_TURNS)
    turns_b = min(turns_b, MAX_TOTAL_TURNS)

    # Base shared prompt (rolling)
    debate_prompt = (
        f"Debate topic:\n{topic}\n\n"
        "Rules:\n"
        "- Be concise (max 6 sentences)\n"
        "- Identify yourself as Model A or Model B at the start of each reply\n"
        "- Address the opponent's last point when possible\n\n"
    )

    def log_entry(model_label: str, text: str):
        ts = datetime.utcnow().isoformat() + "Z"
        entry = {"time": ts, "model": model_label, "text": text}
        transcript.append(entry)
        debates[debate_id]["transcript"] = transcript
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {model_label}\n{text}\n\n")

    # write header
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"Debate started: {datetime.utcnow().isoformat()}Z\nTopic: {topic}\nModels: {model_a} vs {model_b}\n\n")

    # Force an opening move from Model A (guarantees progress)
    if turns_a > 0:
        opening_prompt = debate_prompt + "Model A: Provide the opening argument."
        ok, resp, status = call_ollama_generate(model_a, opening_prompt, options)
        if not ok:
            log_entry(model_a, f"[ERROR] {resp}")
            # abort early on fatal model error
            debates[debate_id]["status"] = "error"
            return
        if not resp:
            resp = "[EMPTY RESPONSE]"
        log_entry(model_a, resp)
        debate_prompt += f"\nModel A:\n{resp}\n"
        turns_a -= 1
        current_turn = 1  # next is Model B
    else:
        current_turn = 1

    # Main loop
    while (turns_a > 0) or (turns_b > 0):
        # Small safety sleep
        time.sleep(0.6)

        if current_turn == 0 and turns_a > 0:
            prompt = debate_prompt + "\nModel A: Your turn to respond."
            ok, resp, status = call_ollama_generate(model_a, prompt, options)
            if not ok:
                log_entry(model_a, f"[ERROR] {resp}")
                debates[debate_id]["status"] = "error"
                return
            resp = resp or "[EMPTY RESPONSE]"
            log_entry(model_a, resp)
            debate_prompt += f"\nModel A:\n{resp}\n"
            turns_a -= 1
            current_turn = 1

        elif current_turn == 1 and turns_b > 0:
            prompt = debate_prompt + "\nModel B: Your turn to respond."
            ok, resp, status = call_ollama_generate(model_b, prompt, options)
            if not ok:
                log_entry(model_b, f"[ERROR] {resp}")
                debates[debate_id]["status"] = "error"
                return
            resp = resp or "[EMPTY RESPONSE]"
            log_entry(model_b, resp)
            debate_prompt += f"\nModel B:\n{resp}\n"
            turns_b -= 1
            current_turn = 0

        else:
            # Nothing to do (shouldn't happen)
            break

    debates[debate_id]["status"] = "finished"
    debates[debate_id]["finished_at"] = datetime.utcnow().isoformat() + "Z"


# ---------- Flask UI & endpoints ----------
INDEX_HTML = """
<!doctype html>
<title>Ollama Debate</title>
<h2>Ollama Debate</h2>
<form method="post" action="/start">
  <label>Topic<br><input name="topic" style="width:720px" required></label><br><br>

  <label>Model A (exact name)<br><input name="model_a" value="{{ default_a }}" style="width:360px"></label>
  <label>Turns for A <input name="turns_a" value="2" type="number" min="1" max="10" style="width:80px"></label><br><br>

  <label>Model B (exact name)<br><input name="model_b" value="{{ default_b }}" style="width:360px"></label>
  <label>Turns for B <input name="turns_b" value="2" type="number" min="1" max="10" style="width:80px"></label><br><br>

  <button type="submit">Start Debate</button>
</form>

<p>Notes: Uses Ollama <code>/api/generate</code>. If a model name is incorrect the debate will log the error.</p>
"""

DEBATE_HTML = """
<!doctype html>
<title>Debate {{ id }}</title>
<h2>Debate: {{ topic }}</h2>
<p>Status: <strong id="status">{{ status }}</strong></p>
<pre id="transcript" style="white-space:pre-wrap; border:1px solid #ccc; padding:10px; max-height:60vh; overflow:auto;">
Loading...
</pre>

<script>
const id = "{{ id }}";
async function fetchUpdate() {
  try {
    const r = await fetch('/debate/' + id + '/json');
    const data = await r.json();
    document.getElementById('status').textContent = data.status;
    let out = '';
    for (const e of data.transcript) {
        out += `[${e.time}] ${e.model}\n${e.text}\n\n`;
    }
    document.getElementById('transcript').textContent = out || "(no transcript yet)";
    if (data.status !== 'finished' && data.status !== 'error') {
      setTimeout(fetchUpdate, 1200);
    }
  } catch (e) {
    document.getElementById('transcript').textContent = "Error polling: " + e;
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

    # validate turns with safe defaults
    try:
        turns_a = int(request.form.get("turns_a") or 1)
    except Exception:
        turns_a = 1
    try:
        turns_b = int(request.form.get("turns_b") or 1)
    except Exception:
        turns_b = 1

    turns_a = max(1, min(turns_a, 10))
    turns_b = max(1, min(turns_b, 10))

    # quick probes to ensure models are reachable; if probe fails, record the message but still start
    ok_a, msg_a = probe_model(model_a)
    ok_b, msg_b = probe_model(model_b)

    debate_id = uuid.uuid4().hex[:12]
    fname = SAVE_DIR / f"debate_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{debate_id}.txt"
    debates[debate_id] = {
        "status": "running",
        "transcript": [],
        "file": str(fname),
        "topic": topic,
        "model_a": model_a,
        "model_b": model_b,
        "probe": {"model_a": (ok_a, msg_a), "model_b": (ok_b, msg_b)}
    }

    # pre-log probe results
    with open(fname, "a", encoding="utf-8") as f:
        f.write(f"Debate start: {datetime.utcnow().isoformat()}Z\nTopic: {topic}\nModels: {model_a} vs {model_b}\n\n")
        f.write(f"Probe model_a: {ok_a} - {msg_a}\n")
        f.write(f"Probe model_b: {ok_b} - {msg_b}\n\n")

    # run debate in background thread
    t = threading.Thread(
        target=run_debate,
        args=(debate_id, topic, turns_a, turns_b, model_a, model_b, DEFAULT_OPTIONS),
        daemon=True
    )
    t.start()

    return redirect(url_for("debate_page", id=debate_id))

@app.route("/debate/<id>", methods=["GET"])
def debate_page(id):
    meta = debates.get(id)
    if not meta:
        return "Debate not found", 404
    return render_template_string(DEBATE_HTML, id=id, topic=meta.get("topic"), status=meta.get("status"))

@app.route("/debate/<id>/json", methods=["GET"])
def debate_json(id):
    meta = debates.get(id)
    if not meta:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "status": meta.get("status"),
        "transcript": meta.get("transcript", []),
        "file": meta.get("file"),
        "topic": meta.get("topic"),
        "model_a": meta.get("model_a"),
        "model_b": meta.get("model_b"),
        "probe": meta.get("probe")
    })

if __name__ == "__main__":
    # Run server
    app.run(host="0.0.0.0", port=3434, threaded=True)
