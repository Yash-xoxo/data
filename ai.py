#!/usr/bin/env python3
"""
curious_debate_app_full.py

Flask web UI (port 3030) that:
- Accepts a topic and two model names (Curious / Explainer) plus Summarizer model
- Runs a sequential conversation: Curious -> Explainer for N rounds
- Writes incremental chat to debate2/<topic_folder>/chat.txt
- Writes final summary to debate2/<topic_folder>/summary.txt
- Provides real-time UI polling endpoints:
    GET  /         -> UI page
    POST /start    -> start conversation (redirects back to UI)
    GET  /state    -> JSON status (used by UI)
    GET  /file/chat    -> current chat file contents
    GET  /file/summary -> summary file contents (when ready)
The UI allows entering a "view model" string to filter which model's messages to display.
"""

from flask import Flask, request, render_template_string, jsonify, redirect, url_for
from pathlib import Path
import threading
import requests
import time
import re

# ---------------- CONFIG ----------------
OLLAMA_URL = "http://localhost:11434/api/generate"

# sensible defaults
DEFAULT_CURIOUS = "Set - Your - Model"
DEFAULT_EXPLAINER = "Set - Your - Model"
DEFAULT_SUMMARIZER = "Set - Your - Model"

NUM_CTX = 512
NUM_PREDICT = 128
DEFAULT_ROUNDS = 3

BASE_DIR = Path("debate2")
BASE_DIR.mkdir(exist_ok=True)
# ----------------------------------------

app = Flask(__name__)

# single shared state (single-session)
state = {
    "running": False,
    "status": "idle",
    "topic": None,
    "topic_dir": None,
    "chat": "",
    "summary": "",
    "model_curious": DEFAULT_CURIOUS,
    "model_explainer": DEFAULT_EXPLAINER,
    "model_summarizer": DEFAULT_SUMMARIZER,
    "rounds": DEFAULT_ROUNDS,
    "view_model": ""  # optional filter string user can enter in UI
}

# ---------- helpers ----------
def safe_topic_folder(topic: str) -> str:
    t = topic.strip().lower()
    t = re.sub(r"[^a-z0-9 _-]", "", t)
    t = re.sub(r"\s+", "_", t)
    return (t[:60] or "untitled_topic")

def ollama_call(model: str, prompt: str, num_ctx: int = NUM_CTX, num_predict: int = NUM_PREDICT, timeout: int = 240) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0.7
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip() or "[EMPTY RESPONSE]"

def write_chat_file(path: Path, header: str = None):
    if header:
        path.write_text(header + "\n\n", encoding="utf-8")
    else:
        if not path.exists():
            path.write_text("", encoding="utf-8")

# ---------- conversation worker ----------
def run_conversation(topic: str, model_curious: str, model_explainer: str, model_summarizer: str, rounds: int, view_model: str):
    # set state
    state["running"] = True
    state["status"] = "running"
    state["topic"] = topic
    state["model_curious"] = model_curious
    state["model_explainer"] = model_explainer
    state["model_summarizer"] = model_summarizer
    state["rounds"] = rounds
    state["view_model"] = view_model or ""
    topic_folder = safe_topic_folder(topic)
    topic_dir = BASE_DIR / topic_folder
    topic_dir.mkdir(parents=True, exist_ok=True)
    state["topic_dir"] = str(topic_dir)

    chat_file = topic_dir / "chat.txt"
    summary_file = topic_dir / "summary.txt"
    tmp_file = topic_dir / "chat.tmp"

    # cleanup previous files for the topic
    for f in (chat_file, summary_file, tmp_file):
        if f.exists():
            f.unlink()

    header = f"Topic: {topic}\nModels: curious={model_curious} explainer={model_explainer} summarizer={model_summarizer}\nRounds: {rounds}\n" + ("-" * 50)
    write_chat_file(chat_file, header)
    tmp_file.write_text("", encoding="utf-8")

    # update in-memory chat (used by /state)
    state["chat"] = header + "\n\n"

    try:
        for rnum in range(1, rounds + 1):
            # Curious model asks a question
            state["status"] = f"round {rnum} - curious"
            curious_prompt = f"""You are a very curious human asking focused questions to learn everything about the topic.

Topic: {topic}

Conversation so far:
{tmp_file.read_text(encoding="utf-8")}

Ask ONE thoughtful, curiosity-driven question (one or two sentences)."""
            question = ollama_call(model_curious, curious_prompt)

            # log
            entry_q = f"[Round {rnum}] [Curious | {model_curious}]\n{question}\n"
            with chat_file.open("a", encoding="utf-8") as fh:
                fh.write(entry_q + "\n")
            tmp_file.write_text(tmp_file.read_text(encoding="utf-8") + f"Curious: {question}\n", encoding="utf-8")
            state["chat"] += entry_q + "\n"

            # small pause
            time.sleep(1)

            # Explainer answers
            state["status"] = f"round {rnum} - explainer"
            explainer_prompt = f"""You are an expert explainer.

Topic: {topic}
Question:
{question}

Give a clear, structured, and detailed explanation."""
            answer = ollama_call(model_explainer, explainer_prompt)

            entry_a = f"[Round {rnum}] [Explainer | {model_explainer}]\n{answer}\n"
            with chat_file.open("a", encoding="utf-8") as fh:
                fh.write(entry_a + "\n")
            tmp_file.write_text(tmp_file.read_text(encoding="utf-8") + f"Explainer: {answer}\n", encoding="utf-8")
            state["chat"] += entry_a + "\n"

            # small pause
            time.sleep(2.5)

        # summarization
        state["status"] = "summarizing"
        summary_prompt = f"""You are a thoughtful human reader.

Below is the conversation. Extract the key ideas and write a clear, human-like conclusion.
Do NOT mention AI, models, or technical internals. Keep a neutral, readable tone.

Conversation:
----------------
{chat_file.read_text(encoding="utf-8")}
----------------

Final conclusion:"""
        summary = ollama_call(model_summarizer, summary_prompt, num_ctx=1024, num_predict=256)
        summary_file.write_text(summary, encoding="utf-8")
        state["summary"] = summary

        state["status"] = "finished"
    except Exception as e:
        # log error
        err_text = f"[ERROR] {e}"
        with chat_file.open("a", encoding="utf-8") as fh:
            fh.write(err_text + "\n")
        state["chat"] += err_text + "\n"
        state["status"] = "error"
    finally:
        # cleanup tmp
        if tmp_file.exists():
            tmp_file.unlink()
        state["running"] = False

# ---------- routes ----------
HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Real-Time Curious Debate</title>
<style>
:root {
  --bg-main: #0b0f14;
  --bg-card: #111827;
  --bg-input: #020617;
  --bg-pre: #020617;
  --text-main: #e5e7eb;
  --text-muted: #9ca3af;
  --accent: #22d3ee;
  --border: #1f2933;
  --success: #10b981;
}
* { box-sizing: border-box; }
body { margin:0; padding:28px; font-family:system-ui, -apple-system, BlinkMacSystemFont, sans-serif; background:linear-gradient(180deg,#020617,#0b0f14); color:var(--text-main); }
.container { max-width:980px; margin:auto; background:var(--bg-card); border:1px solid var(--border); border-radius:12px; padding:22px; box-shadow:0 10px 40px rgba(0,0,0,0.45); }
h2 { margin:0 0 8px 0; font-weight:600; }
label { font-size:13px; color:var(--text-muted); display:block; margin-top:12px; }
input, select { width:100%; padding:10px 12px; border-radius:8px; border:1px solid var(--border); background:var(--bg-input); color:var(--text-main); margin-top:6px; }
.row { display:flex; gap:12px; }
.col { flex:1; }
button { margin-top:12px; padding:10px 16px; border-radius:8px; border:none; cursor:pointer; background:linear-gradient(135deg,#22d3ee,#38bdf8); color:#021022; font-weight:600; }
.meta { margin-top:12px; color:var(--text-muted); font-size:14px; }
pre { margin-top:12px; padding:14px; background:var(--bg-pre); border-radius:10px; border:1px solid var(--border); color:#a7f3d0; font-family:monospace; font-size:13px; max-height:420px; overflow:auto; white-space:pre-wrap; }
.footer { margin-top:14px; color:var(--text-muted); font-size:13px; }
.small { font-size:13px; color:var(--text-muted); }
</style>
</head>
<body>
<div class="container">
  <h2>Curious Debate · Real-Time</h2>

  <form method="post" action="/start" id="startForm">
    <label>Topic</label>
    <input name="topic" placeholder="Enter topic (e.g. Neural Networks)" required>

    <div class="row">
      <div class="col">
        <label>Curious model</label>
        <input name="model_curious" value="{{default_curious}}">
      </div>
      <div class="col">
        <label>Explainer model</label>
        <input name="model_explainer" value="{{default_explainer}}">
      </div>
    </div>

    <div class="row">
      <div class="col">
        <label>Summarizer model</label>
        <input name="model_summarizer" value="{{default_summarizer}}">
      </div>
      <div class="col">
        <label>Rounds</label>
        <input name="rounds" type="number" min="1" max="10" value="{{default_rounds}}">
      </div>
    </div>

    <label>View model filter (optional)</label>
    <input name="view_model" placeholder="Enter model name to filter display (e.g. qwen3:1.7b)">

    <button type="submit">Start Conversation</button>
  </form>

  <div class="meta">
    Status: <strong id="status">idle</strong> &nbsp; | &nbsp;
    Saved dir: <span id="dir">—</span>
  </div>

  <h3 class="small">Live Conversation</h3>
  <pre id="chat">(no conversation yet)</pre>

  <h3 class="small">Final Summary</h3>
  <pre id="summary">(no summary yet)</pre>

  <div class="footer small">Updates every second. Use the view-model filter to show only a particular model's lines in the UI.</div>
</div>

<script>
async function poll() {
  try {
    const r = await fetch('/state');
    const s = await r.json();
    document.getElementById('status').innerText = s.status || 'idle';
    document.getElementById('dir').innerText = s.topic_dir || '—';

    if (s.topic_dir) {
      const chatText = await fetch('/file/chat').then(r => r.text());
      const viewModel = s.view_model || '';
      if (viewModel.trim()) {
        // simple filter: keep lines that include the view model string
        const lines = chatText.split('\\n');
        const filtered = lines.filter(ln => ln.toLowerCase().includes(viewModel.toLowerCase()));
        document.getElementById('chat').innerText = filtered.join('\\n');
      } else {
        document.getElementById('chat').innerText = chatText;
      }

      if (s.status === 'finished' || s.status === 'error') {
        const summaryText = await fetch('/file/summary').then(r => r.text().catch(()=>''));
        if (summaryText) document.getElementById('summary').innerText = summaryText;
      }
    }
  } catch (e) {
    // ignore transient errors
    // optionally display console log
    console.warn('poll error', e);
  } finally {
    setTimeout(poll, 1000);
  }
}
poll();
</script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def root():
    return render_template_string(
        HTML,
        default_curious=DEFAULT_CURIOUS,
        default_explainer=DEFAULT_EXPLAINER,
        default_summarizer=DEFAULT_SUMMARIZER,
        default_rounds=DEFAULT_ROUNDS
    )

@app.route("/start", methods=["POST"])
def start():
    if state["running"]:
        return ("Already running", 409)
    topic = request.form.get("topic", "").strip()
    if not topic:
        return ("Missing topic", 400)
    model_curious = request.form.get("model_curious", DEFAULT_CURIOUS).strip() or DEFAULT_CURIOUS
    model_explainer = request.form.get("model_explainer", DEFAULT_EXPLAINER).strip() or DEFAULT_EXPLAINER
    model_summarizer = request.form.get("model_summarizer", DEFAULT_SUMMARIZER).strip() or DEFAULT_SUMMARIZER
    try:
        rounds = int(request.form.get("rounds", DEFAULT_ROUNDS))
    except Exception:
        rounds = DEFAULT_ROUNDS
    rounds = max(1, min(rounds, 10))
    view_model = request.form.get("view_model", "").strip()

    # start background thread
    th = threading.Thread(
        target=run_conversation,
        args=(topic, model_curious, model_explainer, model_summarizer, rounds, view_model),
        daemon=True
    )
    th.start()

    # redirect back to UI (browser will then poll /state)
    return redirect(url_for('root'))

@app.route("/state", methods=["GET"])
def get_state():
    return jsonify({
        "running": state["running"],
        "status": state["status"],
        "topic": state["topic"],
        "topic_dir": state["topic_dir"],
        "model_curious": state["model_curious"],
        "model_explainer": state["model_explainer"],
        "model_summarizer": state["model_summarizer"],
        "rounds": state["rounds"],
        "view_model": state["view_model"]
    })

@app.route("/file/chat", methods=["GET"])
def get_chat_file():
    if not state["topic_dir"]:
        return ("", 204)
    chat_path = Path(state["topic_dir"]) / "chat.txt"
    if not chat_path.exists():
        return ("", 204)
    return chat_path.read_text(encoding="utf-8")

@app.route("/file/summary", methods=["GET"])
def get_summary_file():
    if not state["topic_dir"]:
        return ("", 204)
    summary_path = Path(state["topic_dir"]) / "summary.txt"
    if not summary_path.exists():
        return ("", 204)
    return summary_path.read_text(encoding="utf-8")

# ---------- run ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3030, threaded=True)
