import requests

OLLAMA_BASE = "http://127.0.0.1:11434"

MODELS = [
    "gemma3:4b",
    "qwen3:4b"
]

def test_model(model):
    url = f"{OLLAMA_BASE}/api/generate"
    payload = {
        "model": model,
        "prompt": "Reply with exactly: MODEL_OK",
        "stream": False
    }

    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[FAIL] {model} -> {e}")
        return

    reply = data.get("response")

    if reply:
        print(f"[OK] {model} responded:")
        print(reply.strip())
    else:
        print(f"[WARN] {model} returned unexpected output:")
        print(data)

if __name__ == "__main__":
    print("=== Ollama model connectivity test ===")
    for m in MODELS:
        print(f"\nTesting {m} ...")
        test_model(m)
