import json, time, urllib.request

BASE = "http://127.0.0.1:8765"
MODEL = "unsloth/Qwen3.8-27B-GGUF"
SYS = "You are a helpful assistant. Answer directly and concisely."

def turn(conv, content):
    body = json.dumps({"model": MODEL, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": content}]}).encode()
    r = urllib.request.Request(f"{BASE}/v1/openai/chat/completions", data=body,
        headers={"Content-Type": "application/json", "X-Strata-Conversation": conv})
    t0 = time.time()
    with urllib.request.urlopen(r, timeout=600) as resp:
        d = json.load(resp)
    return d["choices"][0]["message"]["content"], time.time() - t0
