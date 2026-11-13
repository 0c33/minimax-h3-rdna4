"""Minimal ComfyUI API client: queue a prompt and wait for it."""
import json
import time
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:8188"


def queue(prompt):
    body = json.dumps({"prompt": prompt, "client_id": str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f"{BASE}/prompt", data=body, headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))["prompt_id"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(e.read().decode()[:3000]) from None


def wait(pid, timeout=7200, poll=2.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = json.load(urllib.request.urlopen(f"{BASE}/history/{pid}", timeout=30))
        if pid in h and h[pid].get("status", {}).get("completed") is not None:
            return h[pid], time.time() - t0
        if pid in h and h[pid].get("status", {}).get("status_str") == "error":
            return h[pid], time.time() - t0
        time.sleep(poll)
    raise TimeoutError(pid)


def run(prompt, timeout=7200):
    pid = queue(prompt)
    hist, wall = wait(pid, timeout)
    return hist, wall, hist.get("status", {}).get("status_str")
