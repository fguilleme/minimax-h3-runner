#!/usr/bin/env python3
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8988"
OUT = ROOT / "server-example-results.json"

jobs = [
    (
        "prompt-simple-5s-10fps",
        {
            "prompt": "A red fox walking through a sunlit forest clearing",
            "duration": 5, "fps": 10, "width": 384, "height": 224,
        },
    ),
    (
        "image-plus-prompt-5s-10fps",
        {
            "prompt": "The fox slowly turns its head while leaves move in the breeze",
            "first_frame": str(ROOT / "input/fox-first.png"),
            "duration": 5, "fps": 10, "width": 384, "height": 224,
        },
    ),
    (
        "long-30s-10fps",
        {
            "prompt": "A dancer performs a graceful energetic routine in a studio",
            "first_frame": str(ROOT / "input/dancer-source.jpg"),
            "duration": 30, "fps": 10, "width": 256, "height": 384,
            "audio": "first",
        },
    ),
]

def curl_json(args):
    p = subprocess.run(["curl", "-fsS", "--max-time", "20", *args], check=True,
                       capture_output=True, text=True)
    return json.loads(p.stdout)

results = []
for name, payload in jobs:
    print(f"SUBMIT {name}", flush=True)
    response = curl_json(["-X", "POST", f"{BASE}/generate", "-H", "Content-Type: application/json",
                          "--data", json.dumps(payload)])
    job = response["job"]
    job_id = job["id"]
    print(f"JOB {name} {job_id} {job['output']}", flush=True)
    while True:
        status = curl_json([f"{BASE}/status?job={job_id}"])["job"]
        state = status["state"]
        print(f"STATUS {name} {job_id} {state}", flush=True)
        if state in ("done", "failed"):
            result = {"name": name, "job_id": job_id, **status}
            results.append(result)
            if state == "failed":
                print(f"FAILED {name}: {status.get('error', '')[-1000:]}", flush=True)
                OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
                raise SystemExit(1)
            break
        time.sleep(10)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")

print(f"DONE {OUT}", flush=True)
