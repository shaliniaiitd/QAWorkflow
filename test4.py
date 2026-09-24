from src.workflow import build_graph
import time
from datetime import datetime

app = build_graph()
t_start = time.time()

print("Starting workflow...")
for step in app.stream({"user_story": "As a user, I want to login."}, stream_mode="updates"):
    elapsed = time.time() - t_start
    print(f"[{elapsed:.1f}s] {step}")

print(f"Total: {time.time() - t_start:.1f}s")