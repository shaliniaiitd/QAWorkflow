from src.workflow import build_graph
import time

app = build_graph()
print("Graph compiled OK")

print("Testing invoke with stream...")
t0 = time.time()
for step in app.stream({"user_story": "As a user, I want to login securely."}, stream_mode="updates"):
    print(f"Step: {step}")
    print(f"Elapsed: {time.time()-t0:.1f}s")