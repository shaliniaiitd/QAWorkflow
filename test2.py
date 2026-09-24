from src.workflow import build_graph
import time

app = build_graph()
print("Graph compiled OK")

print("Testing invoke...")
t0 = time.time()
result = app.invoke({"user_story": "As a user, I want to login securely."})
print(f"Took {time.time()-t0:.1f}s")
print("Done")