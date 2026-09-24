from src.workflow import load_static_memory, screen_input, retrieve_memory
import time

# Build minimal state
state = {"user_story": "As a user, I want to login securely."}

# Test the first few nodes
print("Testing load_static_memory...")
state = load_static_memory(state)
print(" load static OK")

print("Testing screen_input...")
state = screen_input(state)
print("screen input OK")

print("Testing retrieve_memory (embedding call)...")
t0 = time.time()
state = retrieve_memory(state)
print(f"Took {time.time()-t0:.1f}s")
print("retrieve memory OK")