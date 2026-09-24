from src.workflow import get_llm, _llm_text
import time

t0 = time.time()
result = _llm_text(get_llm(), "Say hello in one sentence.", node_name="diagnostic_test")
print(f"Took {time.time()-t0:.1f}s")
print(result)