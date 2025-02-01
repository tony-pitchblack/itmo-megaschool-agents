from functools import wraps
import time

def measure_time(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        output = await func(*args, **kwargs)  # Execute the async function
        elapsed_time = time.perf_counter() - start_time
        print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        return output, elapsed_time  # Return both output and timing
    return wrapper