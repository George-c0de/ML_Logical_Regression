import time
from functools import wraps
import psutil
def measure_execution_time(func):
    if not callable(func):
        raise ValueError("The measure_execution_time decorator can only be applied to callable objects.")

    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process()
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_cpu_time = process.cpu_times().user + process.cpu_times().system

        result = func(*args, **kwargs)

        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu_time = process.cpu_times().user + process.cpu_times().system

        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = end_cpu_time - start_cpu_time

        print(f"Execution time of function {func.__name__}: {execution_time} seconds")
        print(f"Memory usage of function {func.__name__}: {memory_usage} bytes")
        print(f"CPU usage of function {func.__name__}: {cpu_usage} seconds")

        return result

    return wrapper