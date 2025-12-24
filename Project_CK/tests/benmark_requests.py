import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/predict"
IMAGE_PATH = "./Background.png"
NUM_REQUESTS = 50
NUM_WORKERS = 10

def send_request():
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": ("Background.png", f, "image/jpeg")}
        start = time.time()
        r = requests.post(URL, files=files)
        return time.time() - start

start_all = time.time()
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as ex:
    latencies = list(ex.map(lambda _: send_request(), range(NUM_REQUESTS)))
total_time = time.time() - start_all

print(f"Concurrent requests: {NUM_REQUESTS}")
print(f"Workers: {NUM_WORKERS}")
print(f"Total time: {total_time:.2f}s")
print(f"Avg latency: {statistics.mean(latencies):.3f}s")
print(f"P95 latency: {statistics.quantiles(latencies, n=20)[18]:.3f}s")
