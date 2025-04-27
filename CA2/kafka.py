import threading
import subprocess
import sys

# def run_producer():
#     subprocess.run(["python3", "producer.py"])

# def run_consumer():
#     subprocess.run(["python3", "consumer.py"])

def run_producer():
    subprocess.run([sys.executable, "producer.py"], check=True)

def run_consumer():
    subprocess.run([sys.executable, "consumer.py"], check=True)

producer_thread = threading.Thread(target=run_producer)
consumer_thread = threading.Thread(target=run_consumer)

producer_thread.start()
consumer_thread.start()

producer_thread.join()
consumer_thread.join()