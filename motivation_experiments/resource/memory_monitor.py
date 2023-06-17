import time
import os
import psutil

container_name = "exp"
pid = os.popen('sudo docker inspect -f "{{.State.Pid}}" ' + container_name).read().strip()
print(pid)
pid = int(pid)
print(f"PID of {container_name} is {pid}")


def total_memory_usage(pid):
    parent=0
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"No such process: {pid}")
        exit(0)
    total_memory = parent.memory_info().rss
    children = parent.children(recursive=True)
    for child in children:
        total_memory += child.memory_info().rss
    total_memory = total_memory / 1024 / 1024
    return total_memory

def monitor_memory_usage(pid, interval=0.1):
    while True:
        print(total_memory_usage(pid))
        time.sleep(interval)

monitor_memory_usage(pid)

