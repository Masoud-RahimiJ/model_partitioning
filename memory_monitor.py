import time
import os
import psutil

container_name = "exp"
pid = os.popen('sudo docker inspect -f "{{.State.Pid}}" ' + container_name).read().strip()
print(pid)
pid = int(pid)
print(f"PID of {container_name} is {pid}")


def total_memory_usage(pid):
    parent = psutil.Process(pid)
    total_memory = parent.memory_info().rss
    children = parent.children(recursive=True)
    for child in children:
        total_memory += child.memory_info().rss
    total_memory = total_memory / 1024 / 1024
    return total_memory

def monitor_memory_usage(pid, interval=0.001):
    max_memory = 0
    previous_memory=-1
    while True:
        try:
            memory_usage = int(total_memory_usage(pid))
        except psutil.NoSuchProcess:
            print(max_memory)
            exit(0)
        if max_memory < memory_usage:
            max_memory = memory_usage
        time.sleep(interval)

monitor_memory_usage(pid)

