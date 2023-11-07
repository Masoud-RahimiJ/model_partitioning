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
        exit(0)
    total_memory = parent.memory_info().rss
    children = parent.children(recursive=True)
    for child in children:
        total_memory += child.memory_info().rss
    total_memory = total_memory / 1024
    return total_memory

def monitor_memory_usage(pid, interval=0.001):
    previous_memory=-1
    while True:
        memory_usage = int(total_memory_usage(pid))
        if memory_usage != previous_memory:
            print(int(10000*time.time()), memory_usage)
            previous_memory = memory_usage
        time.sleep(interval)

monitor_memory_usage(pid)

