from __future__ import annotations

import os
import socket
import time

ip_addr_file = "./ip_address.txt"


def get_ip():
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    os.environ["MASTER_ADDR"] = ip_addr
    with open(os.path.join(ip_addr_file), "w") as f:
        f.write(ip_addr)


try:
    os.environ["NODE_RANK"]
except KeyError:
    os.environ["NODE_RANK"] = "0"

if os.environ["NODE_RANK"] == "0":
    get_ip()
else:
    while not os.path.exists(os.path.join(ip_addr_file)):
        time.sleep(1)
    with open(os.path.join(ip_addr_file)) as f:
        master_ip = f.read()
    os.environ["MASTER_ADDR"] = master_ip
