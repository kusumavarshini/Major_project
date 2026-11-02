import os
import socket
import urllib.request
import ssl
from dotenv import load_dotenv

load_dotenv(".env")
lic = os.getenv("SECUGEN_LIC", "")

print("=== Checking SecuGen Service Ports ===")
for port, proto in [(8443, "https"), (8000, "http")]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    res = sock.connect_ex(("localhost", port))
    sock.close()
    print(f"Port {port} ({proto}): {'OPEN' if res == 0 else 'CLOSED'}")

# Check tasklist for sgibiosrv.exe
import subprocess
try:
    p = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq sgibiosrv.exe'], capture_output=True, text=True)
    if 'sgibiosrv.exe' in p.stdout:
        print("sgibiosrv.exe process is RUNNING.")
    else:
        print("sgibiosrv.exe process is NOT found in tasklist.")
except Exception as e:
    print("Process check error:", e)
