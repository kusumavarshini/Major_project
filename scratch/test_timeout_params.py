import time
import urllib.request
import urllib.parse
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def test_param(name, val):
    params = {
        name: str(val),
        'Quality': '50',
        'licstr': '',
        'fakeDetection': '0',
        'templateFormat': '0',
        'imageWSQRate': '0.75'
    }
    data = urllib.parse.urlencode(params).encode('utf-8')
    req = urllib.request.Request('https://localhost:8443/SGIFPCapture', data=data)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30.0) as resp:
            content = resp.read().decode('utf-8')
            elapsed = time.time() - t0
            print(f"{name}={val} -> Elapsed: {elapsed:.2f}s, Response: {content.strip()}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"{name}={val} -> Elapsed: {elapsed:.2f}s, Error: {e}")

for p_name in ['Timeout', 'timeout', 'TIMEOUT']:
    for p_val in [5, 10, 5000, 10000]:
        test_param(p_name, p_val)
