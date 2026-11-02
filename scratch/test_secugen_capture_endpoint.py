import urllib.request
import urllib.parse
import ssl
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

lic = open('C:/Program Files/SecuGen/SgiBioSrv/sgiwebsrv.lic', 'r').read().strip()
print("License from file:", lic)

params = {
    'Timeout': '500',
    'Quality': '50',
    'licstr': lic,
    'fakeDetection': '0',
    'templateFormat': '0',
    'imageWSQRate': '0.75'
}
data = urllib.parse.urlencode(params).encode('utf-8')
req = urllib.request.Request("https://localhost:8443/SGIFPCapture", data=data)

try:
    with urllib.request.urlopen(req, context=ctx, timeout=3.0) as resp:
        content = resp.read().decode('utf-8')
        print("HTTP Status:", resp.status)
        print("Response:", content)
except Exception as e:
    print("Endpoint call error:", e)
