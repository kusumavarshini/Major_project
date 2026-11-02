import subprocess
import json

cmd = 'Get-Process | Where-Object { $_.ProcessName -match "sgi" -or $_.ProcessName -match "secu" } | Select-Object Id, ProcessName, Path | ConvertTo-Json'
r = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True)
print("Processes:")
print(r.stdout)

# Also check netstat for port 8443 and 8000
cmd2 = 'Get-NetTCPConnection -LocalPort 8443, 8000 -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, OwningProcess, State | ConvertTo-Json'
r2 = subprocess.run(['powershell', '-NoProfile', '-Command', cmd2], capture_output=True, text=True)
print("TCP Connections:")
print(r2.stdout)

