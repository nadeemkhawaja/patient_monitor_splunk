import requests
import os
import urllib3

# Disable the SSL warning for clarity
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# HARDCODED CREDENTIALS (To rule out .env issues)
url = "https://prd-p-jhic9.splunkcloud.com:8088/services/collector"
token = "8d5d7243-6e62-45d7-a2dc-1b7d0d15701e"

print(f"📡 Testing connection to: {url}")

payload = {
    "sourcetype": "ai_monitor",
    "event": {
        "message": "Python Script Test",
        "status": "working"
    }
}

try:
    response = requests.post(
        url, 
        headers={"Authorization": f"Splunk {token}"}, 
        json=payload, 
        verify=False
    )
    print(f"🔹 Status Code: {response.status_code}")
    print(f"🔹 Response: {response.text}")
    
    if response.status_code == 200:
        print("\n✅ SUCCESS! The script sent data.")
        print("👉 Go to Splunk and search: index=* sourcetype=ai_monitor")
    else:
        print("\n❌ FAILED. Check URL or Token.")

except Exception as e:
    print(f"\n❌ CONNECTION ERROR: {e}")