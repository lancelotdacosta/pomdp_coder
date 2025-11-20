import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPEN_ROUTER_KEY")

url = "https://openrouter.ai/api/v1/auth/key"
headers = {
    "Authorization": f"Bearer {api_key}"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    data = response.json()
    print("API Response:")
    print(json.dumps(data, indent=2, sort_keys=True))
else:
    print(f"Error: {response.status_code} - {response.text}")