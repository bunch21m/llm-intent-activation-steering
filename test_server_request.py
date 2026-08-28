import requests
import sys

try:
    response = requests.post("http://localhost:8000/chat", json={"message": "Hello, how are you?"})
    response.raise_for_status()
    data = response.json()
    # Use errors='replace' to handle emoji/unicode that Windows console (cp1252) can't render
    print("Response:", str(data).encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8', errors='replace'))
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
