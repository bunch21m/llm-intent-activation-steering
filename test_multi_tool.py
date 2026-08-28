import requests
import sys

try:
    # "test" -> mention_cat_tool
    # "happy" -> joyful_tool
    prompt = "This is a test and I am so happy right now."
    print(f"Sending prompt: '{prompt}'")
    
    response = requests.post("http://localhost:8000/chat", json={"message": prompt})
    response.raise_for_status()
    print("Response:", response.json())
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
