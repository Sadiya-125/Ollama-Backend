import requests

url = "https://ollama-backend-55ja.onrender.com/chat"
payload = {
    "question": "What is Acne?",
    "chat_history": []
}

res = requests.post(url, json=payload)

print("Status Code:", res.status_code)
print("Raw Text:", res.text)