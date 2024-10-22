import request
import json

url = "http://127.0.0.1:8000/sentiment"

text = input("Masukkan teks: ")
data = {
    "text": text
}

response = requests.post(url, json=data)

if response.status_code == 200:
    print(f"Hasil Sentimen: {response.json()['sentiment']}")
else:
    print(f"Error: {response.status_code} - {response.text}")

