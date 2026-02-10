import os
from mistralai import Mistral

api_key = os.environ.get("MISTRAL_API_KEY_3")
if not api_key:
    raise RuntimeError("MISTRAL_API_KEY_3 is not set")

client = Mistral(api_key=api_key)

resp = client.chat.complete(
    model="mistral-small-2506",
    messages=[{"role": "user", "content": "Say 'ok'."}],
)

print(resp.choices[0].message.content)