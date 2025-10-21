from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

try:
    response = client.chat.completions.create(
        model="codellama:34b-instruct-q5_K_M",
        messages=[{"role": "user", "content": "Write hello world"}],
        max_tokens=50
    )
    print("✅ Connected!")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"❌ Error: {e}")