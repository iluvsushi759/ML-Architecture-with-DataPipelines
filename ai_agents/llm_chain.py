import os

def generate_answer(query: str):
    if os.environ.get("USE_OPENAI", "false").lower() == "true":
        import requests, json
        api_key = os.environ['OPENAI_API_KEY']
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        messages = [
            {"role": "system", "content": "You are a helpful analyst. Ground answers in provided context."},
            {"role": "user", "content": f"Question: {query}"}
        ]
        r = requests.post("https://api.openai.com/v1/chat/completions",
                          headers=headers,
                          data=json.dumps({"model": os.environ.get("CHAT_MODEL", "gpt-4o-mini"),
                                            "messages": messages}))
        r.raise_for_status()
        return r.json()['choices'][0]['message']['content']
    else:
        # Free fallback: just return retrieved context (no LLM)
        return f"[FAISS fallback] Retrieved context for query: {query}"

if __name__ == "__main__":
    print(generate_answer("Which claims show unusually high amounts last week?"))
