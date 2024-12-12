from zhipuai import ZhipuAI


def call_large_model(messages, api_key, model="glm-4-plus", max_tokens=1024, temperature=0.0):
    client = ZhipuAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
