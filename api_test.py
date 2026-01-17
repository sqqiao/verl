from openai import OpenAI

client = OpenAI( api_key="sk-ggULmZVwdXy3OXLUmJIBsywVAQXVK4JhwBdLoYkBGiYxIjDu", base_url="https://a.fe8.cn/v1" )

response = client.chat.completions.create( model="gpt-4o", messages=[ {"role": "system", "content": "You are a helpful assistant."}, { "role": "user", "content": "Explain to me how AI works" } ] )

print(response.choices[0].message)