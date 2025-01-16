import openai

completion = openai.ChatCompletion.create(
  model="gpt-3.5-turbo", 
  messages=[{"role": "user", "content": "What is the OpenAI mission?"}]
)

print(completion)