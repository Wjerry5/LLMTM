

import os
import openai
os.environ["OPENAI_API_KEY"] = "sk-Pth1hQVprCzicKWmB16396007fC44c568a74F3C8Fb484979"
openai.base_url = "https://api.vveai.com/v1/"
openai.default_headers = {"x-foo": "true"}


completion = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": "Hello world!",
        },
    ],
)
print(completion.choices[0].message.content)

# 正常会输出结果：Hello there! How can I assist you today ?