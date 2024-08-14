from utils.load import load
from utils.generate import generate
import models.qwen2 as qwen2


model_name = "mlx-community/Qwen2-0.5B"
# model_name = "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16"


model, tokenizer = load(
    model_name,
    tokenizer_config={"eos_token": "<|im_end|>"},
    mx_prototype=qwen2,
)

prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
input_text = ""
for message in messages:
    if message["role"] == "user":
        input_text = message["content"]
        break


response = generate(
    model,
    tokenizer,
    prompt=input_text,
    verbose=True,
    top_p=0.8,
    temp=0.7,
    repetition_penalty=1.05,
    max_tokens=512,
    mx_prototype=qwen2,
)
