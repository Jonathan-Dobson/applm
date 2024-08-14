# AppLM

AppLM brings Language Model Tools to Apple Silicon.

This project aiming to refactor the [mlx-lm](https://pypi.org/project/mlx-lm/) package with a goal to reorganize it making the features less opinionated and more flexible for developer customization. For example, we are removing the CLI components, and exposing more of the package parts as an developer API.

Warning: This package is still incomplete.

Please do not use it yet in production.

This package was created August 13, 2024

We have only tested this one code example here:

```python
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

```

This working example is meant to showcase the direction we are working toward, however the next change we update will probably break it.

If you like this project, or want to see progress, or want to contribute, please feel free to engage with the repo on github.

We are looking for contributors to help further develop this package.

The refactor is not complete, but we will make regular updates as we progress.

If you are the Apple developer in charge of leading the mlx-lm package development, and you also have the goal to refactor the mlx-lm package to be less opinionated and more developer customizable, feel free to use this package as inspiration, fork it, merge it, or collaborate either on this package or the mlx-lm package.
