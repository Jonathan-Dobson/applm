from typing import Callable, Optional, Union
from mlx import core as mx, nn
from transformers import PreTrainedTokenizer
from .generate_step import generate_step
from .tokenizer.wrapper import TokenizerWrapper


def generate(
    model: nn.Module,
    tokenizer: Union[PreTrainedTokenizer, TokenizerWrapper],
    prompt: str,
    max_tokens: int = 1000,
    verbose: bool = False,
    formatter: Optional[Callable] = None,
    mx_prototype=None,
    **kwargs,
) -> str:
    """
    Generate a complete response from the model.

    Args:
        model (nn.Module): The language model.
        tokenizer (Union[PreTrainedTokenizer, TokenizerWrapper]): The tokenizer.
        prompt (str): The string prompt.
        max_tokens (int, optional): The maximum number of tokens. Defaults to 100.
        verbose (bool, optional): If True, print tokens and timing information. Defaults to False.
        formatter (Optional[Callable], optional): A function which takes a token and a probability. Defaults to None.

    Returns:
        str: The generated text.
    """
    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)

    prompt_tokens = mx.array(tokenizer.encode(prompt))
    detokenizer = tokenizer.detokenizer

    for (token, logprobs), n in zip(
        generate_step(prompt_tokens, model, mx_prototype=None, **kwargs),
        range(max_tokens),
    ):
        if token == tokenizer.eos_token_id:
            break
        detokenizer.add_token(token)

        if verbose and formatter:
            detokenizer.finalize()
            formatter(detokenizer.last_segment, mx.exp(logprobs[token]).item())

    detokenizer.finalize()

    if verbose:
        print(detokenizer.text)

    return detokenizer.text
