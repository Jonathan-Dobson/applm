from typing import Any
from .NaiveStreamingDetokenizer import NaiveStreamingDetokenizer


class TokenizerWrapper:
    """A wrapper that combines an HF tokenizer and a detokenizer.

    Accessing any attribute other than the ``detokenizer`` is forwarded to the
    huggingface tokenizer.
    """

    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer) -> None:
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr):
        if attr == "detokenizer":
            return self._detokenizer
        else:
            return getattr(self._tokenizer, attr)
