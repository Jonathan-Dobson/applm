from .StreamingDetokenizer import StreamingDetokenizer


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    def __init__(self, tokenizer, trim_space=True):
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [""] * (max(tokenizer.vocab.values()) + 1)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        # Replace bytes with their value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        v = self.tokenmap[token]
        if v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        if self.text or not self.trim_space:
            self.text += self._unflushed.replace("\u2581", " ")
        else:
            self.text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


def _remove_space(x):
    if x and x[0] == " ":
        return x[1:]
    return x
