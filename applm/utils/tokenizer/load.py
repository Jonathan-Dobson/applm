import json
from transformers import AutoTokenizer
from functools import partial

from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from .NaiveStreamingDetokenizer import NaiveStreamingDetokenizer
from .wrapper import TokenizerWrapper
from .is_valid_decoder import is_bpe_decoder, is_spm_decoder, is_spm_decoder_no_space
from .SPMStreamingDetokenizer import SPMStreamingDetokenizer
from .BPEStreamingDetokenizer import BPEStreamingDetokenizer


def load_tokenizer(model_path, tokenizer_config_extra={}):
    """Load a huggingface tokenizer and try to infer the type of streaming
    detokenizer to use.

    Note, to use a fast streaming tokenizer, pass a local file path rather than
    a Hugging Face repo ID.
    """
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r") as fid:
            tokenizer_content = json.load(fid)
        if "decoder" in tokenizer_content:
            if is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    return TokenizerWrapper(
        AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
        detokenizer_class,
    )
