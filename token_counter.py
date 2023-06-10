from tokidata import load_data, DataSets
from tokenizer import string_to_tokens, tokens_to_tokipona
from transformers import AutoTokenizer, PreTrainedTokenizer


def calc_len(toks):
    return sum(len(x) for x in toks)


all_tokens = []
for datatype in DataSets:
    datas = load_data(datatype)
    tokens = [string_to_tokens(d) for d in datas]
    all_tokens += tokens

    print(f"{datatype}: {calc_len(tokens)} tokens")

print(f"total: {calc_len(all_tokens)} tokens")


class TokiTokenizer:
    def encode(self, text):
        tokens = string_to_tokens(text)
        return tokens

    def decode(self, text):
        pona = tokens_to_tokipona(text)
        return pona

tokenizer = TokiTokenizer()


