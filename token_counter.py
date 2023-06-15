from tokidata import load_data, DataSets
from tokenizer import string_to_tokens, tokens_to_tokipona


def calc_len(toks):
    # len(toks) is the number of EOFs
    return sum(len(x) for x in toks) + len(toks)


all_tokens = []
for datatype in DataSets:
    datas = load_data(datatype)
    tokens = [string_to_tokens(d) for d in datas]
    all_tokens += tokens

    print(f"{datatype}: {calc_len(tokens)} tokens")

print(f"total: {calc_len(all_tokens)} tokens")
