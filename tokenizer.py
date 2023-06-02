"""
Convert between Toki Pona strings and token ID lists
"""
from vocab import normal_words, punctuation, equivalence
from vocab import no_space_before, no_space_after

token_to_word = dict(enumerate(normal_words + punctuation))
word_to_token = {}
for key, value in token_to_word.items():
    word_to_token[value] = key


def string_to_tokens(string: str) -> list[int]:
    """
    Convert a plain ASCII string to a list of token IDs.
    Preprocesses to ignore garbage and normalize stuff.
    """
    split = string.split(' ')
    tokens = []
    for word in split:
        # punctuation before word
        if word[0] in no_space_after:
            tokens.append(word_to_token[word[0]])
            word = word[1:]  # remove first char

        # actual word
        if word in word_to_token:
            tokens.append(word_to_token[word])
            continue

        # punctuation after, other special cases
        chars = list(word)
        for i, char in enumerate(chars):
            char = equivalence.get(char, char)
            chars[i] = char
        word = ''.join(chars)

        def separate(lst, item):
            lst = lst.split(item)
            result = [item] * (len(lst)*2 - 1)
            result[0::2] = lst
            return result

        chunks = [word]
        for punct in punctuation:
            if punct not in word:
                continue
            new_chunks = []
            for chunk in chunks:
                new_chunks += separate(chunk, punct)
            chunks = [c for c in new_chunks if c != '']

        for chunk in chunks:
            if chunk in word_to_token:
                tokens.append(word_to_token[chunk])

    return tokens


def tokens_to_tokipona(tokens: list[int]) -> str:
    """
    Convert a list of token IDs to a string of Toki Pona.
    Also post-processes for things like spacing.
    """
    strs = [token_to_word[t] for t in tokens]
    to_join = []
    inside_quote = False
    for i, word in enumerate(strs):
        if word in no_space_before and i != 0:
            to_join[-1] += word
            continue
        if word in no_space_after and i != len(strs)-1:
            strs[i+1] = strs[i] + strs[i+1]
            continue
        if word == '"':
            if inside_quote:
                if i != 0:
                    to_join[-1] += word  # no_space_before
            elif i != len(strs)-1:
                strs[i+1] = strs[i] + strs[i+1]  # no_space_after
            elif i == len(strs)-1:
                # special case: sentence ends with open quote
                to_join.append(word)
            inside_quote = not inside_quote
            continue
        to_join.append(word)
    return " ".join(to_join)


def tokens_to_english(tokens: list[int]) -> str:
    """
    Hilarious non-AI translation.
    Convert list of tokens to list of Toki Pona words,
    then look up and substitute a translation of each.
    """
    raise NotImplementedError()
