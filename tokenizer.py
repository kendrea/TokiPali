"""
Convert between Toki Pona strings and token ID lists
"""
from vocab import normal_words, punctuation, equivalence
from vocab import token_to_word, word_to_token
from translations import tokipona_to_en


def string_to_tokens(string: str) -> list[int]:
    """
    Convert a plain ASCII string to a list of token IDs.
    Preprocesses to ignore garbage and normalize stuff.
    """
    split = string.strip(' ').split(' ')
    tokens = [word_to_token["SOS"]]
    for word in split:
        if not word:
            continue

        # make common case fast
        equivalence.get(word, word)
        if word in word_to_token:
            tokens.append(word_to_token[word])
            continue

        # next most common: word with one punctuation after
        no_end = word[:-1]
        equivalence.get(no_end, no_end)
        if no_end in word_to_token:
            tokens.append(word_to_token[no_end])
            if equivalence.get(word[-1], word[-1]) in word_to_token:
                tokens.append(word_to_token[word[-1]])
            continue

        # special cases like repeat punctuation (slow)

        # first normalize everything
        word = word.lower()
        chars = list(word)
        for i, char in enumerate(word):
            if not char.isalpha():
                if char in equivalence or char in punctuation:
                    chars[i] = equivalence.get(char, char)
                else:  # unrecognized character
                    chars[i] = ' '
        word = ''.join(chars)
        print(f"normalized: \"{word}\"")

        def separate(lst, item):
            lst = lst.split(item)
            result = [item] * (len(lst)*2 - 1)
            result[0::2] = lst
            return [r for r in result if r != '']

        chunks = [word]
        for punct in punctuation + [' ']:
            if punct not in word:
                continue
            new_chunks = []
            for chunk in chunks:
                new_chunks += separate(chunk, punct)
            chunks = [c for c in new_chunks if c != '']

        for chunk in chunks:
            chunk = equivalence.get(chunk, chunk)
            if chunk in word_to_token:
                tokens.append(word_to_token[chunk])

    tokens.append(word_to_token["EOS"])
    return tokens


def tokens_to_tokipona(tokens: list[int]) -> str:
    """
    Convert a list of token IDs to a string of Toki Pona.
    Also post-processes for things like spacing.
    """
    strs = [token_to_word[t] for t in tokens]
    to_join = []
    for i, word in enumerate(strs):
        if word in punctuation and i != 0:
            to_join[-1] += word
            continue
        to_join.append(word)
    return " ".join(to_join)


def tokens_to_english(tokens: list[int]) -> None:
    """
    Hilarious non-AI translation.
    Convert list of tokens to list of Toki Pona words,
    then look up and substitute a translation of each.
    """
    for token in tokens:
        word = token_to_word[token]
        print(word, end='')
        if word in normal_words:
            defs = tokipona_to_en[word]
            print(' '*(16-len(word)), str(defs[0]))
            if len(defs) != 1:
                for d in defs[1:]:
                    print(' '*16, str(d))
        else:
            print()
