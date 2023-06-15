"""
[
noun,
verb/pre-verb,
adjective,
particle/preposition,
particle index,
relating to physical object,
numerical,
positive sentiment (0 negative, 1 neutral, 2 positive),
natural,
human,
size (0 n/a, 1 smaller than a breadbox, 2 larger),
color index,
animal index,
indoor (1), outdoor (2), on a person (3),
passive (1), active (2),
final differentiator (0, 1, 2),
punctuation index (1, 2, 3)
]
"""

import torch
import numpy as np
from vocab import token_list


def _make_normalized_tensor(dictionary):
    def normalize(vec_as_list):
        vec = np.array(vec_as_list)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    return torch.Tensor(np.array(list(map(normalize, dictionary.values()))))


def find_duplicate_embeddings(dictionary):
    """
    Ensure no two tokens have the same embeddings
    """
    res = True
    for idx_a, item_a in dictionary.items():
        for idx_b, item_b in dictionary.items():
            if idx_a == idx_b:
                continue
            if item_a == item_b:
                print(f"{idx_a} and {idx_b} have identical embeddings")
                res = False
    return res


def consistency_check(embeddings, tokens):
    """
    Ensure the list of embeddings and the list of tokens are the same
    """
    res = True
    if set(embeddings.keys()) != set(tokens):
        res = False
        print("Definition list and token list don't match...")
        for word in tokens:
            if word not in embeddings:
                print("missing embedding for", word)

        # check to see what's extra
        for key in embeddings:
            if key not in tokens:
                print("extra embedding for", key)

    # check if order is the same as token order
    for i, (e, t) in enumerate(zip(embeddings, token_list)):
        if e != t:
            print(f"Mismatch at idx {i} with embedding {e} and token {t}")
            res = False
            break
    return res


def colinear_check(tensor):
    """
    Same as duplicate check, but for (pre-normalized) rows of a tensor
    """
    res = True
    for idx_a in range(tensor.shape[0]):
        for idx_b in range(tensor.shape[0]):
            if idx_a == idx_b:
                continue
            if np.array_equal(tensor[idx_a, :], tensor[idx_b, :]):
                print(f"{idx_a} and {idx_b} have colinear embeddings")
                res = False
    return res


manual = {
    "EOS":              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3],
    ".":                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3],
    "?":                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
    "!":                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    ",":                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ":":                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],

    "a":                [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    "akesi":            [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 2, 2, 0, 0],
    "ala":              [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "alasa":            [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 2, 2, 0, 0],
    "ale":              [1, 0, 1, 0, 0, 1, 1, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0],
    "anpa":             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    "ante":             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "anu":              [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "awen":             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    "e":                [0, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "en":               [0, 0, 0, 1, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "esun":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 2, 0, 0],
    "ijo":              [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    "ike":              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "ilo":              [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    "insa":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 3, 0, 1, 0],
    "jaki":             [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "jan":              [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 3, 0, 1, 0],
    "jelo":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    "jo":               [0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    "kala":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 2, 2, 2, 0, 0],
    "kalama":           [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 2, 0, 0],
    "kama":             [0, 1, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    "kasi":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 3, 2, 0, 0, 0],
    "ken":              [0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "kepeken":          [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 1, 0],
    "kili":             [1, 0, 0, 0, 0, 1, 0, 2, 1, 1, 1, 0, 0, 0, 1, 0, 0],
    "kiwen":            [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0],
    "ko":               [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    "kon":              [1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "kule":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "kulupu":           [1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 0, 0, 3, 2, 0, 0],
    "kute":             [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 3, 1, 0, 0],
    "la":               [0, 0, 0, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "lape":             [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 3, 1, 0, 0],
    "laso":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    "lawa":             [1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 3, 2, 0, 0],
    "len":              [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 3, 1, 0, 0],
    "lete":             [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    "li":               [0, 0, 0, 1, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "lili":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "linja":            [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    "lipu":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
    "loje":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 3, 0, 0, 0, 0, 0],
    "lon":              [0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "luka":             [1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 3, 2, 0, 0],
    "lukin":            [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 3, 2, 0, 0],
    "lupa":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 0, 0, 0],
    "ma":               [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 2, 0, 0, 0],
    "mama":             [1, 0, 0, 0, 0, 1, 0, 2, 1, 1, 2, 0, 0, 3, 2, 0, 0],
    "mani":             [1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 3, 0, 0, 0],
    "meli":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 3, 2, 1, 0],
    "mi":               [1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 2, 0, 0, 3, 0, 0, 0],
    "mije":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 3, 2, 0, 0],
    "moku":             [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 3, 2, 0, 0],
    "moli":             [0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 3, 1, 0, 0],
    "monsi":            [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "mu":               [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0],
    "mun":              [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0, 0],
    "musi":             [0, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    "mute":             [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    "nanpa":            [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "nasa":             [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "nasin":            [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    "nena":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0],
    "ni":               [0, 0, 1, 0, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "nimi":             [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "noka":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 3, 2, 2, 0],
    "o":                [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    "olin":             [0, 1, 0, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 2, 0, 0],
    "ona":              [1, 0, 0, 0, 2, 1, 0, 1, 0, 1, 2, 0, 0, 3, 0, 0, 0],
    "open":             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    "pakala":           [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "pali":             [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 1, 0],
    "palisa":           [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0],
    "pan":              [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "pana":             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 2, 0],
    "pi":               [0, 0, 0, 1, 4, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pilin":            [1, 0, 1, 0, 0, 1, 0, 2, 1, 1, 0, 0, 0, 3, 0, 0, 0],
    "pimeja":           [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 4, 0, 0, 0, 0, 0],
    "pini":             [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "pipi":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 5, 2, 2, 0, 0],
    "poka":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 3, 1, 0, 0],
    "poki":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0],
    "pona":             [0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "pu":               [0, 0, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 1, 2, 0, 0],
    "sama":             [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "seli":             [1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    "selo":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 2, 0],
    "seme":             [0, 0, 0, 1, 5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sewi":             [1, 0, 1, 0, 0, 0, 0, 2, 1, 1, 2, 0, 0, 0, 0, 0, 0],
    "sijelo":           [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 0, 0, 3, 0, 0, 0],
    "sike":             [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    "sin":              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "sina":             [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 0, 0, 0, 0, 0, 0],
    "sinpin":           [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0],
    "sitelen":          [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    "sona":             [0, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    "soweli":           [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 7, 2, 2, 0, 0],
    "suli":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    "suno":             [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 2, 0, 0],
    "supa":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0],
    "suwi":             [0, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 1, 0, 0],
    "tan":              [0, 0, 0, 1, 6, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "taso":             [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "tawa":             [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    "telo":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    "tenpo":            [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "toki":             [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 2, 0, 0],
    "tomo":             [1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 2, 0, 0, 1, 1, 0, 0],
    "tu":               [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    "unpa":             [0, 1, 0, 0, 0, 1, 0, 2, 1, 1, 0, 0, 0, 1, 2, 0, 0],
    "uta":              [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 3, 0, 0, 0],
    "utala":            [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 0, 0],
    "walo":             [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0],
    "wan":              [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    "waso":             [1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 6, 2, 2, 0, 0],
    "wawa":             [0, 0, 1, 0, 0, 0, 0, 2, 1, 1, 0, 0, 0, 0, 2, 0, 0],
    "weka":             [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    "wile":             [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "leko":             [1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    "monsuta":          [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0],
    # "ali":            [1, 0, 1, 0, 0, 1, 1, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0],
    # "apeja":          [1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # "kijetesantakalu":[1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 4, 0, 2, 0, 0],
    # "kin":            [0, 0, 0, 1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "kipisi":         [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # "majuna":         [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    # "namako":         [1, 1, 1, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    # "oko":            [1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    # "po":             [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 1, 0],
    # "powe":           [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    # "tuli":           [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0],
}


manual_tensor = _make_normalized_tensor(manual)


if __name__ == "__main__":
    if find_duplicate_embeddings(manual) \
            and consistency_check(manual, token_list) \
            and colinear_check(manual_tensor):
        print("Consistency checks passed!")