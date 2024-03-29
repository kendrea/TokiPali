import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from vocab import token_to_word


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("inf")
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None, print_top=0):
    """
    take a conditioning sequence of indices in x (of shape (b, t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilites
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely

        if sample:
            ix = torch.multinomial(probs, num_samples=10)
        else:
            _, ix = torch.topk(probs, k=10, dim=-1)

        if print_top > 0:
            print(f"Top {print_top}:", ' '.join(token_to_word[token.item()] for token in ix[0]))

        # append to the sequence and continue
        x = torch.cat((x, ix[:, :1]), dim=1)

    return x

@torch.no_grad()
def next_distribution(model, x):
    """
    Return distribution over possible next tokens, conditioned on x, a list of context tokens.
    """
    block_size = model.get_block_size()
    # clip context to block_size
    x_cond = (
        x if x.shape[1] <= block_size else x[:, -block_size:]
    )
    logits = model(x_cond)
    if logits.shape[1]:
        logits = logits[:, -1, :]  # just get next token (final) prediction
    probs = F.softmax(logits, dim=-1)
    return probs
