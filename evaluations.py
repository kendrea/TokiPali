import torch
from modelling.utils import sample, next_distribution
from train import load_model
from tokenizer import string_to_tokens, tokens_to_tokipona
from math import log, exp

models_kaplan = {
    "learned embeddings from handcrafted": load_model("trainings/10epochCustom.pt", "Kaplan"),
    "fixed handcrafted embeddings": load_model("trainings/10epochCustomFrozen.pt", "Kaplan"),
    "learned embeddings from random": load_model("trainings/10epochsNormal.pt", "Kaplan"),
    "fixed random embeddings": load_model("trainings/10epochFrozen.pt", "Kaplan"),
    "no training": load_model(None, "Kaplan"),
}

models_chinchilla = {
    "learned embeddings from handcrafted": load_model("trainings/10epochCustomChinchilla.pt", "Chinchilla"),
    "fixed handcrafted embeddings": load_model("trainings/10epochCustomFrozenChinchilla.pt", "Chinchilla"),
    "learned embeddings from random": load_model("trainings/10epochsNormalChinchilla.pt", "Chinchilla"),
    "fixed random embeddings": load_model("trainings/10epochFrozenChinchilla.pt", "Chinchilla"),
    "no training": load_model(None, "Chinchilla"),
}

test_strings = [
    """jan ali li kama lon nasin ni: ona li ken tawa li ken pali. 
        jan ali li kama lon sama. 
        jan ali li jo e ken pi pilin suli. 
        jan ali li ken pali e wile pona ona. 
        jan ali li jo e ken pi sona pona e ken pi pali pona. 
        jan ali li wile pali nasin ni: ona li jan pona pi jan ante.""",
        """jan Meli o,
            kon sewi li suli insa sina.
            wan sewi li poka sina.
            lon meli la, wan sewi li pona e sina.
            kili pona pi insa sina li sewi Jesu.
            jan Meli sewi o!
            mama pi jan sewi o!
            tenpo ni la, tenpo pi moli mi mute la,
            o toki tawa wan sewi tan mi mute jan ike.
            awen.""",
    """mi lon.
        mi lon e tomo mi e ma tomo e ma.
        ma li lon e sike suno.
        suno li lon e ale.
        tenpo ni la mi kepeken lawa e ni: tenpo ni li lon.
        lawa mi li jo e ijo mute tan toki mi li lon ma pona pi toki pona.
        jan ante li lon, kin.
        mi ken toki tawa jan mute tan jan mute li lon.
        mi sona e ni.
    """,
    """tenpo ni li tenpo suno open.
    tenpo pini lili la mi lape ala.
    tenpo pini li lon e tenpo pini.
    ni li lon.
    tenpo kama la mi moku e pan.
    tenpo kama li lon e tenpo kama.
    en pan li lon tenpo kama.
    ni li lon, kin.
    mi sona e ni.
    """,
    """tenpo kama li tenpo ni.
    mi moku e pan.
    mi kepeken lawa e ni: 
    mi moku e pan.
    toki ni li ni: pan li moku.
    pan kama moku tan mi.
    taso, pan kama moku tawa mi, taso.
    tenpo ni la, pan li moku tawa mi.
    mi taso.
    """,
    """mi lon e tomo mi.
    toki ni li ni: tomo mi li lon.
    tomo mi kama lon tan mi.
    taso, tomo mi li lon tawa mi, taso.
    tenpo ni la, tomo mi li lon tawa mi.
    mi taso.
    """,
    """mi kepeken lawa pini.
    tenpo kama lili la… ona li weka.
    lawa mi li weka.
    tomo mi pi sona toki li weka.
    sona pi jan ale li open tawa mi.
    tenpo mi li sama e kulupu e lipu e sona.
    taso… pilin mi li weka.
    pan en moku li lon ala.
    en tomo mi en ma tomo en ma en suno en sike li lon ala.
    tenpo ni en tenpo pini en tenpo kama li lon ala.
    jan ante li lon ala; ma pona pi toki pona li lon ala.
    ijo mute pi lawa mi lon ala.
    toki mi li lon ala.
    lawa mi li lon ala.
    mi lon ala.
    """
]

test_token_seq = list(map(string_to_tokens, test_strings))

def infer(model, x):
    ys = sample(model, torch.as_tensor(string_to_tokens(x))[None, ...], 1, temperature=10.0, sample=False, top_k=10, print_top=10)
    y = ys[0]
    return tokens_to_tokipona(y.tolist())


def perplexity(model, tokens):
    if not len(tokens):
        return None
    accumulated_logprobs = 0
    # note: shape of tokens is 1xsomething
    length = len(tokens)
    for i in range(1, length-1):
        context = tokens[:i]
        correct_next_token = tokens[i]
        probs = next_distribution(model, torch.as_tensor(context, dtype=torch.int)[None, ...])
        # print(probs)
        predicted_prob_of_correct = probs[:,correct_next_token]
        logprob = log(predicted_prob_of_correct)
        accumulated_logprobs += logprob
    scaled_sum = accumulated_logprobs / -length
    return exp(scaled_sum)


def evaluate(models):
    for model_name in models:
        print("Evaluating", model_name)
        model = models[model_name]
        perplexities = []
        for token_seq in test_token_seq:
            perp = perplexity(model, token_seq)
            perplexities.append(perp)
            print("\tPerplexity:", perp)

        print("\tAverage perplexity:", sum(perplexities)/len(perplexities))
    return


if __name__ == "__main__":
    print("Evaluating Kaplan")
    evaluate(models_kaplan)

    print("Evaluating Chinchilla")
    evaluate(models_chinchilla)

    inference = infer(models_kaplan["fixed random embeddings"], "jan ali li kama lon nasin ni: ona li ken tawa li ken pali. jan ali li kama lon sama. jan ali li jo e ken pi pilin suli. jan ali li ken pali e wile pona ona. jan ali li jo e ken pi sona pona e ken pi pali pona. jan ali li wile pali nasin ni: ona li jan pona pi jan")
    print("String inference:", inference)
