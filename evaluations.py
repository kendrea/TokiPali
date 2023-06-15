import torch
from torcheval.metrics import Perplexity
from modelling.utils import sample
from premade_train import model

def infer(x):
    ys = sample(model, torch.as_tensor(string_to_tokens(x))[None, ...], 1, temperature=10.0, sample=False, top_k=10, print_top=10)
    y = ys[0]
    return tokens_to_tokipona(y.tolist())

def evaluate(inference):
	perplexity = Perplexity
	print("it's perfect")
	return

if __name__ == "__main__":
    model.load_state_dict(torch.load("trainings/10epochCustom.pt", map_location=torch.device('cpu')))
    inference = ("jan ali li kama lon nasin ni: ona li ken tawa li ken pali. jan ali li kama lon sama. jan ali li jo e ken pi pilin suli. jan ali li ken pali e wile pona ona. jan ali li jo e ken pi sona pona e ken pi pali pona. jan ali li wile pali nasin ni: ona li jan pona pi jan")
    print(inference)
    evaluate(inference)