# set up logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# make deterministic
from modelling.utils import set_seed
set_seed(42)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from tokenizer import string_to_tokens, word_to_token, tokens_to_tokipona, tokens_to_english

class TokiDataset(Dataset):

    def __init__(self, data: list[str], context_window: int):
        self.data = data
        self.context_window = context_window

        tokenized = []
        for document in data:
            tokenized += string_to_tokens(document) + [word_to_token["EOS"]]

        self.tokenized = tokenized

    def __len__(self):
        return len(self.tokenized) - self.context_window
    
    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        #stream_count = idx // self.block_size
        #index = idx % self.block_size
        #chunk = self.data[stream_count][index:index+1]
        # encode every character to an integer
        dix = self.tokenized[idx: idx + self.context_window + 1]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. 
        
        Notice that the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can parallelize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


#############
# Prep Data #
#############
from tokidata import load_all_data
from vocab import vocab_size

context_window = 128

dataset = TokiDataset(load_all_data().tolist(), context_window)

from modelling.model import GPT, GPTConfig

mconf = GPTConfig(
    vocab_size,
    dataset.context_window,
    n_layer=7,  # 5
    n_head=4,  # 5
    n_embd=72,  # 65,
    freeze_embeds=True,
    custom_embeds=True,
)

model = None

from modelling.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training
tconf = TrainerConfig(
    max_epochs=10,
    batch_size=256*2,
    learning_rate=5e-3,
    lr_decay=0.90,
    warmup_tokens=512*30,
    final_tokens=2*len(dataset)*dataset.context_window,
    num_workers=4,
    ckpt_path="checkpoint.pt",
)

def train_new():
    trainer = Trainer(model, dataset, None, tconf, mconf)
    trainer.train()

def train_continue():
    trainer = Trainer(model, dataset, None, tconf, mconf)
    trainer.model.module.load_state_dict(torch.load("checkpoint.pt", map_location=torch.device('cpu')))
    trainer.train()
    #trainer.save_checkpoint()

def load_model(filename: str):
    new_model = GPT(mconf)
    new_model.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
    return new_model

if __name__ == "__main__":
    model = GPT(mconf)
    print("uncomment to choose what you want this program to even do")
    # train_new()
    # train_continue()
