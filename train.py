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

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import argparse
import pathlib

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

def kaplan_config(freeze=False, custom=False):
    return GPTConfig(
        vocab_size,
        dataset.context_window,
        n_layer=7,  # 5
        n_head=4,  # 5
        n_embd=72,  # 65,
        freeze_embeds=freeze,
        custom_embeds=custom,
    )

def chinchila_config(freeze=False, custom=False):
    return GPTConfig(
        vocab_size,
        dataset.context_window,
        n_layer=4,  # 5
        n_head=4,  # 5
        n_embd=28,  # 65,
        freeze_embeds=freeze,
        custom_embeds=custom,
    )

model = None

from modelling.trainer import Trainer, TrainerConfig

# initialize a trainer instance and kick off training

def get_trainer_config(checkpoint):
    return TrainerConfig(
        max_epochs=1,
        batch_size=256*2,
        learning_rate=5e-3,
        lr_decay=0.9999,
        warmup_tokens=512*30,
        final_tokens=2*len(dataset)*dataset.context_window,
        num_workers=4,
        ckpt_path=checkpoint,
    )

def train_new(mconf, tconf):
    model = GPT(mconf)
    trainer = Trainer(model, dataset, None, tconf, mconf)
    trainer.train()

def train_continue(mconf, tconf):
    model = GPT(mconf)
    trainer = Trainer(model, dataset, None, tconf, mconf)
    trainer.model.module.load_state_dict(torch.load(tconf.ckpt_path))
    trainer.train()
    #trainer.save_checkpoint()

def load_model(filename: str | None, scaling):
    if scaling == "Kaplan":
        new_model = GPT(kaplan_config())
    elif scaling == "Chinchilla":
        new_model = GPT(chinchila_config())
    else:
        raise ValueError(f"Invalid scaling: {scaling}")

    if filename is not None:
        new_model.load_state_dict(torch.load(filename))
    return new_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            prog="Toki Pali",
            description="Scaled GPT for Toki Pona",
    )

    parser.add_argument("task", choices=['new', 'resume'], help='start new or resume training')
    parser.add_argument("--checkpoint", '-c', help='weights to use for training/inference',  type=pathlib.Path, default=pathlib.Path("trainings/checkpoint.pt"))
    parser.add_argument("--scaling", default="kaplan", choices=['kaplan', 'chinchilla'], help="choose scalling laws used for the model.")
    parser.add_argument("--custom-embedding", '-e', action="store_true", help="enable handcrafted word embeddings (random by default)")
    parser.add_argument("--frozen-embedding", '-f', action="store_true", help="freeze embeddings such that they remain unchanged throughout the training process.")
    args = parser.parse_args()

    if args.scaling == "chinchilla":
        mconf = chinchila_config(args.frozen_embedding, args.custom_embedding)
    else:
        mconf = kaplan_config(args.frozen_embedding, args.custom_embedding)

    tconf = get_trainer_config(str(args.checkpoint))

    if args.task == "new":
        logging.info(f"Training using {args.scaling} scaling. Custom Embeds: {args.custom_embedding}, Embeds Frozen: {args.frozen_embedding}")
        train_new(mconf, tconf)

    elif args.task == "resume":
        logging.info(f"Resuming training from {args.checkpoint} with {args.scaling} scaling. Custom Embeds: {args.custom_embedding}, Embeds Frozen: {args.frozen_embedding}")
        train_continue(mconf, tconf)
