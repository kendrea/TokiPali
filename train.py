from gpt2torch.GPT2.model import GPT2Model
from gpt2torch.GPT2.config import GPT2Config

from vocab import vocab_size

from torch.utils.data import Dataset, DataLoader

import torch
import numpy as np
import random

batch_size = 16

def main():
    seed = random.randint(0, 2147483647)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = GPT2Config(vocab_size_or_config_json_file=vocab_size)
    model = GPT2Model(config)
    model.to(device)

    print(model)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    criterion = torch.nn.CrossEntropyLoss()


if __name__ == "__main__":
    main()
