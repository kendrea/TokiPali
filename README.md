# TokiPali: Word Builder
Group project for AI 535: "Word Maker" Toki Pona next-token prediction

## Usage
All required python packages are frozen in `requirements.txt`

### Training
For training the model use `train.py`.

```
usage: Toki Pali [-h] [--checkpoint CHECKPOINT] [--scaling {kaplan,chinchilla}] [--custom-embedding] [--frozen-embedding] {new,resume}

Scaled GPT for Toki Pona

positional arguments:
  {new,resume}          start new or resume training

options:
  -h, --help            show this help message and exit
  --checkpoint CHECKPOINT, -c CHECKPOINT
                        weights to use for training/inference
  --scaling {kaplan,chinchilla}
                        choose scalling laws used for the model.
  --custom-embedding, -e
                        enable handcrafted word embeddings (random by default)
  --frozen-embedding, -f
                        freeze embeddings such that they remain unchanged throughout the training process.
```

### Evaluation
Note that some pretrained weights are available in `trainings/`

Run `evaluations.py` for automatic evaluation using perplexity.
It will also conduct inference on a predefined string.
