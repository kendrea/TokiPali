# Notes

## Scaling Laws
- Chinchilla suggest 20 tokens per parameter
- Kaplan scaling (from OpenAI in 2020) suggests 1.7 tokens per parameter

The Chinchilla authors use three approaches: 1) "fix model sizes and vary number of training tokens"; 
2) fix FLOP counts and vary model size; and
3) model loss as a function of model parameter count and number of seen tokens.

To explore how to make the most of our limited training data and compute, we use approaches 2 and 3. 
Chinchilla has 70B parameters:
- 80 layers
- 64 number heads
- 128 key/value size
- 8192 d_model (bottleneck activation size)
- $10^{-4}$ max LR
- batch size of 1.5M, increasing to 3M halfway through
- feed-forward size of 8192 x 4

We have 558,382 tokens to work with.

### Parameters
- [https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
	- d_model - # expected features in encoder/decoder inputs
	- nhead - # heads in multiheadattention models
	- dim_feedforward
- # layers 


## Embeddings

## Skipping vs. keeping unknown tokens
- ChatGPT says skipping unrecognized text makes for a simpler model that is less likely to give nonsensical output
