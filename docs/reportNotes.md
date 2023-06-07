# Notes

## Scaling Laws

### Questions

- 

### Notes
- Chinchilla suggest 20 tokens per parameter
- Chinchilla: "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled" (1).
- Kaplan scaling (from OpenAI in 2020) suggests 1.7 tokens per parameter

Chinchilla has 70B parameters:
- 80 layers
- 64 number heads
- 128 key/value size
- 8192 d_model (bottleneck activation size)
- $10^{-4}$ max LR
- batch size of 1.5M, increasing to 3M halfway through
- feed-forward size of 8192 x 4

The Chinchilla authors use three approaches: 
1) "fix model sizes and vary number of training tokens" 
2) fix FLOP counts and vary model size
3) model loss as a function of model size and number of seen tokens

They list tokens vs. parameters for fixed FLOP budgets (8). The ratios of tokens to parameters are 20, 20.2, 20.51, 22.388, 21.1428, 21.0714, 21.1538, 21.2, and 21.62. The mean is approximately 21. Fitting a linear regression of tokens vs. ratios yields $3.37687×10^-6 x + 20.9343$, where x is in billions. Substituting our token count of 0.000558382 billion yields a ratio of 20.9343.



### Parameters
- [https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
	- d_model - # expected features in encoder/decoder inputs
	- nhead - # heads in multiheadattention models
	- dim_feedforward
- # layers 


## Embeddings

## Skipping vs. keeping unknown tokens
- ChatGPT says skipping unrecognized text makes for a simpler model that is less likely to give nonsensical output
