## Scaling Laws

### How many parameters?
We have 558,382 training tokens available.

- Chinchilla: "for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled" (1).
- Kaplan scaling (from OpenAI in 2020) suggests 1.7 tokens per parameter (source?)

Chinchilla authors list tokens vs. parameters for fixed FLOP budgets (8). The ratios of tokens to parameters are 20, 20.2, 20.51, 22.388, 21.1428, 21.0714, 21.1538, 21.2, and 21.62. The mean is approximately 21. Fitting a linear regression of tokens vs. ratios yields $3.37687×10^-6 x + 20.9343$, where x is in billions; substituting our token count of 0.000558382 billion yields a ratio of 20.9343. Taking the mean and doing a linear regression are perhaps totally theoretically invalid, but who does theory in DL anyway, and the point is that they agree on a ratio of 21 tokens to parameters.

558,382/21 = **26,590 parameters for our model** by Chinchilla scaling

558,382/1.7 = **328,460 parameters for our model** by Kaplan scaling

Let's try both.

### GPT-2 Notes
- According to [Wikipedia](https://en.wikipedia.org/wiki/GPT-2#/media/File:Full_GPT_architecture.png), GPT-2 consists of
	- input embedding
	- positional encoding
	- transformer blocks (layers)
	- layer norm
	- linear
	- softmax
- According to various internet sources, GPT-2 is decoder-only. But the source code has an [encoder file](https://github.com/openai/gpt-2/tree/master/src).

### Parameter distribution for our model
- [https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
	- d_model - # expected features in encoder/decoder inputs
	- nhead - # heads in multiheadattention models
	- dim_feedforward
- Additional parameters
	- n_layers
- [How to estimate the number of parameters in a transformer](https://towardsdatascience.com/how-to-estimate-the-number-of-parameters-in-transformer-models-ca0f57d8dff0)
	- For a transformer encoder: $4d_{model}^{2}+ 2d_{model}d_{ff} + 9d_{model} + d_{ff}$ 
	- For a transformer decoder: $8d_{model}^{2}+ 2d_{model}d_{ff} + 15d_{model} + d_{ff}$ 
	- For multi-head attention: $4(d_{model}^{2}+ d_{model})$
	- For feed-forward: $2d_{model}d_{ff} + d_{model} + d_{ff}$
	- For layer norm: $2d_{model}$
- So we'll definitely have:
	- input embedding: none
	- positional encoding: none
	- transformer layers: $n_{layers}$ x ==parameters per transformer (encoder or decoder?)==
	- layer norm: $2d_{model}$
	- linear: $2d_{model}d_{ff} + d_{model} + d_{ff}$ ==wait, what about dim_feedforward?==
	- softmax: none


### Chinchilla notes

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

## Embeddings

## Skipping vs. keeping unknown tokens
- ChatGPT says skipping unrecognized text makes for a simpler model that is less likely to give nonsensical output
