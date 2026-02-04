Ever evolving notes for me to get a good understanding of transformers

## Basic building blocks of Transformers

### Attention
- A mechanism that lets a model look at all parts of its input and assign weights to them, showing how relavant each part is for producing a particular output.

### Multi-Head attention
- Multi-head attention is an attention mechanism where the model runs several attention operations ("heads") in parallel on same input, then concatenated and mixes their outputs. 
- Each head can focus on different relationships in the sequence, so together they let the model capture richer patterns than a single attention operation would. 


## Vocab
- Softmax: a fucntion that turns a list of real numbers into a list of probabilities that are all positive and sum to 1.

Given scores $z_1, z_2, \ldots, z_n$, softmax outputs:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

- so larger scores get higher probability but every class gets some non-negative share.

