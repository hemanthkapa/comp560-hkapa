Ever evolving notes for me to get a good understanding of transformers

## Basic building blocks of Transformers

### Attention
- A mechanism that lets a model look at all parts of its input and assign weights to them, showing how relavant each part is for producing a particular output.

### Multi-Head attention
- Multi-head attention is an attention mechanism where the model runs several attention operations ("heads") in parallel on same input, then concatenated and mixes their outputs. 
- Each head can focus on different relationships in the sequence, so together they let the model capture richer patterns than a single attention operation would. 

**My understanding in simple terms:** 
Multi-head attention is like having multiple "experts" look at the same text simultaneously, each focusing on different aspects. For example, when reading a sentence, one expert might focus on grammar relationships, another on semantic meaning, and another on positional context. The function:
1. **Splits** the input into multiple parallel attention "heads" (each head is like a different expert)
2. **Applies** attention independently to each head - each head learns to pay attention to different patterns or relationships
3. **Combines** all the heads' outputs back together, mixing their insights to create a richer, more nuanced understanding

### Feed-Forward networks
- Feed-forward networks are neural networks where information moves only in one direction: from the input layer, through one or more hidden layers to the output layers. 
- The `PositionWiseFeedForward` is a small two-layer feedforward neural network (Linear → ReLU → Linear) that is applied to each token’s vector separately but with the same weights for all positions. It takes the features produced by attention at each position, expands them to a higher dimension, applies a nonlinearity, then projects back to the model dimension, helping the model learn more complex transformations of each token’s representation.

### Postional Encoding
- Positional encoding is a way to add order information to token embeddings so a Transformer can know which token is first, second, etc., even though self‑attention itself is order‑agnostic.
- Here, a fixed matrix of sine and cosine waves at different frequencies is precomputed for positions 0…max_seq_length and dimensions 0…d_model−1, then the corresponding row is simply added to each token embedding, giving each position a unique, smoothly varying pattern the model can learn to interpret as position

## Vocab
- Softmax: a fucntion that turns a list of real numbers into a list of probabilities that are all positive and sum to 1.

Given scores $z_1, z_2, \ldots, z_n$, softmax outputs:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}$$

- so larger scores get higher probability but every class gets some non-negative share.

