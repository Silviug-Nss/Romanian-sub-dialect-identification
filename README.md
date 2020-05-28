# Romanian-sub-dialect-identification
Discriminate between the Moldavian and the Romanian dialects across different text genres (news versus tweets)

  This program implements a RNN using one Embedding layer and 2 stacked
Bidirectional LSTM, ending with one Dense layer in order to get the prediction.

Why this architecture?

  - the reason of choosing a NN is that the size of input in large enough to
  use it for training the neural network
  - the reason of choosing RNN instead of Dense layer is that a Dense layer
  is weak at learning the relationships between words
  - the reason of choosing LSTM instead of GRU is that it has a better
  performance, although the training process lasts longer.
  - the reason of choosing Bidirectional instead of simple LSTM is that texts
  are learned better this way, compared to temperature sequences, where a
  simple RNN is prefered.

Why these hyperparameters?

  The values of hyperparameters where chosen this way because they showed the
best performance during the tuning process, meaning that it showed a good
precision and it finished its learning process fast enough.
  It was possible to choose a smaller batch size, and larger embedding
dimension, number of words in a sample, number of words in the Embedding
dictionary and more weights to learn for these 2 LSTM layers, or even more
layers. Still, because the running time of the program is a very important
factor to consider, I chose "the best trade-off".

Fighting overfitting:

  In order to fight overfitting, I used dropout and recurrent dropout for the
first LSTM layer.

Input preprocessing:

  The input of the model is not the raw data. Before feeding it to the NN,
I considered to be a good idea to remore all the %NE% tokens, because they
are not offering any useful information.
  Also, all the labels are translated from 1-2 to 0-1, in order to have a
0-1 binary classification.
  Because samples and labels files have the same IDs, I use the given IDs
only for the output.

Output:

  After computing the 0-1 classified values, I add one in order to translate
the labels back to 1-2 values. In the output file, these labels are paired with
the appropriate ID.

Can this be improved?

  Probably yes. If I had more computing power, then I would change the
hyperparameters as stated above.
