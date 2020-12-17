#function to encode input sequence to one-hot encoding
def encode(X,seq_len, vocab_size):
    x = np.zeros((len(X),seq_len, vocab_size), dtype=np.float32)
    for ind,batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1
    x = np.reshape(x, (vocab_size, len(X),  seq_len))
    return x

#function to generate batch of input to RNN with given batch_size, seq_length and max no. upto which sequence can occur
def batch_gen(batch_size=32, seq_len=10, max_no=100):

    # Generates a batch of input
    x = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)
    y = np.zeros((batch_size, seq_len, max_no), dtype=np.float32)

    # Randomly generate a batch of integer sequences (X) and its sorted
    # counterpart (Y)
    X = np.random.randint(max_no, size=(batch_size, seq_len))
    Y = np.sort(X, axis=1)

    for ind, batch in enumerate(X):
        for j, elem in enumerate(batch):
            x[ind, j, elem] = 1

    for ind, batch in enumerate(Y):
        for j, elem in enumerate(batch):
            y[ind, j, elem] = 1

    x = np.reshape(x, (max_no, batch_size, seq_len))
    y = np.reshape(y, (max_no, batch_size, seq_len))

    return x, y

