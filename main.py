#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

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


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


#Function to implement a single forward step of the RNN-cell
def rnn_cell_forward(xt, a_prev, parameters):
    """

    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (max_no, batch_size).
    a_prev -- Hidden state at timestep "t-1", numpy array of shape (n_a, batch_size)
    parameters -- python dictionary containing:
        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, max_no)
        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (max_no, n_a)
        ba --  Bias, numpy array of shape (n_a, 1)
        by -- Bias relating the hidden-state to the output, numpy array of shape (max_no, 1)
    Returns:
    a_next -- next hidden state, of shape (n_a, batch_size)
    yt_pred -- prediction at timestep "t", numpy array of shape (max_no, m)
    cache -- tuple of values needed for the backward pass, contains (a_next, a_prev, xt, parameters)
    """

    # Retrieve parameters from "parameters"
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute next activation state
    a_next = np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba)
    # compute output of the current cell
    yt_pred = softmax(np.dot(Wya, a_next) + by)

    # store values you need for backward propagation in cache
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache

# Function to implement unrolled RNN for encoder
def rnn_forward_encoder(x, a0, parameters):

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize "a" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0

    # loop over all time-steps/sequence length
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the prediction in y
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches"
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def rnn_forward_decoder(x, a0, parameters):

    # Initialize "caches" which will contain the list of all caches
    caches = []

    # Retrieve dimensions from shapes of x and Wy
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # initialize "a" and "y" with zeros
    a = np.zeros((n_a, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))

    # Initialize a_next
    a_next = a0
    yt_pred = x[:, :, 0]

    # loop over all time-steps
    for t in range(T_x):
        # Update next hidden state, compute the prediction, get the cache
        x[:, :, t] = yt_pred
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)
        # Save the value of the new "next" hidden state in a
        a[:, :, t] = a_next
        # Save the value of the prediction in y
        y_pred[:, :, t] = yt_pred
        # Append "cache" to "caches"
        caches.append(cache)

    # store values needed for backward propagation in cache
    caches = (caches, x)

    return a, y_pred, caches


def categorical_cross_entropy(y_pred, y):

    x = np.multiply(y, np.log(y_pred))
    loss = x.sum()

    return loss


# Function to implement the backward pass for the decoder RNN-cell (single time-step).
def rnn_cell_backward_deocder(da_next, cache, y_pred, y, dx_prev):
    """

    Arguments:
    da_next -- Gradient of loss with respect to next hidden state
    cache -- python dictionary containing useful values (output of rnn_cell_forward())

    Returns:
    gradients -- python dictionary containing:
        dx -- Gradients of input data, of shape (max_no, batch_size)
        da_prev -- Gradients of previous hidden state, of shape (n_a, batch_size)
        dWax -- Gradients of input-to-hidden weights, of shape (n_a, max_no)
        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
        dba -- Gradients of bias vector, of shape (n_a, 1)
    """

    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute the gradient of tanh with respect to a_next
    da_next += np.dot(Wya.T, (y_pred - y))

    dtanh = (1 - a_next ** 2) * da_next

    # compute the gradient of the loss with respect to Wax
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to Waa
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # compute the gradient with respect to ba
    dba = np.sum(dtanh, axis=1, keepdims=1)

    # compute the gradient with respect to Wya and ba
    dWya = np.dot((y_pred - y), a_next.T)
    dby = np.sum((y_pred - y), axis=1, keepdims=1)

    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba, "dWya": dWya, "dby": dby}

    return gradients


# Function to implement the complete backward pass for the unrolled decoder RNN-cell.
def rnn_backward_decoder(da_next, caches, y_pred, y):

    # Retrieve values from the first cache (t=1) of caches
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, _ = da_next.shape
    n_x, m = x1.shape
    n_y, m, T_x = y.shape

    # initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dWya = np.zeros((n_y, n_a))
    dba = np.zeros((n_a, 1))
    dby = np.zeros((n_y, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = da_next
    dxt = np.zeros((n_y, m))


    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t.
        gradients = rnn_cell_backward_deocder(da_prevt, caches[t], y_pred[:, :, t], y[:, :, t], dxt)
        # Retrieve derivatives from gradients
        dxt, da_prevt, dWaxt, dWaat, dbat, dWyat, dbyt = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients[
            "dWaa"], gradients["dba"], gradients["dWya"], gradients["dby"]
        # Incremented global derivatives w.r.t parameters by adding their derivative at time-step t
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat
        dWya += dWyat
        dby += dbyt

    # Set da0 to the gradient of a which has been backpropagated through all time-steps
    da0 = da_prevt


    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba, "dWya": dWya, "dby": dby}

    return gradients


# Function to implement the backward pass for the encoder RNN-cell (single time-step).
def rnn_cell_backward_encoder(da_next, cache):

    # Retrieve values from cache
    (a_next, a_prev, xt, parameters) = cache

    # Retrieve values from parameters
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # compute the gradient of tanh with respect to a_next
    dtanh = (1 - a_next ** 2) * da_next

    # compute the gradient of the loss with respect to Wax
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # compute the gradient with respect to Waa
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # compute the gradient with respect to b
    dba = np.sum(dtanh, axis=1, keepdims=1)

    # Store the gradients in a python dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients

# Function to implement the complete backward pass for the encoder RNN-cell.
def rnn_backward_encoder(da_next, caches):

    # Retrieve values from the first cache (t=1) of caches
    (caches, x) = caches
    (a1, a0, x1, parameters) = caches[0]

    # Retrieve dimensions from da's and x1's shapes
    n_a, m = da_next.shape
    n_x, m, T_x = x.shape

    # initialize the gradients with the right sizes
    dx = np.zeros((n_x, m, T_x))
    dWax = np.zeros((n_a, n_x))
    dWaa = np.zeros((n_a, n_a))
    dba = np.zeros((n_a, 1))
    da0 = np.zeros((n_a, m))
    da_prevt = da_next

    # Loop through all the time steps
    for t in reversed(range(T_x)):
        # Compute gradients at time step t.
        gradients = rnn_cell_backward_encoder(da_prevt, caches[t])
        # Retrieve derivatives from gradients
        dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
        # Incremented global derivatives w.r.t parameters by adding their derivative at time-step t
        dx[:, :, t] = dxt
        dWax += dWaxt
        dWaa += dWaat
        dba += dbat

    # Set da0 to the gradient of a which has been backpropagated through all time-steps
    da0 = da_prevt

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa ,"dba": dba}

    return gradients

def model_predict(x, a0_encoder, parameters_encoder, parameters_decoder, max_no, batch_size, seq_len):

    # encoder forward pass
    a_encoder, y_pred_encoder, caches_encoder = rnn_forward_encoder(x, a0_encoder, parameters_encoder)

    a0_decoder = a_encoder[:, :, seq_len - 1]  # last a0 of encoder as an input a0 to decoder
    x_decoder = np.zeros((max_no, batch_size, seq_len))
    x_decoder[:, :, 0] = y_pred_encoder[:, :, seq_len - 1]  # last prediction of encoder as an input x to decoder

    # decoder forward pass
    a_decoder, y_pred_decoder, caches_decoder = rnn_forward_decoder(x_decoder, a0_decoder, parameters_decoder)

    return a_encoder, y_pred_encoder, caches_encoder, a_decoder, y_pred_decoder, caches_decoder


def model_backprop(da_next, caches_decoder, caches_encoder, y_pred_decoder, y):

    # backward pass of decoder including backward pass of loss function
    gradients_decoder = rnn_backward_decoder(da_next, caches_decoder, y_pred_decoder, y)

    # backward pass of encoder
    da_next = gradients_decoder["da0"]
    gradients_encoder = rnn_backward_encoder(da_next, caches_encoder)

    return gradients_encoder, gradients_decoder

def update_parameters(parameters, gradients, lr, batch_size, a):

    parameters["Wax"] = parameters["Wax"] - (lr / batch_size) * gradients["dWax"]
    parameters["Waa"] = parameters["Waa"] - (lr / batch_size) * gradients["dWaa"]
    parameters["ba"] = parameters["ba"] - (lr / batch_size) * gradients["dba"]

    if a == "d": # these parameters updated only for decoder(d)
        parameters["Wya"] = parameters["Wya"] - (lr / batch_size) * gradients["dWya"]
        parameters["by"] = parameters["by"] -(lr / batch_size) * gradients["dby"]

    return parameters

def model_test(test, a0_encoder, parameters_encoder, parameters_decoder, max_no, batch_size, seq_len):

    a0_encoder = np.mean(a0_encoder, axis=1, keepdims=True)

    a_encoder, y_pred_encoder, caches_encoder = rnn_forward_encoder(test, a0_encoder, parameters_encoder)

    a0_decoder = a_encoder[:, :, seq_len - 1]
    x_decoder = np.zeros((max_no, batch_size, seq_len))

    a_decoder, y_pred_decoder, caches_decoder = rnn_forward_decoder(x_decoder, a0_decoder, parameters_decoder)

    return y_pred_decoder

if __name__ == "__main__":

    # Initialization

    batch_size = 32
    seq_len = 10        #length of sequence
    max_no = 100        #max number in a sequence
    lr = 0.01           #learning rate
    max_iter = 3000     #maximum iterations
    n_a = 80            #number of parameters in a branch of RNN

    a0_encoder = np.random.normal(0, 1, (n_a, batch_size))    #initial entry of a0 in encoder first cell

    # initialization of parameters of encoder and decoder network between 0 and 1
    parameters_encoder = {"Wax": np.random.normal(0,1,(n_a, max_no)), "Waa": np.random.normal(0,1,(n_a, n_a)),
                          "Wya": np.random.normal(0,1,(max_no, n_a)),"ba": np.random.normal(0,1,(n_a, 1)), "by": np.random.normal(0,1,(max_no, 1))}

    parameters_decoder = {"Wax": np.random.normal(0, 1, (n_a, max_no)), "Waa": np.random.normal(0, 1, (n_a, n_a)),
                          "Wya": np.random.normal(0, 1, (max_no, n_a)),"ba": np.random.normal(0, 1, (n_a, 1)), "by": np.random.normal(0, 1, (max_no, 1))}

    epsilon = 10e-3
    loss = epsilon + 1
    iteration = 0

    # Initialization of loss for each iteration
    loss_iterations = np.zeros((max_iter,1))
    iterations = np.zeros((max_iter,1))

    # Training Loop
    while loss > epsilon or iteration < max_iter:

        # dataset generation
        x, y = batch_gen(batch_size, seq_len, max_no)

        # encoder and decoder output of the model
        a_encoder, y_pred_encoder, caches_encoder, a_decoder, y_pred_decoder, caches_decoder = model_predict(x, a0_encoder, parameters_encoder, parameters_decoder, max_no, batch_size, seq_len)

        # retreive parameters from caches
        (cache_encoder, _) = caches_encoder
        (_, _, _, parameters_encoder) = cache_encoder[0]
        (cache_decoder, _) = caches_decoder
        (_, _, _, parameters_decoder) = cache_decoder[0]

        # Calculation of loss
        loss = categorical_cross_entropy(y_pred_decoder, y)
        loss = loss/batch_size

        # backward pass through network
        da_next = np.zeros((n_a, batch_size))
        gradients_encoder, gradients_decoder = model_backprop(da_next, caches_decoder, caches_encoder, y_pred_decoder, y)

        # updating parameters of encoder and decoder using gradient descent
        parameters_encoder = update_parameters(parameters_encoder, gradients_encoder, lr, batch_size,'e')
        parameters_decoder = update_parameters(parameters_decoder, gradients_decoder, lr, batch_size, 'd')

        # Appending loss for each iteration
        loss_iterations[iteration] = loss
        iterations[iteration] = iteration

        #print iteration number and loss for particular iteration
        print("Iteration number: " + str(iteration) + " Loss: " + str(np.absolute(loss)), end='\r')
        print()

        iteration += 1

    plt.plot( iterations, np.absolute(loss_iterations))
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("Loss_Vs_Iter.png")
    plt.close()

    # Testing the output of network on a test dataset
    batch_size = 1
    testX = np.random.randint(max_no, size=(1, seq_len))
    test = encode(testX, seq_len, max_no)
    print(testX)
    y_pred_decoder = model_test(test, a0_encoder, parameters_encoder, parameters_decoder, max_no, batch_size, seq_len)

    print("actual sorted output is")
    print(np.sort(testX))
    print("sorting done by RNN is")
    print(np.argmax(y_pred_decoder, axis=0))
    print("\n")

