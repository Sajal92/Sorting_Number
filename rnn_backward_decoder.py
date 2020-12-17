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
