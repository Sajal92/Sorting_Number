def update_parameters(parameters, gradients, lr, batch_size, a):

    parameters["Wax"] = parameters["Wax"] - (lr / batch_size) * gradients["dWax"]
    parameters["Waa"] = parameters["Waa"] - (lr / batch_size) * gradients["dWaa"]
    parameters["ba"] = parameters["ba"] - (lr / batch_size) * gradients["dba"]

    if a == "d": # these parameters updated only for decoder(d)
        parameters["Wya"] = parameters["Wya"] - (lr / batch_size) * gradients["dWya"]
        parameters["by"] = parameters["by"] -(lr / batch_size) * gradients["dby"]

    return parameters