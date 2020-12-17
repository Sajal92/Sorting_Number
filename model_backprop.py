def model_backprop(da_next, caches_decoder, caches_encoder, y_pred_decoder, y):

    # backward pass of decoder including backward pass of loss function
    gradients_decoder = rnn_backward_decoder(da_next, caches_decoder, y_pred_decoder, y)

    # backward pass of encoder
    da_next = gradients_decoder["da0"]
    gradients_encoder = rnn_backward_encoder(da_next, caches_encoder)

    return gradients_encoder, gradients_decoder