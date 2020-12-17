def model_predict(x, a0_encoder, parameters_encoder, parameters_decoder, max_no, batch_size, seq_len):

    # encoder forward pass
    a_encoder, y_pred_encoder, caches_encoder = rnn_forward_encoder(x, a0_encoder, parameters_encoder)

    a0_decoder = a_encoder[:, :, seq_len - 1]  # last a0 of encoder as an input a0 to decoder
    x_decoder = np.zeros((max_no, batch_size, seq_len))
    x_decoder[:, :, 0] = y_pred_encoder[:, :, seq_len - 1]  # last prediction of encoder as an input x to decoder

    # decoder forward pass
    a_decoder, y_pred_decoder, caches_decoder = rnn_forward_decoder(x_decoder, a0_decoder, parameters_decoder)

    return a_encoder, y_pred_encoder, caches_encoder, a_decoder, y_pred_decoder, caches_decoder