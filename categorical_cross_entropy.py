
def categorical_cross_entropy(y_pred, y):
    x = np.multiply(y, np.log(y_pred))
    loss = x.sum()

    return loss
