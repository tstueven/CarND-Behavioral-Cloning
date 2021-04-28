def high_value_emphasizing_loss(y_true, y_pred):
    weighted_squared_difference = (y_true - y_pred)**2 * (1 + 100*np.abs(y_true)) # what if true = 0?
    return weighted_squared_difference