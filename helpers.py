# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x, mean_x, std_x



def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        err = y-tx.dot(w)
        loss = 1 / 2 * np.mean(err ** 2)
        gradient = - 1/len(err) * tx.T.dot(err)
        w = w - gamma * gradient

        # store w and loss    
        ws.append(w)
        losses.append(loss)
        print("GD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size = 1
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            err = y_batch-tx_batch.dot(w)
            loss = 1 / 2 * np.mean(err ** 2)
            gradient = - 1/len(err) * tx_batch.T.dot(err)
            w = w - gamma * gradient

            # store w and loss    
            ws.append(w)
            losses.append(loss)
            print("SGD iter. {bi}/{ti}: loss={l}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws

def least_squares(y, tx):
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y-tx.dot(w)
    loss = 1 / ( 2 * y.shape[0] ) * err.T.dot(err)
    return w, loss


def ridge_regression(y, tx, lambda_):
    a = tx.T.dot(tx) + 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    err = y-tx.dot(w)
    loss = 1 / ( 2 * y.shape[0] ) * err.T.dot(err)
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    ws = [initial_w]
    losses = []
    for n_iter in range(max_iters):
        pred = sigmoid(tx.dot(w))
        loss = - 1 / y.shape[0] * ( y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)) )
        grad = tx.T.dot(pred - y) * (1 / y.shape[0])
        w -= gamma * grad
        
        # store w and loss    
        ws.append(w)
        losses.append(loss)

        print("GD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws


def reg_logistic_regression(y, tx, lambda_,initial_w, max_iters, gamma):
    w = initial_w
    ws = [initial_w]
    losses = []
    for n_iter in range(max_iters):
        pred = sigmoid(tx.dot(w))
        loss = - 1 / y.shape[0] * ( y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)) ) + lambda_ * w.T.dot(w)
        grad = tx.T.dot(pred - y) * (1 / y.shape[0]) + 2 * lambda_ * w
        w -= gamma * grad
        
        # store w and loss    
        ws.append(w)
        losses.append(loss)
        
        print("GD iter. {bi}/{ti}: loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss))
    return losses, ws

def sigmoid(t):
    return 1.0 / (1 + np.exp(-t))

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]





# def build_model_data(height, weight):
#     """Form (y,tX) to get regression data in matrix form."""
#     y = weight
#     x = height
#     num_samples = len(y)
#     tx = np.c_[np.ones(num_samples), x] #np-c_ is concatenate column-wise. "np.ones" is just to add an intercept, and the orignial x data
#     # Notice that x is normally a DxN matrix! (D: number of features, N: number of samples) Therefore our concatanation make t transposed.
#     return y, tx


# def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
#     """
#     Generate a minibatch iterator for a dataset.
#     Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
#     Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
#     Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
#     Example of use :
#     for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
#         <DO-SOMETHING>
#     """
#     data_size = len(y)

#     if shuffle:
#         shuffle_indices = np.random.permutation(np.arange(data_size))
#         shuffled_y = y[shuffle_indices]
#         shuffled_tx = tx[shuffle_indices]
#     else:
#         shuffled_y = y
#         shuffled_tx = tx
#     for batch_num in range(num_batches):
#         start_index = batch_num * batch_size
#         end_index = min((batch_num + 1) * batch_size, data_size)
#         if start_index != end_index:
#             yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
