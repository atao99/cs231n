from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = np.dot(X[i],W)
        # print(scores)
        shift_scores = scores - np.max(scores)
        # print(shift_scores)
        for j in range(num_classes):
            soft_max = np.exp(shift_scores[j]) / np.sum(np.exp(shift_scores))
            if j==y[i]:
                loss_i = -np.log(soft_max)
                loss += loss_i
                # print(loss_i,loss)
                dW[:,y[i]] -= X[i]
                dW[:,j] += soft_max*X[i]
            else:
                dW[:,j] += soft_max*X[i]
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW = dW/num_train + 2*reg*W
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = np.dot(X,W)
    shift_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
    soft_max = shift_scores / np.sum(shift_scores,axis=1,keepdims=True)
    loss = -np.log(soft_max[range(num_train),y])
    loss = np.sum(loss) / num_train + reg * np.sum(W * W)
    coef = soft_max.copy()
    coef[range(num_train),y] -= 1
    dW = np.dot(X.T,coef)
    dW = dW/num_train + 2*reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
