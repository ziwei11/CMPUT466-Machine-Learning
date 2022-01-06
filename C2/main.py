from utils import plot_data, generate_data
import numpy as np
"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    alpha = 0.001
    w = np.zeros(X.shape[1])
    b = 0
    for i in range(8000):
        h = sigmoid(np.dot(w.T, X.T) + b)
        w = w - alpha * (np.dot(X.T, (h - t)))
        b = b - alpha * (np.sum(h - t))
    return w, b

def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    y = sigmoid(np.dot(w.T, X.T) + b)
    t = np.zeros(X.shape[0])
    for i in range(len(y)):
      if y[i] >= 0.5:
        t[i] = 1
    return t

def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    n_X = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
    # Closed form solution by matrix-vector representations only
    X_T = np.transpose(n_X)
    # X^T * X
    temp = np.dot(X_T, n_X)
    # (X^T * X)^(-1)
    temp = np.linalg.inv(temp)
    # (X^T * X)^(-1) * X^T
    temp = np.dot(temp, X_T)
    # (X^T * X)^(-1) * X^T * t
    w = np.dot(temp, t)
    b = w[X.shape[1] + 1 - 1]
    w = w[0:(X.shape[1] + 1 - 1)]
    return w, b

def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    n_X = np.concatenate([X, np.ones([X.shape[0], 1])], axis=1)
    w = np.append(w, b)
    y = np.dot(w.T, n_X.T)
    t = np.zeros(X.shape[0])
    for i in range(len(y)):
      if y[i] >= 0.5:
        t[i] = 1
    return t

def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    count = 0
    for i in range(len(t)):
      if t[i] == t_hat[i]:
        count += 1
    acc = count / len(t)
    return acc

def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False, figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True, figure_name='dataset_B_logistic.png')

main()
