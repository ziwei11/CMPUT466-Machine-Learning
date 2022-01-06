# -*- coding: utf-8 -*-


import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

def readMNISTdata():

    with open('t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels




def softmax(z):
    # z:10000*10
    for i in range(len(z)):
      zi_max = z[i][0]
      for j in range(len(z[i])):
        if z[i][j] > zi_max:
          zi_max = z[i][j]
      for j in range(len(z[i])):
        z[i][j] = z[i][j] - zi_max

    array_exp = np.zeros((z.shape[0], z.shape[1]))    
    for i in range(len(z)):
      i_sum = 0
      for j in range(len(z[i])):
        i_sum = i_sum + np.exp(z[i][j])
      for j in range(len(z[i])):
        array_exp[i][j] = np.exp(z[i][j]) / i_sum
    return array_exp


def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    # TODO Your code here
    y = np.dot(X, W)
    z = softmax(y)    
    t_hat = np.argmax(z, axis=1)
    y_hot = np.zeros((len(t), 10))
    # Calculate Y hot
    for i in range(len(y)):
      y_hot[i, t[i]] = 1
    loss = -(1 / X.shape[0]) * np.sum(y_hot * np.log(z))
    t_hat_new = t_hat.reshape(-1,1)
    # Calculate accuracy
    count = 0
    for i in range(len(t)):
      if t_hat_new[i] == t[i]:
        count = count + 1
    acc = count / len(t)
    return y, t_hat, loss, acc


def train(X_train, y_train, X_val, t_val, X_test, t_test):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
    #TODO Your code here
    # Initial val_best, losses_train, validation_performance, epoch
    # w, b, w_best, b_best
    val_best = -100
    losses_train = []
    validation_performance = []
    epoch = 0
    w = np.random.random((X_train.shape[1], 10))
    b = np.random.random(10)
    w_best = None
    b_best = None
    for epoch in range(MaxEpoch):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size)) ):
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]
            y = np.dot(X_batch, w) + b
            z = softmax(y)
            # Calculate Y hot
            y_hot = np.zeros((len(y_batch), 10))
            for i in range(len(y)):
                y_hot[i, y_batch[i]] = 1
            # Update w, b, loss_this_epoch
            w = w - alpha * (1 / X_batch.shape[0]) * np.dot(X_batch.T, (z - y_hot))
            b = b - alpha * (1 / X_batch.shape[0]) * np.sum(z - y_hot)
            loss_this_epoch = loss_this_epoch - (1 / X_batch.shape[0]) * np.sum(y_hot * np.log(z))

        _, _, _, val_acc = predict(X_val, w, t_val)
        # Save loss_this_epoch/int(np.ceil(N_train/batch_size))
        losses_train.append(loss_this_epoch/int(np.ceil(N_train/batch_size)))
        validation_performance.append(val_acc)
        
        if val_best < val_acc:
            val_best = val_acc
            epoch_best = epoch
            w_best = w
            b_best = b 
                       
    return w_best, b_best, epoch_best, val_best, losses_train, validation_performance



##############################
#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()
X_test = X_test / 256
print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 10
alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay

w_best, b_best, epoch_best, acc_best, loss_train, val_acc = train(X_train, t_train, X_val, t_val, X_test, t_test)
_, _, _, accuracy_test = predict(X_test, w_best, t_test)

print('At epoch', epoch_best, 'val: ', acc_best, 'test:', accuracy_test)

# Plot Training Loss / Number of Epoch
plt.plot(np.arange(1, 51), loss_train)
plt.xlabel("Number of Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss / Number of Epoch")
plt.savefig("curve1.jpg")
plt.cla()

# Plot Validation Accuracy / Number of Epoch
plt.plot(np.arange(1, 51), val_acc)
plt.xlabel("Number of Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy / Number of Epoch")
plt.savefig("curve2.jpg")