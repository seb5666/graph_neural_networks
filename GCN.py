import numpy as np
import scipy

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

from utils import load_data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

print(adj.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)
print(train_mask.shape)

print("Training nodes: {}".format(len(np.argwhere(y_train == True))))
print("Validation nodes: {}".format(len(np.argwhere(y_val == True))))
print("Test nodes: {}".format(len(np.argwhere(y_test == True))))

num_nodes = adj.shape[0]
num_features = features.shape[1]
num_classes = y_train.shape[1]

# Hyperparameters
num_epochs = 50
hidden_units = 1024

# preprocessing, compute A
I = scipy.sparse.identity(num_nodes, format='csr')
A_tilde = adj + I
degrees = np.array(np.sum(A_tilde, axis=1))
degrees_inv = np.power(degrees, -1).flatten()
D_inv = scipy.sparse.diags(degrees_inv)
A_hat = D_inv.multiply(A_tilde).multiply(D_inv)
A_hat = tf.constant(np.array(A_hat.todense()), dtype=tf.float32, name='A_hat')

#TODO normalize features?

# Create GCN
X = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes])
mask = tf.placeholder(dtype=tf.bool, shape=[num_nodes])

W_0 = tf.get_variable("W_0", [num_features, hidden_units], dtype=tf.float32, initializer=xavier_initializer())

Z_0 = tf.matmul(X, W_0)
Z_0 = tf.matmul(A_hat, Z_0)
Z_0 = tf.nn.relu(Z_0)

W_1 = tf.get_variable("W_1", shape=[hidden_units, num_classes], dtype=tf.float32, initializer=xavier_initializer())

logits = tf.matmul(tf.matmul(A_hat, Z_0), W_1, name="logits")
Z = tf.nn.softmax(logits=logits, name="probabilities")

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(logits, mask), labels=tf.boolean_mask(y, mask)))

predictions = tf.argmax(Z, axis=-1)

accuracy = tf.metrics.accuracy(labels=tf.argmax(tf.boolean_mask(y,mask), axis=-1), predictions=tf.boolean_mask(predictions, mask))
optimizer = tf.train.AdamOptimizer().minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(num_epochs):
        _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy], 
                feed_dict = {X: np.array(features.todense()), y:y_train, mask: train_mask})
        print("Epoch:{:3d}\tLoss: {:.5f}\tAcc: {:.3f}".format(epoch, loss_, accuracy_[0]))

        val_accuracy_ = sess.run(accuracy, 
                feed_dict = {X: np.array(features.todense()), y:y_val, mask: val_mask})
        print("Validation accuracy: {:.3f}".format(val_accuracy_[0]))


    test_accuracy_ = sess.run(accuracy, 
            feed_dict = {X: np.array(features.todense()), y:y_test, mask: test_mask})
    print("Validation accuracy: {:.3f}".format(test_accuracy_[0]))

