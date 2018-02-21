import numpy as np
import scipy

import tensorflow as tf 
from tensorflow import glorot_uniform_initializer

from sklearn.manifold import TSNE

from matplotlib import pyplot as plt
plt.style.use('ggplot')

from utils import load_data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

print(y_test.shape)
print("Total number of nodes: {}".format(adj.shape[0]))
print("Training nodes: {}".format(len(np.argwhere(y_train == True))))
print("Validation nodes: {}".format(len(np.argwhere(y_val == True))))
print("Test nodes: {}".format(len(np.argwhere(y_test == True))))

num_nodes = adj.shape[0]
num_features = features.shape[1]
num_classes = y_train.shape[1]

# Hyperparameters
num_epochs = 200
hidden_units = 16
initializer = glorot_uniform_initializer
regularization_weight = 5e-4
dropout = 0.5
print_info_every = 20

# preprocessing, compute A
I = scipy.sparse.identity(num_nodes, format='csr')
A_tilde = adj + I
degrees = np.array(np.sum(A_tilde, axis=1))
degrees_inv_sqrt = np.power(degrees, -0.5).flatten()
D_inv_sqrt = scipy.sparse.diags(degrees_inv_sqrt)
A_hat = D_inv_sqrt.dot(A_tilde).dot(D_inv_sqrt)
A_hat = tf.constant(np.array(A_hat.todense()), dtype=tf.float32, name='A_hat')

#TODO normalize features?
def normalize_features(features):
    """row-normalize features"""
    row_sum = np.array(np.sum(features, axis=1))
    r_inv = np.power(row_sum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = scipy.sparse.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

features = normalize_features(features)
# Create GCN
X = tf.placeholder(dtype=tf.float32, shape=[None, num_features])
y = tf.placeholder(dtype=tf.int32, shape=[None, num_classes])
mask = tf.placeholder(dtype=tf.bool, shape=[num_nodes])

dropout_rate = tf.placeholder_with_default(0.0, shape=(), name="dropout")

W_0 = tf.get_variable("W_0", [num_features, hidden_units], dtype=tf.float32, initializer=initializer())
W_1 = tf.get_variable("W_1", shape=[hidden_units, num_classes], dtype=tf.float32, initializer=initializer())

X_dropped = tf.nn.dropout(X, keep_prob = 1-dropout_rate)

Z_0 = tf.matmul(X_dropped, W_0)
Z_0 = tf.matmul(A_hat, Z_0)
Z_0 = tf.nn.relu(Z_0)


Z_0_dropped = tf.nn.dropout(Z_0, keep_prob = 1-dropout_rate)
logits = tf.matmul(tf.matmul(A_hat, Z_0_dropped), W_1, name="logits")
Z = tf.nn.softmax(logits=logits, name="probabilities")

reg_loss = tf.nn.l2_loss(W_0)
sup_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(logits, mask), labels=tf.boolean_mask(y, mask)))
loss = sup_loss + regularization_weight * reg_loss

predictions = tf.boolean_mask(tf.argmax(Z, axis=-1), mask)
labels = tf.boolean_mask(tf.argmax(y, axis=-1), mask)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,  labels), dtype=tf.float32))
accuracy2 = tf.metrics.accuracy(labels=tf.argmax(tf.boolean_mask(y,mask), axis=-1), predictions=tf.boolean_mask(predictions, mask))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)


with tf.Session() as sess:


    # compute mean accuracy over 10 epochs
    test_accuracies = []
    val_accuracies = []

    for i in range(1):
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            _, loss_, accuracy_ = sess.run([optimizer, loss, accuracy],
                    feed_dict = {X: np.array(features.todense()), y:y_train, mask: train_mask, dropout_rate: dropout})

            # if epoch % 1000 == 0:
            #     print("Epoch:{:3d}\tLoss: {:.5f}\tAcc: {:.5f}".format(epoch, loss_, accuracy_))
            #
            #     val_accuracy_ = sess.run(accuracy,
            #             feed_dict = {X: np.array(features.todense()), y:y_val, mask: val_mask})
            #     print("Validation accuracy: {:.5f}".format(val_accuracy_))

        val_accuracy_ = sess.run(accuracy, feed_dict={X: np.array(features.todense()), y: y_val, mask: val_mask})
        print("Val accuracy: {:.5f}".format(val_accuracy_))
        val_accuracies.append(val_accuracy_)

        test_accuracy_, predictions_ = sess.run([accuracy, predictions],
                feed_dict = {X: np.array(features.todense()), y:y_test, mask: test_mask})
        print("Test accuracy: {:.5f}".format(test_accuracy_))

        test_accuracies.append((test_accuracy_))

    print(val_accuracies)
    print("Average val: {}".format(sum(val_accuracies) / len(val_accuracies)))

    print(test_accuracies)
    print("Average test: {}".format(sum(test_accuracies)/len(test_accuracies)))

    tsne = TSNE()

    embeddings_ = sess.run(Z_0, feed_dict={
        X: np.array(features.todense()),
        y: y_train,
        mask: train_mask
    })

    print(embeddings_.shape)

    low_space_X = tsne.fit_transform(embeddings_, None)
    # low_space_X = embeddings_[:, :2]
    print("Computed reduced dimensionality vectors: {}".format(low_space_X.shape))


    def plot_scatter(axes, mask, y, name):
        Xs = low_space_X[mask][:, 0]
        ys = low_space_X[mask][:, 1]
        colors = np.argmax(y[mask], axis=-1)
        axes.set_xticks([])
        axes.set_yticks([])
        axes.scatter(Xs, ys, c=colors, linewidths=0.5)


    fig = plt.figure(figsize=(16, 4))
    # fig.suptitle("t-SNE visualisation of the hidden layer activations of the GCN")

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title("Training nodes")
    plot_scatter(ax1, train_mask, y_train, "train")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title("Validation nodes")
    plot_scatter(ax2, val_mask, y_val, "val")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title("Test nodes")
    plot_scatter(ax3, test_mask, y_test, "test")
    plt.savefig("./figs/gcn/visualisation_tsne.png", bbox_inches='tight')
    plt.show()

    # tsne = TSNE()
    # low_space_X = tsne.fit_transform(embeddings, None)
    #
    # def plot_scatter(mask, y, name):
    #     Xs = low_space_X[mask][:, 0]
    #     ys = low_space_X[mask][:, 1]
    #     colors = np.argmax(y[mask], axis=-1)
    #     plt.figure()
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.scatter(Xs, ys, c=colors, linewidths=0.5)
    #     plt.title(name)
    #     plt.savefig("./figs/gcn/{}_tsne.png".format(name), bbox_inches='tight')
    #
    # plot_scatter(train_mask, y_train, "train")
    # plot_scatter(val_mask, y_val, "val")
    # plot_scatter(test_mask, y_test, "test")
    # plt.show()

