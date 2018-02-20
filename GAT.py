import numpy as np
import scipy

import tensorflow as tf 

from utils import load_data

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

np.set_printoptions(precision=2)

def GAT_layer(scope, features, A, F_out,  K=1, activation=tf.nn.relu, reduction="concat", dropout_p=0.5):

    N = features.get_shape()[0]
    F_in = features.get_shape()[1]

    with tf.variable_scope(scope):
        Ws = tf.get_variable("Ws", shape=[K, F_in, F_out], dtype=tf.float32, initializer=tf.glorot_normal_initializer())
        As = tf.get_variable("as", shape=[K, 2 * F_out], dtype=tf.float32, initializer=tf.glorot_normal_initializer())

        outputs = []
        for head in range(K):

            W = Ws[head]
            a = As[head]

            learned_features = tf.matmul(features, W)

            repeated = tf.reshape(tf.tile(learned_features, [1,N]), shape=[N, N, F_out])
            tiled = tf.reshape(tf.tile(learned_features, [N, 1]), shape=[N, N, F_out])
            combinations = tf.reshape(tf.concat([repeated, tiled], axis=-1), shape=[N, N, 2 * F_out])
            coefficients = tf.reshape(tf.tensordot(combinations, a, axes=1), shape=[N, N])

            transformed_coefficients = tf.nn.leaky_relu(coefficients, alpha=0.2)

            # find elements that are 0 in the adjacency matrix A
            comparison = tf.equal(A, tf.constant(0, dtype=tf.float32))
            # if element is 0, then add big negative number, as during softmax they will disappear (e^(-10e90 = 0)
            mask = tf.where(comparison,
                            tf.multiply(tf.cast(tf.ones_like(A), dtype=tf.float32), tf.constant(-10e9)),
                            tf.cast(tf.zeros_like(A), dtype=tf.float32))
            # ensure only the direct neighbours are taking into consideration
            masked = transformed_coefficients + mask
            softmax = tf.nn.softmax(masked)
            dropout = tf.nn.dropout(softmax, 1 - dropout_p)

            node_features = tf.matmul(dropout, learned_features)

            if reduction == "concat":
                node_features = activation(node_features)

            outputs.append(node_features)

        if reduction == "concat":
            output = tf.concat(outputs, axis=-1)
        else:
            output = tf.add_n(outputs) / len(outputs)
            if activation != None:
                output = activation(output)

        return output

def GAT_model(X, y, mask, dropout_p, l2_coef = 0.0005, lr=0.005):
    dropped_X = tf.nn.dropout(X, 1 - dropout_p)
    f1 = GAT_layer("f1", dropped_X, adj, K=8, F_out=8, activation=tf.nn.elu, reduction="concat", dropout_p = dropout_p)
    dropped_f1 = tf.nn.dropout(f1, 1 - dropout_p)
    logits = GAT_layer("f2", dropped_f1, adj, K=1, F_out=num_classes, activation=None, reduction="average", dropout_p=dropout_p)

    probabilities = tf.nn.softmax(logits)

    vars = tf.trainable_variables()
    L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_coef

    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(logits, mask), labels=tf.boolean_mask(y, mask)))

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss + L2_loss)

    predictions = tf.boolean_mask(tf.argmax(probabilities, axis=-1), mask)
    labels = tf.boolean_mask(tf.argmax(y, axis=-1), mask)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions,  labels), dtype=tf.float32))

    return probabilities, loss, accuracy, train_op, f1

if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

    # add selfconnections to adjacency matrix
    adj = adj + scipy.sparse.eye(adj.shape[0])
    adj = np.array(adj.todense(), dtype=np.float32)
    
    num_nodes = adj.shape[0]
    num_features = features.shape[1]
    num_classes = y_train.shape[1]

    print(y_train.shape)
    print(train_mask.shape)
    print("Total number of nodes: {}".format(adj.shape[0]))
    print("Training nodes: {}".format(len(np.argwhere(y_train == True))))
    print("Validation nodes: {}".format(len(np.argwhere(y_val == True))))
    print("Test nodes: {}".format(len(np.argwhere(y_test == True))))
    print("Num nodes: {}".format(num_nodes))
    print("num_features: {}".format(num_features))

    num_epochs = 100
    l2_coef = 0.0005
    learning_rate = 0.005

    # Create GAT
    X = tf.placeholder(shape=[num_nodes, num_features], name="X", dtype=tf.float32)
    labels = tf.placeholder(shape=[num_nodes, num_classes], name="labels", dtype=tf.float32)
    mask = tf.placeholder(shape=[num_nodes], name="mask", dtype=tf.bool)
    dropout_p = tf.placeholder_with_default(0.0, shape=())

    probabilities, loss, accuracy, train_op, f1 = GAT_model(X, labels, mask, dropout_p, l2_coef=l2_coef, lr=learning_rate)

    with tf.Session() as sess:
        features = np.array(features.todense(), dtype=np.float32)
        checkpoint_file = "./models/gat/model.ckpt"

        saver = tf.train.Saver()
        run_training = False

        if run_training:
            sess.run(tf.global_variables_initializer())
            # sess.run(tf.local_variables_initializer())

            print("Epoch\t\tLoss\t\tTrain acc\t\tVal acc")
            for epoch in range(num_epochs):
                probabilities_, loss_, train_accuracy_, train_op_ = sess.run([probabilities, loss, accuracy, train_op], { X: features, dropout_p: 0.6, labels: y_train, mask: train_mask })

                val_probabilities_, val_accuracy_ = sess.run([probabilities, accuracy], {X: features, labels: y_val, mask: val_mask})
                print("{}\t\t{:.3f}\t\t{:.3f}\t\t\t{:.3f}".format(epoch, loss_, train_accuracy_, val_accuracy_))

            test_probabilities, test_accuracy_ = sess.run([probabilities, accuracy],
                    feed_dict = {X: features, labels:y_test, mask: test_mask})
            print("Test accuracy: {:.5f}".format(test_accuracy_))

            save_path = saver.save(sess, "./models/gat/model.ckpt")

        else: # restore model
            saver.restore(sess, checkpoint_file)
            print("Model restored.")
            print(features.shape)
            print(y_val.shape)
            print(val_mask.shape)

            embeddings_ = sess.run(f1, {X: np.array(features), labels: np.array(y_train), mask: np.array(train_mask)})

            val_probabilities_, val_accuracy_ = sess.run([probabilities, accuracy],
                                                         {X: features, labels: y_val, mask: val_mask})
            print("val acc: {}".format(val_accuracy_))
            test_probabilities_, test_accuracy_ = sess.run([probabilities, accuracy],
                                                           {X: features, labels: y_test, mask: test_mask})
            print("test acc: {}".format(test_accuracy_))

            tsne = TSNE()

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
            fig.suptitle("t-SNE visualisation of 1st layer embeddings")
            
            ax1 = fig.add_subplot(1,3,1)
            ax1.set_title("Training nodes")
            plot_scatter(ax1, train_mask, y_train, "train")

            ax2 = fig.add_subplot(1,3,2)
            ax2.set_title("Validation nodes")
            plot_scatter(ax2, val_mask, y_val, "val")

            ax3 = fig.add_subplot(1,3,3)
            ax3.set_title("Test nodes")
            plot_scatter(ax3, test_mask, y_test, "test")
            plt.savefig("./figs/gat/visualisation_tsne.png",  bbox_inches='tight')
            plt.show()
