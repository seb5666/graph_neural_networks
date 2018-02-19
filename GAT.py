import numpy as np
import scipy

import tensorflow as tf 

from utils import load_data

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


def accuracy(probabilities, y, mask):
    predictions = np.argmax(probabilities, axis=-1)
    predictions = predictions[mask]
    print(predictions.shape)
    print(predictions[:10])
    labels = np.argmax(y[mask], axis=-1)
    print(labels.shape)
    print(labels[:10])
    print("Num examples: {}".format(len(np.argwhere(mask == True))))
    accuracy = len(np.argwhere(predictions == labels)) / len(np.argwhere(mask == True))
    return accuracy

def GAT_model(X, y, mask, l2_coef = 0.0005):
    f1 = GAT_layer("f1", X, adj, K=8, F_out=8, activation=tf.nn.elu, reduction="concat")
    logits = GAT_layer("f2", f1, adj, K=1, F_out=num_classes, activation=None, reduction="average")
    probabilities = tf.nn.softmax(logits)

    vars = tf.trainable_variables()
    L2_loss = lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * l2_coef

    loss = tf.reduce_sum(
        tf.nn.softmax_cross_entropy_with_logits(logits=tf.boolean_mask(logits, mask), labels=tf.boolean_mask(y, mask)))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss + L2_loss)

    return probabilities, loss, train_op

if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
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

    num_epochs = 5

    # add selfconnections to adjacency matrix
    adj = adj + scipy.sparse.eye(adj.shape[0])
    adj = np.array(adj.todense(), dtype=np.float32)

    # Create GAT
    X = tf.placeholder(shape=[num_nodes, num_features], name="X", dtype=tf.float32)
    probabilities, loss, train_op = GAT_model(X, y_train, train_mask)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())

        features = np.array(features.todense(), dtype=np.float32)

        for epoch in range(num_epochs):
            probabilities_, loss_, train_op_ = sess.run([probabilities, loss, train_op], { X: features })
            print("Epoch: {}: Loss: {}, Train accuracy: {}".format(epoch, loss_, accuracy(probabilities_, y_train, train_mask)))
            print("Validation accuracy: {}".format(accuracy(probabilities_, y_val, val_mask)))
            print(probabilities.shape)

            #_, loss_, accuracy_ = sess.run([optimizer, loss, accuracy],
            #        feed_dict = {X: np.array(features.todense()), y:y_train, mask: train_mask, dropout_rate: dropout})
            #if epoch % print_info_every == 0:
            #    print("Epoch:{:3d}\tLoss: {:.5f}\tAcc: {:.5f}".format(epoch, loss_, accuracy_))

            #    val_accuracy_ = sess.run(accuracy,
            #            feed_dict = {X: np.array(features.todense()), y:y_val, mask: val_mask})
            #    print("Validation accuracy: {:.5f}".format(val_accuracy_))


        #test_accuracy_, predictions_ = sess.run([accuracy, predictions],
        #        feed_dict = {X: np.array(features.todense()), y:y_test, mask: test_mask})
        #print("Test accuracy: {:.5f}".format(test_accuracy_))

