import numpy as np

import tensorflow as tf

from utils import load_data

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

print("Total number of nodes: {}".format(adj.shape[0]))
print("Training nodes: {}".format(len(np.argwhere(y_train == True))))
print("Validation nodes: {}".format(len(np.argwhere(y_val == True))))
print("Test nodes: {}".format(len(np.argwhere(y_test == True))))


num_nodes = adj.shape[0]
num_features = features.shape[1]
num_classes = y_train.shape[1]

adj = np.array(adj.todense())
features = np.array(features.todense())

embedding_size = 128

X = tf.placeholder(dtype=tf.float32, shape=(num_nodes, num_features), name="X")

# Single dimension as there is a single feature only, the id of the node..
id_embeddings = tf.get_variable("id_embeddings", shape=(embedding_size), dtype=tf.float32)
print("Id embeddings:", id_embeddings.get_shape())

word_embeddings = tf.get_variable("word_embeddings", shape=(num_features, embedding_size), dtype=tf.float32)
print("Word embeddings:", word_embeddings.get_shape())

sum_embeddings =  tf.matmul(X, word_embeddings) + id_embeddings

h = 1/2 * sum_embeddings

h_tilde = [
    tf.reduce_mean(tf.gather(sum_embeddings, np.argwhere(adj[i] == 1)[:,0]), axis=0)
    for i in range(num_nodes)
]


print(h[0].get_shape())
print(h_tilde[0].get_shape())
