from utils import load_data

from matplotlib import pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from keras.layers import Input, RNN, SimpleRNN, LSTM, concatenate, Dense
from keras.models import Model

from keras.preprocessing.sequence import pad_sequences

dataset = 'cora'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
num_nodes = adj.shape[0]
num_features = features.shape[1]
num_classes = y_train.shape[1]

print("Total number of nodes: {}".format(adj.shape[0]))
print("Training nodes: {}".format(len(np.argwhere(y_train == True))))
print("Validation nodes: {}".format(len(np.argwhere(y_val == True))))
print("Test nodes: {}".format(len(np.argwhere(y_test == True))))
print("Num nodes: {}".format(num_nodes))
print("num_features: {}".format(num_features))

adj = np.array(adj.todense())
features = np.array(features.todense())

#get neighbouring features
neighbour_features = []
print(num_nodes)
for i in range(num_nodes):
    neighbour_features.append([])
    for j in range(num_nodes):
        if adj[i][j] == 1:
            neighbour_features[i].append(features[j])

neighbour_features = pad_sequences(neighbour_features)
print("Neighbour features:", neighbour_features.shape)

max_length = neighbour_features.shape[1]
rnn_units = 128
hidden_units = [32, 32]

neighbour_features_input = Input(shape=(max_length, num_features), name='neighbour_features')
node_features_input = Input(shape=(num_features,), name='node_features')

encoder_outputs = SimpleRNN(units=rnn_units)(neighbour_features_input)
print(node_features_input.shape)
print(encoder_outputs.shape)
merged = concatenate([encoder_outputs, encoder_outputs])

x_1 = Dense(hidden_units[0], activation='relu')(merged)
predictions = Dense(num_classes, activation='softmax')(x_1)

model = Model(inputs=[neighbour_features_input,node_features_input], outputs = predictions)

output = model.predict([neighbour_features, features])
print("Output shape:", output.shape)
