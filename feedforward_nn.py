from utils import load_data
import numpy as np

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2

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


adj = np.array(adj.todense())

# Compute the mean of all neighbouring features for the given node
def neighbouring_features(node):
    neighbours = adj[node]
    neighbour_indices = np.argwhere(neighbours == 1)[:, 0]
    neighbour_features = features[neighbour_indices]
    return np.mean(neighbour_features, axis=0)[0]


def compute_neighbouring_features():
    f = []
    for i in range(num_nodes):
        f.append(neighbouring_features(i))
    # f = [neighbouring_features(i) for i in range(num_nodes)]
    return np.array(f)

def accuracy(predictions_probs, labels):
    predictions = np.argmax(predictions_probs, axis=-1)
    labels = np.argmax(labels, axis=-1)
    assert(predictions.shape == labels.shape)
    num_correct = len(np.argwhere(predictions == labels))
    accuracy = num_correct / len(labels)
    return accuracy

f = compute_neighbouring_features()
print("Computed neighbouring features: {}".format(f.shape))

f = f.reshape(-1, num_features)
dense_features = np.array(features.todense())

merged_features = np.concatenate([f, dense_features], axis=-1)
print("Merged features: {}".format(merged_features.shape))

X_train = merged_features[train_mask]
y_train = y_train[train_mask]
X_val = merged_features[val_mask]
y_val = y_val[val_mask]
X_test = merged_features[test_mask]
y_test = y_test[test_mask]


batch_sizes = [32, 64, 128, 256, num_nodes]
num_epochs = [50, 100, 200, 400]
ps = [0.0, 0.1, 0.2, 0.4, 0.6]
reg_weights = [0.05, 0.005, 0.001, 0.0001]


best_params = None
best_val_acc = 0
best_test_acc = 0

for batch_size in batch_sizes:
    for num_epoch in num_epochs:
        for p in ps:
            for reg_weight in reg_weights:
                print("Batch size: {}\t num_epochs: {}\t p: {}\t reg_weight: {}".format(batch_size, num_epoch, p, reg_weight))

                inputs = Input(shape=(2*num_features, ))
                inputs_dropped = Dropout(p)(inputs)
                x1 = Dense(16, activation='relu', kernel_regularizer=l2(reg_weight))(inputs_dropped)
                x1_dropped = Dropout(p)(x1)
                # x2 = Dense(32, activation='relu', kernel_regularizer=l2(reg_weight))(x1_dropped)
                # x2_dropped = Dropout(p)(x2)
                predictions = Dense(num_classes, activation='softmax')(x1_dropped)

                model = Model(inputs=inputs, outputs=predictions)
                model.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])

                model.fit(x=X_train, y=y_train,
                          batch_size=batch_size,
                          epochs=num_epoch,
                          verbose=0
                          )

                val_probs = model.predict(X_val, batch_size=batch_size)
                test_probs = model.predict(X_test, batch_size=batch_size)

                val_acc = accuracy(val_probs, y_val)
                test_acc = accuracy(test_probs, y_test)
                print("Val accuracy: {}\t Test accuracy: {}".format(val_acc, test_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                    best_params = (batch_size, num_epoch, p, reg_weight)

print()
print("OPTIMAL PARAMS:")
print("Batch size: {}\t num_epochs: {}\t p: {}\t reg_weight: {}".format(batch_size, num_epoch, p, reg_weight))
print("Val acc: {}".format(best_val_acc))
print("Test acc: {}".format(best_test_acc))


