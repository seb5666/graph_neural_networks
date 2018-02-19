from utils import load_data

import numpy as np

from sklearn.svm import SVC

dataset = 'pubmed'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(dataset)
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

adj = adj.todense()

# Compute the mean of all neighbouring features for the given node
def neighbouring_features(node):
    neighbours = adj[node][0]
    neighbour_indices = np.argwhere(neighbours == 1)[:, 1]
    neighbour_features = features[neighbour_indices]
    return np.mean(neighbour_features, axis=0)


def compute_neighbouring_features():
    f = []
    for i in range(num_nodes):
        print(i)
        f.append(neighbouring_features(i))
    # f = [neighbouring_features(i) for i in range(num_nodes)]
    return np.array(f)


# Use a simple SVM on the features only!
X_train = features[train_mask].todense()
y_train = np.argmax(y_train[train_mask], axis=-1)

X_val = features[val_mask].todense()
y_val = np.argmax(y_val[val_mask], axis=-1)

X_test = features[test_mask].todense()
y_test = np.argmax(y_test[test_mask], axis=-1)


best_val_acc = 0
val_accs= []
C_grid = np.arange(1, 20, 0.1)
for C in C_grid:
    clf = SVC(kernel='linear', C=C)
    clf.fit(X_train, y_train)
    val_predictions = clf.predict(X_val)
    val_acc = len(np.argwhere(val_predictions == y_val)) / len(val_predictions)
    val_accs.append(val_acc)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_C = C
        
print(best_C, best_val_acc)

clf = SVC(kernel='linear', C=best_C)
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

accuracy = len(np.argwhere(predictions == y_test)) / len(predictions)

print(accuracy)
