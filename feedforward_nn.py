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


X_train = features[train_mask]
y_train = y_train[train_mask]
X_val = features[val_mask]
y_val = y_val[val_mask]
X_test = features[test_mask]
y_test = y_test[test_mask]

batch_size = 256
num_epochs = 400
p = 0.5
reg_weight = 0.05

inputs = Input(shape=(num_features, ))
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

model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=num_epochs, validation_data=(X_val, y_val))


test_predictions_probabilities = model.predict(X_test, batch_size=num_nodes)
test_predictions = np.argmax(test_predictions_probabilities, axis=-1)
test_accuracy = len(np.argwhere(test_predictions == np.argmax(y_test, axis=-1)))/len(y_test)
print("Test accuracy: {}".format(test_accuracy))