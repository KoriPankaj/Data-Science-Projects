import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
digits = datasets.load_digits()

(X, y) = datasets.load_digits(return_X_y=True)
X = X.astype(np.float32)
y = y.astype(np.int32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Let us define a layer of neurons as
n_inputs = 8*8
n_hidden_1 = 25
n_hidden_2 = 25
n_outputs = 10


class NNLayer(object):
    def __init__(self, n_inputs, n_neurons, activation=None):
        init = tf.random.normal((n_inputs, n_neurons),
                                stddev=2 / np.sqrt(n_inputs))
        self.W = tf.Variable(init, name="kernel")
        self.b = tf.Variable(tf.zeros([n_neurons]), name="bias")
        self.activation = activation

    def __call__(self, X):
        Z = tf.matmul(X, self.W) + self.b
        if self.activation is not None:
            return self.activation(Z)
        else:
            return Z

    def trainable_variables(self):
        return [self.W, self.b]


class Model(object):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    def trainable_variables(self):
        variables = []
        for layer in self.layers:
            variables.extend(layer.trainable_variables())
        return variables


# Instantiate loss function
loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits

# Create our custom model with our layers
model = Model([
    NNLayer(n_inputs, n_hidden_1, activation=tf.nn.relu),
    NNLayer(n_hidden_1, n_hidden_2, activation=tf.nn.relu),
    NNLayer(n_hidden_2, n_outputs)
])


def loss(model, X, y_true):
    y_pred = model(X)
    return tf.reduce_mean(tf.dtypes.cast(loss_function(labels=y_true, logits=y_pred),
                                         tf.float32), name="loss")


l = loss(model, X_train, y_train)
print('Test the starting loss: ', l)


def grad(model, X, y_true):
    with tf.GradientTape() as tape:
        loss_value = loss(model, X, y_true)
    return loss_value, tape.gradient(loss_value, model.trainable_variables())


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, X_train, y_train)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables()))

print("Step: {},         Loss: {}\n\n".format(optimizer.iterations.numpy(),
                                          loss(model, X_train, y_train).numpy()))

train_loss_results = []
train_accuracy_results = []

num_epochs = 2001

for epoch in range(num_epochs):


    loss_value, grads = grad(model, X_train, y_train)
    optimizer.apply_gradients(zip(grads, model.trainable_variables()))

    y_pred = model(X_train)
    correct = tf.equal(tf.math.argmax(y_pred, axis=1), y_train)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    train_loss_results.append(loss_value.numpy())
    train_accuracy_results.append(accuracy)
    
    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    loss_value.numpy(),
                                                                    accuracy))
        
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss")
axes[0].plot(train_loss_results, 'b')

axes[1].set_ylabel("Accuracy")
axes[1].set_xlabel("Epoch")
axes[1].plot(train_accuracy_results, 'r')
plt.show()


# Evaluate the testing accuracy
y_test_pred = model(X_test)
test_correct = tf.equal(tf.math.argmax(y_test_pred, axis=1), y_test)
test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))
print('Test accuracy: {:.3%}'.format(test_accuracy))


y_test_pred[0]

import pandas as pd
output_dist = tf.nn.softmax(y_test_pred[0])
df_output_dist = pd.DataFrame(output_dist).sort_values(0, ascending=False)
print('Inspecting a test example \n\nOutput probability distribution \n', df_output_dist)
df_output_dist.plot.bar(y=0)
plt.show()


from tensorflow import keras

n_inputs = 8*8
n_hidden_1 = 25
n_hidden_2 = 25
n_outputs = 10

# Lets build the same ANN as above
model = keras.Sequential([
    keras.layers.Dense(n_hidden_1, activation='relu', input_dim=n_inputs),
    keras.layers.Dense(n_hidden_2, activation='relu'),
    keras.layers.Dense(n_outputs)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD()

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=2000)


# Evaluate the accuracy
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

y_pred = tf.argmax(probability_model.predict(X_test), axis=1).numpy()

test_correct = tf.equal(y_pred, y_test)
test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))

print('\n\nFinal Testing Accuracy: {:.3%}'.format(test_accuracy))

# Evaluate the testing accuracy
y_test_pred = model(X_test)
test_correct = tf.equal(tf.math.argmax(y_test_pred, axis=1), y_test)
test_accuracy = tf.reduce_mean(tf.cast(test_correct, tf.float32))
print('Test accuracy: {:.3%}'.format(test_accuracy))


# Can also use the Keras API to evaluate the accuracy and loss for us
(loss_value, test_accuracy) = model.evaluate(X_test, y_test, verbose=2)
print('Test accuracy: {:.3%}'.format(test_accuracy))