import numpy as np
import tensorflow as tf
from fate_client import init
from fate_client import components
from fate_client import submit_job

# Define the parties and roles
guest_party = {"name": "guest", "id": 9999}
host_party = {"name": "host", "id": 10000}
guest_role = components.Role(name="guest", party_id=guest_party["id"])
host_role = components.Role(name="host", party_id=host_party["id"])

# Define the FATE client and initialize it
client = init()
client.set_roles([guest_role, host_role])

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess the data
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model using federated learning
federated_learning_model = components.FederatedModel(model, guest_role, host_role)
federated_learning_model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test data
test_loss, test_acc = federated_learning_model.evaluate(x_test, y_test, batch_size=32)
print('Test accuracy:', test_acc)
