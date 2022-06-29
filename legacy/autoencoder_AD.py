# %% [markdown]
# Read dataset

# %% Imports
import datetime
import itertools
import yaml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from AutoEncoder import AutoEncoder
from tensorflow.keras.callbacks import TensorBoard
from yaml.loader import SafeLoader

np.random.seed(1337)

# %% Load configuration

with open('hyperparams_config.yaml') as f:
    full_configuration = yaml.load(f, Loader=SafeLoader)

config = full_configuration['DatasetList'][2]

# %% Import dataset
file_name = config['dataset_file']

dataframe = pd.read_csv(file_name)
dataset = dataframe.values

# %% [markdown]
# Train/Test data split and pre-processing

# %%

# The last element contains the labels
labels = dataset[:, -1]

# The other data points are the features
data = dataset[:, 0:-1]

# Dataset split (train & test)
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=1337
)

# Pre-processing
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

plt.figure(1)
plt.subplot(211)
plt.grid()
plt.plot(np.arange(train_data.shape[1]), normal_train_data[0])
plt.title("A Normal Measure")
plt.subplot(212)
plt.grid()
plt.plot(np.arange(train_data.shape[1]), anomalous_train_data[0])
plt.title("An Anomalous Measure")

# %% [markdown]
# Autoencoder

# %%

hidden_activation = config['hidden_layer_activation_function']
output_activation = config['output_layer_activation_function']
hidden_size = config['hidden_layer_size']
bottleneck_size = config['bottleneck_layer_size']
input_size = train_data.shape[1]

autoencoder = AutoEncoder(hidden_activation, output_activation, hidden_size, bottleneck_size, input_size)

autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

epochs = config['epochs']
batch_size = config['batch_size']

# Logs directory
log_dir = "logs/fit/" + file_name[0:4] + "..." + file_name[len(file_name)-4:len(file_name)] + "/" + hidden_activation + output_activation + "_neurons" + str(hidden_size) + "-" + str(bottleneck_size) + "_epoch" + str(epochs) + "_batch" + str(batch_size) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

trainig_history = autoencoder.fit(normal_train_data, normal_train_data, epochs=epochs, batch_size=batch_size, shuffle=True, validation_data=(test_data,test_data), verbose=1, callbacks=[tensorboard_callback])

plt.figure(2)
plt.plot(trainig_history.history["loss"], label="Training Loss")
plt.plot(trainig_history.history["val_loss"], label="Validation Loss")
plt.legend()


# %% [markdown]
# Validation

# %%
# Functions

def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
   
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mse(reconstructions, data)
  return tf.math.less(loss, threshold), reconstructions, loss

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))

# %%

# Plot the reconstruction error on normal Measures from the training set
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mse(reconstructions, normal_train_data)

# Choose a threshold value that is one standard deviations above the mean
threshold = np.percentile(train_loss, config['threshold_percentile'])
print("Threshold: ", threshold)

# Plot the reconstruction error on anomalous Measures from the test set
reconstructions = autoencoder.predict(anomalous_test_data)
anom_test_loss = tf.keras.losses.mse(reconstructions, anomalous_test_data)

plt.figure(3)
plt.hist(train_loss[None,:], bins=50, alpha=0.8, label="Train (normal)")
plt.hist(anom_test_loss[None, :], bins=50, alpha=0.8, label="Test (anomalous)")
plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")
plt.xlabel("Loss")
plt.ylabel("No of examples")
plt.legend(loc='best')

# Classify an Measure as an anomaly if the reconstruction error is greater than the threshold
pred_labels, pred_reconstructions, pred_loss = predict(autoencoder, test_data, threshold)
print_stats(pred_labels, test_labels)

# Confusion matrix
conf_matrix = confusion_matrix(test_labels, pred_labels)
cmplot = plot_confusion_matrix(conf_matrix, ["Positive","Negative"])

plt.show()
