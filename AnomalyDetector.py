import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.io
import math
from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
from AutoEncoder import AutoEncoder
from tensorflow.keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp

class AnomalyDetector():
    def __init__(self, config):
        super(AnomalyDetector, self).__init__()
        self.__config = config
        self.__rng_seed = 1337
        np.random.seed(self.__rng_seed)

    ### Helper Functions ###
    def plot_confusion_matrix(self, cm, class_names):
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

    def predict(self, model, data, threshold):
        reconstructions = model(data)
        pred_prob = model.predict(data)
        loss = tf.keras.losses.mse(reconstructions, data)
        if self.__config["normal_class"] == 1:
            labels = tf.math.less(loss, threshold)
        else:
            labels = tf.math.greater(loss, threshold)
        return labels, pred_prob, reconstructions, loss

    def print_stats(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        precission = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = 2 * precission * recall / (precission + recall)
        print(colored("Accuracy = {}".format(accuracy), "green"))
        print(colored("Precision = {}".format(precission), "green"))
        print(colored("Recall = {}".format(recall), "green"))
        print(colored("F1 = {}".format(f1), "green"))
    
    ##########################

    def import_dataset(self):
        file_name = self.__config['dataset_file']
        dataframe = pd.read_csv(file_name)
        dataset = dataframe.values
        return dataset

    def test_train_split(self, dataset, split_method):
        # The last element contains the labels
        labels = dataset[:, -1]

        # The other data points are the features
        data = dataset[:, 0:-1]

        # Dataset split (train & test)
        if (split_method == "fixed" and ("train_test_cut_point" in self.__config["train_test_split_method"] or "test_size" in self.__config["train_test_split_method"])):
            print(colored("Using split method: fixed", "yellow"))
            if "train_test_cut_point" in self.__config["train_test_split_method"]:
                train_test_cut_point = self.__config["train_test_split_method"]["train_test_cut_point"]
            elif "test_size" in self.__config["train_test_split_method"]:
                train_test_cut_point = round((1-self.__config["train_test_split_method"]["test_size"])*data.shape[0])
                print(colored("Cut point: {}".format(train_test_cut_point), "yellow"))
            train_data = data[0:train_test_cut_point, :]
            test_data = data[train_test_cut_point:, :]
            train_labels = labels[0:train_test_cut_point]
            test_labels = labels[train_test_cut_point:]
        else:
            print(colored("Using split method: random", "yellow"))
            if ("train_test_split_method" in self.__config and "test_size" in self.__config["train_test_split_method"]):
                test_size = self.__config["train_test_split_method"]["test_size"]
            else:
                test_size = 0.2
            train_data, test_data, train_labels, test_labels = train_test_split(
                data, labels, test_size=test_size, random_state=self.__rng_seed
            )
        

        train_labels = train_labels.astype(bool)
        test_labels = test_labels.astype(bool)

        return (train_data, test_data, train_labels, test_labels)

    def preprocessing(self, train_data, test_data, method):
        
        rows, cols = train_data.shape

        # No preprocessing
        if method == 0:
            prep_name = "NoPrep"
        # Mean-centering
        elif method == 1:
            prep_name = "MC"
            average = np.sum(train_data, axis=0)/rows
            train_data = train_data - average
            test_data = test_data - average
        # Auto-scaling
        elif method == 2:
            prep_name = "AS"
            average = np.sum(train_data, axis=0)/rows
            scale = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(train_data - average), axis=0)/(cols-1))
            train_data = (train_data - average)/scale
            test_data = (test_data - average)/scale
        # Normalization
        elif method == 3:
            prep_name = "Norm"
            min_val = tf.reduce_min(train_data)
            max_val = tf.reduce_max(train_data)
            train_data = (train_data - min_val) / (max_val - min_val)
            test_data = (test_data - min_val) / (max_val - min_val)

        train_data = tf.cast(train_data, tf.float32)
        test_data = tf.cast(test_data, tf.float32)

        return (train_data, test_data, prep_name)

    def generate_anomalies(self, test_data):
        #anom1 = 3*test_data[int(len(test_data)/2):len(test_data)]
        anom2 = test_data[int(len(test_data)/2):len(test_data)]
        n_anom_vars = self.__config["n_anom_vars"]
        # anom_gain = 5*math.sqrt(len(test_data[0]))*math.log10(len(test_data[0])/2)/n_anom_vars
        anom_add = 1.5 * np.sum(np.power(anom2,2), axis=1)/n_anom_vars
        anom2[:,0:n_anom_vars] = np.transpose(np.transpose(anom2[:,0:n_anom_vars]) + anom_add)

        test_data_mod = [np.vstack((test_data[0:int(len(test_data)/2)],anom2))]
        test_labels_mod = [np.hstack((np.ones((int(len(test_data)/2),),dtype=int),np.zeros((len(anom2),),dtype=int)))]
        
        for i in range(len(test_data_mod)):
            test_labels_mod[i] = test_labels_mod[i].astype(bool)

        return (test_data_mod[0], test_labels_mod[0])

    def run(self, hparams, log_dir):
        dataset = self.import_dataset()

        if ("train_test_split_method" in self.__config):
            if ("method_name" in self.__config["train_test_split_method"]):
                split_method = self.__config["train_test_split_method"]["method_name"]
                print(colored("Split method: " + split_method, "yellow"))
            else:
                print(colored("Split method name (method_name) not found in config file", "yellow"))
                split_method = "random"
        else:
            split_method = "random"

        train_data, test_data, train_labels, test_labels = self.test_train_split(dataset, split_method)

        # Generate anomalies
        if self.__config["n_anom_vars"] > 0:
            test_data, test_labels = self.generate_anomalies(test_data)

        # Preprocessing
        if ("prep_method" in self.__config):
            prep_method = self.__config["prep_method"]
        else:
            prep_method = 1 # Mean-centering
        train_data, test_data, prep_name = self.preprocessing(train_data, test_data, prep_method)

        if self.__config["normal_class"] == 1:
            normal_train_data = train_data[train_labels]
            normal_test_data = test_data[test_labels]
            anomalous_train_data = train_data[~train_labels]
            anomalous_test_data = test_data[~test_labels]
        else:
            normal_train_data = train_data[~train_labels]
            normal_test_data = test_data[~test_labels]
            anomalous_train_data = train_data[train_labels]
            anomalous_test_data = test_data[test_labels]

        if ("sparsity" in self.__config):
            sparsity = self.__config["sparsity"]
            print(colored("Sparsity = {}".format(sparsity), "red"))
        else:
            sparsity = 0

        # plt.figure(10)
        # plt.plot(np.arange(train_data.shape[1]), normal_train_data[0])
        # plt.title("ECG signal")
        # plt.xlabel("Time")
        # plt.ylabel("Amplitude")

        # plt.figure(1)
        # plt.subplot(211)
        # plt.grid()
        # if (len(normal_train_data) > 0):
        #     plt.plot(np.arange(train_data.shape[1]), normal_train_data[0])
        # plt.title("A Normal Measure")
        # plt.subplot(212)
        # plt.grid()
        # if (len(anomalous_train_data) > 0):
        #     plt.plot(np.arange(train_data.shape[1]), anomalous_train_data[0])
        # plt.title("An Anomalous Measure")

        hidden_activation = self.__config['hidden_layer_activation_function']
        output_activation = self.__config['output_layer_activation_function']
        hidden_size = self.__config['hidden_layer_size']
        bottleneck_size = self.__config['bottleneck_layer_size']
        input_size = train_data.shape[1]

        autoencoder = AutoEncoder(hidden_activation, output_activation, hidden_size, bottleneck_size, input_size, sparsity)

        learning_rate = self.__config['learning_rate']

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

        epochs = self.__config['epochs']
        batch_size = self.__config['batch_size']

        callbacks = []
        if ("enable_logging" in self.__config and self.__config['enable_logging'] == True):
            # Logs directory
            file_name = self.__config['dataset_file'].split('/')[-1]
            #log_dir = "logs/fit/" + file_name[0:4] + "..." + file_name[len(file_name)-4:len(file_name)] + "/" + hidden_activation + output_activation + "_neurons" + str(hidden_size) + "-" + str(bottleneck_size) + "_epoch" + str(epochs) + "_batch" + str(batch_size) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            #log_dir = "logs/fit/" + file_name + "/" + hidden_activation + output_activation + "_neurons" + str(hidden_size) + "-" + str(bottleneck_size) + "_epoch" + str(epochs) + "_batch" + str(batch_size) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            callbacks.append(tensorboard_callback)
            if hparams:
                callbacks.append(hp.KerasCallback(log_dir, hparams))

        trainig_history = autoencoder.fit(normal_train_data, normal_train_data, epochs=epochs, batch_size=batch_size, shuffle=False, validation_data=(test_data,test_data), verbose=1, callbacks=callbacks)

        plt.figure(2)
        plt.plot(trainig_history.history["loss"], label="Training Loss")
        plt.plot(trainig_history.history["val_loss"], label="Validation Loss")
        plt.legend(loc='best')

        # Plot the reconstruction error on normal Measures from the training set
        reconstructions = autoencoder.predict(normal_train_data)
        train_loss = tf.keras.losses.mse(reconstructions, normal_train_data)

        # Choose a threshold value that is one standard deviations above the mean
        threshold = np.percentile(train_loss, self.__config['threshold_percentile'])
        print(colored("Threshold: " + str(threshold), "green"))

        # Plot the reconstruction error on anomalous Measures from the test set
        reconstructions = autoencoder.predict(anomalous_test_data)
        anom_test_loss = tf.keras.losses.mse(reconstructions, anomalous_test_data)
        reconstructions = autoencoder.predict(normal_test_data)
        normal_test_loss = tf.keras.losses.mse(reconstructions, normal_test_data)

        plt.figure(3)
        plt.hist(train_loss[None,:], bins=50, alpha=0.8, label="Train")
        plt.hist(normal_test_loss[None, :], bins=50, alpha=0.8, label="Test (normal)")
        plt.hist(anom_test_loss[None, :], bins=50, alpha=0.8, label="Test (anomalous)")
        plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")
        plt.xlabel("Loss")
        plt.ylabel("No of examples")
        plt.legend(loc='best')

        # Classify an Measure as an anomaly if the reconstruction error is greater than the threshold
        pred_labels, pred_prob, pred_reconstructions, pred_loss = self.predict(autoencoder, test_data, threshold)

        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels, pred_labels)
        cmplot = self.plot_confusion_matrix(conf_matrix, ["Positive","Negative"])

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(~test_labels, pred_loss)
        roc_auc = auc(fpr, tpr)
        closest_threshold = np.argmin(np.abs(thresholds - threshold))
        self.print_stats(pred_labels, test_labels)
        print(colored("AUC = {}".format(roc_auc), "cyan"))
        sparsity_string = "S-" if sparsity>0 else ""
        #scipy.io.savemat("logs/mat/"+self.__config['dataset_file'].split('/')[-1].replace(".csv","_"+str(self.__config["n_anom_vars"])+"gen_"+prep_name+"-"+sparsity_string+"Autoencoder.mat"), dict(test_labels=np.array(test_labels.astype(float)), pred_loss=np.array(pred_loss).astype(float), threshold=threshold)) #, autoencoder_fpr=fpr, autoencoder_tpr=tpr, autoencoder_thresholds=thresholds, autoencoder_closest_threshold=closest_threshold, autoencoder_roc_auc=roc_auc))

        plt.figure(5)
        plt.plot(fpr, tpr, label='Autoencoder AUC = {:.4f}'.format(roc_auc))
        plt.plot([0,1],[0,1],'r--', label='Random AUC = 0.5')
        plt.plot(fpr[closest_threshold],tpr[closest_threshold],'o', label='Selected threshold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc='best')

        if ("show_figures" in self.__config and self.__config["show_figures"] == True):
            plt.show()
        
        return roc_auc


