# Deep Anomaly Detection Toolkit

This repository contains a framework and a set of tools to build a basic anomaly detection (AD) system, based on Deep Neural Networks (DNN), like an autoencoder, and lineal methods like Principal Components Analysis (PCA).

## Setup

It is recommended to install all the python dependencies on a virtual environment. You can create it running the following commands on Linux:

```bash
conda create --name tf python=3.9
conda activate tf
pip install -r requirements.txt
```

 ## Get started

First, you need to configure some datasets to work with and the hyper-parameters needed to generate the AD system.

Download a test dataset:

```bash
mkdir ../Datasets
wget http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv
mv ecg.csv ../Datasets
```

Then, delete the template hyperparams_config.yaml and create a new one, for example:

```yaml
DatasetList:
  - dataset_file: ../Datasets/ecg.csv
    normal_class: 1
    prep_method: 2 # Auto-scaling
    hidden_layer_activation_function: relu
    output_layer_activation_function: sigmoid
    hidden_layer_size: 64
    bottleneck_layer_size: 4
    PCs: 3
    epochs: 200
    batch_size: 8
    learning_rate: 0.0005
    sparsity: 0.0001
    threshold_percentile: 93
    show_figures: true
    enable_logging: true
    n_anom_vars: 0
    train_test_split_method:
      method_name: random
      test_size: 0.2
```

Now edit *main.py* and set CONFIG_ID to the ID in the hyperparams_config.yaml *DatasetList* array. In the example above, it is CONFIG_ID=0.

```python
# File: main.py
...
# Load configuration
CONFIG_ID = 2
...
```

Finally, you are ready to run the anomaly detector.

```bash
python main.py
```

