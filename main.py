# %% Imports
import yaml
import json
import datetime
import tensorflow as tf
from yaml.loader import SafeLoader
from AnomalyDetector import AnomalyDetector
from tensorboard.plugins.hparams import api as hp
from itertools import product

# %% Load configuration

CONFIG_ID = 2

with open('hyperparams_config.yaml') as f:
    full_configuration = yaml.load(f, Loader=SafeLoader)

# Save configuration as JSON
with open('hyperparams_config.json', 'w') as f:
    json.dump(full_configuration, f)

config = full_configuration['DatasetList'][CONFIG_ID]
log_dir = 'logs/fit/'+config["dataset_file"].split("/")[-1].replace(".csv","") + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

# %% Run the main program

METRIC_PERFORMANCE = 'AUC'
HPARAMS = {
    # 'hidden_layer_size': [256,512,600],
    # 'bottleneck_layer_size': [8,16,32],
    # 'learning_rate': [0.0001,0.0005,0.001],
    # 'n_anom_vars': [100,200,500],
    # 'sparsity': [0.0001,0.0],
    # 'prep_method': [0,1,2],
}

if HPARAMS: # hyper-parameters grid testing
    hparams_list=[]
    for hparam in product(HPARAMS.items()):
        hparams_list.append(hp.HParam(hparam[0][0], hp.Discrete(hparam[0][1])))
    with tf.summary.create_file_writer(log_dir).as_default():
        hp.hparams_config(
            # hparams=[HP_hidden_layer_size,HP_bottleneck_layer_size,HP_batch_size],
            hparams=hparams_list,
            metrics=[hp.Metric(METRIC_PERFORMANCE, display_name='ROC_AUC')],
        )

    def run(run_dir, config, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # Record the values used in this trial
            anomaly_detector = AnomalyDetector(config)
            roc_auc = anomaly_detector.run(hparams, run_dir)
            tf.summary.scalar(METRIC_PERFORMANCE, roc_auc, step=1)

    keys, values = zip(*HPARAMS.items())
    hparams_variations = [dict(zip(keys, v)) for v in product(*values)]

    session_num = 0

    for hparams in hparams_variations:
        config = {**config, **hparams} # Merge original config with hyperparams combination
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        run(log_dir + run_name, config, hparams)
        session_num += 1
else: # Normal run
    anomaly_detector = AnomalyDetector(config)
    roc_auc = anomaly_detector.run({}, log_dir+"single_run")
