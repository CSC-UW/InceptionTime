import sys
import os
import re
#sys.path = [p for p in sys.path if p.find('/opt/apps/software/') == -1]
from glob import glob
from IPython.display import display, HTML
from matplotlib import pyplot as plt

from utils.constants import UNIVARIATE_ARCHIVE_NAMES as ARCHIVE_NAMES
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.utils import read_all_datasets, transform_labels, create_directory, run_length_xps, generate_results_csv, plot_epochs_metric, rmse
from utils.data_loading import get_multiple_data_cf, predict, shifted_zscore_cf, zscore, print_metric, tf_rmse, tf_pmse_cf
import utils
from classifiers import inception

import numpy as np
import pandas as pd
import sklearn
import keras
# keras.backend.tensorflow_backend._get_available_gpus()

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def tf_pmse_DA(y_true, y_pred):
    return tf_pmse_cf(y_true, y_pred, idx=0)

def tf_pmse_5HT(y_true, y_pred):
    return tf_pmse_cf(y_true, y_pred, idx=1)

def tf_pmse_pH(y_true, y_pred):
    return tf_pmse_cf(y_true, y_pred, idx=2)

def tf_pmse_NE(y_true, y_pred):
    return tf_pmse_cf(y_true, y_pred, idx=3)


parser = argparse.ArgumentParser()
population = parser.add_mutually_exclusive_group()
population.add_argument('--slow', default=False, action='store_true')
population.add_argument('--fast', default=False, action='store_true')
parser.add_argument('--epochs', nargs='?', default=100, type=int)
parser.add_argument('--probe', required=True, type=int)

args = parser.parse_args()
print(args, flush=True)

if args.slow:
    speed = 'slow'
if args.fast:
    speed = 'fast'

names = ['DA', '5HT', 'pH', 'NE']
data_prefix = '/mnt/nfs/proj/in-vitro/Leonardo/cf_data'

probes = [
    'CF025', 'CF027', 'CF057', 'CF064', 'CF066', 'CF078', 'CF081', 'CF082'
]

hold_probe = probes[args.probe]

# jitter = None
jitter = 500
output_directory = f'/mnt/nfs/proj/in-vitro/Leonardo/inception/results/cf/J{jitter}/E{epochs}/{hold_probe}/'

if not (os.path.exists(output_directory)):
    os.makedirs(output_directory, exist_ok=True)

# val_probe=None
if args.probe == 0
    val_probe=probes[]
else
    val_probe=probes[0]

print(f'Leaving out probe {hold_probe}', flush=True)
print(f'Validation probe {val_probe}', flush=True)
print(f'Loading data', flush=True)

# normalize_data = minmax
# revert_data = lambda x: minmax(x, inverse=True)

normalize_data = shifted_zscore_cf
revert_data = lambda x: shifted_zscore_cf(x, inverse=True)

# normalize_data = lambda x: x
# revert_data = lambda x: x

# this is actually the number of records per UNIQUE CONCENTRATIONS per probe
n_records_per_probe = -1 # all
# n_records_per_probe = 1

x_train, y_train, x_val, y_val, x_test, y_test = get_multiple_data_cf(data_prefix,
                                                                      probes=probes,
                                                                      hold_probe=hold_probe,
                                                                      val_probe=val_probe,
                                                                      normalize_data=lambda x: x,
                                                                      n_records_per_probe=n_records_per_probe,
                                                                      jitter=jitter)

print('Data loaded')

# adds singleton dimension to input due to inception time implementation
# TODO: update input to 2D matrix
if len(x_train.shape) == 2:  # if univariate
    print('adding singleton')
    # add a dimension to make it multivariate with one dimension
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

output_shape = y_train.shape[1]
input_shape = x_train.shape[1:]

for x in [x_train, y_train, x_val, y_val, x_test, y_test]:
    print(x.shape)
print(output_shape)
print(input_shape)

# build inception time model, 5 Resnet blocks each with 5 convulional layers of increasing kernel sizes (5, 10, 20, 40 and 80)
# Adam optimizer at default learning rate with schedule to half every 50 unsuccessful epochs (monitor loss)
# TODO: monitor val_loss
classifier = inception.Regression_INCEPTION(output_directory, input_shape, output_shape, verbose=1, build=True, nb_epochs=args.epochs, 
                                            metrics=[tf_pmse_DA, tf_pmse_5HT, tf_pmse_pH, tf_pmse_NE], normalize_y=(normalize_data, revert_data))
model_path = classifier.output_directory + 'best_model.hdf5'
if os.path.isfile(model_path):
    print('Best model already fit: %s'%model_path)
    best_model = classifier.get_best_model()
else:
    print('Model not fit yet')
    best_model = None

# Fit inception time the model
if best_model is None:
    print('Fitting new model...')
    metrics = classifier.fit(x_train, y_train, x_val, y_val, plot_test_acc=True)
    best_model = classifier.get_best_model()
else:
    print('Model alread fit, computing prediction of validation data')
    metrics = classifier.predict(x_val, y_val, x_train, y_train, return_df_metrics=True)

print('Validation metrics')
display(HTML(metrics.to_html()))

print('Hold out metrics')
metrics = classifier.predict(x_test, y_test, x_train, y_train, return_df_metrics=True)
display(HTML(metrics.to_html()))




