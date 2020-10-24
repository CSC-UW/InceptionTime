import argparse
import os
from glob import glob
import pickle

import numpy as np
from sklearn.metrics import r2_score

import tensorflow as tf
import tensorflow_probability as tfp

import sys
sys.path.append('/home/leonardo.barbosa/projects/voltametry/Voltammetry_Modeling/mymodels/')
from four_analyte_model_bounded import (conv, ds_to_array,
                                        get_multiple_data, make_bijector,
                                        negloglik, predict_non_prob, print_metric, rmse)

parser = argparse.ArgumentParser()
population = parser.add_mutually_exclusive_group()
population.add_argument('--slow', default=False, action='store_true')
population.add_argument('--fast', default=False, action='store_true')
parser.add_argument('--kernel_size', nargs='?', default=5, type=int)
parser.add_argument('--base_depth', nargs='?', default=15, type=int)
parser.add_argument('--learning_rate', nargs='?', default=0.001, type=float)
parser.add_argument('--optimizer', nargs='?', default='Adadelta', type=str)
parser.add_argument('--epochs', nargs='?', default=2000, type=int)
parser.add_argument('--probe', required=True, type=int)
parser.add_argument('--batch_size', nargs='?', default=2048, type=int)
parser.add_argument('--force_reload', default=False, action='store_true')
parser.add_argument('--extra_pooling', default=False, action='store_true')
parser.add_argument('--weight_samples', default=False, action='store_true')
parser.add_argument('--dont_save', default=False, action='store_true')

args = parser.parse_args()
print(args, flush=True)

if args.slow:
    speed = 'slow'
if args.fast:
    speed = 'fast'

bi_type = 5

tfk = tf.keras
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers
tfpl = tfp.layers
tfb = tfp.bijectors
dtype = tf.float32
tf.keras.backend.set_floatx('float32')
npdt = np.float32
names = ['DA', '5HT', 'pH', 'NE']
prefix = os.path.join('/mnt/nfs/proj/in-vitro/Mark/four_analyte/', speed,
                      'allin')
probes = [x.split('/')[-1] for x in glob(prefix + '/CF*')]
probes.sort()
print(probes, flush=True)

mirrored_strategy = tf.distribute.MirroredStrategy()

good_probes = [
    'CF025', 'CF027', 'CF057', 'CF064', 'CF066', 'CF078', 'CF081', 'CF082'
]

# Only use good data
print('hold out probe: %s'%good_probes[args.probe], flush=True)
probes = good_probes  # probes.pop(probes.index(good_probes[args.probe]))
hold_probe = probes.pop(args.probe)

print(f'Training model ({bi_type} {args.optimizer}) on probes {probes}.', flush=True)
print(f'Leaving out probe {hold_probe}', flush=True)
print(f'Loading {speed} data', flush=True)
lab_bijector = make_bijector(4, bi_type=bi_type)
train_data, val_data, hold_data = get_multiple_data(prefix,
                                                    probes,
                                                    hold_probe,
                                                    batch_size=args.batch_size,
                                                    weight=True,
                                                    allin=True,
                                                    bi_type=bi_type)
if args.dont_save:
    print()
    print('NOT SAVING RESULTS!')
    print()

print('BEING DEBUG')

print('kernel_size: %d'%args.kernel_size)

try:
    x_arr, y_arr = list(zip(*hold_data))
except ValueError:
    x_arr, y_arr, _ = list(zip(*hold_data))
y_arr = np.concatenate(y_arr, axis=0)
x_arr = np.concatenate(x_arr, axis=0)
print(y_arr[-10:,])
print('END DEBUG')

print('Data loaded')

input_shape = 999
output_shape = 4
with mirrored_strategy.scope():
    vae = conv(input_dim=input_shape,
               encoded_size=output_shape,
               base_depth=args.base_depth,
               kernel_size=args.kernel_size,
               norm=True,
               interpool=args.extra_pooling,
               dense_depth=64)
    # vae.build(input_shape=(None, input_shape))

    if not args.dont_save:
        checkpoint = (
            f'four_analyte/checkpoints/{speed}'
            f'/conv_holdout_allin'
            f'_{hold_probe}_{len(probes)}_{args.base_depth}_{args.kernel_size}_{bi_type}_{args.optimizer}.h5')
    else:
        checkpoint = (
            f'four_analyte/checkpoints/{speed}'
            f'/debug'
            f'_{hold_probe}_{len(probes)}_{args.base_depth}_{args.kernel_size}_{bi_type}_{args.optimizer}.h5')

    if not args.force_reload:
        try:
            vae.load_weights(checkpoint)
            print('Found previous model weights.')
        except (OSError, ValueError):
            print('Did not find usable previous model weights.')

    print(f'optimizer {args.optimizer}', flush=True)
    print(f'learning rate {args.learning_rate}', flush=True)
    if args.optimizer == 'Adam':
        opt = tfk.optimizers.Adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'RMSprop':
        opt = tfk.optimizers.RMSprop(learning_rate=args.learning_rate)
    elif args.optimizer == 'Adadelta':
        opt = tfk.optimizers.Adadelta(learning_rate=args.learning_rate,
                                  epsilon=tfk.backend.epsilon())
    else:
        raise ValueError('Unknown optmizer')

    vae.compile(optimizer=opt,
                loss='mse',
                metrics=[tfk.metrics.MeanAbsolutePercentageError(),'mse'],
                experimental_run_tf_function=False)

    if not args.dont_save:
        model_file = (
            f'four_analyte/models/{speed}'
            f'/conv_holdout_allin'
            f'_{hold_probe}_{len(probes)}_{args.base_depth}_{args.kernel_size}_{bi_type}_{args.optimizer}.h5')
    else:
        model_file = (
            f'four_analyte/models/{speed}'
            f'/debug'
            f'_{hold_probe}_{len(probes)}_{args.base_depth}_{args.kernel_size}_{bi_type}_{args.optimizer}.h5')

    tfk.models.save_model(vae, model_file)

cb = [
    tfk.callbacks.ModelCheckpoint(
        checkpoint,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        save_freq='epoch',
    ),
    tfk.callbacks.TerminateOnNaN(),
    tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join('four_analyte', 'checkpoints', speed, 'logs',
                             'holdout', 'allin', hold_probe),
        histogram_freq=1,
        write_graph=False,
        write_images=True,
        update_freq='epoch',
        # profile_batch="20, 25",
        embeddings_freq=0,
        embeddings_metadata=None),
    # tfk.callbacks.ReduceLROnPlateau(monitor='val_loss',
    #                                 factor=0.8,
    #                                 patience=10,
    #                                 verbose=1,
    #                                 mode='min',
    #                                 min_delta=0.0001,
    #                                 cooldown=0,
    #                                 min_lr=0.05)
]
verb = 2
remainder = args.epochs
all_history = []
# all_stats = []
while remainder > 10:
    history = vae.fit(train_data,
                      validation_data=val_data,
                      epochs=remainder,
                      verbose=verb,
                      callbacks=cb)
    all_history.append(history.history)
    stats = vae.evaluate(hold_data)
#     all_stats.append(stats)
    print(stats)
    vae.load_weights(checkpoint)
    remainder -= len(history.history['loss'])

stats = vae.evaluate(hold_data)
print(stats)

out_prefix = os.path.join('/home/leonardo.barbosa/projects/voltametry/results/',
                          'four_analyte/good_probe_holdout/', 'pred_csvs',
                          'allin/non_prob')

if not (os.path.exists(out_prefix)):
    os.makedirs(out_prefix, exist_ok=True)

print('Saving history...', flush=True)

with open(os.path.join(out_prefix, f'{speed}_{hold_probe}_{bi_type}_{args.optimizer}_history.pickle'), 'wb') as handle:
    pickle.dump(all_history, handle)

print('History Saved.', flush=True)

print('Predicting holdout data...')
hold_conc_pred = predict_non_prob(lab_bijector, vae, hold_data)
_, hold_y_arr = ds_to_array(lab_bijector, hold_data)
print('done.')

print('----------------------------------------------------\n\n')
print(f'Holdout {hold_probe} Results: ')
print('RMSE:')
print_metric(rmse, names, hold_y_arr, hold_conc_pred)
print('R2 Score:')
print_metric(r2_score, names, hold_y_arr, hold_conc_pred)
print('----------------------------------------------------\n\n')

if not args.dont_save:
    print('Saving predictions for hold out...')
    np.savetxt(
        os.path.join(out_prefix, f'{speed}_{hold_probe}_{bi_type}_{args.optimizer}_outprobe_ytest.csv'),
        hold_y_arr, delimiter=',')
    np.savetxt(
        os.path.join(out_prefix, f'{speed}_{hold_probe}_{bi_type}_{args.optimizer}_outprobe_ypred.csv'),
        hold_conc_pred, delimiter=',')
    print('done.')

print('predicting validation data...')
val_x_arr, val_y_arr = ds_to_array(lab_bijector, val_data)
val_conc_pred = predict_non_prob(lab_bijector, vae, val_data)
print('----------------------------------------------------\n\n')
print('Validation Results: ')
print('RMSE:')
print_metric(rmse, names, val_y_arr, val_conc_pred)
print('R2 Score:')
print_metric(r2_score, names, val_y_arr, val_conc_pred)
print('----------------------------------------------------\n\n')

if not args.dont_save:
    print('Saving predictions for validation...')
    np.savetxt(os.path.join(out_prefix, f'{speed}_{hold_probe}_{bi_type}_{args.optimizer}_yval.csv'),
               val_y_arr, delimiter=',')
    np.savetxt(os.path.join(out_prefix, f'{speed}_{hold_probe}_{bi_type}_{args.optimizer}_yval_pred.csv'),
               val_conc_pred, delimiter=',')
    print('Predictions Saved.')