import os
import re
from itertools import compress
from glob import glob
import warnings

import h5py as h5
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.io import loadmat

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# # import tensorflow_io as tfio
# import tensorflow_probability as tfp

# tfk = tf.keras
# tfkl = tf.keras.layers
# tfkr = tf.keras.regularizers
# tfpl = tfp.layers
# tfd = tfp.distributions
# tfb = tfp.bijectors
# tf.keras.backend.set_floatx('float32')

dtype = tf.float32
npdt = np.float32

names = ['DA', '5HT', 'pH', 'NE']

__all__ = [
    'tf_pmse', 'tf_rmse', 'rmse', 'get_data', 'get_multiple_data', 'conv', 'predict', 'shifted_zscore', 'zscore', 'get_multiple_data_cf', 'shifted_zscore_cf', 'tf_pmse_cf'
]

def tf_mmse(y_true, y_pred):
    xmin = np.array([-1,-1,6.8,-1])
    xmax = np.array([4000.,4000.,8.,4000.])
    a = tf.convert_to_tensor(xmax - xmin, dtype=dtype)
    var = a**2
    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
    dp = tf.math.multiply(d, var)
    return tf.sqrt(tf.reduce_mean(dp))

def tf_pmse(y_true, y_pred, idx=-1):
#     mean = tf.convert_to_tensor([-8613.974243164062, -8613.973999023438, 4.2091034054756165, -8613.974853515625], dtype=dtype)
#     std = tf.convert_to_tensor([996.99524, 996.99524, 0.31908962, 996.9953], dtype=dtype)
    std = tf.convert_to_tensor([1.0397461e+03, 1.0397556e+03, 3.1908679e-01, 1.0397499e+03], dtype=dtype)
    var = std**2

    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
    dp = tf.math.multiply(d,var)
    
    if idx == -1:
        return tf.sqrt(tf.reduce_mean(dp))
    else:
        return tf.sqrt(dp[idx])
    
def tf_pmse_cf(y_true, y_pred, idx=-1):
    std = tf.convert_to_tensor([1266.3062, 1266.7249, 0.13912384, 1266.6195], dtype=dtype)
    var = std**2

    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
    dp = tf.math.multiply(d,var)
    
    if idx == -1:
        return tf.sqrt(tf.reduce_mean(dp))
    else:
        return tf.sqrt(dp[idx])

def tf_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))

def rmse(ytrue, ypred, multioutput='raw_values'):
    '''Wrapper around sklearn's mean_squared_error.
    
    Arguments:
        ytrue : np.ndarray
            Array containing true values, shape (samples, analytes)
        ypred : np.ndarray
            Array containing predicted values, shape (samples, analytes)
        multioutput : str, optional
            See sklearn.metrics multioutput options

    Returns:
        np.ndarray containing RMSE across samples, shape (analytes)
    '''
    return np.sqrt(
        mean_squared_error(
            ytrue,
            ypred,
            # squared=False,
            multioutput=multioutput))

def print_metric(metric, names, ytrue, ypred):
    '''Prints regression performance metric

    Arguments:
        metric : callable
            Metric function to apply, must follow sklearn.metrics signature
        names : list(str)
            Analyte names, length must equal ytrue.shape[1]
        ytrue : np.ndarraytfp.bijectors.NormalCDF
            Array containing true values, shape (samples, analytes)
        ypred : np.ndarray
            Array containing predicted values, shape (samples, analytes)
    '''
    if len(names) != ytrue.shape[1]:
        names = ['Var ' + str(i) for i in range(ytrue.shape[1])]

    for name, score in zip(names, metric(ytrue,
                                         ypred,
                                         multioutput='raw_values')):
        print(f'{name}: {score}')

def predict(invert, model, dataset, samples=10):
    '''Draw inferences from probabilistic model.

    Arguments:
        invert : function
            receives a 4d vector and returns rescaled 4d vector
        model : tf.keras.Model
            trained model created by prob_conv
        dataset : tf.data.Dataset
            dataset to draw inferences from
        samples : int, optional
            number of times to run model on each batch
    
    Returns:
        tempmean : np.ndarray
            median of samples number of distribution means'''
    tempmean = []
    for x, *_ in dataset:
        ys = [model(x) for _ in range(10)]
        print(len(ys))
        tempmean.append(np.median([y for y in ys], axis=0))
    tempmean = np.concatenate(tempmean, axis=0)
    tempmean = np.apply_along_axis(invert, axis=1, arr=tempmean) 

    return tempmean

def minmax(x, inverse=False):
    xmin = np.array([-1,-1,6.8,-1])
    xmax = np.array([4000.,4000.,8.,4000.])
    
    a = xmax - xmin

    if inverse:
        x = x*a + xmin
    else:
        x = (x-xmin)/a

    return x

def zscore(x, inverse=False):
    mean =  [945.0833,    945.0833,      7.4008675, 945.0833   ]
    std = [1.0397461e+03, 1.0397556e+03, 3.1908679e-01, 1.0397499e+03]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def shifted_zscore(x, inverse=False):
#     mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
#     std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]
#     mean = [-8613.974243164062, -8613.973999023438, 4.2091034054756165, -8613.974853515625]
#     std = [996.99524, 996.99524, 0.31908962, 996.9953]
    mean = [-9.4523779e+03, -9.4524736e+03,  4.2099996e+00, -9.4524160e+03,]
    std = [1.0397461e+03, 1.0397556e+03, 3.1908679e-01, 1.0397499e+03]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def load_matlab(vfiles, n_records_per_label_per_probe=10, split=1.0, asnumpy=False, proj_y=shifted_zscore):
    yv = []
    yl = []
    for v in vfiles:
        bv = np.array(loadmat(v)['voltammograms'])
        bl = np.array(loadmat(v.replace('voltammograms.mat', 'labels.mat'))['labels'])

        # iterate through matlab cell contents (one label per cell element?)
        for (xv, xl) in zip(bv,bl):
            v = xv[0].T # voltammetry
            v = np.apply_along_axis(np.diff, axis=1, arr=v) 
            l = xl[0][:,:4] # labels (concentrations)
            l = np.apply_along_axis(proj_y, axis=1, arr=l) 
            v = v.astype(npdt)
            l = l.astype(npdt)

            # get n unique records per label (tuple), or all data (-1)
            if n_records_per_label_per_probe > -1:
                # just to make sure there is only once concertraiton tuple in this record
                _, ulidx = np.unique(l, return_index=True, axis=0)
                u_v = []
                u_l = []
                for idx in ulidx:
                    # TODO: I think this can come from unique
                    one_label_idxs = np.where((l == l[idx,:]).all(axis=1))[0]
                    u_v.append(v[one_label_idxs[:n_records_per_label_per_probe], :])
                    u_l.append(l[one_label_idxs[:n_records_per_label_per_probe], :])
                v = np.concatenate(u_v)
                l = np.concatenate(u_l)
            yv.append(v)
            yl.append(l)

    x = np.vstack(yv)
    print(x.shape)
    y = np.vstack(yl)
    print(y.shape)

    idxs = np.random.permutation(x.shape[0])
    
    lim = int(x.shape[0]*split)
    d1idx = idxs[idxs[:lim]]
    d2idx = idxs[idxs[lim:]]
    x_1, y_1 = x[d1idx,:], y[d1idx,:]
    x_2, y_2 = x[d2idx,:], y[d2idx,:]

    x = x[idxs,:]
    y = y[idxs,:]
    
    if asnumpy:
        if split == 1.0:
            return x, y
        else:
            return x_1, y_1, x_2, y_2
    else:
        if split == 1.0:
            d1 = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            d1 = tf.data.Dataset.from_tensor_slices((x_1, y_1))
            d2 = tf.data.Dataset.from_tensor_slices((x_2, y_2))

def get_multiple_data(prefix,
                      probes=[],
                      holdout='MMA019',
                      val_probe=None,
                      data_split = .9,
                      number_of_minibatches=20,
                      normalize_data=shifted_zscore,
                      filter_files=None,
                      nrecords_per_session=1,
                      input_interval=None):
    '''Obtains tfrecords files for specific probes under a prefix directory
    Arguments:
        prefix : str
            absolute path (on Hawking) of prefix directory
        holdout : str
            single probe ID to test on
        val_probe : str, optional
            single probe ID to validate training (defaults to random permutation with (1-data_split) size)
        number_of_minibatches : int, optional
            number of minibatches (data will be spli accordingly)
        normalize_data:
            function to normalize labels (default to a z-scoring shifted by 10 std)
        filter_files: function, optional
            remove files matching filter

    Returns:
        train_data : tf.data.Dataset
        val_data : tf.data.Dataset
        hold_data : tf.data.Dataset    
    '''

    voltammograms = glob(prefix + '/*MMA*/voltammograms.mat')
    print('number of voltammograms files %d'%len(voltammograms))
    voltammograms = list(compress(voltammograms, [filter_files(x) for x in voltammograms]))
    print('after filter %d'%len(voltammograms))
    
    if probes:
        keep = np.array([False]*len(voltammograms))
        for probe in probes:
            keep = keep | np.array([x.find(probe) > -1 for x in voltammograms])
        voltammograms = list(compress(voltammograms, keep))
        print('after removing bad probes %d'%len(voltammograms))
    
    if holdout is None:
        warnings.warn('NO HOLD OUT DATA!')
        hold_data = None
        train_val_files = voltammograms
    else:
        holdout_files = list(compress(voltammograms, [x.find(holdout) > -1 for x in voltammograms]))
        print('number of holdout files %d'%len(holdout_files))
        x_test, y_test = load_matlab(holdout_files, nrecords_per_session, asnumpy=True, proj_y=normalize_data)

        train_val_files = list(compress(voltammograms, [x.find(holdout) == -1 for x in voltammograms]))

    if val_probe is None:
        print('number of train/val files %d'%len(train_val_files))
        x_train, y_train, x_val, y_val = load_matlab(train_val_files, nrecords_per_session, split=data_split, asnumpy=True, proj_y=normalize_data)
    else:
        print('validation probe: %s'%val_probe)
        
        val_files = list(compress(train_val_files, [x.find(val_probe) > -1 for x in train_val_files]))
        print('number of validation files %d'%len(val_files))
        x_val, y_val = load_matlab(val_files, nrecords_per_session, asnumpy=True, proj_y=normalize_data)
        
        train_files = list(compress(train_val_files, [x.find(val_probe) == -1 for x in train_val_files]))
        print('number of train files %d'%len(train_files))
        x_train, y_train = load_matlab(train_files, nrecords_per_session, asnumpy=True, proj_y=normalize_data)

    if not input_interval is None:
        print('selecting x values from %d to %d'%(input_interval[0], input_interval[1]))
        x_train = x_train[:, input_interval[0]:input_interval[1]]
        x_val = x_val[:, input_interval[0]:input_interval[1]]
        x_test = x_test[:, input_interval[0]:input_interval[1]]
        
    return x_train, y_train, x_val, y_val, x_test, y_test



def shifted_zscore_cf(x, inverse=False):
    # without zeros shifted by 10 std to avoid negative values
    mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
    std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def preprocess(serialized_example):
    
    features = tf2.io.parse_example(
        serialized_example,
        features={
            'gram': tf2.io.FixedLenFeature([], tf2.string),
            'label': tf2.io.FixedLenFeature([], tf2.string)
        })
    data = tf2.io.decode_raw(features['gram'], tf2.float32)
    label = tf2.io.decode_raw(features['label'], tf2.float32)
    data.set_shape((None, 999))
    label.set_shape((None, 4))
    return data, label

def merge_datasets(vfiles, batch_size, prep):
    
    tf2.config.experimental_run_functions_eagerly(True)
    print('wtf')
    print(tf2.executing_eagerly())
    
    yv = []
    yl = []
    for filename in vfiles:
        ds = tf2.data.TFRecordDataset(filename)
        ds = ds.batch(batch_size=batch_size)
        ds = ds.map(map_func=preprocess)
        for v,l in ds:
            v = np.array(v).astype(npdt)
            l = np.array(l).astype(npdt)
            yv.append(v)
            l = np.apply_along_axis(prep, axis=1, arr=l) 
            yl.append(l)
        
    x = np.vstack(yv)
    y = np.vstack(yl)

    return x,y

    
def get_multiple_data_cf(prefix,
                      probes,
                      hold_probe,
                      val_probe=None,
                      normalize_data=shifted_zscore_cf,
                      val_ratio=.1,
                      n_records_per_probe=-1):
    
    x_train_probes = []
    y_train_probes = []
    for probe in probes:
        print('loading probe %s'%probe)
        probe_file = os.path.join(prefix, f'{probe}.npz')
        probe_data = np.load(probe_file)
        x_probe = probe_data['x']
        y_probe = probe_data['y']
        
        # get n unique records per label
        if n_records_per_probe > -1:
            _, ulidx = np.unique(y_probe, return_index=True, axis=0)
            u_x = []
            u_y = []
            for idx in ulidx:
                all_idx = np.where((y_probe == y_probe[idx,:]).all(axis=1))[0]
                u_x.append(x_probe[all_idx[:n_records_per_probe], :])
                u_y.append(y_probe[all_idx[:n_records_per_probe], :])
            x_probe = np.concatenate(u_x)
            y_probe = np.concatenate(u_y)

        y_probe = np.apply_along_axis(normalize_data, axis=1, arr=y_probe) 
#         print(x_probe.shape, y_probe.shape)
        
        if probe == hold_probe:
            x_test = x_probe
            y_test = y_probe
        elif probe == val_probe:
            x_val = x_probe
            y_val = y_probe
        else:
            x_train_probes.append(x_probe)
            y_train_probes.append(y_probe)
            
    x_train = np.concatenate(x_train_probes)
    y_train = np.concatenate(y_train_probes)
    
    idxs = np.random.permutation(x_train.shape[0])

    if val_probe is None:
        print('Splitting training and validation datasets (validation ratio: %2.2f).'%val_ratio)
        lim = int(x_train.shape[0]*val_ratio)
        x_val, y_val = x_train[idxs[:lim],:], y_train[idxs[:lim],:]
        x_train, y_train = x_train[idxs[lim:],:], y_train[idxs[lim:],:]
    else:
        print('Shuffling training dataset')
        x_train, y_train = x_train[idxs,:], y_train[idxs,:]

    return x_train, y_train, x_val, y_val, x_test, y_test


def conv_block(base_depth=15,
               kernel_size=5,
               prob=False,
               pool=True,
               norm=False,
               num=0,
               drop_prob=0.1,
               l2_reg=None):
    '''Parametrically generate "convolutional blocks" for model as lists of layers.
    
    Arguments:
        base_depth : int
            Number of filters for convolution
        kernel_size : int
            Length of convolutional kernel
        prob : bool, optional default False
            Flag that uses Bayesian convolution layer when true
        pool : bool, optional default True
            Flag that adds pooling layer if True
        norm : bool, optional default False
            Flag that adds batch normalization when True
        num : int
            Block index in model, must be unique
    
    Returns:
        list
            List of layers corresponding to a convolutional block'''
    layers = []
    
    if prob:
        layers.append(
            tfpl.Convolution1DFlipout(base_depth,
                                      kernel_size,
                                      padding='valid',
                                      activation=tf.nn.swish,
                                      name=f'conv_prob_{num}'))
    else:
        layers.append(
            tfkl.Conv1D(base_depth,
                        kernel_size,
                        padding='valid',
                        activation=tf.nn.swish,
                        kernel_initializer='he_uniform',
#                         kernel_regularizer=l2_reg,
                        name=f'conv_{num}'))

    if isinstance(drop_prob,list):
        drop_prob = drop_prob[num]

    if drop_prob > 0.:
        print('layer %d has drop_prob %2.2f'%(num, drop_prob))
        layers.append(tfkl.SpatialDropout1D(drop_prob, name=f'conv_drop_{num}'))
    else:
        print('Dropout disable for layer %d'%num)

    if pool:
        layers.append(tfkl.MaxPool1D(2, name=f'conv_pool_{num}'))

    if norm:
        layers.append(tfkl.experimental.SyncBatchNormalization(name=f'conv_norm_{num}', epsilon=0.1))

    return layers


def conv(input_dim=999,
              encoded_size=4,
              base_depth=15,
              kernel_size=10,
              dense_depth=256,
              norm=False,
              interpool=False,
              name='conv',
              drop_prob=.1,
              l2_reg=None):
    '''Parametrically generate a probabilistic convolutional model.
    
    Arguments:
        input_dim : int
            Length of 1-D input
        encoded_size : int
            Size of output space/number of analytes
        base_depth : int
            Number of filters for first convolutional block
        kernel_size : int
            Length of kernel for first convolutional block
        dense_depth : int
            Size of first dense layer
        norm : bool, optional default False
            Flag that uses batch normalization in conv blocks if true
        interpool : bool, optional default False
            Flag that uses pooling in all conv blocks rather than just the even ones
        name : str, optional default 'vae' 
            name for model in tensorflow scope
    
    Returns:
        tf.keras.Model
            Probabilistic convolutional model'''
    model_input = tfk.Input(shape=(input_dim, ))
    x = tfkl.Reshape([input_dim, 1])(model_input)
    blocks = [
        conv_block(base_depth,
                   kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=0,
                   drop_prob=drop_prob,
                   l2_reg=l2_reg),
        conv_block(base_depth,
                   2 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=1,
                   drop_prob=drop_prob,
                   l2_reg=l2_reg),
        conv_block(2 * base_depth,
                   3 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=2,
                   drop_prob=drop_prob,
                   l2_reg=l2_reg),
        conv_block(2 * base_depth,
                   4 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=3,
                   drop_prob=drop_prob,
                   l2_reg=l2_reg),
        conv_block(4 * base_depth,
                   5 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=4,
                   drop_prob=drop_prob,
                   l2_reg=l2_reg),
    ]
    for block in blocks:
        for layer in block:
            x = layer(x)
    x = tfkl.Flatten(name='flatten')(x)
        
    if not l2_reg is None:
        print("L2 regularization for dense layers: %2.4f"%(l2_reg))
        reg = tfk.regularizers.l2(l2_reg)
    else:
        reg=None
        print("No regularization")

#     x = tfkl.Dense(dense_depth, activation=tf.nn.swish, name='dense_0')(x)
#     x = tfkl.Dense(int(dense_depth / 2), activation=tf.nn.swish, name='dense_1')(x)

    x = tfkl.Dense(dense_depth, activation=tf.nn.swish, kernel_regularizer=reg, name='dense_0')(x)

    if drop_prob > 0.:
        print('Dense layer 1 has drop_prob %2.2f'%(drop_prob))
        x = tfkl.Dropout(drop_prob, name='dense_drop_1')(x)
    
#     x = tfkl.Dense(dense_depth, activation=tf.nn.swish, kernel_regularizer=reg, name='dense_1')(x)
    x = tfkl.Dense(int(dense_depth / 2), activation=tf.nn.swish, kernel_regularizer=reg, name='dense_1')(x)

    if drop_prob > 0.:
        print('Dense layer 2 has drop_prob %2.2f'%(drop_prob/2))
        x = tfkl.Dropout(drop_prob/2, name='dense_drop_2')(x)
    
    model_output = tfkl.Dense(encoded_size, activation=tf.nn.softplus, name='loc')(x)

    model = tfk.Model(inputs=model_input, outputs=model_output, name=name)
    return model
