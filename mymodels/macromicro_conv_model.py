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

# import tensorflow_io as tfio
# import tensorflow_probability as tfp

tfk = tf.keras
tfkl = tf.keras.layers
tfkr = tf.keras.regularizers
# tfpl = tfp.layers
# tfd = tfp.distributions
# tfb = tfp.bijectors
dtype = tf.float32
tf.keras.backend.set_floatx('float32')
npdt = np.float32
names = ['DA', '5HT', 'pH', 'NE']

__all__ = [
    'get_probes', 'rmse', 'make_bijector', 'get_data',
    'ds_to_array',
    'get_multiple_data', 'fromh5', 
    'datachain', 'conv', 'predict', 'shifted_zscore', 'zscore'
]

def tf_mmse(y_true, y_pred):
    xmin = np.array([-1,-1,6.8,-1])
    xmax = np.array([4000.,4000.,8.,4000.])
    a = tf.convert_to_tensor(xmax - xmin, dtype=dtype)
    var = a**2
    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
    dp = tf.math.multiply(d, var)
    return tf.sqrt(tf.reduce_mean(dp))

def tf_pmse(y_true, y_pred):
#     mean = tf.convert_to_tensor([-8613.974243164062, -8613.973999023438, 4.2091034054756165, -8613.974853515625], dtype=dtype)
    std = tf.convert_to_tensor([996.99524, 996.99524, 0.31908962, 996.9953], dtype=dtype)
    var = std**2
#     inverse = lambda x: tf.math.add(tf.math.multiply(x,std),mean)

#     tf.print(y_true)
#     tf.print(y_pred)
    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
#     tf.print(d)
    dp = tf.math.multiply(d,var)
#     tf.print(dp)
    return tf.sqrt(tf.reduce_mean(dp))
#     return tf.sqrt( (dp[0] + dp[1] + dp[3])/3 )

#     t = tf.reduce_mean(y_true, axis=0)
#     p = tf.reduce_mean(y_pred, axis=0)
#     t = tf.map_fn(inverse, t)
#     p = tf.map_fn(inverse, p)
#     return tf.sqrt(tf.reduce_mean((t - p)**2))

#     t = tf.map_fn(inverse, y_true)
#     p = tf.map_fn(inverse, y_pred)
#     return tf_rmse(t, p)

def tf_rmse(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean((y_true - y_pred)**2))


def get_probes(voltammograms):
# l -d /mnt/nfs/proj/in-vitro/iterate/results_014/model_style_008/training-A-RBE-97Hz/*DA_*MMA* | awk '{print $9}' | awk -F "_" '{print $19}' | sort -u
    return ['MMA003W01R04', 'MMA004W01R04', 'MMA013W01R04', 'MMA018W01R04', 'MMA019W01R04', 'MMA022W01R04', 'MMA023W01R04']

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
    # without zeros shifted by 10 std to avoid negative values
    # from CF probes
#     mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
#     std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]
    mean = [-8613.974243164062, -8613.973999023438, 4.2091034054756165, -8613.974853515625]
    std = [996.99524, 996.99524, 0.31908962, 996.9953]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def load_matlab(vfiles, nrecords_per_session=150, split=1.0, asnumpy=False, proj_y=shifted_zscore):
    yv = []
    yl = []
    for v in vfiles:
#         print()
#         print(v)
        bv = np.array(loadmat(v)['voltammograms'])
        bl = np.array(loadmat(v.replace('voltammograms.mat', 'labels.mat'))['labels'])

        for (xv, xl) in zip(bv,bl):
            a = xv[0][:,:nrecords_per_session].T
            a = np.apply_along_axis(np.diff, axis=1, arr=a) 
            b = xl[0][:nrecords_per_session,:4]
            b = np.apply_along_axis(proj_y, axis=1, arr=b) 
            yv.append(a.astype(npdt))
            yl.append(b.astype(npdt))

    x = np.vstack(yv)
    print(x.shape)
    y = np.vstack(yl)
    print(y.shape)
    
    if asnumpy:
        return x, y
    else:
        if split == 1.0:
            d1 = tf.data.Dataset.from_tensor_slices((x, y))
        else:
            idxs = np.random.permutation(x.shape[0])
            lim = int(x.shape[0]*split)
            d1idx = idxs[idxs[:lim]]
            d2idx = idxs[idxs[lim:]]
            d1 = tf.data.Dataset.from_tensor_slices((x[d1idx,:], y[d1idx,:]))
            d2 = tf.data.Dataset.from_tensor_slices((x[d2idx,:], y[d2idx,:]))

        if split == 1.0:
            return d1
        else:
            return d1, d2

def get_multiple_data(prefix,
                      probes=[],
                      holdout='MMA019',
                      val_probe=None,
                      batch_size=2048,
                      normalize_data=shifted_zscore,
                      weight=False,
                      filter_files=None):
    '''Obtains tfrecords files for specific probes under a prefix directory

    Arguments:
        prefix : str
            absolute path (on Hawking) of prefix directory
        holdout : str
            single probe ID to test on, currently does not check for overlap
            recommend to generate list, and pop() holdout ID to ensure its not in
            training set
        batch_size : int, optional
            desired batch size for data
        weight : bool, optional
            whether or not to add sample weights, see make_preprocess
        filter_files: function, optional
            remove files matching filter

    Returns:
        train_data : tf.data.Dataset
        val_data : tf.data.Dataset
        (test_data) : tf.data.Dataset if allin=False
        hold_data : tf.data.Dataset    
    '''

    nrecords_per_session = 300 # * 4 * 5
#     nrecords_per_session = 1000 # * 4 * 5
    data_split = .9

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
        hold_data = load_matlab(holdout_files, nrecords_per_session, proj_y=normalize_data)
        hold_data = hold_data.batch(batch_size)    
        hold_data = hold_data.prefetch(tf.data.experimental.AUTOTUNE)    
        train_val_files = list(compress(voltammograms, [x.find(holdout) == -1 for x in voltammograms]))

    if val_probe is None:
        print('number of train/val files %d'%len(train_val_files))
        train_data, val_data = load_matlab(train_val_files, nrecords_per_session, split=data_split, proj_y=normalize_data)
    else:
        print('validation probe: %s'%val_probe)
        
        val_files = list(compress(train_val_files, [x.find(val_probe) > -1 for x in train_val_files]))
        print('number of validation files %d'%len(val_files))
        val_data = load_matlab(val_files, nrecords_per_session, proj_y=normalize_data)
        
        train_files = list(compress(train_val_files, [x.find(val_probe) == -1 for x in train_val_files]))
        print('number of train files %d'%len(train_files))
        train_data = load_matlab(train_files, nrecords_per_session, proj_y=normalize_data)
        
    train_data = train_data.shuffle(buffer_size=100_000)
    train_data = train_data.batch(batch_size, drop_remainder=True)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    val_data = val_data.batch(batch_size, drop_remainder=True)    
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
        
    return train_data, val_data, hold_data

def ds_to_array(bijector, dataset):
    '''Creates numpy arrays from tf.data.dataset output by get_data or get_multiple_data.

    Arguments:
        bijector : tfp.Bijector
            bijector used to scale labels
        dataset : tf.data.Dataset
            preprocessed dataset
    
    Returns:
        x_arr : np.ndarray of voltammograms
        y_arr : np.ndarray of real space labels
    '''
    try:
        x_arr, y_arr = list(zip(*dataset))
    except ValueError:
        x_arr, y_arr, _ = list(zip(*dataset))
    y_arr = bijector.inverse(np.concatenate(y_arr, axis=0)).numpy()
    x_arr = np.concatenate(x_arr, axis=0)
    return x_arr, y_arr


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
