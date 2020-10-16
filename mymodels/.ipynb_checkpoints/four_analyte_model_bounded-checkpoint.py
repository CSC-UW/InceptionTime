import os
import re

import h5py as h5
import numpy as np
from sklearn.metrics import mean_squared_error

import tensorflow as tf
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
    'rmse', 'print_metric', 'make_bijector', 'make_center_func', 'get_data',
    'ds_to_array', 'negloglik', 'predict', 'ConvBlock', 'ProbConv',
    'get_multiple_data', 'fromh5', 'get_multiple_data', 'make_preprocess',
    'datachain', 'prob_conv', 'conv', 'predict_non_prob', 'merge_datasets'
]

def tf_pmse(y_true, y_pred):
    std = tf.convert_to_tensor([1266.3062, 1266.7249, 0.13912384, 1266.6195], dtype=dtype)
    var = std**2

    d = tf.reduce_mean((y_true - y_pred)**2, axis=0)
    dp = tf.math.multiply(d,var)
    return tf.sqrt(tf.reduce_mean(dp))


def atoi(text):
    '''Converts text to integer if it contains an integer, does nothing otherwise.
    
    Arguments:
        text : str
            input text to be converted
    
    Returns:
        int if text contains an int value, else a string
    '''
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


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

def shifted_zscore(x, inverse=False):
    # without zeros shifted by 10 std to avoid negative values
    mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
    std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def make_bijector(n_dims, bi_type=0):
    '''Creates a tfp.Bijector for data
    
    Arguments:
        n_dims : int or list(tuple)
            if n_dims is an int, uses hardcoded prior ranges if defined
            if a list of tuples, creates min-max ranges from each
    
    Returns:
        tfp.Bijector such that lab_bijector.forward() goes from "real" -> scaled
        and lab.bijector.inverse() goes from scaled -> "real"
    '''

    print(bi_type)
    if bi_type==0:
        print('min max')
        if n_dims == 4:
            prior_min = tf.constant(np.array([-1, -1, 6, -1]), dtype=dtype)
            prior_max = tf.constant(np.array([5000, 5000, 8.8, 5000]), dtype=dtype)
        elif n_dims == 3:
            prior_min = tf.constant(np.array([-1, -1, 6]), dtype=dtype)
            prior_max = tf.constant(np.array([5000, 5000, 8.8]), dtype=dtype)
        else:
            minlist, maxlist = zip(*n_dims)
            prior_min = tf.constant(np.array(minlist), dtype=dtype)
            prior_max = tf.constant(np.array(maxlist), dtype=dtype)
        prior_range = prior_max - prior_min
        lab_bijector = tfb.Chain([
            tfb.Scale(1 / prior_range),
            tfb.Shift(-1 * prior_min),
        ])
    elif bi_type==1:
        print('no projection')
        lab_bijector = tfb.Identity()
    elif bi_type==2:
        print('z values')

        # without zeros
        mean = [2252.3855, 2252.6833, 7.392215, 2253.0007]
        std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]

#         # with zeros
#         mean = [887.8015747070312, 887.9190063476562, 7.392214298248291, 888.0441284179688]
#         std = [1357.765869140625, 1358.0380859375, 0.1391238272190094, 1358.125]

        if n_dims != 4:
            raise ValueError('Not implemented')
        
        prior_mean = tf.constant(np.array(mean), dtype=dtype)
        prior_std = tf.constant(np.array(std), dtype=dtype)
        
        lab_bijector = tfb.Chain([ 
            tfb.Scale(1 / prior_std), 
            tfb.Shift(-1 * prior_mean ), 
        ])
    elif bi_type==3:
        print('CDF')

        # without zeros shifted by 10 std to avoid negative values
        mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
        std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]
    
        prior_mean = tf.constant(np.array(mean), dtype=dtype)
        prior_std = tf.constant(np.array(std), dtype=dtype)
        
        lab_bijector = tfb.Chain([
            tfb.NormalCDF(),
            tfb.Scale(1 / prior_std), 
            tfb.Shift(-1 * prior_mean ), 
        ])
#         lab_bijector = tfb.NormalCDF()
    elif bi_type==4:
        print('"z values" between 0 and 0.5')
        if n_dims != 4:
            raise ValueError('Not implemented')

        # to transform to zvalues in range [0,1]
    #     mean01 = [0., 0., 6.699999809265137, 0.]
    #     std01 = [4510.0, 4492.0, 1.4000005722045898, 4517.0]
    #     mean01 = [0.0, 0.0, 5., 0.0]
    #     std01 = [10000.0, 10000.0, 5., 10000.0]
        mean01 = [0.0, 0.0, 5., 0.0]
        std01 = [100000.0, 100000.0, 50., 100000.0]
        
        prior_mean = tf.constant(np.array(mean01), dtype=dtype)
        prior_std = tf.constant(np.array(std01), dtype=dtype)
        
        lab_bijector = tfb.Chain([ 
            tfb.Scale(1 / prior_std), 
            tfb.Shift(-1 * prior_mean ), 
        ])
    elif bi_type==5:
        print('z values shifted by 10 std')
        if n_dims != 4:
            raise ValueError('Not implemented')

        # without zeros shifted by 10 std to avoid negative values
        mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
        std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]

        prior_mean = tf.constant(np.array(mean), dtype=dtype)
        prior_std = tf.constant(np.array(std), dtype=dtype)
        
        lab_bijector = tfb.Chain([ 
            tfb.Scale(1 / prior_std), 
            tfb.Shift(-1 * prior_mean ), 
        ])
 
    return lab_bijector


def make_center_func(bijector, weight=False):
    '''Creates a label preprocessing function from a bijector, acts on loaded data, not tfrecords files

    Arguments:
        bijector : tfp.Bijector
            scaling function to apply, forward() should move to scaled space
        weight : bool, optional
            whether or not to add sample weight, choice of weight function is somewhat arbitrary
    
    Returns:
        callable
            a function that takes zipped tf.data.Dataset values (voltammograms and labels here)
            and applies a scaling to the labels
    '''
    if weight:

        def center_func(x, y):
            ret = bijector.forward(y)
            return x, ret, tf.reduce_sum(tf.math.square(ret) + 1., axis=-1)
    else:

        def center_func(x, y):
            ret = bijector.forward(y)
            return x, ret  # tf.reduce_sum(tf.math.square(ret) + 1., axis=-1)

    return center_func

# Do not use at the moment, tensorflow-io has been having issues recently
def fromh5(prefix, probe, mode):
    pass
    # dsx = tfio.IODataset.from_hdf5(os.path.join(prefix, probe,
    #                                             probe + '_' + mode + '.h5'),
    #                                dataset='/voltammograms',
    #                                spec=dtype)
    # dsy = tfio.IODataset.from_hdf5(os.path.join(prefix, probe,
    #                                             probe + '_' + mode + '.h5'),
    #                                dataset='/labels',
    #                                spec=dtype)
    # return tf.data.Dataset.zip((dsx, dsy))


def make_preprocess(bijector, weight=True, n_dims=4):
    '''Creates a label preprocessing function from a bijector, acts on tfrecords files

    Arguments:
        bijector : tfp.Bijector
            scaling function to apply, forward() should move to scaled space
        weight : bool, optional
            whether or not to add sample weight, choice of weight function is somewhat arbitrary
    
    Returns:
        callable
            a function that takes zipped tf.data.Dataset values (voltammograms and labels here)
            and applies a scaling to the labels after parsing the data
    '''
    if weight:

        def preprocess(serialized_example):
            features = tf.io.parse_example(
                serialized_example,
                features={
                    'gram': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string)
                })

            data = tf.io.decode_raw(features['gram'], tf.float32)
            label = tf.io.decode_raw(features['label'], tf.float32)
            data.set_shape((None, 999))
            label.set_shape((None, n_dims))
            ry = bijector.forward(label)
            return data, ry, tf.reduce_sum(tf.math.square(ry) + 1., axis=-1)
    else:

        def preprocess(serialized_example):
            features = tf.io.parse_example(
                serialized_example,
                features={
                    'gram': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string)
                })

            data = tf.io.decode_raw(features['gram'], tf.float32)
            label = tf.io.decode_raw(features['label'], tf.float32)
            data.set_shape((None, 999))
            label.set_shape((None, n_dims))
            ry = bijector.forward(label)
            return data, ry

    return preprocess


def datachain(dataset, center_func, mode='train', batch_size=2048,
              cache=False):
    '''Applies preprocessing and other data handling functions.

    Arguments:
        dataset : tf.data.Dataset
            a tf.data.Dataset
        center_func : callable
            a function that applies any non-builtin preprocessing, see make_preprocess
        mode : str
            will be either 'train', 'val', 'test', or 'hold'; determines the "chain"
        batch_size : int, optional, default 2048
            determines the number of samples per batch in the output dataset
        cache : bool, optional
            whether or not to cache data in memory, DOES NOT check if there is 
            enough before hand, but its a safe bet
    
    Returns:
        tf.data.Dataset
            transformed dataset
    '''
    if mode == 'train':
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()

        if cache:
            dataset = dataset.cache()

        dataset = dataset.shuffle(buffer_size=100_000)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset
    elif mode == 'val':
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)  # drop_remainder=True
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if cache:
            dataset = dataset.cache()

        return dataset
    else:
        dataset = dataset.batch(batch_size=batch_size)  # , drop_remainder=True
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset


def shifted_zscore(x, inverse=False):
    # without zeros shifted by 10 std to avoid negative values
    mean = [-10410.677490234375, -10414.565185546875, 6.00097793340683, -10413.194091796875]
    std = [1266.3062, 1266.7249, 0.13912384, 1266.6195]

    if inverse:
        x = (x*std)+mean
    else:
        x = (x-mean)/std
    
    return x

def preprocess(serialized_example):
    features = tf.io.parse_example(
        serialized_example,
        features={
            'gram': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string)
        })
    data = tf.io.decode_raw(features['gram'], tf.float32)
    label = tf.io.decode_raw(features['label'], tf.float32)
    data.set_shape((None, 999))
    label.set_shape((None, 4))
    return data, label

def merge_datasets(vfiles, batch_size, prep, asnumpy=False):
    yv = []
    yl = []
    for filename in vfiles:
        ds = tf.data.TFRecordDataset(filename)
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

    if asnumpy:
        return x,y
    else:
        d = tf.data.Dataset.from_tensor_slices((x, y))
        return d

    
def get_multiple_data_v2(prefix,
                      probes,
                      holdout,
                      val_probe=None,
                      batch_size=2048,
                      weight=False,
                      allin=False,
                      n_dims=4,
                      bi_type=0,
                      cache=False):
    '''Obtains tfrecords files for specific probes under a prefix directory

    Arguments:
        prefix : str
            absolute path (on Hawking) of prefix directory
        probes : list(str)
            list of probe IDs put in train, val, and optionally test sets
        holdout : str
            single probe ID to test on, currently does not check for overlap
            recommend to generate list, and pop() holdout ID to ensure its not in
            training set
        batch_size : int, optional
            desired batch size for data
        weight : bool, optional
            whether or not to add sample weights, see make_preprocess
        n_dims : int or list(tuple)
            passed to make_bijector (dimensionality of the data being projected)
        bi_type: int
            passed to make_bijector (type of projection)
        cache : bool, optional
            passed to datachain
        
    Returns:
        train_data : tf.data.Dataset
        val_data : tf.data.Dataset
        (test_data) : tf.data.Dataset if allin=False
        hold_data : tf.data.Dataset    
    '''
    train_list = sum([
        sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'train_records', '*')),
               key=natural_keys) for probe in probes
    ], [])
    
    if val_probe is None:
        val_list = sum([
            sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'val_records', '*')),
                   key=natural_keys) for probe in probes
        ], [])
    else:
        print('using %s probe for validation and merging TFRecords original validation data into training'%val_probe)
        val_list = sum([
            sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'total_records', '*')),
                   key=natural_keys) for probe in [val_probe]
        ], [])
        # if keeping extra validation probe out, use validation data of other probes for training
        train_list = sum([ 
            sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'val_records', '*')), 
                   key=natural_keys) for probe in probes
        ], train_list)

    if not allin:
        test_list = sum([
            sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'test_records', '*')),
                   key=natural_keys) for probe in probes
        ], [])
    hold_list = sum([
        sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'total_records', '*')),
               key=natural_keys) for probe in [holdout]
    ], [])

    train_data = merge_datasets(train_list, batch_size, shifted_zscore)
    train_data = train_data.shuffle(buffer_size=100_000)
    train_data = train_data.batch(batch_size)
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    val_data = merge_datasets(val_list, batch_size, shifted_zscore)
    val_data = val_data.batch(batch_size)    
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    
    if not allin:
        test_data = merge_datasets(test_list, batch_size, shifted_zscore)
        test_data = test_data.batch(batch_size)    
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)

    hold_data = merge_datasets(hold_list, batch_size, shifted_zscore)
    hold_data = hold_data.batch(batch_size)    
    hold_data = hold_data.prefetch(tf.data.experimental.AUTOTUNE)

    if not allin:
        return train_data, val_data, test_data, hold_data
    else:
        return train_data, val_data, hold_data
    
def get_multiple_data(prefix,
                      probes,
                      holdout,
                      batch_size=2048,
                      weight=False,
                      allin=False,
                      n_dims=4,
                      bi_type=0,
                      cache=False):
    '''Obtains tfrecords files for specific probes under a prefix directory

    Arguments:
        prefix : str
            absolute path (on Hawking) of prefix directory
        probes : list(str)
            list of probe IDs put in train, val, and optionally test sets
        holdout : str
            single probe ID to test on, currently does not check for overlap
            recommend to generate list, and pop() holdout ID to ensure its not in
            training set
        batch_size : int, optional
            desired batch size for data
        weight : bool, optional
            whether or not to add sample weights, see make_preprocess
        allin : bool, optional
            whether prefix contains tfrecords prepared with
            in-probe test sets of novel concentrations. Allin=true
            means only train_data, val_data, and hold_data are returned
        n_dims : int or list(tuple)
            passed to make_bijector (dimensionality of the data being projected)
        bi_type: int
            passed to make_bijector (type of projection)
        cache : bool, optional
            passed to datachain
        
    Returns:
        train_data : tf.data.Dataset
        val_data : tf.data.Dataset
        (test_data) : tf.data.Dataset if allin=False
        hold_data : tf.data.Dataset    
    '''
    lab_bijector = make_bijector(n_dims, bi_type=bi_type)
    prep = make_preprocess(lab_bijector, weight=weight, n_dims=n_dims)

    train_list = sum([
        sorted(tf.io.gfile.glob(
            os.path.join(prefix, probe, 'train_records', '*')),
               key=natural_keys) for probe in probes
    ], [])
    val_list = sum([
        sorted(tf.io.gfile.glob(os.path.join(prefix, probe, 'val_records',
                                             '*')),
               key=natural_keys) for probe in probes
    ], [])
    if not allin:
        test_list = sum([
            sorted(tf.io.gfile.glob(
                os.path.join(prefix, probe, 'test_records', '*')),
                   key=natural_keys) for probe in probes
        ], [])
    hold_list = sum([
        sorted(tf.io.gfile.glob(
            os.path.join(prefix, probe, 'total_records', '*')),
               key=natural_keys) for probe in [holdout]
    ], [])

    train_list = [tf.data.TFRecordDataset(filename) for filename in train_list]
    train_list = [
        datachain(dataset,
                  prep,
                  mode='train',
                  batch_size=batch_size,
                  cache=cache) for dataset in train_list
    ]
    train_data = tf.data.experimental.sample_from_datasets(train_list)
    val_data = tf.data.TFRecordDataset(val_list)
    if not allin:
        test_data = tf.data.TFRecordDataset(test_list)
    hold_data = tf.data.TFRecordDataset(hold_list)

    val_data = datachain(val_data,
                         prep,
                         mode='val',
                         batch_size=batch_size,
                         cache=cache)
    if not allin:
        test_data = datachain(test_data,
                              prep,
                              mode='test',
                              batch_size=batch_size)
    hold_data = datachain(hold_data, prep, mode='test', batch_size=batch_size)

    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.prefetch(tf.data.experimental.AUTOTUNE)
    if not allin:
        test_data = test_data.prefetch(tf.data.experimental.AUTOTUNE)
    hold_data = hold_data.prefetch(tf.data.experimental.AUTOTUNE)

    if not allin:
        return train_data, val_data, test_data, hold_data
    else:
        return train_data, val_data, hold_data


def get_data(prefix, probe, center_func, mode='train', batch_size=2048):
    '''Obtains data for specific probe and set type under a prefix directory

    Arguments:
        prefix : str
            absolute path (on Hawking) of prefix directory
        probe : str
            probe ID to use
        center_func : callable
            see make_center_func
        mode : str, optional, default 'train'
            one of 'train', 'val', or 'test'
        batch_size : int, optional
            desired batch size for data
        
    Returns:
        tf.data.Dataset  
    '''
    if mode not in ['train', 'val', 'test']:
        return
    repeat = 4
    with h5.File(os.path.join(prefix, probe, probe + '_' + mode + '.h5'),
                 'r') as f:
        buff_size = f['labels'].shape[0]
        kl_weight = tf.constant(batch_size / (repeat * buff_size), dtype=dtype)
        if mode == 'train':
            prior_loc = tf.constant(np.mean(f['labels'], axis=0), dtype=dtype)
            prior_scale = tf.constant(np.sqrt(np.var(f['labels'], axis=0)),
                                      dtype=dtype)
        dataset = tf.data.Dataset.from_tensor_slices(
            (np.diff(np.array(f['voltammograms'], dtype=npdt),
                     axis=1), np.array(f['labels'], dtype=npdt)))
    if mode == 'train':
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.unbatch()
        dataset = dataset.cache()
        dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=repeat * buff_size)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, prior_loc, prior_scale, kl_weight
    elif mode == 'val':
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.cache()
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
    else:
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.map(map_func=center_func,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


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


def predict(bijector, model, dataset, samples=10):
    '''Draw inferences from probabilistic model.

    Arguments:
        bijector : tfp.Bijector
            bijector used to scale labels
        model : tf.keras.Model
            trained model created by prob_conv
        dataset : tf.data.Dataset
            dataset to draw inferences from
        samples : int, optional
            number of times to run model on each batch
    
    Returns:
        tempmean : np.ndarray
            median of samples number of distribution means
        temp05 : np.ndarray
            median of samples number of distribution 0.05 quantiles
        temp95 : np.ndarray
            median of samples number of distribution 0.95 quantiles'''
    tempmean = []
    temp05 = []
    temp95 = []
    #total = 10#tf.data.experimental.cardinality(dataset)
    #with tqdm(total=total) as pbar:
    for x, *_ in dataset:
        dists = [model(tf.convert_to_tensor(x)) for _ in range(samples)]
        tempmean.append(
            np.median([np.squeeze(dist.mean()) for dist in dists], axis=0))
        temp05.append(
            np.median([
                np.squeeze(dist.submodules[0].quantile(.05)) for dist in dists
            ],
                      axis=0))
        temp95.append(
            np.median([
                np.squeeze(dist.submodules[0].quantile(.95)) for dist in dists
            ],
                      axis=0))
        #pbar.update(1)
    # print(temp[0])
    tempmean = bijector.inverse(np.concatenate(tempmean, axis=0)).numpy()
    temp05 = bijector.inverse(np.concatenate(temp05, axis=0)).numpy()
    temp95 = bijector.inverse(np.concatenate(temp95, axis=0)).numpy()
    return tempmean, temp05, temp95


def predict_non_prob(bijector, model, dataset, samples=10):
    '''Draw inferences from probabilistic model.

    Arguments:
        bijector : tfp.Bijector
            bijector used to scale labels
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
#     print('training Not Set.') # , training=False
    for x, *_ in dataset:
        ys = [model(x) for _ in range(10)]
        print(len(ys))
        tempmean.append(np.median([y for y in ys], axis=0))
    tempmean = bijector.inverse(np.concatenate(tempmean, axis=0)).numpy()
    return tempmean

@tf.function
def harmonic_number(x):
    '''Computes analaytic continuation of the Harmonic number function.
    
    Arguments:
        x : tf.Tensor
    
    Returns:
        H(x) : tf.Tensor
    '''
    # taken from tensorflow probability, which uses it but doesn't export it
    one = tf.ones([], dtype=x.dtype)
    return tf.math.digamma(x + one) - tf.math.digamma(one)


# @tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=dtype), tf.TensorSpec(shape=None, dtype=dtype)])
# def kl_math(a, b):
#     '''Computes the kl divergence of a (batch of) Kumaraswamy distribution(s) with parameters a and b
#     from a uniform distribution, e.g. Kumaraswamy(1,1). Applies in the case of a, b >= 1.
    
#     Arguments:
#         a : tf.Tensor
#             Parameter a, corresponding to concentration1 in tfp
#         b : tf.Tensor
#             Parameter b, corresponding to concentration0 in tfp
    
#     Returns:
#         KL(Kumaraswamy(a,b), Kumaraswamy(1,1)) : tf.Tensor
#     '''
#     # I couldn't find a general analytical formula for the kl divergence between two kumaraswamy distributions
#     # this formula applies in the special case of KL(P, U(0,1)) where P is a unimodal kumaraswamy distribution (a,b > 1)
#     # and U(0,1) ~ Kumaraswamy(1,1)
#     one = tf.ones_like(a)
#     return -1 * one + tf.math.divide(one, b) + (-1 * one + tf.math.divide(
#         one, a)) * harmonic_number(b) + tf.math.log(a) + tf.math.log(b)


# def uniform_kl(rv_x):
#     '''Calls kl_math for a tpf Kumaraswamy distribution. Defined like this to avoid function retracing.
    
#     Arguments:
#         rv_x : tfp.distributions.Kumaraswamy
#             output of model
    
#     Returns:
#         KL(Kumaraswamy(a,b), Kumaraswamy(1,1)) : tf.Tensor'''
#     a = tf.convert_to_tensor(rv_x.concentration1)
#     b = tf.convert_to_tensor(rv_x.concentration0)
#     return kl_math(a, b)


# # Hackish way of registering KL, should check if U is really uniform,
# # but would slow it down.
# @tfp.distributions.RegisterKL(tfd.Kumaraswamy, tfd.Kumaraswamy)
# def kumara_uniform_kl(p, U, name=None):
#     return uniform_kl(p)


# @tf.function
# def negloglik(x, rv_x):
#     '''Negative log likelihood of obtaining x from distribution rv_x.
    
#     Arguments:
#         x : tf.Tensor
#             Tensor of scaled concentration tuples
#         rv_x : tfp.distributions.Distribution
#             Distribution capable of producing a tensor of shape (x.shape)
    
#     Returns:
#         negloglik : tf.Tensor'''
#     return -1 * rv_x.log_prob(x)


def conv_block(base_depth=15,
               kernel_size=5,
               prob=False,
               pool=True,
               norm=False,
               num=0,
               drop_prob=0.1):
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
                        name=f'conv_{num}'))

    if isinstance(drop_prob,list):
        drop_prob = drop_prob[num]
    
    if drop_prob > 0.:
        print('layer %d has drop_prob %2.2f'%(num, drop_prob))
        layers.append(tfkl.SpatialDropout1D(drop_prob, name=f'conv_drop_{num}'))
    else:
        print('Dropout disabled for layer %d'%num)

    if pool:
        layers.append(tfkl.MaxPool1D(2, name=f'conv_pool_{num}'))
        
    if norm:
        layers.append(tfkl.experimental.SyncBatchNormalization(name=f'conv_norm_{num}', epsilon=0.1))

    return layers


def prob_conv(input_dim=999,
              encoded_size=4,
              base_depth=15,
              kernel_size=10,
              dense_depth=256,
              prior=None,
              norm=False,
              deep_prob=False,
              dense_prob=False,
              kl_weight=0.001,
              interpool=False,
              name='prob_conv'):
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
        prior : tfp.distributions.Independent(tfp.distributions.Kumaraswamy) or None
            Prior distribution, only uniform prior works
        norm : bool, optional default False
            Flag that uses batch normalization in conv blocks if true
        deep_prob : bool, optional default False
            Flag that uses Bayesian conv layers in all blocks (rather than just the first)
            if true
        dense_prob : bool, optional default False
            Flag that uses Bayesian dense layers if true
        kl_weight : float
            Scaling factor to apply to kl divergence regularization
            Technically should be batch_size/number of batches in training data, but that can
            be difficult to compute while tf.data.Dataset.cardinality is still experimental
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
                   prob=True,
                   norm=norm,
                   pool=interpool,
                   num=0),
        conv_block(base_depth,
                   2 * kernel_size,
                   prob=deep_prob,
                   norm=norm,
                   pool=True,
                   num=1),
        conv_block(2 * base_depth,
                   3 * kernel_size,
                   prob=deep_prob,
                   norm=norm,
                   pool=interpool,
                   num=2),
        conv_block(2 * base_depth,
                   4 * kernel_size,
                   prob=deep_prob,
                   norm=norm,
                   pool=True,
                   num=3),
        conv_block(4 * base_depth,
                   5 * kernel_size,
                   prob=deep_prob,
                   norm=norm,
                   pool=interpool,
                   num=4),
    ]
    for block in blocks:
        for layer in block:
            x = layer(x)
    x = tfkl.Flatten(name='flatten')(x)
    if dense_prob:
        x = tfpl.DenseFlipout(dense_depth,
                              activation=tf.nn.swish,
                              name='dense_prob_0')(x)
        x = tfpl.DenseFlipout(int(dense_depth / 2),
                              activation=tf.nn.swish,
                              name='dense_prob_1')(x)
    else:
        x = tfkl.Dense(dense_depth, activation=tf.nn.swish, name='dense_0')(x)
        x = tfkl.Dense(int(dense_depth / 2),
                       activation=tf.nn.swish,
                       name='dense_1')(x)
    scale = tfkl.Dense(encoded_size, activation=tf.nn.softplus,
                       name='scale')(x)
    loc = tfkl.Dense(encoded_size, activation=tf.nn.softplus, name='loc')(x)
    x = tfkl.Concatenate(name='concatenate')([loc, scale])
    if prior is not None:
        reg = tfpl.KLDivergenceRegularizer(prior,
                                           weight=kl_weight,
                                           use_exact_kl=True)
    else:
        reg = None
    dist = tfpl.DistributionLambda(
        lambda t: tfd.Independent(tfd.Kumaraswamy(1. + t[..., :encoded_size],
                                                  1. + t[..., encoded_size:]),
                                  reinterpreted_batch_ndims=1),
        activity_regularizer=reg)(x)
    model = tfk.Model(inputs=model_input, outputs=dist, name=name)
    return model


def conv(input_dim=999,
              encoded_size=4,
              base_depth=15,
              kernel_size=10,
              dense_depth=256,
              norm=False,
              interpool=False,
              name='conv',
              drop_prob=.1):
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
                   drop_prob=drop_prob),
        conv_block(base_depth,
                   2 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=1,
                   drop_prob=drop_prob),
        conv_block(2 * base_depth,
                   3 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=2,
                   drop_prob=drop_prob),
        conv_block(2 * base_depth,
                   4 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=3,
                   drop_prob=drop_prob),
        conv_block(4 * base_depth,
                   5 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=4,
                   drop_prob=drop_prob),
    ]
    for block in blocks:
        for layer in block:
            x = layer(x)
    x = tfkl.Flatten(name='flatten')(x)
    x = tfkl.Dense(dense_depth, activation=tf.nn.swish, name='dense_0')(x)
    x = tfkl.Dense(int(dense_depth / 2), activation=tf.nn.swish, name='dense_1')(x)
    model_output = tfkl.Dense(encoded_size, activation=tf.nn.softplus, name='loc')(x)
    model = tfk.Model(inputs=model_input, outputs=model_output, name=name)
    return model

def conv_drop_last(input_dim=999,
              encoded_size=4,
              base_depth=15,
              kernel_size=10,
              dense_depth=256,
              norm=False,
              interpool=False,
              name='conv',
              drop_prob=.1):
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
                   drop_prob=0.),
        conv_block(base_depth,
                   2 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=1,
                   drop_prob=0.),
        conv_block(2 * base_depth,
                   3 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=2,
                   drop_prob=0.),
        conv_block(2 * base_depth,
                   4 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=True,
                   num=3,
                   drop_prob=0.),
        conv_block(4 * base_depth,
                   5 * kernel_size,
                   prob=False,
                   norm=norm,
                   pool=interpool,
                   num=4,
                   drop_prob=0.),
    ]
    for block in blocks:
        for layer in block:
            x = layer(x)
    x = tfkl.Flatten(name='flatten')(x)
    x = tfkl.Dense(dense_depth, activation=tf.nn.swish, name='dense_0')(x)
    
    if drop_prob > 0.:
        print('Dense layers with drop_prob %2.2f and %2.2f'%(drop_prob, drop_prob/2))
    
    x = tfkl.Dropout(drop_prob, name='dense_drop_0')(x)
    x = tfkl.Dense(int(dense_depth / 2), activation=tf.nn.swish, name='dense_1')(x)
    x = tfkl.Dropout(drop_prob / 2, name='dense_drop_1')(x)
    model_output = tfkl.Dense(encoded_size, activation=tf.nn.softplus, name='loc')(x)
    model = tfk.Model(inputs=model_input, outputs=model_output, name=name)
    return model

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self,
                 base_depth=15,
                 kernel_size=5,
                 prob=False,
                 pool=True,
                 norm=False,
                 name=None,
                 **kwargs):
        super(ConvBlock, self).__init__()
        if prob:
            self.conv = tfpl.Convolution1DFlipout(base_depth,
                                                  kernel_size,
                                                  padding='valid',
                                                  activation=tf.nn.swish)
        else:
            self.conv = tfkl.Conv1D(base_depth,
                                    kernel_size,
                                    padding='valid',
                                    activation=tf.nn.swish)
        self.normed = norm
        if norm:
            self.norm = tfkl.experimental.SyncBatchNormalization()
        self.dropout = tfkl.SpatialDropout1D(0.1)
        self.pooling = pool
        if pool:
            self.pool = tfkl.MaxPool1D(2)

    def call(self, inputs, training=None):
        x = self.conv(inputs, training=training)
        x = self.dropout(x, training=training)
        if self.pooling:
            x = self.pool(x, training=training)
        if self.normed:
            x = self.norm(x, training=training)
        return x


class ProbConv(tf.keras.Model):
    def __init__(self,
                 input_dim=999,
                 encoded_size=4,
                 base_depth=15,
                 kernel_size=10,
                 dense_depth=256,
                 prior=None,
                 norm=False,
                 loc_prior=None,
                 scale_prior=None,
                 deep_prob=False,
                 dense_prob=False,
                 kl_weight=0.001,
                 interpool=False,
                 name='autoencoder',
                 **kwargs):
        super(ProbConv, self).__init__()
        self.enc_params = tfpl.IndependentNormal.params_size(encoded_size)
        # self.scale_tril = tfb.FillScaleTriL(diag_bijector=None,
        #                                     diag_shift=0.002)
        self.reshape = tfkl.Reshape([input_dim, 1])
        self.block1 = ConvBlock(base_depth,
                                kernel_size,
                                prob=True,
                                norm=norm,
                                pool=interpool)
        self.block2 = ConvBlock(base_depth,
                                2 * kernel_size,
                                prob=deep_prob,
                                norm=norm,
                                pool=True)
        self.block3 = ConvBlock(2 * base_depth,
                                3 * kernel_size,
                                prob=deep_prob,
                                norm=norm,
                                pool=interpool)
        self.block4 = ConvBlock(2 * base_depth,
                                4 * kernel_size,
                                prob=deep_prob,
                                norm=norm,
                                pool=True)
        self.block5 = ConvBlock(4 * base_depth,
                                5 * kernel_size,
                                prob=deep_prob,
                                norm=norm,
                                pool=interpool)
        # self.block6 = ConvBlock(4 * base_depth,
        #                         6 * kernel_size,
        #                         prob=deep_prob,
        #                         norm=norm,
        #                         pool=True)
        self.reducer = tfkl.Flatten()
        # self.norm1 = tfkl.BatchNormalization()
        # self.norm2 = tfkl.BatchNormalization()

        # self.drop1 = tfkl.Dropout(0.1)
        # self.drop2 = tfkl.Dropout(0.1)

        if dense_prob:
            self.dense1 = tfpl.DenseFlipout(dense_depth,
                                            activation=tf.nn.swish)
            self.dense2 = tfpl.DenseFlipout(int(dense_depth / 2),
                                            activation=tf.nn.swish)
        else:
            self.dense1 = tfkl.Dense(dense_depth, activation=tf.nn.swish)
            self.dense2 = tfkl.Dense(int(dense_depth / 2),
                                     activation=tf.nn.swish)

        self.scale = tfkl.Dense(encoded_size, activation=tf.nn.softplus)
        # self.scaledist = tfpl.DistributionLambda(
        #    lambda t:
        #    # tfd.Independent(
        #    tfd.InverseGamma(concentration=2. + t[..., :encoded_size],
        #                     scale=0.001 + t[..., encoded_size:]),
        #    #    reinterpreted_batch_ndims=1)
        # )
        self.loc = tfkl.Dense(encoded_size, activation=tf.nn.softplus)
        # self.lam = tfkl.Dense(2 * encoded_size,
        #                      activation=tf.nn.softplus,
        #                      bias_initializer='ones')
        # self.lamdist = tfpl.DistributionLambda(lambda t: tfd.InverseGamma(
        #    concentration=2. + t[..., :encoded_size],
        #    scale=0.001 + t[..., encoded_size:]))
        self.concat = tfkl.Concatenate()
        if prior is not None:
            reg = tfpl.KLDivergenceRegularizer(prior, weight=kl_weight)
        else:
            reg = None
        # self.dist = tfpl.DistributionLambda(
        #     lambda t:  # tfd.Independent(
        #     tfd.Normal(
        #         loc=t[..., :encoded_size],
        #         scale=0.0002 + tf.math.divide(
        #             t[..., 2 * encoded_size:],
        #             0.0001 + t[..., encoded_size:2 * encoded_size]),
        #     ),
        #     # reinterpreted_batch_ndims=1),
        #     activity_regularizer=reg)
        self.dist = tfpl.DistributionLambda(
            lambda t: tfd.Independent(tfd.Kumaraswamy(
                1e-6 + t[..., :encoded_size], 1e-6 + t[..., encoded_size:]),
                                      reinterpreted_batch_ndims=1),
            activity_regularizer=reg)

    def call(self, inputs, training=None):
        x = self.reshape(inputs)
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        x = self.block5(x, training=training)
        # x = self.block6(x, training=training)
        x = self.reducer(x)
        x = self.dense1(x, training=training)
        # x = self.drop1(x, training=training)
        # x = self.norm1(x, training=training)
        x = self.dense2(x, training=training)
        # x = self.drop2(x, training=training)
        # x = self.norm2(x, training=training)
        scale = self.scale(x, training=training)
        # scaledist = self.scaledist(scale, training=training)
        loc = self.loc(x, training=training)
        # lam = self.lam(x, training=training)
        # lamdist = self.lamdist(lam, training=training)
        # x = self.concat([loc, lamdist, scaledist])
        x = self.concat([loc, scale])
        return self.dist(x, training=training)
