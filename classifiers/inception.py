# resnet model
import keras
import numpy as np
import time
import pandas as pd

from utils.utils import save_logs, plot_epochs_metric, calculate_metrics, save_test_duration, rmse
from utils.data_loading import tf_pmse, tf_pmse_cf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class Classifier_INCEPTION:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=1500):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

        x = input_layer
        input_res = input_layer

        for d in range(self.depth):

            x = self._inception_module(x)

            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

    def fit(self, x_train, y_train, x_val, y_val, y_true, plot_test_acc=False):
        if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        start_time = time.time()

        if plot_test_acc:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val,
                              return_df_metrics=False)

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        df_metrics = save_logs(self.output_directory, hist, y_pred, y_true, duration,
                               plot_test_acc=plot_test_acc)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        start_time = time.time()
        model_path = self.output_directory + 'best_model.hdf5'
        print(model_path)
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred


class Regression_INCEPTION:

    def __init__(self, output_directory, input_shape, output_shape, verbose=False, build=True, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, nb_epochs=100,
                metrics=None, pre_model=None, normalize_y=(lambda x: x, lambda x: x)):

        self.output_directory = output_directory

        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.nb_epochs = nb_epochs
        self.metrics = metrics
        self.pre_model = pre_model
        self.normalize_data = normalize_y[0]
        self.revert_data = normalize_y[1]

        if build == True:
            self.model = self.build_model(input_shape, output_shape, pre_model=pre_model)
#             if (verbose == True):
#                 self.model.summary()
            self.verbose = verbose
            self.model.save(self.output_directory + 'model_init.hdf5')

            
    def _calculate_metrics(self, y_true, y_pred, duration):
        res = pd.DataFrame(data=np.zeros((1, 5), dtype=np.float), index=[0],
                           columns=['rmse_DA', 'rmse_5HT', 'rmse_pH', 'rmse_NE', 'duration'])
        y_pred = np.apply_along_axis(self.revert_data, axis=1, arr=y_pred) 
        y_true = np.apply_along_axis(self.revert_data, axis=1, arr=y_true) 
        rmse4 = rmse(y_true, y_pred)
        res['rmse_DA'] = rmse4[0]
        res['rmse_5HT'] = rmse4[1]
        res['rmse_pH'] = rmse4[2]
        res['rmse_NE'] = rmse4[3]
        res['duration'] = duration
        return res

    def _save_logs(self, hist, y_pred, y_true, duration,
                  lr=True, plot_test_acc=True):
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(self.output_directory + 'history.csv', index=False)

        df_metrics = self._calculate_metrics(y_true, y_pred, duration)
        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

        if plot_test_acc:
            print('using val_loss to find best metrics')
            index_best_model = hist_df['val_loss'].idxmin()
        else:
            print('using loss to find best metrics')
            index_best_model = hist_df['loss'].idxmin()

        row_best_model = hist_df.loc[index_best_model]

        df_best_model = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float), index=[0],
                                     columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_learning_rate', 'best_model_nb_epoch'])

        df_best_model['best_model_train_loss'] = row_best_model['loss']
        if plot_test_acc:
            df_best_model['best_model_val_loss'] = row_best_model['val_loss']

        if lr == True:
            df_best_model['best_model_learning_rate'] = row_best_model['lr']
        df_best_model['best_model_nb_epoch'] = index_best_model

        df_best_model.to_csv(self.output_directory + 'df_best_model.csv', index=False)

        if plot_test_acc:
            # plot losses
            plot_epochs_metric(hist, self.output_directory + 'epochs_loss.png')
            plot_epochs_metric(hist, self.output_directory + 'epochs_DA.png', metric='tf_pmse_DA')
            plot_epochs_metric(hist, self.output_directory + 'epochs_5HT.png', metric='tf_pmse_5HT')
            plot_epochs_metric(hist, self.output_directory + 'epochs_pH.png', metric='tf_pmse_pH')
            plot_epochs_metric(hist, self.output_directory + 'epochs_NE.png', metric='tf_pmse_NE')

        return df_metrics

    def _inception_module(self, input_tensor, stride=1, activation='linear'):

        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor

        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]

        conv_list = []

        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(
                input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)

        conv_list.append(conv_6)

        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x

    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x

    def build_model(self, input_shape, output_shape, pre_model=None):
        
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():

            input_layer = keras.layers.Input(input_shape)

            x = input_layer
            input_res = input_layer

            for d in range(self.depth):

                x = self._inception_module(x)

                if self.use_residual and d % 3 == 2:
                    x = self._shortcut_layer(input_res, x)
                    input_res = x

    #         print('')
    #         print('NO GAP LAYER!!!')
    #         print('')
    #         gap_layer = x

            gap_layer = keras.layers.GlobalAveragePooling1D()(x)

    #         output_layer = keras.layers.Dense(output_shape, activation='relu')(gap_layer)
            output_layer = keras.layers.Dense(output_shape, activation='softplus')(gap_layer)

            model = keras.models.Model(inputs=input_layer, outputs=output_layer)

            if not pre_model is None:
                print('loading previous weights (L-1 layers)...')
                for i in range(len(model.layers)-1):
                    model.layers[i].set_weights(pre_model.layers[i].get_weights())
            else:
                print('starting model from scratch...')

    #         model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=[])
    #         model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=[tf_pmse])
            if self.metrics is None:
                metrics = []
            else:
                metrics = self.metrics

    #         print('Compiling with Adadelta and metrics: ', [m.__name__ for m in metrics])
    #         model.compile(loss='mse', optimizer=keras.optimizers.Adadelta(), metrics=metrics)
            print('Compiling with Adam and metrics: ', [m.__name__ for m in metrics])
            model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=metrics)

    #         model.compile(loss='mse', optimizer=keras.optimizers.Adam(), metrics=['root_mean_squared_error'])

            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

            file_path = self.output_directory + 'best_model.hdf5'
            model_checkpoint_val = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)

            file_path = self.output_directory + 'best_train_model.hdf5'
            model_checkpoint_train = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True)

            self.callbacks = [reduce_lr, model_checkpoint_train, model_checkpoint_val]

        return model

    def fit(self, x_train, y_train, x_val, y_val, plot_test_acc=False):
        if len(keras.backend.tensorflow_backend._get_available_gpus()) == 0:
            print('error no gpu')
            exit()
        # x_val and y_val are only used to monitor the test loss and NOT for training

        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 20, 16))
        else:
            mini_batch_size = self.batch_size
        
        print(f'mini batch size: {mini_batch_size}')
            
        start_time = time.time()

        print(f'projecting y_train and y_val with {self.normalize_data.__name__}')
        y_train = np.apply_along_axis(self.normalize_data, axis=1, arr=y_train) 
        y_val = np.apply_along_axis(self.normalize_data, axis=1, arr=y_val) 
        
        if plot_test_acc:
            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:

            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                                  verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

        self.model.save(self.output_directory + 'last_model.hdf5')

        print('predicting validation set...', end = '')
        y_pred = self.predict(x_val, y_val, x_train, y_train, return_df_metrics=False)
        print(' done.')

        # save predictions
        np.save(self.output_directory + 'y_pred.npy', np.apply_along_axis(self.revert_data, axis=1, arr=y_pred))
        np.save(self.output_directory + 'y_true.npy', np.apply_along_axis(self.revert_data, axis=1, arr=y_true))

        df_metrics = self._save_logs(hist, y_pred, y_val, duration, plot_test_acc=plot_test_acc)

        keras.backend.clear_session()

        return df_metrics

    def predict(self, x_test, y_test, x_train, y_train, return_df_metrics=True):
        start_time = time.time()
        model = self.get_best_model()
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        if return_df_metrics:
            df_metrics = self._calculate_metrics(y_test, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            save_test_duration(self.output_directory + 'test_duration.csv', test_duration)
            return y_pred

    def get_best_model(self):
        model_path = self.output_directory + 'best_model.hdf5'
#         "tf_pmse_DA": tf_pmse_DA, "tf_pmse_5HT": tf_pmse_5HT, "tf_pmse_pH": tf_pmse_pH, "tf_pmse_NE": tf_pmse_NE
        custom_objects = {}
        if not self.metrics is None:
            for metric in self.metrics:
                custom_objects[metric.__name__] = metric
#         print('custom_objects: ', custom_objects)
        return keras.models.load_model(model_path, custom_objects=custom_objects)  
