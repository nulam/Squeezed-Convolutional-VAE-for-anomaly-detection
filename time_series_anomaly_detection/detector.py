from typing import Optional, Iterable, Callable

import pandas as pd
import numpy as np

from time_series_anomaly_detection.abstractions import (
    TimeSeriesAnomalyDetector
)

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import glorot_normal  # this is xavier initializer as described in the paper
from tensorflow.keras import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class MultipleTimeseriesGenerator(Sequence):
    """
    Analogue to tf.keras.preprocessing.sequence.TimeseriesGenerator, except we allow
    multiple timeseries to be specified on the input, in the form of a list of timeseries.

    Parameters
    ----------
    df_list: Iterable[pd.DataFrame]
        List of DataFrames, each containing one time series.
    label_list:
        Complying with TimeseriesGenerator interface, it is possible to supply target labels for the dataset.
    time_window: int
        Width of the resulting samples.
    shuffle: bool
        True if dataset should be iterated randomly and not in order.
    batch_size: int
        How many time window samples will the generator provide in each iteration.
    """
    def __init__(self, df_list: Iterable[pd.DataFrame], label_list=None, time_window=4, shuffle=False, batch_size=32):
        super().__init__()
        # drop remainder (batch with len < batch_size)
        df_list = [np.array(series)[:(len(series) - len(series) % batch_size)] for series in df_list]

        if label_list is None:
            self.label_list = [np.zeros(len(series)) for series in df_list]
        else:
            self.label_list = label_list

        self.batch_size = batch_size
        self.generators = [TimeseriesGenerator(np.array(df_list[i]), np.array(self.label_list[i]), length=time_window,
                                               batch_size=batch_size, shuffle=shuffle) for i in range(len(df_list))]

        self.generator_lengths = [len(g) for g in self.generators]
        self.generator_indexes = np.cumsum(self.generator_lengths)
        self.len = np.sum(self.generator_lengths)

    def __len__(self) -> int:
        """
        Returns the number of samples in all the datasets combined
        """
        return self.len

    def __getitem__(self, index: int):
        """
        Fetches a single batch. Implemented by respecting we have multiple single series generators stored.
        The function decides which generator should be used and position within it.
        Parameters
        ----------
        index: int
            Position in the generator
        """
        # which series contains this index
        time_series_index = np.where(self.generator_indexes > index)[0][0]

        # get generator for the series, calculate position within than series and get its element
        element = self.generators[time_series_index][index % self.generator_indexes[max(0, time_series_index - 1)]]
        return element


class CustomFunctionCallback(tf.keras.callbacks.Callback):
    """
    Makes it possible to call an arbitrary function from within the tf.keras.Model.fit training loop.
    Supports only calling the function each `epoch_frequency` epochs.
    Parameters
    ----------
    fun : Callable
        The function to be called.
    epoch_frequency: int, optional
        How often the function should be called. It will called at the end of every `epoch_frequency`th epoch.
        The default is each 50 epochs.
    """
    def __init__(self, fun: Callable, epoch_frequency: Optional[int] = 50):
        self.fun = fun
        self.epoch_frequency = epoch_frequency

    def on_epoch_end(self, epoch, logs = None):
        """
        This function is called by Keras framework at the end of each epoch.
        This implementation checks whether it's the right time and calls the function stored in the callback if it is.
        """
        if epoch % self.epoch_frequency == 0:
            print(f"Epoch {epoch}")
            self.fun()


class SCVAEDetector(TimeSeriesAnomalyDetector):
    """
    Anomaly detector implemented as described in https://arxiv.org/pdf/1712.06343.pdf

    Parameters
    ----------
    id_columns: Iterable[str], optional
        ID columns used to identify individual time series.

        Should be specified in case the detector is provided with
        time series during training or inference with ID columns
        included. Using these columns the detector can separate individual
        time series and not use ID columns as feature columns.
        In case they are not specified, all columns are regarded as feature
        columns and the provided data is regarded as a single time series.

        To ensure the predict_anomaly_score function works correctly, the values in id_columns
        should be ascending, as there is a groupby operation used, that possibly would mix them up.

    latent_dim: int, optional
        The dimension of the latent layer, e.g. the layer between the encoder and the decoder.
        The default value is 5.

    time_window: int, optional
        The length of time window samples supplied to the model.
        The default value is 8.

    batch_size: int, optional
        The number of samples in each training batch.
        This model uses a reparametrization layer where we need to manually sample from the normal distribution.
        Because of this, current version of the model requires you to supply the training `batch_size` in advance.
        The default value is 16.

    use_probability_reconstruction: bool, optional
        If True, a probabilistic anomaly scoring method is used, as suggested in the paper.
        As it turns out, the score is very unstable doing it this way. Since MSE is used during the training,
        a MSE-based anomaly detection method is used by default.
        The default value is False.
    """

    def __init__(
            self,
            id_columns: Optional[Iterable[str]] = None,
            latent_dim: Optional[int] = 4,
            time_window: Optional[int] = 4,
            batch_size: Optional[int] = 4,
            use_probability_reconstruction: Optional[bool] = False
    ):
        super().__init__()
        self._latent_dim = latent_dim
        self._time_window = time_window
        self._id_columns = id_columns if id_columns is not None else []
        self._batch_size = batch_size
        self._real_samples = None
        self._mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
        self._scaler = StandardScaler()
        self._use_probability_reconstruction = use_probability_reconstruction

    def _split_multiple_timeseries_by_id(self, df: pd.DataFrame) -> Iterable[pd.DataFrame]:
        """
        Function that splits one pd.DataFrame with multiple time series identified by `id_columns` into
        a list of dataframes, with all the id columns removed.
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with all the series in one continuous table, identified by `id_columns` stored in this SCVAE object.

        Returns
        -------
        Iterable[pd.DataFrame]
            List of pd.DataFrame objects, one for each of the time series, without the ID columns.
        """

        # the case when id_columns were not supplied on init, just return a list with the original time series
        if self._id_columns == []:
            return [df]

        # group them by id, drop id_cols, make them into a list
        return [pd.DataFrame(y).drop(self._id_columns, axis=1) for x, y in df.groupby(self._id_columns, as_index=False)]

    def predict_anomaly_scores(
            self, X: pd.DataFrame
    ) -> pd.Series:
        """
        Scores each time moment in all the time series provided with a scalar anomaly score. These scores are returned
        as a pd.Series of the same length as `X`.
        We still have to respect the possibility of `X` containing multiple timeseries, so we first split them by
        `id_columns`, rank each of their windows, average the scores of windows for samples and finally concat
        all the anomaly scores back into the original shape.

        Please note that if the values in the `id_columns` columns in `X` are not ascending, the groupby operation
        can permutate the timeseries within the DataFrame. I recommend splitting and scoring them manually using
        the `_split_multiple_timeseries_by_id` function and multiple calls to this function.

        Parameters
        ----------
        X: pd.DataFrame
            DataFrame to be scored. Can contain multiple timeseries (with no overlaps!) identified by `id_columns`.

        Returns
        -------
        pd.Series
            Series with anomaly scores for each time sample.
        """
        # split time series into multiple by ID
        df_list = self._split_multiple_timeseries_by_id(X)

        # for each time series, generate all the possible windows
        all_time_windows = [list(pd.DataFrame(self._scaler.transform(df)).rolling(self._time_window)) for df in df_list]

        res_dfs = []

        # each iteration produces a pd.Series with results for each time series included in X
        for time_windows in all_time_windows:
            # pick scoring method (decided to default to MSE based one)
            get_score = self._reconstruction_probability if self._use_probability_reconstruction else self._reconstruction_score_mse

            # score every valid window with an anomaly score
            window_scores = pd.Series([get_score(window) for window in time_windows])
            # anomaly score for one time sample calculated as mean of all windows which contain the sample
            res = pd.Series([np.mean(window_scores[i:min(len(window_scores), i + self._time_window)]) for i in
                             range(len(window_scores))])
            res_dfs.append(res)

        # concat results for all the timeseries
        return pd.concat(res_dfs)

    def _scale_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scales the supplied dataframe `X` using a trained scaler. The data is scaled per-feature, as expected.
        Parameters
        ----------
        X: pd.DataFrame
            Data to be transformed.

        Returns
        -------
        pd.DataFrame
            Scaled data, same shape and form like the original `X` DataFrame.
        """
        res = X.copy()
        for i in range(len(X)):
            res[i][res[i].columns] = self._scaler.transform(X[i][X[i].columns])
        return X

    def plot_real_vs_generated(self) -> None:
        """
        Picks a static sample from the timeseries generator, plots it, runs it through the model and plots several
        reconstructions. Can be used as a callback during training to see the training progressing.
        """
        # if no samples were yet used, pick some
        if self._real_samples is None:
            self._real_samples = self._timeseries_generator[0][0]  # pick element, pick only timeseries
        res = self._model(self._real_samples)

        # multiple things are output by the model, actual reconstructions are the first element
        output = res[0]
        fig, axs = plt.subplots(2, 4, figsize=(15, 10))

        # the model outputs both its input and output merged into single tensor, split them and plot them
        _, output = tf.split(output, num_or_size_splits=2, axis=0)
        for i in range(4):
            axs[0, i].plot(self._real_samples[i])
            axs[1, i].plot(output[i])
        plt.show()
        pass

    def _reparametrization_latent(self, args) -> tf.Tensor:
        """
        VAE repamatrization trick, used on the encoder's output.
        """
        mean, logvar = args
        # Adding Gaussian noise, avoiding backpropagation path
        eps = tf.random.normal(shape=(self._batch_size, self._latent_dim))
        return eps * tf.exp(logvar * 0.5) + mean

    def _reparametrization_series(self, args) -> tf.Tensor:
        """
        VAE repamatrization trick, used on the decoder's output.
        """
        mean, logvar = args
        eps = tf.random.normal(shape=(self._batch_size, self._time_window * self._feature_count))
        return eps * tf.exp(logvar * 0.5) + mean

    def _kl_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Kullback–Leibler divergence loss, pushes both networks's output to resemble a multivariate N(0,1) distribution.
        """
        mean, logvar = tf.split(y_pred, num_or_size_splits=2, axis=0)
        loss = -0.5 * tf.keras.backend.sum(1 + logvar - mean ** 2 - tf.exp(logvar), axis=-1)
        return loss

    def _reconstruction_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Reconstruction loss, MSE is used in this model due to having unbounded continuous data.
        Attempts were made with binary-cross entropy, but it didn't perform well.
        """
        input, output = tf.split(y_pred, num_or_size_splits=2, axis=0)
        return self._mse(input, output)

    def _build_encoder(self) -> Model:
        """
        Builds a keras.Model instance of SCVAE encoder. It is implemented exactly as described in the paper, using
        parallel Fire modules, instead of sequential 1D convolutions.
        """
        # Encoder
        input = Input(shape=(self._time_window, self._feature_count), batch_size=self._batch_size)
        # Squeeze convolution
        x = Conv1D(16, kernel_size=1, strides=1, kernel_initializer=glorot_normal, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Extend (expand) convolutions
        extend1 = Conv1D(16, kernel_size=1, strides=1, kernel_initializer=glorot_normal, padding='same')(x)
        extend1 = Activation('relu')(extend1)
        extend1 = BatchNormalization(momentum=0.9)(extend1)
        extend2 = Conv1D(32, kernel_size=3, strides=1, kernel_initializer=glorot_normal, padding='same')(x)
        extend2 = Activation('relu')(extend2)
        extend2 = BatchNormalization(momentum=0.9)(extend2)
        x = Concatenate()([extend1, extend2])

        # Fully connected layers
        x = Flatten()(x)
        mean = Dense(self._latent_dim)(x)
        logvar = Dense(self._latent_dim)(x)
        return Model(name='scvae_encoder', inputs=input, outputs=[mean, logvar])

    def _build_decoder(self) -> Model:
        """
        Builds a keras.Model instance of SCVAE decoder. It is implemented exactly as described in the paper, using
        parallel Fire modules, instead of sequential 1D transposed convolutions.
        """
        # Encoder
        input = Input(shape=(self._latent_dim, 1), batch_size=self._batch_size)

        # Squeeze convolution
        x = Conv1DTranspose(16, kernel_size=1, strides=1, kernel_initializer=glorot_normal, padding='same')(input)
        x = Activation('relu')(x)
        x = BatchNormalization(momentum=0.9)(x)

        # Extend (expand) convolutions
        extend1 = Conv1DTranspose(16, kernel_size=1, strides=1, kernel_initializer=glorot_normal, padding='same')(x)
        extend1 = Activation('relu')(extend1)
        extend1 = BatchNormalization(momentum=0.9)(extend1)
        extend2 = Conv1DTranspose(1, kernel_size=3, strides=1, kernel_initializer=glorot_normal, padding='same')(x)
        extend2 = Activation('relu')(extend2)
        extend2 = BatchNormalization(momentum=0.9)(extend2)

        x = Concatenate()([extend1, extend2])

        x = Flatten()(x)
        # Fully connected layers
        mean = Dense(self._time_window * self._feature_count)(x)
        logvar = Dense(self._time_window * self._feature_count)(x)
        return Model(name='scvae_decoder', inputs=input, outputs=[mean, logvar])

    def _build_model(self) -> Model:
        """
        Builds a keras.Model instance of the entire SCVAE. It is build by joining encoder and decoder and putting a
        random sampling process with a reparametrization trick on the output of both of the networks.
        """
        encoder = self._build_encoder()
        decoder = self._build_decoder()

        input = Input(shape=(self._time_window, self._feature_count), batch_size=self._batch_size)
        mean_encoder, logvar_encoder = encoder(input)

        # "Sampling" using reparametrization trick
        x = Lambda(self._reparametrization_latent, output_shape=(self._latent_dim,))([mean_encoder, logvar_encoder])
        x = Reshape((self._latent_dim, 1))(x)
        mean_decoder, logvar_decoder = decoder(x)

        # "Sampling" using reparametrization trick
        x = Lambda(self._reparametrization_series, output_shape=(self._time_window * self._feature_count, 1))(
            [mean_decoder, logvar_decoder])
        output = Reshape((self._time_window, self._feature_count,))(x)
        out_reconstruction = Concatenate(axis=0, name='reconstruction')([input, output])
        out_encoder = Concatenate(axis=0, name='encoder_kl')([mean_encoder, logvar_encoder])
        out_decoder = Concatenate(axis=0, name='decoder_kl')([mean_decoder, logvar_decoder])
        model = Model(inputs=input, outputs=[out_reconstruction, out_encoder, out_decoder])
        return model

    def _init_empty_model(self, X: pd.DataFrame) -> None:
        """
        Initialises this instance of SCVAE using information from the training dataset.
        These steps are basically the same as the first steps of the `fit` function.
        The use case for this function is initialisation before loading a trained model manually from a saved file.         
        Parameters
        ----------
        X: pd.DataFrame
            Training data that were or would be used in model.fit
        """
        self._feature_count = len(X.columns) - len(self._id_columns)
        self._model = self._build_model()
        self._scaler = StandardScaler()
        self._scaler.fit(X)
        pass

    def fit(self, X: pd.DataFrame, learning_rate: float = 0.001, epochs: int = 50, *args, **kwargs) -> None:
        """
        Wrapper for the tf.keras.Model.fit function. 
        Parameters
        ----------
        X: pd.DataFrame
            The training dataset. For training purposes, NaN values are zeroed out.
        learning_rate: float
            Learning rate supplied to both of the optimizers.
        epochs: int
            Number of iterations through the training dataset.
        """
        self._feature_count = len(X.columns) - len(self._id_columns)
        X = X.fillna(0)
        X = self._split_multiple_timeseries_by_id(X)
        self._scaler = StandardScaler()
        self._scaler.fit(pd.concat(X))
        X = self._scale_columns(X)
        self._model = self._build_model()
        self._timeseries_generator = MultipleTimeseriesGenerator(X, batch_size=self._batch_size,
                                                                 time_window=self._time_window, shuffle=True)

        losses = {
            "reconstruction": self._reconstruction_loss,
            "encoder_kl": self._kl_loss,
            "decoder_kl": self._kl_loss
        }

        def reconstruct():
            print(self._reconstruction_score_mse(X[0][:self._time_window], debug_plots=True))

        callbacks = [
            # enables tracking losses in tensorboard
            #             tf.keras.callbacks.TensorBoard(log_dir=f'./logs'),

            # show subplots with real and reconstructed samples to visually compare them
            CustomFunctionCallback(self.plot_real_vs_generated, epoch_frequency=100),

            # shows several reconstructions and calculates reconstruction probability for one sample
            CustomFunctionCallback(reconstruct, epoch_frequency=100)
        ]
        self._model.compile(loss=losses, optimizer=RMSprop(learning_rate=learning_rate))
        self._model.fit(self._timeseries_generator, epochs=epochs, verbose=1, callbacks=callbacks)
        pass

    # https://github.com/Michedev/VAE_anomaly_detection/blob/master/VAE.py
    # not used by default, poor anomaly detection
    def _reconstruction_probability(self, X: pd.DataFrame, debug_plots: bool = False) -> float:
        """
        Probabilistic anomaly score calculation of a time window using multivariate probability density function.
        Turned off by default, but used in the papers.
        Parameters
        ----------
        X: pd.DataFrame
            DataFrame with a time window queried for anomalies.
        debug_plots: bool, optional
            If True, this function plots the input time window and several of its reconstructions.
            Default value is False.
        Returns
        -------
        float
            Returns the reconstruction probability. Anomaly score should be interpreted as 1 - reconstruction_probability
            according to the paper. If the input does not match the expected time window format or contains a NaN value,
            `np.nan` is returned.
        """
        if np.array(X).shape != (self._time_window, self._feature_count) or X.isnull().values.any():
            return np.nan
        L = 100
        mean_latent, logvar_latent = self._model.get_layer('scvae_encoder')(tf.reshape(X, (1, X.shape[0], X.shape[1])))
        mean_latent = tf.reshape(mean_latent, shape=[-1])  # flatten to vector
        logvar_latent = tf.reshape(logvar_latent, shape=[-1])
        latent_distribution = multivariate_normal.rvs(mean=mean_latent, cov=np.diag(np.exp(logvar_latent)),
                                                      size=L)  # logvar vector -> diagonal covariance matrix
        latent_distribution = tf.reshape(latent_distribution, (L, self._latent_dim, 1))
        mean_series, logvar_series = self._model.get_layer('scvae_decoder')(latent_distribution)
        # debug lines for what is the visual generated recosntruction
        if debug_plots:
            print("Real sample")
            plt.plot(X)
            plt.show()
            reconstructing_distributions = [multivariate_normal.rvs(mean=tf.reshape(mean, shape=[-1]),
                                                                    cov=np.diag(np.exp(tf.reshape(logvar, shape=[-1]))),
                                                                    size=3) for mean, logvar in
                                            zip(mean_series, logvar_series)]
            print("Sampled from network output")
            for i in range(3):
                plt.plot(np.reshape(reconstructing_distributions[i][0], (X.shape[0], X.shape[1])))
                plt.show()
        probabilities = [multivariate_normal.pdf(tf.reshape(X, shape=[-1]), mean=tf.reshape(mean, shape=[-1]),
                                                 cov=np.diag(np.exp(tf.reshape(logvar, shape=[-1])))) for mean, logvar
                         in zip(mean_series, logvar_series)]
        return np.mean(probabilities)


    def _reconstruction_score_mse(self, X: pd.DataFrame, debug_plots: bool = False) -> np.ndarray:
        """
        Probabilistic anomaly score calculation of a time window using multivariate probability density function.
        Used by default, implemented as an alternative that better reflects this model's training.
        Parameters
        ----------
        X: pd.DataFrame
            DataFrame with a time window queried for anomalies.
        debug_plots: bool, optional
            If True, this function plots the input time window and several of its reconstructions.
            Default value is False.
        Returns
        -------
        float
            The returned value represents anomaly score. It is calculated as the average MSE of 100 reconstructions.
            We encode the input into a latent distribution, sample ten times from it and then feed each of these ten
            samples into decoder. Altogether we obtain ten time series distributions on the decoder's output.
            Again, we sample ten times from each of these and obtain 100 samples. Then we calculate average MSE
            against the input and return it.
            If the input does not match the expected time window format or contains a NaN value,
            `np.nan` is returned.
        """
        if np.array(X).shape != (self._time_window, self._feature_count) or X.isnull().values.any():
            return np.nan
        L = 10
        mean_latent, logvar_latent = self._model.get_layer('scvae_encoder')(tf.reshape(X, (1, X.shape[0], X.shape[1])))
        mean_latent = tf.reshape(mean_latent, shape=[-1])  # flatten to vector
        logvar_latent = tf.reshape(logvar_latent, shape=[-1])
        latent_sampled = multivariate_normal.rvs(mean=mean_latent, cov=np.diag(np.exp(logvar_latent)),
                                                 size=L)  # logvar vector -> diagonal covariance matrix for independent vars
        latent_sampled = tf.reshape(latent_sampled, (L, self._latent_dim, 1))
        mean_series, logvar_series = self._model.get_layer('scvae_decoder')(latent_sampled)
        series_sampled = [multivariate_normal.rvs(mean=cur_mean, cov=np.diag(np.exp(cur_logvar)), size=L) for
                          cur_mean, cur_logvar in zip(mean_series, logvar_series)]
        # debug lines for what is the visual generated reconstruction
        if debug_plots:
            print("Real sample")
            plt.plot(X)
            plt.show()
            print("Sampled from network output")
            for i in range(3):
                plt.plot(np.reshape(series_sampled[i][0], (X.shape[0], X.shape[1])))
                plt.show()
        error_scores = [[self._mse(X, np.reshape(reconstruction, (X.shape[0], X.shape[1]))) for reconstruction in case]
                        for case in series_sampled]
        return np.mean(error_scores)