from abc import abstractmethod

import pandas as pd


class TimeSeriesAnomalyDetector:
    """
    Base class for anomaly detectors working on time series data.
    """

    @abstractmethod
    def predict_anomaly_scores(
        self, X: pd.DataFrame, *args, **kwargs
    ) -> pd.Series:
        """
        Predicts an anomaly score of the input samples. Samples should be
        ordered by their timestamps.

        An anomaly score is a measure of normality. The higher the score,
        the more abnormal the measured sample is.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_columns)
            The samples whose anomaly scores are to be predicted.
            The columns contain samples' features and possibly
            samples' identifiers.

        Returns
        -------
        scores : pd.Series, shape (n_samples,)
            The anomaly score of the input samples. The higher, the more
            abnormal.
        """

    @abstractmethod
    def fit(self, X: pd.DataFrame, *args, **kwargs) -> None:
        """
        Fits the anomaly detector according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_columns)
            The training samples. The columns contain samples' features and
            possibly samples' identifiers.
        """
