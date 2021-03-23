from typing import Optional, Iterable

import pandas as pd

from time_series_anomaly_detection.abstractions import (
    TimeSeriesAnomalyDetector
)


class TemplateDetector(TimeSeriesAnomalyDetector):
    """
    Time series anomaly detector template.

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
    """

    def __init__(
        self,
        id_columns: Optional[Iterable[str]] = None,
    ):
        super().__init__()
        self._id_columns = id_columns

    def predict_anomaly_scores(
        self, X: pd.DataFrame, *args, **kwargs
    ) -> pd.Series:
        # TODO: return predicted anomaly scores for the given samples
        pass

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> None:
        # TODO: perform training
        pass
