from typing import Optional, List

import pandas as pd

from time_series_anomaly_detection.abstractions import (
    TimeSeriesAnomalyDetector
)


class TemplateDetector(TimeSeriesAnomalyDetector):
    """
    Time series anomaly detector template.

    Parameters
    ----------
    id_columns: List[str], optional
        ID columns used to identify individual time series. Should be specified
        in case the detector is going to be working with multiple time series
        so that the detector can separate them. In case they are not specified,
        the provided data is regarded as a single time series.
    """

    def __init__(
        self,
        id_columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self._id_columns = id_columns

    def predict_anomaly_scores(self, X: pd.DataFrame, *args,
                               **kwargs) -> pd.Series:
        # TODO: return predicted anomaly scores for the given samples
        pass

    def fit(self, X: pd.DataFrame, *args, **kwargs) -> None:
        # TODO: perform training
        pass
