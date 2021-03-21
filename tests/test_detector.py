from typing import Tuple, Iterable, List

import numpy as np
import pandas as pd

from time_series_anomaly_detection.detector import TemplateDetector

DETECTOR_CLASS = TemplateDetector


def _get_simple_random_dataset(
    size: int,
    n_data_cols: int,
    n_id_cols: int = 2,
    multiple_series: bool = False,
    train_nan_locs: Iterable[int] = (),
    test_nan_locs: Iterable[int] = ()
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    train_nan_locs = list(train_nan_locs)
    X = np.random.randn(size, n_data_cols) * 0.1 + 10
    X[train_nan_locs, 0] = np.nan

    data_cols = [f'col_{i}' for i in range(n_data_cols)]
    id_cols = [f'id_{i}' for i in range(n_id_cols)]

    X_df = pd.DataFrame(data=X, columns=data_cols)

    X_test = np.random.randn(size, n_data_cols) + 10
    X_test[test_nan_locs, 0] = np.nan
    X_test_df = pd.DataFrame(data=X_test, columns=data_cols)

    for idx, id_col in enumerate(id_cols):
        ids = (
            [2 * idx] * (size // 2) +
            [2 * idx + multiple_series] * (size // 2)
        )
        X_df[id_col] = ids
        X_test_df[id_col] = ids

    return X_df, X_test_df, id_cols


def test_single_univariate_time_series():
    X_df, X_test_df, id_columns = _get_simple_random_dataset(
        size=20, n_data_cols=1
    )
    detector = DETECTOR_CLASS(id_columns=id_columns)
    detector.fit(X_df)
    scores = detector.predict_anomaly_scores(X_test_df)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 20


def test_multiple_univariate_time_series():
    X_df, X_test_df, id_columns = _get_simple_random_dataset(
        size=20, n_data_cols=1, multiple_series=True
    )
    detector = DETECTOR_CLASS(id_columns=id_columns)
    detector.fit(X_df)
    scores = detector.predict_anomaly_scores(X_test_df)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 20


def test_single_multivariate_time_series():
    X_df, X_test_df, id_columns = _get_simple_random_dataset(
        size=20, n_data_cols=5
    )
    detector = DETECTOR_CLASS(id_columns=id_columns)
    detector.fit(X_df)
    scores = detector.predict_anomaly_scores(X_test_df)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 20


def test_multiple_multivariate_time_series():
    X_df, X_test_df, id_columns = _get_simple_random_dataset(
        size=20, n_data_cols=5, multiple_series=True
    )
    detector = DETECTOR_CLASS(id_columns=id_columns)
    detector.fit(X_df)
    scores = detector.predict_anomaly_scores(X_test_df)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 20


def test_nans():
    nan_locs = (1, 5, 6, 23)
    X_df, X_test_df, id_columns = _get_simple_random_dataset(
        size=20,
        n_data_cols=1,
        multiple_series=True,
        train_nan_locs=nan_locs,
        test_nan_locs=nan_locs
    )
    detector = DETECTOR_CLASS(id_columns=id_columns)
    detector.fit(X_df)
    scores = detector.predict_anomaly_scores(X_test_df)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 20
