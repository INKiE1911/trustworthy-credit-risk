"""Tests for creditrisk.data.to_parquet. Uses a tiny made-up table, so no real data is needed."""

import numpy as np
import pandas as pd

from creditrisk.data.to_parquet import memory_mb, shrink


def make_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [100002, 456255, 300000],
            "TARGET": [0, 1, 0],
            "DAYS_BIRTH": [-9461, -25229, -16000],
            "AMT_CREDIT": [406597.5, 1234567.8, np.nan],
            "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans", "Cash loans"],
        }
    )


def test_shrink_picks_small_types():
    df = shrink(make_table())
    assert df["SK_ID_CURR"].dtype == "int32"
    assert df["TARGET"].dtype == "int8"
    assert df["DAYS_BIRTH"].dtype == "int16"
    assert df["AMT_CREDIT"].dtype == "float32"
    assert isinstance(df["NAME_CONTRACT_TYPE"].dtype, pd.CategoricalDtype)


def test_shrink_keeps_the_values():
    original, small = make_table(), shrink(make_table())
    for col in ["SK_ID_CURR", "TARGET", "DAYS_BIRTH", "AMT_CREDIT"]:
        np.testing.assert_allclose(
            small[col].to_numpy(dtype=float), original[col].to_numpy(dtype=float), rtol=1e-6
        )
    text_col = "NAME_CONTRACT_TYPE"
    assert small[text_col].astype(str).tolist() == original[text_col].tolist()


def test_shrink_saves_memory():
    big = pd.concat([make_table()] * 1000, ignore_index=True)
    before = memory_mb(big)
    assert memory_mb(shrink(big)) < before


def test_parquet_round_trip_keeps_types_and_values(tmp_path):
    small = shrink(make_table())
    path = tmp_path / "table.parquet"
    small.to_parquet(path, index=False)
    back = pd.read_parquet(path)
    pd.testing.assert_frame_equal(back, small)