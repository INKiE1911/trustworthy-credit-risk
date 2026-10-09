"""Step 9: the scoring service, demo bundle, API and Streamlit app, all on fake data."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import make_fake_application, make_fake_history
from pydantic import ValidationError

from creditrisk.features.application import SENSITIVE
from creditrisk.features.build import KEY_COLS, build_features
from creditrisk.models import baselines
from creditrisk.models.final import save_final_model
from creditrisk.serving.demo import make_demo_bundle
from creditrisk.serving.service import (
    RAW_PREFIX,
    Changes,
    ScoringService,
    feature_label,
    format_value,
    sigmoid,
)

APP_FILE = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    train = make_fake_application(2_000, True, seed=11)
    test = make_fake_application(300, False, seed=12, first_id=800_000)
    tables = {"application_train": train, "application_test": test,
              **make_fake_history(np.r_[train["SK_ID_CURR"], test["SK_ID_CURR"]], seed=13)}

    def read(name, columns=None):
        return tables[name] if columns is None else tables[name][columns]

    features = build_features(verbose=False, read=read)
    columns = [c for c in features.columns if c not in KEY_COLS + SENSITIVE]
    labelled = features[features["is_test"] == 0]
    model = baselines.lightgbm(seed=0, n_estimators=60, n_jobs=1)
    model.fit(labelled[columns], labelled["TARGET"].astype(int))
    folder = tmp_path_factory.mktemp("serving")
    save_final_model(model, columns, {"n_estimators": 60, "cv_roc_auc": 0.5},
                     model_dir=folder / "models")
    bundle = make_demo_bundle(features, test, n=50, seed=0)
    bundle.to_parquet(folder / "demo.parquet", index=False)
    service = ScoringService(model_dir=folder / "models", demo_path=folder / "demo.parquet")
    return service, model, bundle.set_index("SK_ID_CURR"), folder


def test_bundle_holds_raw_and_feature_columns(served):
    service, _, bundle, _ = served
    assert len(bundle) == 50 and len(service.applicant_ids) == 50
    assert sum(c.startswith(RAW_PREFIX) for c in bundle.columns) == 120  # all raw test columns
    assert set(service.columns) <= set(bundle.columns)
    assert (bundle["is_test"] == 1).all()  # only Kaggle test applicants, never our splits


def test_no_change_rebuilds_exactly_the_stored_features(served):
    service, model, bundle, _ = served
    for applicant_id in service.applicant_ids[:25]:
        rebuilt = service.build_features(applicant_id)
        stored = bundle.loc[[applicant_id], service.columns]
        pd.testing.assert_frame_equal(rebuilt, stored, check_exact=False, rtol=1e-6)
        expected = model.predict_proba(stored)[0, 1]
        assert abs(service.score(applicant_id).probability - expected) < 1e-6


def test_reasons_add_up_to_the_score(served):
    service, model, _, _ = served
    applicant_id = service.applicant_ids[3]
    result = service.score(applicant_id)
    X = service.build_features(applicant_id)
    contributions = model.predict(X, pred_contrib=True)[0]
    assert abs(sigmoid(contributions.sum()) - result.probability) < 1e-5
    up = [r.impact for r in result.raises_risk]
    down = [r.impact for r in result.lowers_risk]
    assert len(up) <= 4 and len(down) <= 4 and all(v > 0 for v in up) and all(v < 0 for v in down)
    assert up == sorted(up, reverse=True) and down == sorted(down)


def test_changes_rebuild_the_dependent_features(served):
    service, _, bundle, _ = served
    applicant_id = service.applicant_ids[0]
    before = service.build_features(applicant_id)
    after = service.build_features(applicant_id, Changes(income=500_000, age_years=60,
                                                         ext_source_2=0.9))
    assert after["APP_AMT_INCOME_TOTAL"].iloc[0] == pytest.approx(500_000)
    credit = after["APP_AMT_CREDIT"].iloc[0]
    assert after["APP_CREDIT_TO_INCOME"].iloc[0] == pytest.approx(credit / 500_000, rel=1e-5)
    assert after["APP_AGE_YEARS"].iloc[0] == pytest.approx(60, abs=0.01)
    assert after["APP_EXT_SOURCE_2"].iloc[0] == pytest.approx(0.9, abs=1e-6)
    debt = bundle.loc[applicant_id, "BUR_ACTIVE_DEBT_TOTAL"]
    if pd.notna(debt):
        assert after["BUR_DEBT_TO_INCOME"].iloc[0] == pytest.approx(debt / 500_000, rel=1e-4)
    history = [c for c in service.columns if c.startswith(("INST_", "POS_", "CC_", "BB_"))]
    pd.testing.assert_frame_equal(after[history], before[history])  # history stays the same
    no_record = bundle.index[bundle[RAW_PREFIX + "DAYS_EMPLOYED"] == 365243]
    if len(no_record):
        X = service.build_features(int(no_record[0]), Changes(years_employed=3))
        assert X["APP_DAYS_EMPLOYED_ANOM"].iloc[0] == 0
        assert X["APP_EMPLOYED_YEARS"].iloc[0] == pytest.approx(3, abs=0.01)


def test_bad_input_is_rejected(served):
    service, _, _, _ = served
    for bad in ({"ext_source_1": 1.5}, {"age_years": 10}, {"income": -5}, {"salary": 1}):
        with pytest.raises(ValidationError):
            Changes(**bad)
    with pytest.raises(KeyError):
        service.score(1)
    with pytest.raises(ValueError, match="age minus 14"):
        service.score(service.applicant_ids[0], Changes(age_years=25, years_employed=20))


def test_profile_and_labels(served):
    service, _, bundle, _ = served
    profile = service.profile(service.applicant_ids[1])
    assert set(profile.current) == {"income", "loan_amount", "annuity", "age_years",
                                    "years_employed", "ext_source_1", "ext_source_2",
                                    "ext_source_3"}
    assert len(profile.history) == 5
    no_record = bundle.index[bundle[RAW_PREFIX + "DAYS_EMPLOYED"] == 365243]
    if len(no_record):
        assert service.profile(int(no_record[0])).current["years_employed"] is None
    assert feature_label("APP_EXT_MEAN") == "External credit scores (average)"
    assert feature_label("INST_LATE_SHARE").startswith("Repayment history:")
    assert format_value(np.nan) == "missing" and format_value(123456.7) == "123,457"
    assert format_value(0.12345) == "0.123" and format_value("Working") == "Working"


@pytest.mark.skipif(importlib.util.find_spec("httpx") is None, reason="httpx is not installed")
@pytest.mark.filterwarnings("ignore:Using `httpx`")  # a notice from newer Starlette versions
def test_api(served, monkeypatch):
    from fastapi.testclient import TestClient

    from creditrisk.serving import api

    service, _, _, folder = served
    monkeypatch.setenv("CREDITRISK_MODEL_DIR", str(folder / "models"))
    monkeypatch.setenv("CREDITRISK_DEMO_PATH", str(folder / "demo.parquet"))
    api.get_service.cache_clear()
    client = TestClient(api.app)
    applicant_id = service.applicant_ids[0]
    assert client.get("/health").json()["applicants"] == 50
    assert client.get("/applicants", params={"limit": 5}).json()["applicant_ids"] == \
        service.applicant_ids[:5]
    assert client.get(f"/applicants/{applicant_id}").status_code == 200
    assert client.get("/applicants/1").status_code == 404
    ok = client.post("/score", json={"applicant_id": applicant_id, "changes": {"income": 1e5}})
    assert ok.status_code == 200
    body = ok.json()
    assert 0 < body["probability"] < 1 and body["changed"] == {"income": 100000.0}
    assert {"raises_risk", "lowers_risk", "base_probability", "model"} <= set(body)
    bad = client.post("/score", json={"applicant_id": applicant_id,
                                      "changes": {"ext_source_1": 1.5}})
    assert bad.status_code == 422
    assert client.post("/score", json={"applicant_id": 1}).status_code == 404
    clash = client.post("/score", json={"applicant_id": applicant_id,
                                        "changes": {"age_years": 25, "years_employed": 20}})
    assert clash.status_code == 422
    api.get_service.cache_clear()


@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None,
                    reason="streamlit is not installed")
def test_streamlit_app_runs(served, monkeypatch):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    service, _, _, folder = served
    monkeypatch.setenv("CREDITRISK_MODEL_DIR", str(folder / "models"))
    monkeypatch.setenv("CREDITRISK_DEMO_PATH", str(folder / "demo.parquet"))
    monkeypatch.delenv("SCORING_API_URL", raising=False)
    st.cache_resource.clear()
    app = AppTest.from_file(str(APP_FILE), default_timeout=60).run()
    assert not app.exception
    assert app.metric[0].value.endswith("%") and app.metric[0].delta in (None, "")
    first = service.applicant_ids[0]
    app.slider(key=f"income_{first}_0").set_value(600_000.0).run()
    assert not app.exception
    assert "points" in app.metric[0].delta  # the what-if score is compared with the original
    st.cache_resource.clear()
