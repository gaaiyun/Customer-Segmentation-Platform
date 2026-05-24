"""clv_uncertainty.py 测试 —— bootstrap + parameter perturbation。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bgnbd import BGNBDModel, GammaGammaModel
from clv_uncertainty import (
    CLVBootstrapResult, bootstrap_clv, parameter_perturbation_clv,
)


def _synth_rfm(n: int = 80, seed: int = 0) -> pd.DataFrame:
    """合成符合 BG/NBD 假设的客户数据，足够稳定 fit。"""
    rng = np.random.default_rng(seed)
    T = rng.uniform(120, 365, n)
    lam = rng.gamma(2.0, 0.05, n)
    p = rng.beta(1.0, 5.0, n)

    rows = []
    for i in range(n):
        n_purch = rng.poisson(lam[i] * T[i])
        if n_purch == 0:
            rows.append({"frequency": 0, "recency": 0.0, "T": T[i],
                          "monetary_value": 0.0})
            continue
        times = np.sort(rng.uniform(0, T[i], n_purch))
        dropouts = rng.random(n_purch) < p[i]
        if dropouts.any():
            kill = np.argmax(dropouts)
            times = times[: kill + 1]
        if len(times) <= 1:
            rows.append({"frequency": 0, "recency": 0.0, "T": T[i],
                          "monetary_value": 0.0})
            continue
        rows.append({
            "frequency": len(times) - 1,
            "recency": float(times[-1] - times[0]),
            "T": float(T[i] - times[0]),
            "monetary_value": float(rng.gamma(5.0, 10.0)),
        })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def synth_rfm() -> pd.DataFrame:
    return _synth_rfm(n=80, seed=42)


# --- bootstrap_clv ---------------------------------------------------

def test_bootstrap_returns_result(synth_rfm):
    result = bootstrap_clv(synth_rfm, n_bootstrap=15, seed=0)
    assert isinstance(result, CLVBootstrapResult)
    assert result.n_customers == len(synth_rfm)


def test_bootstrap_customer_summary_columns(synth_rfm):
    result = bootstrap_clv(synth_rfm, n_bootstrap=15, alpha=0.10, seed=0)
    cols = set(result.customer_clv.columns)
    assert "clv_mean" in cols
    assert "clv_std" in cols
    assert "clv_p50" in cols
    # alpha=0.10 → percentile p05 / p95
    assert "clv_p05" in cols
    assert "clv_p95" in cols


def test_bootstrap_customer_clv_index_matches_input(synth_rfm):
    result = bootstrap_clv(synth_rfm, n_bootstrap=10, seed=0)
    assert list(result.customer_clv.index) == list(synth_rfm.index)


def test_bootstrap_total_clv_within_ci(synth_rfm):
    result = bootstrap_clv(synth_rfm, n_bootstrap=15, seed=0)
    assert result.total_clv_ci_low <= result.total_clv_mean <= result.total_clv_ci_high


def test_bootstrap_rejects_low_n_bootstrap(synth_rfm):
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_clv(synth_rfm, n_bootstrap=5)


def test_bootstrap_rejects_bad_alpha(synth_rfm):
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_clv(synth_rfm, n_bootstrap=10, alpha=1.5)


def test_bootstrap_rejects_bad_columns():
    bad = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="frequency"):
        bootstrap_clv(bad, n_bootstrap=10)


def test_bootstrap_to_dict_serializable(synth_rfm):
    import json
    result = bootstrap_clv(synth_rfm, n_bootstrap=12, seed=0)
    json.dumps(result.to_dict())


def test_bootstrap_mean_close_to_point_estimate(synth_rfm):
    """Bootstrap mean 应与 single-fit 点估计相近（在合理误差内）。"""
    bg = BGNBDModel(); bg.fit(synth_rfm)
    gg_input = synth_rfm[synth_rfm["frequency"] >= 1].reset_index(drop=True)
    gg = GammaGammaModel(); gg.fit(gg_input)
    from bgnbd import clv
    point = clv(bg, gg, synth_rfm, horizon=365.0).sum()

    result = bootstrap_clv(synth_rfm, n_bootstrap=20, seed=0)
    # 应在 CI 内或接近（bootstrap mean 是有偏估计，但应同量级）
    assert result.total_clv_mean > 0
    assert abs(result.total_clv_mean - point) / max(point, 1) < 1.0


def test_bootstrap_with_discount_rate(synth_rfm):
    no_discount = bootstrap_clv(synth_rfm, n_bootstrap=12,
                                 discount_rate=0.0, seed=0)
    with_discount = bootstrap_clv(synth_rfm, n_bootstrap=12,
                                   discount_rate=0.5, seed=0)
    # 折现率高 → mean CLV 更低
    assert with_discount.total_clv_mean < no_discount.total_clv_mean


def test_bootstrap_horizon_scales_clv(synth_rfm):
    short = bootstrap_clv(synth_rfm, horizon=90, n_bootstrap=12, seed=0)
    long_ = bootstrap_clv(synth_rfm, horizon=365, n_bootstrap=12, seed=0)
    # 1 年的 CLV 应大于 90 天
    assert long_.total_clv_mean > short.total_clv_mean


# --- parameter_perturbation_clv -------------------------------------

def test_perturbation_returns_dataframe(synth_rfm):
    bg = BGNBDModel(); bg.fit(synth_rfm)
    gg_input = synth_rfm[synth_rfm["frequency"] >= 1].reset_index(drop=True)
    gg = GammaGammaModel(); gg.fit(gg_input)

    df = parameter_perturbation_clv(bg, gg, synth_rfm,
                                     n_perturbations=30, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(synth_rfm)
    assert set(["clv_mean", "clv_p05", "clv_p95", "clv_std"]).issubset(df.columns)


def test_perturbation_zero_noise_collapses_to_point(synth_rfm):
    """relative_noise → 0：所有 perturbation 应该几乎相等。

    （由于 rng.normal(0, eps) 几乎为 0，所有 CLV 应几乎一致。）
    """
    bg = BGNBDModel(); bg.fit(synth_rfm)
    gg_input = synth_rfm[synth_rfm["frequency"] >= 1].reset_index(drop=True)
    gg = GammaGammaModel(); gg.fit(gg_input)

    df = parameter_perturbation_clv(bg, gg, synth_rfm,
                                     n_perturbations=20,
                                     relative_noise=1e-6, seed=0)
    # std 应该非常小
    assert (df["clv_std"] < 0.01 * df["clv_mean"].abs() + 0.01).all()


def test_perturbation_restores_original_params(synth_rfm):
    """扰动后原模型参数应被恢复。"""
    bg = BGNBDModel(); bg.fit(synth_rfm)
    gg_input = synth_rfm[synth_rfm["frequency"] >= 1].reset_index(drop=True)
    gg = GammaGammaModel(); gg.fit(gg_input)

    orig_r = bg.params_.r
    orig_p = gg.params_.p
    parameter_perturbation_clv(bg, gg, synth_rfm,
                                n_perturbations=10, relative_noise=0.1, seed=0)
    assert bg.params_.r == orig_r
    assert gg.params_.p == orig_p


def test_perturbation_unfit_raises():
    bg = BGNBDModel()
    gg = GammaGammaModel()
    with pytest.raises(ValueError, match="fit"):
        parameter_perturbation_clv(bg, gg, pd.DataFrame({
            "frequency": [1, 2], "recency": [10, 20], "T": [50, 60],
            "monetary_value": [10, 20],
        }))


def test_perturbation_rejects_zero_noise():
    bg = BGNBDModel()
    # 强制赋一个 fake params 让 bg.params_ 不是 None
    from bgnbd import BGNBDParams
    bg.params_ = BGNBDParams(r=1, alpha=1, a=1, b=1)
    gg = GammaGammaModel()
    from bgnbd import GammaGammaParams
    gg.params_ = GammaGammaParams(p=1, q=2, gamma=1)
    with pytest.raises(ValueError, match="relative_noise"):
        parameter_perturbation_clv(bg, gg,
                                    pd.DataFrame({"frequency": [1], "recency": [10],
                                                   "T": [50], "monetary_value": [10]}),
                                    relative_noise=0)


def test_perturbation_higher_noise_wider_ci(synth_rfm):
    bg = BGNBDModel(); bg.fit(synth_rfm)
    gg_input = synth_rfm[synth_rfm["frequency"] >= 1].reset_index(drop=True)
    gg = GammaGammaModel(); gg.fit(gg_input)

    low = parameter_perturbation_clv(bg, gg, synth_rfm,
                                      n_perturbations=30,
                                      relative_noise=0.02, seed=0)
    high = parameter_perturbation_clv(bg, gg, synth_rfm,
                                       n_perturbations=30,
                                       relative_noise=0.20, seed=0)
    # 高噪声 → std 更大
    assert high["clv_std"].mean() > low["clv_std"].mean()
