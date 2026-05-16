"""bgnbd.py 测试 — 概率 CLV 模型。"""
import numpy as np
import pandas as pd
import pytest

from bgnbd import (
    BGNBDModel,
    GammaGammaModel,
    clv,
    rfm_to_bgnbd_input,
)


def _synth_rfm(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """合成符合 BG/NBD 假设的客户数据。"""
    rng = np.random.default_rng(seed)
    T = rng.uniform(60, 365, n)
    # 客户有不同 lambda（购买率）
    lam = rng.gamma(2.0, 0.05, n)
    # 流失概率
    p = rng.beta(1.0, 5.0, n)

    rows = []
    for i in range(n):
        # 用泊松+几何近似生成 frequency, recency
        n_purchases = rng.poisson(lam[i] * T[i])
        if n_purchases == 0:
            rows.append({"frequency": 0, "recency": 0.0, "T": T[i], "monetary_value": 0.0})
            continue
        # 在 [0, T[i]] 中均匀生成购买时刻
        times = np.sort(rng.uniform(0, T[i], n_purchases))
        # 模拟流失：每次交易后有 p 概率流失
        dropouts = rng.random(n_purchases) < p[i]
        if dropouts.any():
            kill_idx = np.argmax(dropouts)
            times = times[: kill_idx + 1]
        if len(times) <= 1:
            rows.append({"frequency": 0, "recency": 0.0, "T": T[i], "monetary_value": 0.0})
            continue
        rows.append({
            "frequency": len(times) - 1,
            "recency": float(times[-1] - times[0]),
            "T": float(T[i] - times[0]),
            "monetary_value": float(rng.gamma(5.0, 10.0)),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# BG/NBD
# ----------------------------------------------------------------------------

def test_bgnbd_fits_synthetic_data():
    df = _synth_rfm(n=300, seed=1)
    df = df[df["frequency"] >= 0]  # 全部都符合
    m = BGNBDModel()
    m.fit(df)
    assert m.params_ is not None
    assert m.params_.r > 0
    assert m.params_.alpha > 0
    assert m.params_.a > 0
    assert m.params_.b > 0


def test_bgnbd_predict_purchases_non_negative():
    df = _synth_rfm(n=200, seed=2)
    m = BGNBDModel().fit(df)
    preds = m.predict_purchases(t=30, df=df)
    assert (preds >= 0).all()
    assert np.isfinite(preds).all()


def test_bgnbd_predict_alive_in_unit_interval():
    df = _synth_rfm(n=200, seed=3)
    m = BGNBDModel().fit(df)
    alive = m.predict_alive(df)
    assert (alive >= 0).all() and (alive <= 1).all()
    # frequency = 0 的客户没有"还活着"信号 → 应该 = 1
    zero_mask = df["frequency"] == 0
    if zero_mask.any():
        assert np.allclose(alive[zero_mask], 1.0)


def test_bgnbd_more_recent_buyers_more_likely_alive():
    """recency 接近 T 的客户应当被 model 判定为更可能仍活跃。"""
    df = _synth_rfm(n=400, seed=4)
    df_active = df[df["frequency"] >= 2].copy()
    if len(df_active) < 20:
        pytest.skip("合成数据太稀疏")
    m = BGNBDModel().fit(df)
    alive = m.predict_alive(df_active)
    # ratio = recency / T 越大，alive 概率应该越高
    ratio = df_active["recency"] / df_active["T"]
    corr = np.corrcoef(ratio, alive)[0, 1]
    assert corr > 0.2, f"recency/T 与 P(alive) 应正相关，实际 {corr:.2f}"


def test_bgnbd_invalid_input():
    m = BGNBDModel()
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame({"x": [1, 2]}))  # 缺列
    with pytest.raises(ValueError):
        m.fit(pd.DataFrame({"frequency": [1], "recency": [10], "T": [5]}))  # recency>T


def test_predict_before_fit_raises():
    m = BGNBDModel()
    with pytest.raises(RuntimeError):
        m.predict_purchases(30, pd.DataFrame({"frequency": [1], "recency": [10], "T": [30]}))


# ----------------------------------------------------------------------------
# Gamma-Gamma
# ----------------------------------------------------------------------------

def test_gg_fits_synthetic_monetary():
    df = _synth_rfm(n=400, seed=10)
    df = df[df["frequency"] >= 1].copy()  # GG 仅对 freq >= 1 有意义
    if len(df) < 50:
        pytest.skip("合成数据太稀疏")
    g = GammaGammaModel().fit(df)
    assert g.params_ is not None
    assert g.params_.p > 0
    assert g.params_.q > 0
    assert g.params_.gamma > 0


def test_gg_conditional_mean_between_observed_and_population():
    df = _synth_rfm(n=400, seed=11)
    df = df[df["frequency"] >= 1].copy()
    if len(df) < 50:
        pytest.skip("合成数据太稀疏")
    g = GammaGammaModel().fit(df)
    pred = g.predict_conditional_mean(df)
    assert (pred > 0).all()
    # 预测值应当在合理范围内
    assert pred.mean() > 0


def test_gg_rejects_zero_monetary():
    df = pd.DataFrame({
        "frequency": [3, 5, 2],
        "monetary_value": [10.0, 0.0, 20.0],  # 第二个无效
    })
    with pytest.raises(ValueError):
        GammaGammaModel().fit(df)


# ----------------------------------------------------------------------------
# CLV (端到端)
# ----------------------------------------------------------------------------

def test_clv_combines_two_models():
    df = _synth_rfm(n=400, seed=20)
    df = df[df["frequency"] >= 1].copy()
    if len(df) < 50:
        pytest.skip("合成数据太稀疏")
    bg = BGNBDModel().fit(df)
    gg = GammaGammaModel().fit(df)
    series = clv(bg, gg, df, horizon=365)
    assert series.name == "clv_365d"
    assert (series >= 0).all()
    assert series.shape[0] == len(df)


def test_clv_horizon_monotone():
    """更长的预测窗口应当给出更大或相等的 CLV。"""
    df = _synth_rfm(n=400, seed=21)
    df = df[df["frequency"] >= 1].copy()
    if len(df) < 50:
        pytest.skip("合成数据太稀疏")
    bg = BGNBDModel().fit(df)
    gg = GammaGammaModel().fit(df)
    clv_30 = clv(bg, gg, df, horizon=30)
    clv_365 = clv(bg, gg, df, horizon=365)
    assert (clv_365 >= clv_30 - 1e-6).all()


# ----------------------------------------------------------------------------
# rfm_to_bgnbd_input
# ----------------------------------------------------------------------------

def test_rfm_input_basic():
    transactions = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "B", "C"],
        "date": pd.to_datetime([
            "2024-01-01", "2024-01-15", "2024-02-01",
            "2024-01-10", "2024-03-01",
            "2024-02-20",
        ]),
        "amount": [10, 20, 15, 30, 25, 40],
    })
    out = rfm_to_bgnbd_input(
        transactions, observation_end=pd.Timestamp("2024-04-01")
    )
    out_by_cust = out.set_index("customer_id")
    assert out_by_cust.loc["A", "frequency"] == 2  # 3 次 - 第 1 次
    assert out_by_cust.loc["A", "recency"] == 31
    assert out_by_cust.loc["A", "T"] == 91
    assert out_by_cust.loc["C", "frequency"] == 0
    assert out_by_cust.loc["C", "monetary_value"] == 0.0
    assert out_by_cust.loc["A", "monetary_value"] == pytest.approx(15.0)
