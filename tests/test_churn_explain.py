"""churn_explain.py 测试 — 模型解释。"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from churn_explain import (
    FeatureContribution,
    compute_shap_values,
    permutation_importance_table,
    top_features_for_customer,
)


@pytest.fixture
def fitted_lr_and_rf():
    rng = np.random.default_rng(0)
    n = 500
    X = pd.DataFrame({
        "recency_days": rng.uniform(0, 365, n),
        "frequency": rng.poisson(3, n),
        "monetary": rng.gamma(2, 50, n),
        "age": rng.uniform(18, 75, n),
    })
    # 给 recency 更大的真实权重，方便测出来
    logits = -0.005 * X["recency_days"] + 0.3 * X["frequency"] - 1.0
    p = 1 / (1 + np.exp(-logits))
    y = pd.Series((rng.random(n) < p).astype(int))

    lr = LogisticRegression(max_iter=200).fit(X, y)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
    return X, y, lr, rf


def test_permutation_importance_for_lr(fitted_lr_and_rf):
    X, y, lr, _ = fitted_lr_and_rf
    table = permutation_importance_table(lr, X, y, n_repeats=5)
    assert {"feature", "importance_mean", "importance_std"}.issubset(table.columns)
    # recency_days 与 frequency 都是真信号，至少有一个进 top-2
    top2 = set(table.head(2)["feature"])
    assert top2 & {"recency_days", "frequency"}, f"top2={top2}"
    # 噪声特征 age 不应该排第一
    assert table.iloc[0]["feature"] != "age"


def test_permutation_importance_for_rf(fitted_lr_and_rf):
    X, y, _, rf = fitted_lr_and_rf
    table = permutation_importance_table(rf, X, y, n_repeats=5)
    assert len(table) == X.shape[1]
    # 全部 importance 应为有限实数
    assert np.isfinite(table["importance_mean"]).all()


def test_top_features_for_customer_returns_objects(fitted_lr_and_rf):
    X, _, lr, _ = fitted_lr_and_rf
    means = X.mean()
    contribs = top_features_for_customer(
        lr, X.iloc[0], list(X.columns), feature_means=means, top_k=3
    )
    assert len(contribs) == 3
    for c in contribs:
        assert isinstance(c, FeatureContribution)
        assert c.feature in X.columns


def test_top_features_for_rf(fitted_lr_and_rf):
    X, _, _, rf = fitted_lr_and_rf
    means = X.mean()
    contribs = top_features_for_customer(rf, X.iloc[10], list(X.columns), feature_means=means, top_k=2)
    assert len(contribs) == 2


def test_unknown_model_type_raises(fitted_lr_and_rf):
    X, y, _, _ = fitted_lr_and_rf

    class NoWeights:
        pass

    with pytest.raises(TypeError):
        top_features_for_customer(NoWeights(), X.iloc[0], list(X.columns))


def test_shap_returns_none_if_not_installed(fitted_lr_and_rf):
    """compute_shap_values 在 shap 未安装时应优雅返回 None，而不是 raise。"""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        X, _, _, rf = fitted_lr_and_rf
        result = compute_shap_values(rf, X.head(20))
        assert result is None
    finally:
        builtins.__import__ = real_import
