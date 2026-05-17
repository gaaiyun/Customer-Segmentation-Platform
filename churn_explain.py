"""
流失模型的解释性 — permutation importance + (可选) SHAP。

v1 的 `churn_predictor.py` 训练了 LR / RF / GBDT，但只输出 metric 与
ranking，没有"为什么这个客户被标为流失"的解释。本模块补上：

- `permutation_importance_table()`：sklearn 自带的 PI，列出每个特征对
  AUC / accuracy 的影响（模型无关，零额外依赖）
- `top_features_for_customer()`：给单个客户的 prediction 拆解贡献
  - 默认：用「该客户特征值 - 全体均值」× 模型系数（仅对线性模型）
  - 高级：装了 `shap` 后调用 TreeExplainer / LinearExplainer
- `compute_shap_values()`：可选。requires `pip install shap`

设计考虑
-------
- 不强制要 shap：很多场景下 PI + 简单 contribution 足够
- 不接管训练：你已经训好的 sklearn 模型直接传进来就行
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class FeatureContribution:
    feature: str
    value: float
    contribution: float

    def to_dict(self) -> dict:
        return {"feature": self.feature, "value": self.value, "contribution": self.contribution}


def permutation_importance_table(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    scoring: str = "roc_auc",
    random_state: Optional[int] = 0,
) -> pd.DataFrame:
    """
    返回每个特征对 scoring 的 permutation importance。

    Parameters
    ----------
    model : 已 fit 的 sklearn-compatible classifier（有 predict_proba）
    X, y : 评估集（不应是训练集）
    scoring : 'roc_auc' / 'accuracy' / 'f1' 等
    """
    if not hasattr(model, "predict_proba"):
        scoring = "accuracy"
    r = permutation_importance(
        model, X, y, n_repeats=n_repeats, scoring=scoring, random_state=random_state, n_jobs=1
    )
    out = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std": r.importances_std,
    }).sort_values("importance_mean", ascending=False)
    return out.reset_index(drop=True)


def top_features_for_customer(
    model: Any,
    row: pd.Series,
    feature_names: Sequence[str],
    feature_means: Optional[pd.Series] = None,
    top_k: int = 5,
) -> List[FeatureContribution]:
    """
    线性 / 树模型通用的"单客户解释"。

    对线性模型：contribution = coef * (x - x_mean)
    对树模型：用 feature_importances_ × (x - x_mean) 近似（不精确，但够看趋势）
    """
    if feature_means is None:
        feature_means = pd.Series(0.0, index=feature_names)

    coefs = _extract_feature_weights(model, feature_names)
    deltas = row[feature_names].values - feature_means[feature_names].values
    contributions = coefs * deltas

    df = pd.DataFrame({
        "feature": feature_names,
        "value": row[feature_names].values,
        "contribution": contributions,
    })
    df["abs"] = df["contribution"].abs()
    df = df.sort_values("abs", ascending=False).head(top_k)
    return [
        FeatureContribution(
            feature=str(r.feature), value=float(r.value), contribution=float(r.contribution)
        )
        for r in df.itertuples()
    ]


def _extract_feature_weights(model: Any, feature_names: Sequence[str]) -> np.ndarray:
    """从不同类型的 sklearn 模型里抽出"每特征权重"。"""
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim == 2:
            coef = coef[0]  # 二分类
        return coef
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    raise TypeError(f"无法从 {type(model)} 中抽取特征权重；请改用 SHAP")


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> Optional[Dict[str, np.ndarray]]:
    """
    （可选）用 shap 给一批样本算 SHAP 值。

    依赖：pip install shap
    返回 None 表示 shap 未安装；调用方应回退到 `permutation_importance_table`。
    """
    try:
        import shap  # type: ignore
    except ImportError:
        return None

    if len(X) > max_samples:
        X = X.sample(max_samples, random_state=0)

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    else:
        # 线性模型
        explainer = shap.LinearExplainer(model, X)
        shap_values = explainer.shap_values(X)

    sv = np.array(shap_values)
    # 二分类的 TreeExplainer 可能返回 list of [neg, pos]，取 pos
    if sv.ndim == 3:
        sv = sv[1]

    return {
        "shap_values": sv,
        "feature_names": np.array(list(X.columns)),
        "X_used": X,
    }
