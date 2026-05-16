"""
BG/NBD + Gamma-Gamma CLV — 概率客户生命周期价值模型。

来源：
- Fader, Hardie, Lee (2005), "Counting Your Customers" the Easy Way: An
  Alternative to the Pareto/NBD Model. *Marketing Science*.
- Fader, Hardie (2013), The Gamma-Gamma Model of Monetary Value.

为什么不用线性回归 / RF 预测 CLV？
--------------------------------
电商客户行为里 RFM 三个值跟 CLV 是非线性的：
- 一个买了 1 次但很贵的客户 ≠ 买了 100 次每次便宜的客户
- "最近活跃"对未来交易概率的影响有衰减
- 离散计数（购买次数）+ 连续金额，应分两个模型

BG/NBD（购买频率）+ Gamma-Gamma（每次消费金额）是经典的两段式概率模型，
工业界从 Fader 1980s 一路传到现在，CLV 估计的稳定性比 ML 黑盒强很多。

模型假设
--------
**BG/NBD（购买频率）**

每个客户有两个隐变量：
- λ ~ Gamma(r, α)         —— 购买率
- p ~ Beta(a, b)           —— 每次交易后流失的概率

观测到的 (frequency, recency, T) 通过 MLE 反推总体参数 (r, α, a, b)。

**Gamma-Gamma（金额）**

每个客户每次的金额 X ~ Gamma(p, ν)，ν 因客户而异：
- ν ~ Gamma(q, γ)

要求购买频率与金额**条件独立**（即客户买几次 vs 客户单次花多少没相关）。
现实中存在弱相关，但对实务足够稳。

输入
----
RFM 风格的客户交易汇总：

| customer_id | frequency | recency | T   | monetary_value |
|---|---|---|---|---|
| 1 | 5 | 30 | 60 | 80.0 |

- frequency: 重复交易次数（不算第一次！第 1 次不计入，第 2 次起算）
- recency: 首次交易到最近一次的天数
- T: 首次交易到当前的天数（观测窗口长度）
- monetary_value: 平均每次交易金额（仅对 frequency >= 1 的客户）

依赖：只用 numpy + scipy（无 lifetimes / tensorflow）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import gammaln, hyp2f1


# ============================================================================
# BG/NBD — purchase frequency model
# ============================================================================

@dataclass
class BGNBDParams:
    """BG/NBD 的四个总体参数。"""

    r: float       # purchase-rate prior shape
    alpha: float   # purchase-rate prior scale
    a: float       # dropout-prob prior alpha
    b: float       # dropout-prob prior beta

    def to_dict(self) -> dict:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}


class BGNBDModel:
    """
    BG/NBD 模型 (Fader, Hardie, Lee 2005)。

    Usage
    -----
    >>> m = BGNBDModel()
    >>> m.fit(rfm_df)                                # rfm_df 必须含 frequency/recency/T
    >>> exp_purchases = m.predict_purchases(t=30, rfm_df=rfm_df)
    >>> alive_prob = m.predict_alive(rfm_df)         # P(customer still active)
    """

    def __init__(self, penalty: float = 1e-3):
        # 一个小 L2 惩罚，防止参数飘到不合理的极端
        self.penalty = penalty
        self.params_: Optional[BGNBDParams] = None

    # --------- log-likelihood ---------

    @staticmethod
    def _log_likelihood(params: np.ndarray, x: np.ndarray, t_x: np.ndarray, T: np.ndarray) -> float:
        """对单个客户的 BG/NBD 对数似然（向量化）。"""
        r, alpha, a, b = params
        if min(r, alpha, a, b) <= 0:
            return -1e10

        # 公式：Fader, Hardie, Lee (2005), Eq. (A2)
        ln_A1 = (
            gammaln(r + x)
            - gammaln(r)
            + r * np.log(alpha)
            - (r + x) * np.log(alpha + T)
        )
        ln_A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
        ln_A3 = (
            np.log(a)
            - np.log(b + x - 1 + 1e-300)
            + (r + x) * np.log((alpha + T) / (alpha + t_x))
        )
        ln_A4 = ln_A1 + ln_A2

        # 当 x = 0 时退化（无第二次交易就没"流失发生"路径）
        delta = (x > 0).astype(float)
        ll_each = ln_A4 + np.log1p(delta * np.exp(ln_A3))
        return float(ll_each.sum())

    def fit(self, df: pd.DataFrame) -> "BGNBDModel":
        """从 RFM 表中 fit。df 需有 frequency / recency / T 列。"""
        for col in ("frequency", "recency", "T"):
            if col not in df.columns:
                raise ValueError(f"BGNBD.fit 缺列: {col}")

        x = df["frequency"].to_numpy(dtype=float)
        t_x = df["recency"].to_numpy(dtype=float)
        T = df["T"].to_numpy(dtype=float)

        # 检查约束：recency <= T
        if np.any(t_x > T + 1e-9):
            raise ValueError("recency > T 的行存在，请检查输入")

        def neg_ll(theta: np.ndarray) -> float:
            ll = self._log_likelihood(theta, x, t_x, T)
            return -ll + self.penalty * (theta ** 2).sum()

        # 多起点搜，免得卡局部
        # a > 1 是必要约束：predict_purchases 的 (a-1) 在分母上，a<=1 会让预测
        # 走出"非负"区。理论可证 BG/NBD 在 a > 1 时是 well-defined。
        bounds = [(1e-4, 1e4), (1e-4, 1e4), (1.001, 1e4), (1e-4, 1e4)]
        best = None
        best_val = np.inf
        for init in [(1.0, 1.0, 1.5, 1.0), (0.5, 10.0, 1.2, 2.0), (2.0, 5.0, 2.0, 3.0)]:
            try:
                res = optimize.minimize(
                    neg_ll,
                    x0=np.array(init, dtype=float),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": 500, "ftol": 1e-8},
                )
            except Exception:
                continue
            if res.success and res.fun < best_val:
                best_val = res.fun
                best = res
        if best is None:
            raise RuntimeError("BG/NBD fit 全部初始值都失败")

        r, alpha, a, b = best.x
        self.params_ = BGNBDParams(r=float(r), alpha=float(alpha), a=float(a), b=float(b))
        return self

    # --------- prediction: expected # transactions in t periods ---------

    def predict_purchases(self, t: float, df: pd.DataFrame) -> np.ndarray:
        """每个客户在未来 t 个周期的预期交易次数 E[X(t)|x, t_x, T]。"""
        if self.params_ is None:
            raise RuntimeError("先 fit 再 predict")
        r, alpha, a, b = self.params_.r, self.params_.alpha, self.params_.a, self.params_.b
        x = df["frequency"].to_numpy(dtype=float)
        t_x = df["recency"].to_numpy(dtype=float)
        T = df["T"].to_numpy(dtype=float)

        # Fader-Hardie-Lee (2005) Eq. (10)
        # E[Y(t) | x, t_x, T] =
        #   (a + b + x - 1) / (a - 1) * [1 - ((alpha + T)/(alpha + T + t))^(r+x) * 2F1(...)]
        #   / (1 + δ_{x>0} * a/(b + x - 1) * ((alpha + T)/(alpha + t_x))^(r+x))
        if a <= 1:
            # 模型病态：返回保守的"过去频率外推"
            return x * (t / np.maximum(T, 1e-9))
        z = t / (alpha + T + t)
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, z)
        numer_part = 1.0 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp
        coef = (a + b + x - 1) / (a - 1)
        delta = (x > 0).astype(float)
        denom = 1.0 + delta * a / np.maximum(b + x - 1, 1e-9) * (
            (alpha + T) / (alpha + t_x)
        ) ** (r + x)
        result = coef * numer_part / denom
        # 极端边界下 hyp2f1 数值可能略小于该走的精确值，导致负小数；clamp 到 0
        return np.maximum(result, 0.0)

    def predict_alive(self, df: pd.DataFrame) -> np.ndarray:
        """P(客户当前仍未流失 | x, t_x, T)。"""
        if self.params_ is None:
            raise RuntimeError("先 fit 再 predict")
        r, alpha, a, b = self.params_.r, self.params_.alpha, self.params_.a, self.params_.b
        x = df["frequency"].to_numpy(dtype=float)
        t_x = df["recency"].to_numpy(dtype=float)
        T = df["T"].to_numpy(dtype=float)

        delta = (x > 0).astype(float)
        log_ratio = (r + x) * (np.log(alpha + T) - np.log(alpha + t_x))
        # P(alive) = 1 / (1 + delta * a/(b+x-1) * (alpha+T)/(alpha+t_x))^(r+x) ) 简化版
        return 1.0 / (1.0 + delta * a / np.maximum(b + x - 1, 1e-9) * np.exp(log_ratio))


# ============================================================================
# Gamma-Gamma — monetary value model
# ============================================================================

@dataclass
class GammaGammaParams:
    """Gamma-Gamma 的三个总体参数。"""

    p: float
    q: float
    gamma: float

    def to_dict(self) -> dict:
        return {"p": self.p, "q": self.q, "gamma": self.gamma}


class GammaGammaModel:
    """
    Gamma-Gamma 单次消费金额模型 (Fader, Hardie 2013)。

    重要前提：购买频率与每次消费金额条件独立。fit 之前请 check。
    """

    def __init__(self, penalty: float = 1e-3):
        self.penalty = penalty
        self.params_: Optional[GammaGammaParams] = None

    @staticmethod
    def _log_likelihood(theta: np.ndarray, x: np.ndarray, m: np.ndarray) -> float:
        """以 (frequency=x, monetary=m) 的客户为单位。"""
        p, q, gamma = theta
        if min(p, q, gamma) <= 0:
            return -1e10

        # 仅对 x >= 1 的客户计算（m 才有定义）
        mask = x >= 1
        x = x[mask]
        m = m[mask]
        if len(x) == 0:
            return -1e10

        # Fader, Hardie (2013) Eq. (5)
        ll = (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(gamma)
            + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(gamma + x * m)
        )
        return float(ll.sum())

    def fit(self, df: pd.DataFrame) -> "GammaGammaModel":
        for col in ("frequency", "monetary_value"):
            if col not in df.columns:
                raise ValueError(f"GG.fit 缺列: {col}")

        x = df["frequency"].to_numpy(dtype=float)
        m = df["monetary_value"].to_numpy(dtype=float)

        if np.any(m[x >= 1] <= 0):
            raise ValueError("monetary_value 必须 > 0（对 frequency>=1 的客户）")

        def neg_ll(theta: np.ndarray) -> float:
            return -self._log_likelihood(theta, x, m) + self.penalty * (theta ** 2).sum()

        best = None
        best_val = np.inf
        for init in [(1.0, 1.0, 1.0), (3.0, 4.0, 5.0), (5.0, 10.0, 30.0)]:
            try:
                res = optimize.minimize(
                    neg_ll,
                    x0=np.array(init, dtype=float),
                    method="L-BFGS-B",
                    bounds=[(1e-4, 1e4)] * 3,
                    options={"maxiter": 500},
                )
            except Exception:
                continue
            if res.success and res.fun < best_val:
                best_val = res.fun
                best = res
        if best is None:
            raise RuntimeError("GG fit 失败")

        p, q, gamma = best.x
        self.params_ = GammaGammaParams(p=float(p), q=float(q), gamma=float(gamma))
        return self

    def predict_conditional_mean(self, df: pd.DataFrame) -> np.ndarray:
        """E[avg monetary | observed frequency & monetary] —— Bayesian shrinkage 后的均值。"""
        if self.params_ is None:
            raise RuntimeError("先 fit 再 predict")
        p, q, gamma = self.params_.p, self.params_.q, self.params_.gamma
        x = df["frequency"].to_numpy(dtype=float)
        m = df["monetary_value"].to_numpy(dtype=float)

        # Eq. (6): E[M | p, q, gamma, x, m] = (p * x * m + gamma * (q - 1) * m_bar) / (p * x + q - 1)
        # 这里 m_bar = E[M] = gamma * p / (q - 1)，仅 q > 1 时有定义
        if q <= 1:
            m_bar = m.mean() if len(m) else 0.0
        else:
            m_bar = gamma * p / (q - 1)
        return (p * x * m + gamma * (q - 1) * m_bar) / np.maximum(p * x + q - 1, 1e-9)


# ============================================================================
# CLV — 把两个模型组合起来
# ============================================================================

def clv(
    bgnbd: BGNBDModel,
    gg: GammaGammaModel,
    df: pd.DataFrame,
    horizon: float = 365.0,
    discount_rate: float = 0.0,
) -> pd.Series:
    """
    每个客户的预期 CLV（horizon 周期，年化折现率 discount_rate）。

    简化：忽略时间内的折现复利，用一次性折现。生产环境建议改 numerical
    integration 做真正的连续折现。
    """
    expected_purchases = bgnbd.predict_purchases(horizon, df)
    expected_value_per_tx = gg.predict_conditional_mean(df)
    raw_clv = expected_purchases * expected_value_per_tx
    if discount_rate > 0:
        raw_clv = raw_clv / (1 + discount_rate) ** (horizon / 365.0)
    return pd.Series(raw_clv, index=df.index, name=f"clv_{int(horizon)}d")


def rfm_to_bgnbd_input(transactions: pd.DataFrame,
                       customer_col: str = "customer_id",
                       date_col: str = "date",
                       value_col: str = "amount",
                       observation_end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    把原始交易表转成 BG/NBD/GG 需要的形态。

    Parameters
    ----------
    transactions : DataFrame[customer_id, date, amount]
    observation_end : pd.Timestamp
        观测窗口截止时间。默认是交易里最晚的那天。
    """
    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if observation_end is None:
        observation_end = df[date_col].max()

    grouped = df.groupby(customer_col)
    out = pd.DataFrame(index=grouped.groups.keys())
    out["first_purchase"] = grouped[date_col].min()
    out["last_purchase"] = grouped[date_col].max()
    out["frequency"] = grouped.size() - 1   # 重复交易次数（不算第 1 次）
    out["T"] = (observation_end - out["first_purchase"]).dt.days
    out["recency"] = (out["last_purchase"] - out["first_purchase"]).dt.days

    # 平均金额仅对 frequency>=1 的客户有定义
    out["monetary_value"] = grouped[value_col].mean()
    out.loc[out["frequency"] == 0, "monetary_value"] = 0.0

    out.index.name = customer_col
    return out.reset_index()
