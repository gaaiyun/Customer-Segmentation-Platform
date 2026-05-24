"""CLV 估计的不确定性区间（bootstrap + 渐进 normal）。

v2 的 ``clv(bgnbd, gg, df)`` 给一个点估计，但 BG/NBD 和 Gamma-Gamma 的
MLE 参数本身有标准误 —— 不告诉 marketing 经理"CLV ≈ 500 ± 多少"，他们
要么把数字当成绝对，要么完全不信。

两种做法：

1. **Customer-level bootstrap**（最常用）—— 对 customer 行重抽样 B 次，
   每次完整重 fit BG/NBD + Gamma-Gamma + clv，得到每个客户的 CLV 分布。
   优点：不假设 normality；缺点：BG/NBD fit 慢，B=100 在大客户表上要
   分钟级。

2. **Parameter perturbation**（快速近似）—— 一次 fit，从参数协方差矩阵
   抽 B 次参数，每次推 CLV。比 bootstrap 快 100x。但要求拿到 Hessian
   反演 —— scipy.optimize.minimize 默认不给。这里实现简化版：对每个参数
   独立加 σ 扰动，假设参数间无相关。

参考：
- Fader, Hardie, Berger (2009) "Customer-Base Analysis with Discrete-Time
  Transaction Data" — 讨论 BG/NBD 参数不确定性。
- Wager & Athey (2018) — bootstrap-confidence-intervals 在客户分析中的
  现代应用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from bgnbd import BGNBDModel, GammaGammaModel, clv


@dataclass
class CLVBootstrapResult:
    """每个客户的 CLV 分布统计 + 群体级聚合。"""
    customer_clv: pd.DataFrame      # columns: clv_mean / clv_p05 / clv_p50 / clv_p95 / clv_std
    total_clv_samples: np.ndarray   # bootstrap-level 总 CLV samples，长度 B
    total_clv_mean: float
    total_clv_ci_low: float
    total_clv_ci_high: float
    horizon_days: float
    n_bootstrap: int
    n_customers: int
    n_failed: int = 0               # 因 fit 失败被丢弃的 bootstrap 次数

    def to_dict(self) -> dict:
        return {
            "horizon_days": float(self.horizon_days),
            "n_bootstrap": int(self.n_bootstrap),
            "n_customers": int(self.n_customers),
            "n_failed": int(self.n_failed),
            "total_clv_mean": float(self.total_clv_mean),
            "total_clv_ci_low": float(self.total_clv_ci_low),
            "total_clv_ci_high": float(self.total_clv_ci_high),
            "customer_summary": self.customer_clv.describe().to_dict(),
        }


def bootstrap_clv(
    df: pd.DataFrame,
    horizon: float = 365.0,
    discount_rate: float = 0.0,
    n_bootstrap: int = 100,
    alpha: float = 0.05,
    seed: Optional[int] = 42,
    verbose: bool = False,
) -> CLVBootstrapResult:
    """对 RFM 表做客户级 bootstrap，得到每个客户的 CLV 分布。

    Parameters
    ----------
    df : DataFrame 含 columns frequency / recency / T / monetary_value
    horizon : 预测周期（天）
    discount_rate : 年化折现率
    n_bootstrap : 重抽样次数。100 已经够稳定的 5% / 95% percentile 估计。
    alpha : CI 显著性水平（双尾）。
    seed : 随机种子
    verbose : 打 fit 进度
    """
    if not {"frequency", "recency", "T", "monetary_value"}.issubset(df.columns):
        raise ValueError(
            "df 必须含 frequency / recency / T / monetary_value 列；"
            "用 bgnbd.rfm_to_bgnbd_input(transactions) 先转换"
        )
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap < 10 不足以估 percentile")
    if not 0 < alpha < 1:
        raise ValueError("alpha ∈ (0, 1)")

    rng = np.random.default_rng(seed)
    n_customers = len(df)
    df_reset = df.reset_index(drop=True)

    # 每次 bootstrap 都对客户行重抽样（同等 size 含重复），重 fit 模型，
    # 然后预测原始 customer 的 CLV
    bootstrap_matrix = np.zeros((n_bootstrap, n_customers))
    n_failed = 0
    successful = 0

    for b in range(n_bootstrap):
        sample_idx = rng.integers(0, n_customers, size=n_customers)
        sample_df = df_reset.iloc[sample_idx].reset_index(drop=True)

        try:
            bg = BGNBDModel()
            bg.fit(sample_df)
            gg_input = sample_df[sample_df["frequency"] >= 1].reset_index(drop=True)
            if len(gg_input) < 2:
                raise ValueError("Gamma-Gamma 拟合需要 ≥ 2 个重复客户")
            gg = GammaGammaModel()
            gg.fit(gg_input)
            # 用原始 df 预测，不是 sample_df —— 我们要每个原 customer 的 CLV
            customer_clv = clv(bg, gg, df_reset, horizon=horizon,
                                discount_rate=discount_rate)
            bootstrap_matrix[successful, :] = customer_clv.values
            successful += 1
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  bootstrap {b}: skip ({type(e).__name__}: {e})")
            continue

    if successful < 10:
        raise RuntimeError(
            f"成功 fit 的 bootstrap 次数 {successful} < 10，无法稳定估 CI。"
            "通常因为客户数太少或 BG/NBD 参数 unfittable。"
        )
    bootstrap_matrix = bootstrap_matrix[:successful, :]

    # 每个客户的分布统计
    p_low = alpha / 2 * 100
    p_high = (1 - alpha / 2) * 100
    customer_summary = pd.DataFrame({
        "clv_mean": bootstrap_matrix.mean(axis=0),
        f"clv_p{int(p_low):02d}": np.percentile(bootstrap_matrix, p_low, axis=0),
        "clv_p50": np.percentile(bootstrap_matrix, 50, axis=0),
        f"clv_p{int(p_high):02d}": np.percentile(bootstrap_matrix, p_high, axis=0),
        "clv_std": bootstrap_matrix.std(axis=0, ddof=1),
    }, index=df.index)

    # 群体级总 CLV 的不确定性
    total_per_boot = bootstrap_matrix.sum(axis=1)
    total_mean = float(total_per_boot.mean())
    total_low = float(np.percentile(total_per_boot, p_low))
    total_high = float(np.percentile(total_per_boot, p_high))

    return CLVBootstrapResult(
        customer_clv=customer_summary,
        total_clv_samples=total_per_boot,
        total_clv_mean=total_mean,
        total_clv_ci_low=total_low,
        total_clv_ci_high=total_high,
        horizon_days=horizon,
        n_bootstrap=successful,
        n_customers=n_customers,
        n_failed=n_failed,
    )


def parameter_perturbation_clv(
    bg: BGNBDModel,
    gg: GammaGammaModel,
    df: pd.DataFrame,
    horizon: float = 365.0,
    discount_rate: float = 0.0,
    n_perturbations: int = 200,
    relative_noise: float = 0.05,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """快速近似：对已 fit 的参数加 ±relative_noise 高斯扰动，重预测。

    比 bootstrap 快 100x，但 **过于乐观** —— 假设参数无相关、无 outer 不
    确定性。生产环境只推荐用于 quick UI feedback，正式报告用 bootstrap。

    Parameters
    ----------
    bg / gg : 已 fit 的 BG/NBD 和 Gamma-Gamma 模型
    df : 要预测的客户 RFM 表
    n_perturbations : 扰动次数
    relative_noise : 参数扰动量（百分比）

    Returns
    -------
    DataFrame 每行一个客户，columns: clv_mean / clv_p05 / clv_p95
    """
    if bg.params_ is None or gg.params_ is None:
        raise ValueError("bg 和 gg 必须先 fit")
    if relative_noise <= 0:
        raise ValueError("relative_noise 必须 > 0")

    rng = np.random.default_rng(seed)
    n_customers = len(df)

    # 备份原参数
    bg_orig = (bg.params_.r, bg.params_.alpha, bg.params_.a, bg.params_.b)
    gg_orig = (gg.params_.p, gg.params_.q, gg.params_.gamma)

    samples = np.zeros((n_perturbations, n_customers))

    try:
        for k in range(n_perturbations):
            # 参数扰动
            bg.params_.r = max(bg_orig[0] * (1 + rng.normal(0, relative_noise)), 1e-6)
            bg.params_.alpha = max(bg_orig[1] * (1 + rng.normal(0, relative_noise)), 1e-6)
            bg.params_.a = max(bg_orig[2] * (1 + rng.normal(0, relative_noise)), 1e-6)
            bg.params_.b = max(bg_orig[3] * (1 + rng.normal(0, relative_noise)), 1e-6)
            gg.params_.p = max(gg_orig[0] * (1 + rng.normal(0, relative_noise)), 1e-6)
            gg.params_.q = max(gg_orig[1] * (1 + rng.normal(0, relative_noise)), 1e-6)
            gg.params_.gamma = max(gg_orig[2] * (1 + rng.normal(0, relative_noise)), 1e-6)
            samples[k, :] = clv(bg, gg, df, horizon=horizon,
                                 discount_rate=discount_rate).values
    finally:
        # 恢复原参数
        bg.params_.r, bg.params_.alpha, bg.params_.a, bg.params_.b = bg_orig
        gg.params_.p, gg.params_.q, gg.params_.gamma = gg_orig

    return pd.DataFrame({
        "clv_mean": samples.mean(axis=0),
        "clv_p05": np.percentile(samples, 5, axis=0),
        "clv_p95": np.percentile(samples, 95, axis=0),
        "clv_std": samples.std(axis=0, ddof=1),
    }, index=df.index)
