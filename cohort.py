"""
Cohort × Retention 分析。

把客户按"首次购买月"分组，看每个 cohort 在第 1/2/3/... 月的留存率/复购金额。
这是电商运营的基本盘报表，但 v1 没有。

输入约定
-------
原始交易表：
    customer_id | date | amount

不要求宽表（matrix），内部自己 pivot。

输出
----
两张表：
- retention_matrix: 行 = cohort（首购月），列 = 后续月份偏移，单元 = 留存率
- revenue_matrix: 同样维度，但单元 = 该 cohort 在该月偏移上的人均消费
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class CohortReport:
    retention: pd.DataFrame      # cohort × period_offset → retention rate
    arpu: pd.DataFrame           # cohort × period_offset → ARPU
    cohort_sizes: pd.Series      # cohort → 客户数


def build_cohort_report(
    transactions: pd.DataFrame,
    customer_col: str = "customer_id",
    date_col: str = "date",
    value_col: str = "amount",
    period: str = "M",  # 'M' 月度, 'W' 周, 'D' 日
) -> CohortReport:
    """构建 cohort 留存与人均消费两张矩阵。"""
    if period not in {"D", "W", "M", "Q"}:
        raise ValueError("period 必须是 D/W/M/Q")

    df = transactions.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # 客户首购周期
    first = df.groupby(customer_col)[date_col].min().rename("first_purchase")
    df = df.merge(first, on=customer_col, how="left")

    df["cohort"] = df["first_purchase"].dt.to_period(period)
    df["period"] = df[date_col].dt.to_period(period)
    df["period_offset"] = (df["period"] - df["cohort"]).apply(lambda x: x.n)

    # 1. cohort 大小
    cohort_sizes = df.drop_duplicates(customer_col).groupby("cohort").size().rename("cohort_size")

    # 2. 留存：每 (cohort, period_offset) 的独立客户数 / cohort 大小
    active = df.groupby(["cohort", "period_offset"])[customer_col].nunique().rename("active")
    retention = active.unstack(fill_value=0).divide(cohort_sizes, axis=0)
    retention = retention.fillna(0)

    # 3. ARPU：每 (cohort, period_offset) 的客户人均消费
    rev = df.groupby(["cohort", "period_offset"])[value_col].sum().rename("rev").unstack(fill_value=0)
    arpu = rev.divide(cohort_sizes, axis=0).fillna(0)

    # 按时间排序
    retention = retention.sort_index().reindex(sorted(retention.columns), axis=1)
    arpu = arpu.sort_index().reindex(sorted(arpu.columns), axis=1)

    return CohortReport(
        retention=retention,
        arpu=arpu,
        cohort_sizes=cohort_sizes.sort_index(),
    )


def cohort_retention_curves(report: CohortReport, max_offset: int = 12) -> pd.DataFrame:
    """裁剪到前 max_offset 期，方便画线。"""
    cols = [c for c in report.retention.columns if c <= max_offset]
    return report.retention.loc[:, cols]


def average_retention_curve(report: CohortReport, max_offset: int = 12) -> pd.Series:
    """跨 cohort 取平均 — 给"整体留存曲线"用。"""
    sliced = cohort_retention_curves(report, max_offset)
    # 只在该 offset 有数据的 cohort 上求均值（避免新 cohort 拉低均值）
    return sliced.replace(0, np.nan).mean(axis=0, skipna=True).fillna(0)
