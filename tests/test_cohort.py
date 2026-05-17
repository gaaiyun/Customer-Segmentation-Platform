"""cohort.py 测试 — 留存矩阵与 ARPU 矩阵。"""
import numpy as np
import pandas as pd
import pytest

from cohort import (
    average_retention_curve,
    build_cohort_report,
    cohort_retention_curves,
)


def _make_transactions():
    """三个 cohort，每月一次的交易序列。"""
    return pd.DataFrame({
        "customer_id": [
            # cohort A：2024-01 首购，每月都来
            "A1", "A1", "A1", "A1",
            "A2", "A2",                 # 只来了 2 个月
            # cohort B：2024-02 首购
            "B1", "B1", "B1",
            "B2",                       # 只来 1 个月
            # cohort C：2024-03 首购
            "C1", "C1",
        ],
        "date": pd.to_datetime([
            "2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05",
            "2024-01-20", "2024-02-15",
            "2024-02-10", "2024-03-15", "2024-04-20",
            "2024-02-22",
            "2024-03-08", "2024-04-09",
        ]),
        "amount": [100, 100, 100, 100,
                   50, 50,
                   80, 80, 80,
                   60,
                   90, 90],
    })


def test_cohort_sizes_correct():
    r = build_cohort_report(_make_transactions(), period="M")
    cohorts = r.cohort_sizes
    # 2024-01 有 2 个新客户（A1, A2）
    # 2024-02 有 2 个新客户（B1, B2）
    # 2024-03 有 1 个新客户（C1）
    assert cohorts[pd.Period("2024-01", "M")] == 2
    assert cohorts[pd.Period("2024-02", "M")] == 2
    assert cohorts[pd.Period("2024-03", "M")] == 1


def test_retention_first_offset_is_100pct():
    """offset=0 是 cohort 自身月份，应当 100% 留存。"""
    r = build_cohort_report(_make_transactions(), period="M")
    assert all(abs(v - 1.0) < 1e-9 for v in r.retention.iloc[:, 0])


def test_retention_decreases_over_time():
    """随 offset 增大，留存非严格递减（允许 plateau）。"""
    r = build_cohort_report(_make_transactions(), period="M")
    # cohort A：offset=0 时 2 人，offset=1 时 2 人, offset=2 时 1 人, offset=3 时 1 人
    a = r.retention.loc[pd.Period("2024-01", "M")]
    # 在出现过的 offset 上，留存应当 1, 1, 0.5, 0.5
    assert a[0] == 1.0
    assert a[1] == 1.0
    assert a[2] == 0.5
    assert a[3] == 0.5


def test_arpu_matrix_sane():
    r = build_cohort_report(_make_transactions(), period="M")
    # cohort 2024-01 在 offset=0 时人均消费 = (100+50)/2 = 75
    val = r.arpu.loc[pd.Period("2024-01", "M"), 0]
    assert abs(val - 75.0) < 1e-6


def test_retention_curve_helper():
    r = build_cohort_report(_make_transactions(), period="M")
    curves = cohort_retention_curves(r, max_offset=2)
    assert curves.shape[1] == 3  # offset = 0, 1, 2


def test_average_retention_curve_drops():
    r = build_cohort_report(_make_transactions(), period="M")
    avg = average_retention_curve(r, max_offset=3)
    # offset 0 应当是 1.0（每个 cohort 自身月份都全员活跃）
    assert avg.iloc[0] == 1.0


def test_invalid_period():
    with pytest.raises(ValueError):
        build_cohort_report(_make_transactions(), period="X")
