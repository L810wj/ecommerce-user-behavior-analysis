"""
计算电商用户行为业务指标，并输出 Power BI / Excel 可用的 CSV 和基础图表。

本脚本是 Pandas 版本，不依赖 Spark，适合作为本地快速分析和看板数据输出入口。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"
DASHBOARD_DATA_DIR = Path("dashboard") / "dashboard_data"
METRICS_DIR = Path("outputs") / "metrics"
FIGURES_DIR = Path("outputs") / "figures"

STAGE_ORDER = ["pv", "cart", "fav", "buy"]
RETENTION_DAYS = [1, 3, 7]


def safe_divide(numerator: float, denominator: float) -> float:
    """安全除法，避免指标计算时出现除零错误。"""
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def load_dwd_data(project_root: Path) -> pd.DataFrame:
    """读取 DWD 明细数据，并做基础类型转换。"""
    input_path = project_root / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入文件不存在: {INPUT_FILE}，请先运行 python src/02_clean_data_pandas.py"
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df["behavior_time"] = pd.to_datetime(df["behavior_time"], errors="coerce")
    df["behavior_date"] = pd.to_datetime(df["behavior_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["hour"] = df["hour"].astype(int)

    if df["behavior_time"].isna().any() or df["behavior_date"].isna().any():
        raise ValueError("DWD 数据存在无法解析的时间字段")

    print(f"读取 DWD 数据成功: {INPUT_FILE}")
    print(f"输入行数: {len(df):,}")
    return df


def calculate_daily_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算每日核心指标，包括 PV、UV、DAU、购买用户数和购买转化率。"""
    rows = []
    for behavior_date, group in df.groupby("behavior_date"):
        pv = int(group["is_pv"].sum())
        uv = int(group["user_id"].nunique())
        dau = uv
        cart_users = int(group.loc[group["is_cart"] == 1, "user_id"].nunique())
        fav_users = int(group.loc[group["is_fav"] == 1, "user_id"].nunique())
        buyers = int(group.loc[group["is_buy"] == 1, "user_id"].nunique())
        buy_count = int(group["is_buy"].sum())
        rows.append(
            {
                "behavior_date": behavior_date,
                "pv": pv,
                "uv": uv,
                "dau": dau,
                "cart_users": cart_users,
                "fav_users": fav_users,
                "buyers": buyers,
                "buy_count": buy_count,
                "avg_pv_per_user": safe_divide(pv, uv),
                "user_buy_rate": safe_divide(buyers, dau),
            }
        )

    return pd.DataFrame(rows).sort_values("behavior_date").reset_index(drop=True)


def calculate_conversion_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """计算整体行为转化漏斗。"""
    metrics = []
    previous_user_count = None
    pv_user_count = None

    for stage_order, behavior_stage in enumerate(STAGE_ORDER, start=1):
        stage_df = df[df["behavior_type"] == behavior_stage]
        user_count = int(stage_df["user_id"].nunique())
        behavior_count = int(len(stage_df))

        if behavior_stage == "pv":
            pv_user_count = user_count

        metrics.append(
            {
                "behavior_stage": behavior_stage,
                "user_count": user_count,
                "behavior_count": behavior_count,
                "stage_order": stage_order,
                "conversion_rate_from_pv": safe_divide(user_count, pv_user_count or 0),
                "conversion_rate_from_previous_stage": 1.0
                if previous_user_count is None
                else safe_divide(user_count, previous_user_count),
            }
        )
        previous_user_count = user_count

    return pd.DataFrame(metrics)


def calculate_retention(df: pd.DataFrame) -> pd.DataFrame:
    """计算 1 日、3 日、7 日活跃留存。"""
    user_date = df[["user_id", "behavior_date"]].drop_duplicates().copy()
    user_date["behavior_date"] = pd.to_datetime(user_date["behavior_date"])
    active_map = set(zip(user_date["user_id"], user_date["behavior_date"]))
    rows = []

    for base_date, base_group in user_date.groupby("behavior_date"):
        base_users = set(base_group["user_id"])
        for retention_day in RETENTION_DAYS:
            target_date = base_date + pd.Timedelta(days=retention_day)
            retained_users = {
                user_id
                for user_id in base_users
                if (user_id, target_date) in active_map
            }
            rows.append(
                {
                    "base_date": base_date.strftime("%Y-%m-%d"),
                    "retention_day": retention_day,
                    "base_active_users": len(base_users),
                    "retained_users": len(retained_users),
                    "retention_rate": safe_divide(len(retained_users), len(base_users)),
                }
            )

    return pd.DataFrame(rows).sort_values(["base_date", "retention_day"]).reset_index(drop=True)


def score_recency(recency_days: int) -> int:
    """R 分数：最近购买越近，分数越高。"""
    if recency_days <= 3:
        return 5
    if recency_days <= 7:
        return 4
    if recency_days <= 14:
        return 3
    if recency_days <= 21:
        return 2
    return 1


def score_frequency(value: int) -> int:
    """F/M 分数：购买次数越多，分数越高。"""
    if value >= 5:
        return 5
    if value >= 3:
        return 4
    if value >= 2:
        return 3
    if value == 1:
        return 2
    return 1


def segment_user(row: pd.Series) -> str:
    """根据 RFM 分数划分用户价值层级。"""
    if row["frequency"] == 0:
        return "未购买用户"
    if row["r_score"] >= 4 and row["f_score"] >= 4 and row["m_score"] >= 4:
        return "高价值用户"
    if row["r_score"] >= 4 and (row["f_score"] >= 3 or row["m_score"] >= 3):
        return "潜力用户"
    if row["r_score"] >= 3 and row["f_score"] >= 2:
        return "一般用户"
    if row["r_score"] <= 2 and row["frequency"] > 0:
        return "流失风险用户"
    return "低活跃用户"


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """计算 RFM 用户价值分层，金额字段暂用购买次数近似。"""
    observation_date = pd.to_datetime(df["behavior_date"]).max()
    buy_df = df[df["is_buy"] == 1].copy()
    all_users = pd.DataFrame({"user_id": sorted(df["user_id"].unique())})

    if buy_df.empty:
        rfm = all_users.copy()
        rfm["last_buy_date"] = pd.NaT
        rfm["frequency"] = 0
    else:
        buy_group = (
            buy_df.groupby("user_id")
            .agg(
                last_buy_date=("behavior_date", "max"),
                frequency=("is_buy", "sum"),
            )
            .reset_index()
        )
        rfm = all_users.merge(buy_group, on="user_id", how="left")

    rfm["last_buy_date"] = pd.to_datetime(rfm["last_buy_date"], errors="coerce")
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    rfm["monetary"] = rfm["frequency"]
    rfm["recency_days"] = (
        observation_date - rfm["last_buy_date"]
    ).dt.days.fillna(999).astype(int)

    rfm["r_score"] = rfm["recency_days"].apply(score_recency)
    rfm["f_score"] = rfm["frequency"].apply(score_frequency)
    rfm["m_score"] = rfm["monetary"].apply(score_frequency)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["user_value_segment"] = rfm.apply(segment_user, axis=1)

    return rfm[
        [
            "user_id",
            "recency_days",
            "frequency",
            "monetary",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
            "user_value_segment",
        ]
    ].sort_values(["rfm_score", "frequency", "user_id"], ascending=[False, False, True])


def calculate_top_items(df: pd.DataFrame) -> pd.DataFrame:
    """计算热门商品 Top 20。"""
    item_metrics = (
        df.groupby(["item_id", "category_id"])
        .agg(
            pv_count=("is_pv", "sum"),
            cart_count=("is_cart", "sum"),
            fav_count=("is_fav", "sum"),
            buy_count=("is_buy", "sum"),
            buyer_count=("user_id", lambda x: x[df.loc[x.index, "is_buy"] == 1].nunique()),
        )
        .reset_index()
    )
    item_metrics["click_to_buy_rate"] = item_metrics.apply(
        lambda row: safe_divide(row["buy_count"], row["pv_count"]), axis=1
    )
    item_metrics = item_metrics.sort_values(
        ["pv_count", "buy_count", "item_id"], ascending=[False, False, True]
    ).head(20)
    item_metrics["rank_no"] = range(1, len(item_metrics) + 1)
    return item_metrics.reset_index(drop=True)


def calculate_top_categories(df: pd.DataFrame) -> pd.DataFrame:
    """计算热门类目 Top 20。"""
    category_metrics = (
        df.groupby("category_id")
        .agg(
            pv_count=("is_pv", "sum"),
            cart_count=("is_cart", "sum"),
            fav_count=("is_fav", "sum"),
            buy_count=("is_buy", "sum"),
            buyer_count=("user_id", lambda x: x[df.loc[x.index, "is_buy"] == 1].nunique()),
        )
        .reset_index()
    )
    category_metrics["click_to_buy_rate"] = category_metrics.apply(
        lambda row: safe_divide(row["buy_count"], row["pv_count"]), axis=1
    )
    category_metrics = category_metrics.sort_values(
        ["pv_count", "buy_count", "category_id"], ascending=[False, False, True]
    ).head(20)
    category_metrics["rank_no"] = range(1, len(category_metrics) + 1)
    return category_metrics.reset_index(drop=True)


def calculate_hourly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """计算小时级行为趋势。"""
    hourly = (
        df.groupby("hour")
        .agg(
            pv_count=("is_pv", "sum"),
            cart_count=("is_cart", "sum"),
            fav_count=("is_fav", "sum"),
            buy_count=("is_buy", "sum"),
            behavior_count=("user_id", "size"),
            active_users=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values("hour")
    )
    return hourly


def save_metric_table(df: pd.DataFrame, file_name: str, project_root: Path) -> None:
    """同时保存一份到 dashboard/dashboard_data 和 outputs/metrics。"""
    for directory in [DASHBOARD_DATA_DIR, METRICS_DIR]:
        output_dir = project_root / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"输出 CSV: {output_path.relative_to(project_root)}，行数: {len(df):,}")


def setup_plot_style() -> None:
    """设置基础图表风格，使用英文标题规避 Windows 中文字体问题。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def save_daily_pv_uv_chart(daily: pd.DataFrame, output_path: Path) -> None:
    """保存每日 PV / UV 趋势图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(daily["behavior_date"])
    ax.plot(x, daily["pv"], marker="o", label="PV")
    ax.plot(x, daily["uv"], marker="o", label="UV")
    ax.set_title("Daily PV and UV Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_conversion_funnel_chart(funnel: pd.DataFrame, output_path: Path) -> None:
    """保存转化漏斗图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(funnel["behavior_stage"], funnel["user_count"], color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("User Behavior Conversion Funnel")
    ax.set_xlabel("Stage")
    ax.set_ylabel("User Count")
    for idx, row in funnel.iterrows():
        ax.text(idx, row["user_count"], f"{row['conversion_rate_from_pv']:.2%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_retention_heatmap(retention: pd.DataFrame, output_path: Path) -> None:
    """保存留存热力图。"""
    matrix = retention.pivot(index="base_date", columns="retention_day", values="retention_rate").fillna(0)
    fig_height = max(6, min(12, len(matrix) * 0.28))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.01, matrix.values.max()))
    ax.set_title("Retention Heatmap")
    ax.set_xlabel("Retention Day")
    ax.set_ylabel("Base Date")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(im, ax=ax, label="Retention Rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_rfm_segments_chart(rfm: pd.DataFrame, output_path: Path) -> None:
    """保存 RFM 用户分层柱状图。"""
    segment_counts = rfm["user_value_segment"].value_counts()
    label_mapping = {
        "高价值用户": "High Value",
        "潜力用户": "Potential",
        "一般用户": "Normal",
        "低活跃用户": "Low Active",
        "流失风险用户": "Churn Risk",
        "未购买用户": "No Purchase",
    }
    display_labels = [label_mapping.get(label, label) for label in segment_counts.index]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(display_labels, segment_counts.values, color="#72B7B2")
    ax.set_title("RFM User Segments")
    ax.set_xlabel("Segment")
    ax.set_ylabel("User Count")
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(segment_counts.values):
        ax.text(idx, value, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_hourly_trend_chart(hourly: pd.DataFrame, output_path: Path) -> None:
    """保存小时级活跃趋势图。"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly["hour"], hourly["behavior_count"], marker="o", label="Behavior Count")
    ax.plot(hourly["hour"], hourly["active_users"], marker="o", label="Active Users")
    ax.set_title("Hourly Behavior Trend")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_figures(metrics: dict[str, pd.DataFrame], project_root: Path) -> None:
    """生成并保存基础图表。"""
    figures_dir = project_root / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    chart_jobs = [
        ("daily_pv_uv_trend.png", save_daily_pv_uv_chart, metrics["daily_core_metrics"]),
        ("conversion_funnel.png", save_conversion_funnel_chart, metrics["conversion_funnel"]),
        ("retention_heatmap.png", save_retention_heatmap, metrics["retention_analysis"]),
        ("rfm_user_segments.png", save_rfm_segments_chart, metrics["rfm_user_value"]),
        ("hourly_behavior_trend.png", save_hourly_trend_chart, metrics["hourly_behavior_trend"]),
    ]

    for file_name, chart_func, data in chart_jobs:
        output_path = figures_dir / file_name
        chart_func(data, output_path)
        print(f"输出图表: {output_path.relative_to(project_root)}")


def main() -> None:
    """脚本入口函数。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        df = load_dwd_data(project_root)

        metrics = {
            "daily_core_metrics": calculate_daily_core_metrics(df),
            "conversion_funnel": calculate_conversion_funnel(df),
            "retention_analysis": calculate_retention(df),
            "rfm_user_value": calculate_rfm(df),
            "top_items": calculate_top_items(df),
            "top_categories": calculate_top_categories(df),
            "hourly_behavior_trend": calculate_hourly_trend(df),
        }

        output_files = {
            "daily_core_metrics": "daily_core_metrics.csv",
            "conversion_funnel": "conversion_funnel.csv",
            "retention_analysis": "retention_analysis.csv",
            "rfm_user_value": "rfm_user_value.csv",
            "top_items": "top_items.csv",
            "top_categories": "top_categories.csv",
            "hourly_behavior_trend": "hourly_behavior_trend.csv",
        }

        for metric_name, file_name in output_files.items():
            save_metric_table(metrics[metric_name], file_name, project_root)

        save_figures(metrics, project_root)
        print("\n业务指标和图表输出完成")
    except Exception as exc:
        print(f"业务指标计算失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
