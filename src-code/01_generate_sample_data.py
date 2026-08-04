"""
生成更接近真实电商业务的模拟用户行为数据。

本脚本输出 ODS 原始行为日志，字段结构保持稳定，供后续 Pandas 清洗、
SQL 数仓、PySpark ETL、看板指标继续使用。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20240611
ROW_COUNT = 120_000
USER_COUNT = 6_000
ITEM_COUNT = 2_500
CATEGORY_COUNT = 120
DAY_COUNT = 30
START_DATE = "2024-11-01"
OUTPUT_FILE = Path("data") / "raw" / "user_behavior_raw.csv"

BEHAVIOR_COUNTS = {
    "pv": 111_000,   # 92.50%
    "cart": 4_800,  # 4.00%
    "fav": 3_000,   # 2.50%
    "buy": 1_200,   # 1.00%
}

CART_USER_COUNT = 720  # pv 用户的 12%左右
FAV_USER_COUNT = 540   # pv 用户的 9%左右
BUY_USER_COUNT = 300   # pv 用户的 5%左右


def make_long_tail_weights(size: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """生成长尾分布权重，让少数对象拥有更高概率。"""
    ranks = np.arange(1, size + 1, dtype=float)
    weights = 1 / np.power(ranks, alpha)
    rng.shuffle(weights)
    return weights / weights.sum()


def sample_with_full_coverage(
    base_values: np.ndarray,
    total_size: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """抽样时先保证每个基础对象至少出现一次，再按权重补齐剩余样本。"""
    if total_size < len(base_values):
        raise ValueError("total_size 不能小于 base_values 的数量")

    sampled_values = rng.choice(
        base_values,
        size=total_size - len(base_values),
        replace=True,
        p=weights,
    )
    result = np.concatenate([base_values, sampled_values])
    rng.shuffle(result)
    return result


def weighted_choice_without_replacement(
    values: np.ndarray,
    size: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """按权重不放回抽样，用于挑选更可能转化的用户或商品。"""
    normalized_weights = weights / weights.sum()
    return rng.choice(values, size=size, replace=False, p=normalized_weights)


def build_user_profile(rng: np.random.Generator) -> pd.DataFrame:
    """构建用户活跃度、购买倾向和用户分层。"""
    user_ids = np.array([f"U{i:06d}" for i in range(1, USER_COUNT + 1)])
    segments = rng.choice(
        ["low", "medium", "high", "vip"],
        size=USER_COUNT,
        p=[0.72, 0.20, 0.06, 0.02],
    )
    segment_multiplier = {
        "low": 0.35,
        "medium": 1.0,
        "high": 3.5,
        "vip": 8.0,
    }

    base_activity = rng.lognormal(mean=0.0, sigma=1.0, size=USER_COUNT)
    activity_score = base_activity * np.array([segment_multiplier[s] for s in segments])
    activity_weight = activity_score / activity_score.sum()

    # 购买倾向和活跃度相关，但加入噪声，避免所有高活跃用户都购买。
    propensity_noise = rng.beta(2.0, 8.0, size=USER_COUNT)
    purchase_propensity = (
        0.65 * (activity_score / activity_score.max())
        + 0.35 * propensity_noise
    )
    conversion_weight = purchase_propensity / purchase_propensity.sum()

    return pd.DataFrame(
        {
            "user_id": user_ids,
            "segment": segments,
            "activity_weight": activity_weight,
            "conversion_weight": conversion_weight,
        }
    )


def build_item_catalog(rng: np.random.Generator) -> pd.DataFrame:
    """构建商品、类目、商品热度和转化能力。"""
    item_ids = np.array([f"I{i:06d}" for i in range(1, ITEM_COUNT + 1)])
    category_ids = np.array([f"C{i:03d}" for i in range(1, CATEGORY_COUNT + 1)])

    category_weights = make_long_tail_weights(CATEGORY_COUNT, alpha=0.85, rng=rng)
    mandatory_categories = category_ids.copy()
    extra_categories = rng.choice(
        category_ids,
        size=ITEM_COUNT - CATEGORY_COUNT,
        replace=True,
        p=category_weights,
    )
    item_categories = np.concatenate([mandatory_categories, extra_categories])
    rng.shuffle(item_categories)

    popularity = make_long_tail_weights(ITEM_COUNT, alpha=1.08, rng=rng)
    attractiveness = rng.lognormal(mean=-0.2, sigma=0.7, size=ITEM_COUNT)

    # 商品转化能力不完全等同于热度：热门商品更容易被互动，但小部分非头部商品也可能转化较好。
    cart_weight = popularity * np.power(attractiveness, 0.45)
    fav_weight = popularity * np.power(attractiveness, 0.35)
    buy_weight = popularity * np.power(attractiveness, 0.75)

    return pd.DataFrame(
        {
            "item_id": item_ids,
            "category_id": item_categories,
            "pv_weight": popularity / popularity.sum(),
            "cart_weight": cart_weight / cart_weight.sum(),
            "fav_weight": fav_weight / fav_weight.sum(),
            "buy_weight": buy_weight / buy_weight.sum(),
        }
    )


def build_user_pools(user_profile: pd.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """按用户购买倾向选择加购、收藏、购买人群，控制用户级转化率。"""
    user_ids = user_profile["user_id"].to_numpy()
    conversion_weight = user_profile["conversion_weight"].to_numpy()

    buy_users = weighted_choice_without_replacement(
        user_ids,
        BUY_USER_COUNT,
        conversion_weight,
        rng,
    )

    buy_user_set = set(buy_users)
    non_buy_mask = ~user_profile["user_id"].isin(buy_user_set)
    non_buy_users = user_profile.loc[non_buy_mask, "user_id"].to_numpy()
    non_buy_weights = user_profile.loc[non_buy_mask, "conversion_weight"].to_numpy()

    cart_extra = weighted_choice_without_replacement(
        non_buy_users,
        CART_USER_COUNT - BUY_USER_COUNT,
        non_buy_weights,
        rng,
    )
    cart_users = np.concatenate([buy_users, cart_extra])

    fav_buy_part = rng.choice(buy_users, size=min(210, BUY_USER_COUNT), replace=False)
    fav_existing = set(fav_buy_part)
    fav_extra_candidates = user_profile.loc[
        ~user_profile["user_id"].isin(fav_existing),
        "user_id",
    ].to_numpy()
    fav_extra_weights = user_profile.loc[
        ~user_profile["user_id"].isin(fav_existing),
        "conversion_weight",
    ].to_numpy()
    fav_extra = weighted_choice_without_replacement(
        fav_extra_candidates,
        FAV_USER_COUNT - len(fav_buy_part),
        fav_extra_weights,
        rng,
    )
    fav_users = np.concatenate([fav_buy_part, fav_extra])

    return {
        "pv": user_ids,
        "cart": cart_users,
        "fav": fav_users,
        "buy": buy_users,
    }


def sample_users_for_behavior(
    behavior_type: str,
    user_profile: pd.DataFrame,
    user_pools: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """为指定行为抽样用户，兼顾用户全覆盖和长尾活跃度。"""
    pool = user_pools[behavior_type]
    pool_profile = user_profile.set_index("user_id").loc[pool]

    if behavior_type == "pv":
        weights = pool_profile["activity_weight"].to_numpy()
    else:
        # 非浏览行为更偏向高活跃、高购买倾向用户。
        weights = (
            pool_profile["activity_weight"].to_numpy() * 0.35
            + pool_profile["conversion_weight"].to_numpy() * 0.65
        )

    weights = weights / weights.sum()
    return sample_with_full_coverage(pool, BEHAVIOR_COUNTS[behavior_type], weights, rng)


def sample_items_for_behavior(
    behavior_type: str,
    item_catalog: pd.DataFrame,
    rng: np.random.Generator,
) -> np.ndarray:
    """为指定行为抽样商品，体现商品热度和转化能力差异。"""
    item_ids = item_catalog["item_id"].to_numpy()
    weight_column = {
        "pv": "pv_weight",
        "cart": "cart_weight",
        "fav": "fav_weight",
        "buy": "buy_weight",
    }[behavior_type]
    weights = item_catalog[weight_column].to_numpy()

    if behavior_type == "pv":
        return sample_with_full_coverage(item_ids, BEHAVIOR_COUNTS[behavior_type], weights, rng)
    return rng.choice(item_ids, size=BEHAVIOR_COUNTS[behavior_type], replace=True, p=weights)


def generate_behavior_times(total_size: int, rng: np.random.Generator) -> pd.Series:
    """生成带晚高峰、凌晨低谷、周末提升、活动日峰值和随机噪声的行为时间。"""
    start_time = pd.Timestamp(START_DATE)
    day_offsets = np.arange(DAY_COUNT)
    dates = start_time + pd.to_timedelta(day_offsets, unit="D")
    weekdays = dates.weekday

    day_weights = np.where(weekdays >= 5, 1.16, 1.0).astype(float)
    campaign_dates = {
        pd.Timestamp("2024-11-11"): 1.35,
        pd.Timestamp("2024-11-17"): 1.20,
        pd.Timestamp("2024-11-24"): 1.18,
    }
    for idx, date_value in enumerate(dates):
        day_weights[idx] *= campaign_dates.get(date_value, 1.0)

    day_weights *= rng.normal(loc=1.0, scale=0.035, size=DAY_COUNT).clip(0.9, 1.12)
    day_weights = day_weights / day_weights.sum()
    sampled_days = sample_with_full_coverage(day_offsets, total_size, day_weights, rng)

    hour_weights = np.array(
        [
            0.030, 0.010, 0.008, 0.006, 0.006, 0.008,
            0.015, 0.025, 0.035, 0.045, 0.050, 0.052,
            0.055, 0.050, 0.048, 0.050, 0.056, 0.062,
            0.070, 0.086, 0.094, 0.092, 0.078, 0.064,
        ],
        dtype=float,
    )
    hour_weights *= rng.normal(loc=1.0, scale=0.045, size=24).clip(0.86, 1.15)
    hour_weights = hour_weights / hour_weights.sum()

    sampled_hours = rng.choice(np.arange(24), size=total_size, replace=True, p=hour_weights)
    sampled_minutes = rng.integers(0, 60, size=total_size)
    sampled_seconds = rng.integers(0, 60, size=total_size)

    behavior_times = (
        start_time
        + pd.to_timedelta(sampled_days, unit="D")
        + pd.to_timedelta(sampled_hours, unit="h")
        + pd.to_timedelta(sampled_minutes, unit="m")
        + pd.to_timedelta(sampled_seconds, unit="s")
    )
    return pd.Series(behavior_times, name="behavior_time")


def ensure_unique_behavior_rows(df: pd.DataFrame) -> pd.DataFrame:
    """避免完全重复的行为记录，保持原始行为日志干净。"""
    key_columns = ["user_id", "item_id", "category_id", "behavior_type", "behavior_time"]
    start_time = pd.Timestamp(START_DATE)
    max_time = start_time + pd.Timedelta(days=DAY_COUNT) - pd.Timedelta(seconds=1)

    for attempt in range(10):
        duplicated_indices = df.index[df.duplicated(subset=key_columns, keep="first")]
        if len(duplicated_indices) == 0:
            return df

        for offset, row_index in enumerate(duplicated_indices, start=1):
            seconds_to_shift = offset + attempt * len(duplicated_indices)
            new_time = df.at[row_index, "behavior_time"] + pd.Timedelta(seconds=seconds_to_shift)

            if new_time > max_time:
                new_time = df.at[row_index, "behavior_time"] - pd.Timedelta(seconds=seconds_to_shift)
            if new_time < start_time:
                new_time = start_time + pd.Timedelta(seconds=seconds_to_shift)

            df.at[row_index, "behavior_time"] = new_time

    raise ValueError("去除重复行为记录失败，请调整数据生成参数")


def generate_sample_data() -> pd.DataFrame:
    """生成更接近真实业务漏斗的模拟电商用户行为数据。"""
    rng = np.random.default_rng(RANDOM_SEED)
    user_profile = build_user_profile(rng)
    item_catalog = build_item_catalog(rng)
    user_pools = build_user_pools(user_profile, rng)
    item_to_category = dict(zip(item_catalog["item_id"], item_catalog["category_id"]))

    frames = []
    for behavior_type in ["pv", "cart", "fav", "buy"]:
        sampled_users = sample_users_for_behavior(behavior_type, user_profile, user_pools, rng)
        sampled_items = sample_items_for_behavior(behavior_type, item_catalog, rng)
        frames.append(
            pd.DataFrame(
                {
                    "user_id": sampled_users,
                    "item_id": sampled_items,
                    "category_id": [item_to_category[item_id] for item_id in sampled_items],
                    "behavior_type": behavior_type,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["behavior_time"] = generate_behavior_times(len(df), rng)
    df = ensure_unique_behavior_rows(df)

    # 衍生日期、小时、星期字段，保持后续清洗和指标脚本字段依赖不变。
    df["date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["behavior_time"].dt.hour
    df["weekday"] = df["behavior_time"].dt.weekday

    df = df.sort_values("behavior_time").reset_index(drop=True)
    df["behavior_time"] = df["behavior_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df[
        [
            "user_id",
            "item_id",
            "category_id",
            "behavior_type",
            "behavior_time",
            "date",
            "hour",
            "weekday",
        ]
    ]


def validate_data(df: pd.DataFrame) -> None:
    """检查生成数据是否满足第 3.5 轮真实性增强要求。"""
    required_behavior_types = {"pv", "cart", "fav", "buy"}
    actual_behavior_types = set(df["behavior_type"].unique())
    behavior_ratio = df["behavior_type"].value_counts(normalize=True)
    user_counts = df.groupby("behavior_type")["user_id"].nunique()
    pv_users = user_counts["pv"]

    item_category_count = df.groupby("item_id")["category_id"].nunique().max()
    checks = [
        (len(df) == ROW_COUNT, "数据行数不等于 120000"),
        (df["user_id"].nunique() == USER_COUNT, "用户数不等于 6000"),
        (df["item_id"].nunique() == ITEM_COUNT, "商品数不等于 2500"),
        (df["category_id"].nunique() == CATEGORY_COUNT, "类目数不等于 120"),
        (required_behavior_types.issubset(actual_behavior_types), "行为类型不完整"),
        (df["date"].nunique() == DAY_COUNT, "日期没有覆盖连续 30 天"),
        (not df.duplicated().any(), "存在重复行"),
        (item_category_count == 1, "存在一个 item_id 对应多个 category_id"),
        (0.88 <= behavior_ratio["pv"] <= 0.94, "pv 记录占比不在目标范围"),
        (0.03 <= behavior_ratio["cart"] <= 0.06, "cart 记录占比不在目标范围"),
        (0.02 <= behavior_ratio["fav"] <= 0.05, "fav 记录占比不在目标范围"),
        (0.008 <= behavior_ratio["buy"] <= 0.025, "buy 记录占比不在目标范围"),
        (0.08 <= user_counts["cart"] / pv_users <= 0.18, "cart 用户转化率不在目标范围"),
        (0.05 <= user_counts["fav"] / pv_users <= 0.15, "fav 用户转化率不在目标范围"),
        (0.02 <= user_counts["buy"] / pv_users <= 0.08, "buy 用户转化率不在目标范围"),
    ]

    for passed, error_message in checks:
        if not passed:
            raise ValueError(error_message)


def print_summary(df: pd.DataFrame, output_path: Path, project_root: Path) -> None:
    """打印数据基本信息，便于运行后自检。"""
    behavior_counts = df["behavior_type"].value_counts()
    behavior_ratios = df["behavior_type"].value_counts(normalize=True).mul(100).round(2)
    behavior_summary = pd.DataFrame({"count": behavior_counts, "ratio_percent": behavior_ratios})
    user_counts = df.groupby("behavior_type")["user_id"].nunique().reindex(["pv", "cart", "fav", "buy"])
    pv_users = user_counts["pv"]
    user_funnel = (user_counts / pv_users * 100).round(2)

    print("数据生成完成")
    print(f"数据行数: {len(df):,}")
    print(f"用户数: {df['user_id'].nunique():,}")
    print(f"商品数: {df['item_id'].nunique():,}")
    print(f"类目数: {df['category_id'].nunique():,}")
    print(f"时间范围: {df['behavior_time'].min()} 至 {df['behavior_time'].max()}")
    print("行为类型分布:")
    print(behavior_summary.to_string())
    print("用户级漏斗:")
    print(pd.DataFrame({"user_count": user_counts, "conversion_from_pv_percent": user_funnel}).to_string())
    print("每小时行为分布:")
    print(df["hour"].value_counts().sort_index().to_string())
    print(f"输出文件路径: {output_path.relative_to(project_root)}")


def main() -> None:
    """脚本入口函数。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = generate_sample_data()
        validate_data(df)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print_summary(df, output_path, project_root)
    except Exception as exc:
        print(f"数据生成失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
