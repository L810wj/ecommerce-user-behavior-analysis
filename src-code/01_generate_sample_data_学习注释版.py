"""
第 1 步：生成更接近真实电商业务的模拟用户行为数据。

【本文件在项目中的位置】
本文件：从零构造用户、商品、行为和时间，输出 ODS 风格的原始行为日志。
下游一：02_clean_data_pandas.py 读取该 CSV，生成清洗后的 DWD 明细。
下游二：项目中的 SQL / PySpark 流程也可以把同一 CSV 当作 ODS 输入。
最终：04_calculate_metrics.py 基于 DWD 数据计算业务指标。

输出：data/raw/user_behavior_raw.csv
字段：user_id、item_id、category_id、behavior_type、behavior_time、
      date、hour、weekday。

【生成思路】
1. 先生成具有长尾活跃度的用户画像和商品目录；
2. 再限定浏览、加购、收藏、购买的人群规模与记录数量；
3. 分行为抽样用户和商品，形成更像真实业务的转化漏斗；
4. 最后生成带周末、活动日、晚高峰特征的时间，并完成质量校验。

注意：这是“可复现的模拟数据”，不是从真实电商平台采集的数据。
"""

# 延迟处理类型注解，提高不同 Python 版本下的兼容性。
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------- 全局生成参数 -------------------------
# 固定随机种子后，多次运行会得到相同结果，便于复现和对照学习。
RANDOM_SEED = 20240611
ROW_COUNT = 120_000  #总数据
USER_COUNT = 6_000  #用户数
ITEM_COUNT = 2_500 #商品数
CATEGORY_COUNT = 120  #类目数
DAY_COUNT = 30  #总天数
START_DATE = "2024-11-01"

# 使用 Path 拼接跨平台路径；main() 中还会在前面加项目根目录。
OUTPUT_FILE = Path("data") / "raw" / "user_behavior_raw.csv"

# 四种行为的“记录数”之和必须等于 ROW_COUNT。
# 这控制的是事件级分布：浏览事件最多，购买事件最少。
BEHAVIOR_COUNTS = {
    "pv": 111_000,   # 92.50% 浏览
    "cart": 4_800,  # 4.00% 加购
    "fav": 3_000,   # 2.50% 收藏
    "buy": 1_200,   # 1.00% 购买
}

# 下面控制的是“去重用户数”，用于形成用户级漏斗。
# 例如 BUY_USER_COUNT=300 表示约 5% 的浏览用户最终进入购买人群。
CART_USER_COUNT = 720  # pv 用户的 12%左右
FAV_USER_COUNT = 540   # pv 用户的 9%左右
BUY_USER_COUNT = 300   # pv 用户的 5%左右


def make_long_tail_weights(size: int, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """
    生成长度为 size 的长尾概率权重。

    长尾含义：少数用户/商品非常活跃或热门，大多数对象频率较低。
    alpha 越大，头部与尾部差异越明显；rng 负责可复现的随机打乱。
    返回值之和为 1，可直接作为 np.random.choice 的参数 p。
    """
    # ranks 为 1, 2, ..., size；1 / rank^alpha 会随排名增大而快速下降。
    ranks = np.arange(1, size + 1, dtype=float)
    weights = 1 / np.power(ranks, alpha)
    # 打乱后，高权重不会永远固定分配给编号靠前的用户或商品。
    rng.shuffle(weights)
    return weights / weights.sum()


def sample_with_full_coverage(
    base_values: np.ndarray,
    total_size: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    按权重有放回抽样，同时保证每个 base_values 至少出现一次。
    典型用途：
    - 保证 6000 名用户都至少有一次 pv；
    - 保证 2500 个商品、30 个日期都能被数据覆盖。
    与普通 rng.choice 的区别是：先放入一份全部基础值，再抽剩余数量。
    """
    if total_size < len(base_values):
        raise ValueError("total_size 不能小于 base_values 的数量")

    # replace=True 表示同一个用户、商品或日期可以被重复抽中。
    sampled_values = rng.choice(
        base_values,
        size=total_size - len(base_values),
        replace=True,
        p=weights,
    )
    # 合并“全覆盖部分”和“按概率补齐部分”，再打乱，避免基础值集中在开头。
    result = np.concatenate([base_values, sampled_values])
    rng.shuffle(result)
    return result


def weighted_choice_without_replacement(
    values: np.ndarray,
    size: int,
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    按权重进行不放回抽样。
    不放回意味着同一对象最多被选一次，适合挑选“哪些用户属于购买人群”；
    weights 越高的用户被选中的概率越大，但不会重复占用名额。
    """
    normalized_weights = weights / weights.sum()
    return rng.choice(values, size=size, replace=False, p=normalized_weights)


def build_user_profile(rng: np.random.Generator) -> pd.DataFrame:
    """
    构建 6000 名用户的模拟画像。
    输出一行一个用户，包含：
    - segment：低/中/高/VIP 活跃层级；
    - activity_weight：发生行为的相对概率；
    - conversion_weight：进入加购、收藏、购买人群的相对概率。
    该表仅用于生成数据，不会直接写入最终行为 CSV。
    """
    # f"U{i:06d}" 把数字补齐为 6 位，例如 1 -> U000001。
    user_ids = np.array([f"U{i:06d}" for i in range(1, USER_COUNT + 1)])
    # p 指定四种层级的总体比例，模拟“大多数普通用户、少数高活跃用户”。
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

    # 对数正态分布天然具有右偏长尾，再乘分层系数拉开用户活跃差异。
    base_activity = rng.lognormal(mean=0.0, sigma=1.0, size=USER_COUNT)
    activity_score = base_activity * np.array([segment_multiplier[s] for s in segments])
    activity_weight = activity_score / activity_score.sum()

    # 购买倾向和活跃度相关，但加入噪声，避免所有高活跃用户都购买。
    # Beta(2, 8) 多数取值偏低，符合购买用户少于浏览用户的常见情况。
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
    """
    构建商品目录以及不同行为下的商品抽样权重。
    输出一行一个商品，包含所属类目，以及 pv/cart/fav/buy 四套概率权重。
    同一个 item_id 在整个项目中只对应一个 category_id，后续 validate_data 会校验。
    """
    item_ids = np.array([f"I{i:06d}" for i in range(1, ITEM_COUNT + 1)])
    category_ids = np.array([f"C{i:03d}" for i in range(1, CATEGORY_COUNT + 1)])

    # 先放入每个类目一次，确保 120 个类目都有商品；再按长尾概率分配剩余商品。
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

    # popularity 表示流量热度；attractiveness 表示商品发生深层转化的潜力。
    popularity = make_long_tail_weights(ITEM_COUNT, alpha=1.08, rng=rng)
    attractiveness = rng.lognormal(mean=-0.2, sigma=0.7, size=ITEM_COUNT)

    # 商品转化能力不完全等同于热度：热门商品更容易被互动，但小部分非头部商品也可能转化较好。
    # 不同行为使用不同指数，让“热门”和“高转化”并非完全等价。
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
    """
    从全部用户中选出四种行为各自允许出现的用户池。
    关键关系：
    - pv：全部 6000 名用户；
    - buy：按转化倾向选 300 人；
    - cart：购买用户全部纳入，再补到 720 人；
    - fav：部分购买用户与其他高倾向用户组成，共 540 人。
    这样既能控制用户级转化率，也保留行为之间的交叉关系。
    """
    user_ids = user_profile["user_id"].to_numpy()
    conversion_weight = user_profile["conversion_weight"].to_numpy()

    # 先确定最小的购买人群；conversion_weight 高的用户更可能入选。
    buy_users = weighted_choice_without_replacement(
        user_ids,
        BUY_USER_COUNT,
        conversion_weight,
        rng,
    )

    # set 用于快速判断用户是否已在购买池；先排除，再补充非购买的加购用户。
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
    # 让所有购买用户都有加购资格，使总体漏斗更符合业务直觉。
    cart_users = np.concatenate([buy_users, cart_extra])

    # 收藏与购买并非严格前后关系，因此仅让部分购买用户与收藏用户重叠。
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
    """
    为某种行为生成指定数量的 user_id。

    输入 behavior_type 后，先取对应用户池，再计算池内抽样概率：
    - pv 主要由活跃度决定；
    - cart/fav/buy 同时考虑活跃度与购买倾向，后者权重更高。
    最后调用 sample_with_full_coverage，确保池内每位用户至少出现一次。
    """
    pool = user_pools[behavior_type]
    # 以 user_id 为索引并按 pool 顺序取行，保证权重与用户数组一一对应。
    pool_profile = user_profile.set_index("user_id").loc[pool]

    if behavior_type == "pv":
        weights = pool_profile["activity_weight"].to_numpy()
    else:
        # 非浏览行为更偏向高活跃、高购买倾向用户。
        # 0.35/0.65 是人为设置的业务模拟权重，不是模型训练得到的参数。
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
    """
    为某种行为生成指定数量的 item_id。

    字典把行为名称映射到对应权重列。pv 要保证所有商品至少被浏览一次；
    其他行为允许部分商品从未加购/收藏/购买，因此直接按概率有放回抽样。
    """
    item_ids = item_catalog["item_id"].to_numpy()
    # 用映射表替代多段 if/elif，使“行为 → 权重列”的关系一眼可见。
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
    """
    生成 total_size 个带业务规律的行为时间。
    日期层面：周末概率略高，双 11 等活动日进一步放大；
    小时层面：凌晨低、白天逐步上升、晚间达到高峰；
    分钟和秒：在 0~59 之间均匀随机。
    返回 pd.Series[datetime64]，稍后会赋给行为明细的 behavior_time。
    """
    start_time = pd.Timestamp(START_DATE)
    # day_offsets=0~29，和 START_DATE 相加得到连续 30 天。
    day_offsets = np.arange(DAY_COUNT)
    dates = start_time + pd.to_timedelta(day_offsets, unit="D")
    weekdays = dates.weekday

    # np.where：周六/周日赋 1.16，工作日赋 1.0。
    day_weights = np.where(weekdays >= 5, 1.16, 1.0).astype(float)
    campaign_dates = {
        pd.Timestamp("2024-11-11"): 1.35,
        pd.Timestamp("2024-11-17"): 1.20,
        pd.Timestamp("2024-11-24"): 1.18,
    }
    # 活动日系数继续乘到当天权重上；普通日期通过 get 默认取 1.0。
    for idx, date_value in enumerate(dates):
        day_weights[idx] *= campaign_dates.get(date_value, 1.0)

    # 乘少量随机噪声，避免每天完全按照固定模板分布；clip 限制波动范围。
    day_weights *= rng.normal(loc=1.0, scale=0.035, size=DAY_COUNT).clip(0.9, 1.12)
    day_weights = day_weights / day_weights.sum()
    # 使用全覆盖抽样，确保 30 天每一天至少有一条记录。
    sampled_days = sample_with_full_coverage(day_offsets, total_size, day_weights, rng)

    # 24 个数依次对应 0~23 时；数值越大，该小时被抽中的概率越高。
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

    # 日期、小时、分钟、秒分别抽样，再通过 Timedelta 相加组成完整时间戳。
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
    """
    消除五个业务关键字段完全相同的重复行为。

    用户和商品都可能重复出现，这是正常业务现象；只有 user/item/category/
    behavior_type/behavior_time 五项同时相同，才视为同一条重复事件。

    处理方法：最多循环 10 轮，把重复行时间向前或向后移动若干秒。
    返回修改后的 df；若仍无法消除重复则抛错，不输出不可靠数据。
    """
    key_columns = ["user_id", "item_id", "category_id", "behavior_type", "behavior_time"]
    start_time = pd.Timestamp(START_DATE)
    max_time = start_time + pd.Timedelta(days=DAY_COUNT) - pd.Timedelta(seconds=1)

    # 每轮重新寻找重复项，因为移动时间后理论上仍可能与另一条记录碰撞。
    for attempt in range(10):
        duplicated_indices = df.index[df.duplicated(subset=key_columns, keep="first")]
        if len(duplicated_indices) == 0:
            return df

        for offset, row_index in enumerate(duplicated_indices, start=1):
            # 不同重复行移动不同秒数；attempt 使下一轮使用更大的偏移量。
            seconds_to_shift = offset + attempt * len(duplicated_indices)
            new_time = df.at[row_index, "behavior_time"] + pd.Timedelta(seconds=seconds_to_shift)

            # 时间不能越过 30 天数据边界；超过上界就反向移动。
            if new_time > max_time:
                new_time = df.at[row_index, "behavior_time"] - pd.Timedelta(seconds=seconds_to_shift)
            if new_time < start_time:
                new_time = start_time + pd.Timedelta(seconds=seconds_to_shift)

            df.at[row_index, "behavior_time"] = new_time

    raise ValueError("去除重复行为记录失败，请调整数据生成参数")


def generate_sample_data() -> pd.DataFrame:
    """
    组织本文件全部生成函数，返回最终 12 万行行为明细。

    调用链：
    用户画像 + 商品目录 → 行为用户池 → 分行为抽样 → 拼接四类事件
    → 生成时间 → 去重 → 衍生字段 → 排序并固定输出列。
    """
    # Generator 是 NumPy 推荐的新式随机数接口；固定种子保证结果可复现。
    rng = np.random.default_rng(RANDOM_SEED)
    user_profile = build_user_profile(rng)
    item_catalog = build_item_catalog(rng)
    user_pools = build_user_pools(user_profile, rng)
    # 字典用于把抽到的 item_id 快速映射到其唯一 category_id。
    item_to_category = dict(zip(item_catalog["item_id"], item_catalog["category_id"]))

    frames = []
    # 每种行为单独抽样，以精确控制行为记录数及所用概率分布。
    for behavior_type in ["pv", "cart", "fav", "buy"]:
        sampled_users = sample_users_for_behavior(behavior_type, user_profile, user_pools, rng)
        sampled_items = sample_items_for_behavior(behavior_type, item_catalog, rng)
        # 当前行为形成一个临时 DataFrame，最后统一纵向拼接。
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

    # concat 默认按行纵向拼接；ignore_index=True 生成新的连续索引。
    df = pd.concat(frames, ignore_index=True)
    # 先打乱四类行为，避免所有 pv/cart/fav/buy 分块排列。
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    df["behavior_time"] = generate_behavior_times(len(df), rng)
    df = ensure_unique_behavior_rows(df)

    # 衍生日期、小时、星期字段，保持后续清洗和指标脚本字段依赖不变。
    # 这些列在 02 脚本中会根据 behavior_time 重新生成；
    # 原始层仍保留它们，便于直观看数据，并测试下游“重建衍生字段”的逻辑。
    df["date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["behavior_time"].dt.hour
    df["weekday"] = df["behavior_time"].dt.weekday

    # 时间排序让 CSV 更接近按事件时间写入的日志，也便于人工检查。
    df = df.sort_values("behavior_time").reset_index(drop=True)
    # 写 CSV 前统一转换成稳定的字符串格式。
    df["behavior_time"] = df["behavior_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # 显式返回规定列，构成稳定的 ODS 输入接口。
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
    """
    对生成结果做数据质量校验。

    校验分为四类：
    - 规模：总行数、用户/商品/类目数；
    - 完整性：四种行为、30 天是否覆盖；
    - 一致性：无重复、一个商品只属于一个类目；
    - 业务合理性：事件占比和用户级转化率是否落在目标范围。

    任一条件失败就抛出 ValueError，main() 将不会写出 CSV。
    """
    required_behavior_types = {"pv", "cart", "fav", "buy"}
    actual_behavior_types = set(df["behavior_type"].unique())
    behavior_ratio = df["behavior_type"].value_counts(normalize=True)
    user_counts = df.groupby("behavior_type")["user_id"].nunique()
    pv_users = user_counts["pv"]

    # 对每个商品统计类目去重数，再取最大值；最大值为 1 才满足一对一映射。
    item_category_count = df.groupby("item_id")["category_id"].nunique().max()
    # 每个元组为 (是否通过, 失败提示)，集中管理比多段 if 更易阅读。
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
    """
    打印运行摘要，帮助学习者快速确认数据是否符合预期。

    此函数不改变数据，只展示规模、行为事件分布、用户级漏斗、小时分布和路径。
    """
    # value_counts() 统计事件数；normalize=True 时统计比例。
    behavior_counts = df["behavior_type"].value_counts()
    behavior_ratios = df["behavior_type"].value_counts(normalize=True).mul(100).round(2)
    behavior_summary = pd.DataFrame({"count": behavior_counts, "ratio_percent": behavior_ratios})
    # nunique 统计每种行为的去重用户数；reindex 固定漏斗展示顺序。
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
    """
    脚本入口：定位目录 → 生成数据 → 校验 → 保存 CSV → 打印摘要。

    具体业务逻辑拆在其他函数中，main() 只负责编排执行顺序。
    """
    try:
        # 当前文件通常位于项目 src/ 目录，所以 parents[1] 取得项目根目录。
        project_root = Path(__file__).resolve().parents[1]
        output_path = project_root / OUTPUT_FILE
        # 若 data/raw 不存在则自动递归创建。
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = generate_sample_data()
        validate_data(df)
        # index=False 不写出无业务意义的 DataFrame 索引；
        # utf-8-sig 兼容 Windows Excel 直接打开中文 CSV。
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print_summary(df, output_path, project_root)
    except Exception as exc:
        # 先输出易读提示，再重新抛错以保留调试所需的完整堆栈。
        print(f"数据生成失败: {exc}", file=sys.stderr)
        raise


# 直接执行本文件才运行 main；被 import 时仅提供函数，不自动生成数据。
if __name__ == "__main__":
    main()
