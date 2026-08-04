"""
第 4 步：计算电商用户行为指标，并输出看板 CSV 与基础图表。

【本文件在项目中的位置】
上游：02_clean_data_pandas.py
      输出 data/processed/dwd_user_behavior_detail.csv（DWD 行为明细）。
本文件：从同一份 DWD 明细分别计算每日指标、转化漏斗、留存、RFM、
        热门商品、热门类目和小时趋势。
下游：输出到 dashboard/dashboard_data 供 Power BI / Excel 建看板；
      同时输出到 outputs/metrics 供结果检查，并在 outputs/figures 保存图表。

【核心数据流】
一份 DWD DataFrame
  ├─按日期分组 → PV / UV / DAU / 购买转化率
  ├─按行为类型 → 整体转化漏斗
  ├─按用户和日期 → 1/3/7 日活跃留存
  ├─按用户购买行为 → RFM 用户价值分层
  ├─按商品/类目分组 → Top 20
  └─按小时分组 → 小时级行为趋势

本脚本是 Pandas 本地分析版本，不依赖 Spark。学习时重点关注每个指标的
“统计粒度、分子、分母和去重口径”，而不是只记 Pandas 语法。
"""

# 延迟处理类型注解，提高不同 Python 版本下的兼容性。
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

# Agg 是不需要图形界面的绘图后端，适合服务器或命令行批量保存 PNG。
# 必须在 import pyplot 之前设置。
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------- 输入与输出路径 -------------------------
# 都是相对项目根目录的路径，main() / load_dwd_data() 中再拼成绝对路径。
INPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"
DASHBOARD_DATA_DIR = Path("dashboard") / "dashboard_data"
METRICS_DIR = Path("outputs") / "metrics"
FIGURES_DIR = Path("outputs") / "figures"

# 漏斗必须按业务顺序展示，而不能依赖 value_counts 的默认排序。
STAGE_ORDER = ["pv", "cart", "fav", "buy"]
# 当前项目计算次日、第 3 日、第 7 日的活跃留存。
RETENTION_DAYS = [1, 3, 7]


def safe_divide(numerator: float, denominator: float) -> float:
    """
    计算 numerator / denominator，并统一保留 4 位小数。

    当分母为 0 时返回 0.0，避免转化率等指标出现 ZeroDivisionError。
    这是项目内所有比率指标的统一口径入口。
    """
    if denominator == 0:
        return 0.0
    return round(float(numerator) / float(denominator), 4)


def load_dwd_data(project_root: Path) -> pd.DataFrame:
    """
    读取 02 脚本生成的 DWD CSV，并恢复计算所需的数据类型。

    CSV 本身不保存 datetime 类型，所以行为时间读入后默认是字符串；
    本函数将 behavior_time 恢复为时间类型，将 behavior_date 规范成日期字符串，
    并确保 hour 为整数。返回值是后续所有指标函数共享的输入 DataFrame。
    """
    input_path = project_root / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入文件不存在: {INPUT_FILE}，请先运行 python src/02_clean_data_pandas.py"
        )

    # utf-8-sig 与上游写出编码保持一致。
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    # errors="coerce" 把无法解析的值转成 NaT，随后统一校验。
    df["behavior_time"] = pd.to_datetime(df["behavior_time"], errors="coerce")
    # 先转 datetime 校验格式，再输出 YYYY-MM-DD 字符串，方便 groupby 和写 CSV。
    df["behavior_date"] = pd.to_datetime(df["behavior_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["hour"] = df["hour"].astype(int)

    # any() 表示只要有一个无法解析的时间，就阻止后续指标计算。
    if df["behavior_time"].isna().any() or df["behavior_date"].isna().any():
        raise ValueError("DWD 数据存在无法解析的时间字段")

    print(f"读取 DWD 数据成功: {INPUT_FILE}")
    print(f"输入行数: {len(df):,}")
    return df


def calculate_daily_core_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 behavior_date 计算每日核心指标。

    指标口径：
    - PV：当日 is_pv 之和，即浏览事件次数；
    - UV：当日发生任意行为的去重用户数；
    - DAU：因数据没有登录事件，用当日行为 UV 近似；
    - buyers：当日发生购买的去重用户数；
    - user_buy_rate：buyers / dau，是用户级购买转化率。

    返回一行一个日期的结果表，可直接绘制每日趋势或接入看板。
    """
    rows = []
    # group 是某一天的全部用户行为明细。
    for behavior_date, group in df.groupby("behavior_date"):
        # is_pv/is_buy 是 02 脚本生成的 0/1 列，sum 即事件次数。
        pv = int(group["is_pv"].sum())
        # nunique 是“去重计数”；同一用户当日多次行为只计一个 UV。
        uv = int(group["user_id"].nunique())
        dau = uv
        # loc[条件, 列]：先筛选指定行为，再对 user_id 去重计数。
        cart_users = int(group.loc[group["is_cart"] == 1, "user_id"].nunique())
        fav_users = int(group.loc[group["is_fav"] == 1, "user_id"].nunique())
        buyers = int(group.loc[group["is_buy"] == 1, "user_id"].nunique())
        buy_count = int(group["is_buy"].sum())
        # 每天先组装成字典，再追加到列表，最后一次性转为 DataFrame。
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

    # 按日期升序输出，并重建连续索引。
    return pd.DataFrame(rows).sort_values("behavior_date").reset_index(drop=True)


def calculate_conversion_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算全观察期的用户行为漏斗：pv → cart → fav → buy。

    每一层同时保留：
    - user_count：发生该行为的去重用户数（用户级漏斗主要使用它）；
    - behavior_count：该行为的事件总次数；
    - conversion_rate_from_pv：当前层用户数 / 浏览用户数；
    - conversion_rate_from_previous_stage：当前层 / 前一层。

    注意：这是按“各行为去重用户数”计算的简化漏斗，不要求同一用户严格按照
    pv→cart→fav→buy 的先后路径逐层流转。
    """
    metrics = []
    previous_user_count = None
    pv_user_count = None

    # enumerate(..., start=1) 同时得到业务顺序编号和行为名。
    for stage_order, behavior_stage in enumerate(STAGE_ORDER, start=1):
        stage_df = df[df["behavior_type"] == behavior_stage]
        user_count = int(stage_df["user_id"].nunique())
        behavior_count = int(len(stage_df))

        # 第一层保存为统一基准，供后面三层计算“相对 pv 转化率”。
        if behavior_stage == "pv":
            pv_user_count = user_count

        metrics.append(
            {
                "behavior_stage": behavior_stage,
                "user_count": user_count,
                "behavior_count": behavior_count,
                "stage_order": stage_order,
                "conversion_rate_from_pv": safe_divide(user_count, pv_user_count or 0),
                # 第一层没有上一层，约定自身转化率为 1.0（100%）。
                "conversion_rate_from_previous_stage": 1.0
                if previous_user_count is None
                else safe_divide(user_count, previous_user_count),
            }
        )
        previous_user_count = user_count

    return pd.DataFrame(metrics)


def calculate_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 1 日、3 日、7 日活跃留存。

    项目没有注册时间，因此采用“活跃留存”口径：
    某基准日活跃过的用户，在基准日后的第 N 天是否再次发生任意行为。

    retention_rate = retained_users / base_active_users
    输出一行对应“一个基准日 + 一个留存天数”。
    """
    # 同一用户同一天无论有多少条行为，都只保留一个活跃标记。
    user_date = df[["user_id", "behavior_date"]].drop_duplicates().copy()
    user_date["behavior_date"] = pd.to_datetime(user_date["behavior_date"])
    # set 中存 (user_id, date) 元组，可快速判断某用户在目标日是否活跃。
    active_map = set(zip(user_date["user_id"], user_date["behavior_date"]))
    rows = []

    for base_date, base_group in user_date.groupby("behavior_date"):
        base_users = set(base_group["user_id"])
        for retention_day in RETENTION_DAYS:
            # 例如基准日 11-01、retention_day=3，则目标日是 11-04。
            target_date = base_date + pd.Timedelta(days=retention_day)
            # 集合推导式：遍历基准用户，只保留目标日仍在 active_map 中的用户。
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
    """
    把“距离最近购买的天数”映射为 1~5 分的 R 分数。

    天数越小代表最近购买越近、活跃价值越高，因此得分越高。
    这些阈值是本项目的演示口径，真实业务应结合购买周期重新设定。
    """
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
    """
    把购买次数映射为 1~5 分。

    本项目 F 使用购买次数，M 因缺少订单金额也暂用购买次数近似，
    所以 F 和 M 共用这套评分函数。
    """
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
    """
    根据单个用户的一行 RFM 结果，返回用户价值标签。

    pd.DataFrame.apply(..., axis=1) 会把每行作为 Series 传入本函数。
    判断顺序从特殊情况和高价值人群开始，命中后立即 return。
    """
    # 未购买用户必须优先单独判断，否则其默认分数可能被误分到其他层级。
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
    """
    计算每位用户的 RFM 指标、得分和价值分层。

    R（Recency）：最近一次购买距观察截止日的天数；
    F（Frequency）：购买事件次数；
    M（Monetary）：通常为消费金额，但本数据没有价格，暂用购买次数近似。

    输出一行一个用户，并保留未购买用户；未购买用户的 recency_days=999。
    """
    # 观察截止日取数据中的最大日期，而不是机器当前日期，保证结果可复现。
    observation_date = pd.to_datetime(df["behavior_date"]).max()
    buy_df = df[df["is_buy"] == 1].copy()
    # 从全量行为取用户全集，避免最终 RFM 只剩购买用户。
    all_users = pd.DataFrame({"user_id": sorted(df["user_id"].unique())})

    if buy_df.empty:
        # 极端情况下没有购买行为，也要构造结构完整的用户表。
        rfm = all_users.copy()
        rfm["last_buy_date"] = pd.NaT
        rfm["frequency"] = 0
    else:
        # Named Aggregation：每个用户聚合最近购买日期和购买次数。
        buy_group = (
            buy_df.groupby("user_id")
            .agg(
                last_buy_date=("behavior_date", "max"),
                frequency=("is_buy", "sum"),
            )
            .reset_index()
        )
        # left join 保留全体用户；未购买者的购买字段暂时为空。
        rfm = all_users.merge(buy_group, on="user_id", how="left")

    rfm["last_buy_date"] = pd.to_datetime(rfm["last_buy_date"], errors="coerce")
    rfm["frequency"] = rfm["frequency"].fillna(0).astype(int)
    # 当前 CSV 无商品价格/订单金额，因此 monetary 只能用 frequency 近似。
    rfm["monetary"] = rfm["frequency"]
    # 日期相减得到 Timedelta，再取 .dt.days；未购买者填 999 作为特殊值。
    rfm["recency_days"] = (
        observation_date - rfm["last_buy_date"]
    ).dt.days.fillna(999).astype(int)

    # Series.apply 把评分函数应用到每一个值。
    rfm["r_score"] = rfm["recency_days"].apply(score_recency)
    rfm["f_score"] = rfm["frequency"].apply(score_frequency)
    rfm["m_score"] = rfm["monetary"].apply(score_frequency)
    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    # axis=1 表示逐行调用 segment_user。
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
    # 优先展示总分高、购买频次高的用户；同分时按 user_id 升序保证结果稳定。
    ].sort_values(["rfm_score", "frequency", "user_id"], ascending=[False, False, True])


def calculate_top_items(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 item_id + category_id 粒度汇总并筛选热门商品 Top 20。

    主排序依据是 pv_count，其次是 buy_count；同时保留加购、收藏、购买次数、
    购买用户数及点击购买率，方便看板同时判断“流量”和“转化”。
    """
    # groupby 后每个分组代表一个商品；由于上游保证商品与类目一对一，
    # category_id 可以安全地作为分组键一起保留。
    item_metrics = (
        df.groupby(["item_id", "category_id"])
        .agg(
            pv_count=("is_pv", "sum"),
            cart_count=("is_cart", "sum"),
            fav_count=("is_fav", "sum"),
            buy_count=("is_buy", "sum"),
            # x 是当前商品分组内的 user_id Series；用相同索引回到原 df 找购买行，
            # 再对购买用户去重，得到 buyer_count。
            buyer_count=("user_id", lambda x: x[df.loc[x.index, "is_buy"] == 1].nunique()),
        )
        .reset_index()
    )
    # 商品点击购买率 = 购买事件数 / 浏览事件数；safe_divide 处理 pv=0。
    item_metrics["click_to_buy_rate"] = item_metrics.apply(
        lambda row: safe_divide(row["buy_count"], row["pv_count"]), axis=1
    )
    item_metrics = item_metrics.sort_values(
        ["pv_count", "buy_count", "item_id"], ascending=[False, False, True]
    ).head(20)
    # 排序截取后再生成 1~20 的名次。
    item_metrics["rank_no"] = range(1, len(item_metrics) + 1)
    return item_metrics.reset_index(drop=True)


def calculate_top_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 category_id 粒度汇总并筛选热门类目 Top 20。

    计算方式与热门商品一致，只是统计粒度从商品提升到类目。
    """
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
    """
    按 0~23 时汇总四种行为次数、总行为数和活跃用户数。

    behavior_count 是事件量；active_users 是去重用户量。
    两者结合可以区分“更多用户来了”和“同一批用户操作更频繁”。
    """
    hourly = (
        df.groupby("hour")
        .agg(
            pv_count=("is_pv", "sum"),
            cart_count=("is_cart", "sum"),
            fav_count=("is_fav", "sum"),
            buy_count=("is_buy", "sum"),
            # size 统计分组行数，即该小时全部行为事件数。
            behavior_count=("user_id", "size"),
            active_users=("user_id", "nunique"),
        )
        .reset_index()
        .sort_values("hour")
    )
    return hourly


def save_metric_table(df: pd.DataFrame, file_name: str, project_root: Path) -> None:
    """
    把同一指标表同时写到两个目录。

    - dashboard/dashboard_data：给 Power BI / Excel 等看板工具读取；
    - outputs/metrics：作为本地分析结果和质量检查副本。

    数据内容完全相同，只是下游用途不同。
    """
    for directory in [DASHBOARD_DATA_DIR, METRICS_DIR]:
        output_dir = project_root / directory
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / file_name
        # index=False 避免写出无业务意义的行索引；utf-8-sig 方便 Excel 打开中文。
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"输出 CSV: {output_path.relative_to(project_root)}，行数: {len(df):,}")


def setup_plot_style() -> None:
    """统一设置 Matplotlib 字体和负号显示规则。"""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def save_daily_pv_uv_chart(daily: pd.DataFrame, output_path: Path) -> None:
    """根据 daily_core_metrics 保存每日 PV / UV 折线图。"""
    # fig 是整张画布，ax 是具体坐标轴；后续绘图方法主要调用 ax。
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(daily["behavior_date"])
    ax.plot(x, daily["pv"], marker="o", label="PV")
    ax.plot(x, daily["uv"], marker="o", label="UV")
    ax.set_title("Daily PV and UV Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)
    # 自动旋转日期刻度，减少重叠；tight_layout 自动调整边距。
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    # 脚本批量出图时及时关闭，避免多张图累计占用内存。
    plt.close(fig)


def save_conversion_funnel_chart(funnel: pd.DataFrame, output_path: Path) -> None:
    """根据 conversion_funnel 保存用户数柱状漏斗图。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(funnel["behavior_stage"], funnel["user_count"], color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_title("User Behavior Conversion Funnel")
    ax.set_xlabel("Stage")
    ax.set_ylabel("User Count")
    # 在每根柱顶标注“相对 pv 的转化率”；.2% 会显示两位百分数。
    for idx, row in funnel.iterrows():
        ax.text(idx, row["user_count"], f"{row['conversion_rate_from_pv']:.2%}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_retention_heatmap(retention: pd.DataFrame, output_path: Path) -> None:
    """根据 retention_analysis 保存“基准日 × 留存天数”热力图。"""
    # pivot 把长表转成矩阵：行=base_date，列=retention_day，值=retention_rate。
    matrix = retention.pivot(index="base_date", columns="retention_day", values="retention_rate").fillna(0)
    # 日期越多图越高，同时把高度限制在 6~12 英寸内。
    fig_height = max(6, min(12, len(matrix) * 0.28))
    fig, ax = plt.subplots(figsize=(7, fig_height))
    # vmax 至少为 0.01，避免全 0 数据导致颜色范围退化。
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
    """根据 rfm_user_value 统计各价值层级人数并保存柱状图。"""
    # value_counts 默认按人数从多到少排列。
    segment_counts = rfm["user_value_segment"].value_counts()
    # 图中使用英文标签，避免运行环境缺少中文字体而显示方框。
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
    """根据 hourly_behavior_trend 保存事件量与活跃用户数折线图。"""
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
    """
    统一调度五个绘图函数。

    metrics 字典由 main() 生成；chart_jobs 把“文件名、绘图函数、数据表”
    配成任务列表，避免重复写五段相似的保存代码。
    """
    figures_dir = project_root / FIGURES_DIR
    figures_dir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    # 函数在 Python 中也是对象，可以放进列表，随后通过 chart_func(...) 调用。
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
    """
    串联完整指标分析流程。

    1. 定位项目根目录并读取 DWD；
    2. 调用七个互相独立的指标函数；
    3. 将结果保存为 CSV；
    4. 基于部分指标表生成 PNG 图表。
    """
    try:
        # 当前脚本通常放在项目 src/ 下，parents[1] 取得项目根目录。
        project_root = Path(__file__).resolve().parents[1]
        df = load_dwd_data(project_root)

        # 字典把“指标名”与“计算结果 DataFrame”统一管理。
        # 所有函数只读取 df，不会互相覆盖或修改彼此结果。
        metrics = {
            "daily_core_metrics": calculate_daily_core_metrics(df),
            "conversion_funnel": calculate_conversion_funnel(df),
            "retention_analysis": calculate_retention(df),
            "rfm_user_value": calculate_rfm(df),
            "top_items": calculate_top_items(df),
            "top_categories": calculate_top_categories(df),
            "hourly_behavior_trend": calculate_hourly_trend(df),
        }

        # 指标名与输出文件名分开配置，便于下游稳定引用固定 CSV 文件。
        output_files = {
            "daily_core_metrics": "daily_core_metrics.csv",
            "conversion_funnel": "conversion_funnel.csv",
            "retention_analysis": "retention_analysis.csv",
            "rfm_user_value": "rfm_user_value.csv",
            "top_items": "top_items.csv",
            "top_categories": "top_categories.csv",
            "hourly_behavior_trend": "hourly_behavior_trend.csv",
        }

        # items() 同时遍历字典的键和值；用键从 metrics 取对应 DataFrame。
        for metric_name, file_name in output_files.items():
            save_metric_table(metrics[metric_name], file_name, project_root)

        save_figures(metrics, project_root)
        print("\n业务指标和图表输出完成")
    except Exception as exc:
        # 输出易读提示后重新抛错，保留完整 traceback 供定位问题。
        print(f"业务指标计算失败: {exc}", file=sys.stderr)
        raise


# 直接运行本文件时才执行 main；被 import 时只提供可复用的指标函数。
if __name__ == "__main__":
    main()
