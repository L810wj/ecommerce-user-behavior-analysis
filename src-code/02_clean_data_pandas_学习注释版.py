"""
第 2 步：使用 Pandas 清洗电商用户行为明细数据。

【本文件在项目中的位置】
上游：01_generate_sample_data.py
      生成 data/raw/user_behavior_raw.csv（可视为 ODS 原始日志）。
本文件：读取原始日志，去重、去除关键字段缺失/非法记录、统一时间字段，
        再生成行为标记列。
下游：04_calculate_metrics.py
      读取本文件输出的 DWD 明细，计算 PV、UV、漏斗、留存、RFM 等指标。

输入：data/raw/user_behavior_raw.csv
输出：data/processed/dwd_user_behavior_detail.csv

【学习时抓住三个问题】
1. 为什么先 copy：避免直接修改传入的原 DataFrame。
2. 为什么清洗后重算时间衍生列：保证 date/hour/weekday 与 behavior_time 一致。
3. 为什么生成 is_pv 等 0/1 列：让后续聚合可以直接 sum，减少重复判断。
"""

# 让类型注解按“延迟求值”方式处理，兼容部分较低版本的 Python。
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# 这里只保存“相对项目根目录”的路径；main() 中会再与 project_root 拼接。
# Path("data") / "raw" / ... 比手写 "data/raw/..." 更适合跨操作系统使用。
RAW_FILE = Path("data") / "raw" / "user_behavior_raw.csv"
OUTPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"

# 合法行为枚举。set（集合）的成员判断速度快，适合配合 isin 做过滤。
VALID_BEHAVIOR_TYPES = {"pv", "cart", "fav", "buy"}

# 这些字段共同决定一条行为日志是否还能被业务使用；任一缺失就无法准确归属。
#字段分别是：用户ID, 商品ID, 类别ID, 行为类型， 行为时间
KEY_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "behavior_time"]

# 最终输出字段及顺序。显式限定列可防止临时辅助列意外写入 DWD 文件。
OUTPUT_COLUMNS = [
    "user_id",
    "item_id",
    "category_id",
    "behavior_type",
    "behavior_time",
    "date",
    "hour",
    "weekday",
    "behavior_date",
    "is_pv",
    "is_cart",
    "is_fav",
    "is_buy",
]


def print_data_profile(df: pd.DataFrame, title: str) -> None:
    """
    打印数据概况，便于清洗前后对比。

    参数：
    - df：待检查的 DataFrame；既可以是原始数据，也可以是清洗结果。
    - title：打印标题，用来区分“清洗前”和“清洗后”。

    返回：无。此函数只负责观察数据，不修改 df，也不参与业务计算。
    """
    print(f"\n[{title}]")
    print(f"数据行数: {len(df):,}")
    print(f"字段列表: {list(df.columns)}")
    print("缺失值数量:")
    # isna() 得到逐单元格 True/False；sum() 按列统计 True 的数量。
    print(df.isna().sum().to_string())
    # duplicated() 默认判断整行是否重复；这里仅用于报告，不在此处删除。
    print(f"重复行数量: {df.duplicated().sum():,}")

    if "behavior_type" in df.columns:
        print("behavior_type 分布:")
        # 同时统计各行为的“数量”和“比例”，便于检查业务漏斗是否合理。
        # dropna=False 表示缺失值也要被统计出来，避免脏数据被悄悄忽略。
        behavior_summary = pd.DataFrame(
            {
                "count": df["behavior_type"].value_counts(dropna=False),
                "ratio_percent": df["behavior_type"]
                .value_counts(normalize=True, dropna=False)
                .mul(100)
                .round(2),
            }
        )
        print(behavior_summary.to_string())

    # 同一套去重统计逻辑应用于三个 ID 列，避免重复写三段代码。
    for column_name, label in [
        ("user_id", "user_id 去重数量"),
        ("item_id", "item_id 去重数量"),
        ("category_id", "category_id 去重数量"),
    ]:
        if column_name in df.columns:
            print(f"{label}: {df[column_name].nunique():,}")

    if "behavior_time" in df.columns:
        # errors="coerce"：无法解析的时间不会让这里报错，而会变成 NaT。
        behavior_time = pd.to_datetime(df["behavior_time"], errors="coerce")
        print(f"时间范围: {behavior_time.min()} 至 {behavior_time.max()}")


def clean_user_behavior(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    执行 DWD 明细层所需的基础清洗和字段标准化。
    输入：01 脚本生成或从 ODS 读取的原始行为 DataFrame。
    输出：字段顺序固定、可直接进入后续指标计算的 DWD DataFrame。

    处理顺序很重要：
    去重 → 删除关键字段缺失 → 过滤非法行为 → 解析时间 → 重建衍生字段
    → 生成行为标记 → 按时间排序。
    """
    # copy() 后，本函数对 df 的修改不会污染调用方持有的 raw_df。
    df = raw_df.copy()

    # 删除完全重复行，避免日志重复采集影响后续指标。
    # 默认按“所有列完全一致”判断重复，并保留第一次出现的记录。
    df = df.drop_duplicates()

    # 删除关键字段缺失记录，这些字段缺失会导致无法识别用户、商品、行为和时间。
    # subset 表示只针对关键字段判断缺失；非关键扩展字段缺失不必整行删除。
    df = df.dropna(subset=KEY_COLUMNS)

    # 只保留合法行为类型，防止脏值进入 DWD 明细层。
    # isin 返回布尔掩码，只留下 pv/cart/fav/buy 四种业务认可的行为。
    # 末尾 copy() 避免后续赋值触发 Pandas 的链式赋值问题。
    df = df[df["behavior_type"].isin(VALID_BEHAVIOR_TYPES)].copy()

    # 转换行为时间，无法解析的时间会变成 NaT，并在下一步删除。
    # errors="coerce" 会把错误日期（如 2024-13-50）转成 NaT，方便统一删除。
    df["behavior_time"] = pd.to_datetime(df["behavior_time"], errors="coerce")
    df = df.dropna(subset=["behavior_time"])

    # 重新生成所有时间衍生字段，确保它们和 behavior_time 保持一致。
    # .dt 是 Pandas 时间序列访问器；它可以批量提取日期、小时、星期等信息。
    df["date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["behavior_date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["behavior_time"].dt.hour.astype(int)
    # weekday 口径：周一=0，周日=6。
    df["weekday"] = df["behavior_time"].dt.weekday.astype(int)

    # 行为标记字段用于后续 DWS 聚合，避免 SQL/Pandas 反复写 CASE。
    # 条件比较先得到 True/False，再转成 1/0。
    # 例如后续 group["is_buy"].sum() 就能直接得到购买次数。
    df["is_pv"] = (df["behavior_type"] == "pv").astype(int)
    df["is_cart"] = (df["behavior_type"] == "cart").astype(int)
    df["is_fav"] = (df["behavior_type"] == "fav").astype(int)
    df["is_buy"] = (df["behavior_type"] == "buy").astype(int)

    # 按行为时间排序后重建 0~n-1 的连续索引；drop=True 表示不保留旧索引列。
    df = df.sort_values("behavior_time").reset_index(drop=True)
    # CSV 不保存 Pandas datetime 类型，因此写出前统一转成可读、稳定的字符串格式。
    df["behavior_time"] = df["behavior_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # 仅返回规定列，并同时固定列顺序，形成稳定的下游数据接口。
    return df[OUTPUT_COLUMNS]


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """
    校验清洗后的 DWD 数据是否满足下游使用要求。

    与 clean_user_behavior 的关系：
    - clean_user_behavior 负责“处理”；
    - validate_cleaned_data 负责“证明处理结果符合约定”。

    校验失败直接抛出 ValueError，阻止错误数据继续写入文件并污染后续指标。
    """
    if df.empty:
        raise ValueError("清洗后数据为空")

    # 两次 sum：第一次按列统计缺失数，第二次得到所有关键列的缺失总数。
    missing_key_count = df[KEY_COLUMNS].isna().sum().sum()
    if missing_key_count > 0:
        raise ValueError(f"关键字段仍存在缺失值: {missing_key_count}")

    # “实际值集合 - 合法值集合”得到所有非法行为；若为空集合则通过。
    invalid_behavior_types = set(df["behavior_type"].unique()) - VALID_BEHAVIOR_TYPES
    if invalid_behavior_types:
        raise ValueError(f"存在非法 behavior_type: {invalid_behavior_types}")

    behavior_time = pd.to_datetime(df["behavior_time"], errors="coerce")
    if behavior_time.isna().any():
        raise ValueError("behavior_time 仍存在无法解析的值")

    # 根据 behavior_time 重新计算一份“标准答案”，随后逐列与清洗结果比较。
    # 这样能发现日期、小时、星期字段虽然不为空，但内容与原时间不一致的情况。
    checks = pd.DataFrame(
        {
            "date": behavior_time.dt.strftime("%Y-%m-%d"),
            "hour": behavior_time.dt.hour,
            "weekday": behavior_time.dt.weekday,
            "behavior_date": behavior_time.dt.strftime("%Y-%m-%d"),
        }
    )

    if not (df["date"].astype(str).reset_index(drop=True) == checks["date"]).all():
        raise ValueError("date 字段与 behavior_time 不一致")
    if not (df["behavior_date"].astype(str).reset_index(drop=True) == checks["behavior_date"]).all():
        raise ValueError("behavior_date 字段与 behavior_time 不一致")
    if not (df["hour"].astype(int).reset_index(drop=True) == checks["hour"]).all():
        raise ValueError("hour 字段与 behavior_time 不一致")
    if not (df["weekday"].astype(int).reset_index(drop=True) == checks["weekday"]).all():
        raise ValueError("weekday 字段与 behavior_time 不一致")

    # 逐一校验四个 0/1 行为标记；每行理论上只有对应行为列为 1。
    for behavior_type, flag_column in [
        ("pv", "is_pv"),
        ("cart", "is_cart"),
        ("fav", "is_fav"),
        ("buy", "is_buy"),
    ]:
        expected_flag = (df["behavior_type"] == behavior_type).astype(int)
        if not (df[flag_column].astype(int).reset_index(drop=True) == expected_flag).all():
            raise ValueError(f"{flag_column} 字段生成不正确")


def main() -> None:
    """
    串联“定位项目目录 → 读文件 → 清洗 → 校验 → 写文件”的完整流程。

    main() 只负责编排，各具体功能拆到独立函数中，便于测试和复用。
    """
    try:
        # __file__ 是当前脚本路径；脚本位于 src/ 下，因此 parents[1] 是项目根目录。
        # 使用绝对项目根目录后，无论从哪个工作目录启动脚本，读写路径都稳定。
        project_root = Path(__file__).resolve().parents[1]
        raw_path = project_root / RAW_FILE
        output_path = project_root / OUTPUT_FILE

        # 提前给出带操作建议的错误，而不是让 read_csv 抛出不易理解的底层异常。
        if not raw_path.exists():
            raise FileNotFoundError(
                f"输入文件不存在: {RAW_FILE}，请先运行 python src/01_generate_sample_data.py"
            )

        # utf-8-sig 兼容带 BOM 的 CSV，在 Windows Excel 中打开中文也更稳定。
        raw_df = pd.read_csv(raw_path, encoding="utf-8-sig")
        print_data_profile(raw_df, "清洗前数据概况")

        # 先处理，再校验；只有校验通过，才继续输出。
        cleaned_df = clean_user_behavior(raw_df)
        validate_cleaned_data(cleaned_df)
        print_data_profile(cleaned_df, "清洗后数据概况")

        # parents=True 会递归创建 data/processed；exist_ok=True 表示目录已存在也不报错。
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # index=False：不把 DataFrame 行索引保存成多余的 CSV 列。
        cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("\n清洗完成")
        print(f"清洗前行数: {len(raw_df):,}")
        print(f"清洗后行数: {len(cleaned_df):,}")
        print(f"减少行数: {len(raw_df) - len(cleaned_df):,}")
        print(f"输出文件路径: {output_path.relative_to(project_root)}")
    except Exception as exc:
        # 统一在标准错误流中输出上下文后重新抛出，既便于人看，也保留完整报错堆栈。
        print(f"数据清洗失败: {exc}", file=sys.stderr)
        raise


# 只有“直接运行本文件”时才执行 main；被其他脚本 import 时不会自动开始清洗。
if __name__ == "__main__":
    main()
