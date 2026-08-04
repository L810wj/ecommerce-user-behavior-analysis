"""
Pandas 清洗电商用户行为明细数据。

输入：data/raw/user_behavior_raw.csv
输出：data/processed/dwd_user_behavior_detail.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


RAW_FILE = Path("data") / "raw" / "user_behavior_raw.csv"
OUTPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"
VALID_BEHAVIOR_TYPES = {"pv", "cart", "fav", "buy"}
KEY_COLUMNS = ["user_id", "item_id", "category_id", "behavior_type", "behavior_time"]
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
    """打印数据概况，便于清洗前后对比。"""
    print(f"\n[{title}]")
    print(f"数据行数: {len(df):,}")
    print(f"字段列表: {list(df.columns)}")
    print("缺失值数量:")
    print(df.isna().sum().to_string())
    print(f"重复行数量: {df.duplicated().sum():,}")

    if "behavior_type" in df.columns:
        print("behavior_type 分布:")
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

    for column_name, label in [
        ("user_id", "user_id 去重数量"),
        ("item_id", "item_id 去重数量"),
        ("category_id", "category_id 去重数量"),
    ]:
        if column_name in df.columns:
            print(f"{label}: {df[column_name].nunique():,}")

    if "behavior_time" in df.columns:
        behavior_time = pd.to_datetime(df["behavior_time"], errors="coerce")
        print(f"时间范围: {behavior_time.min()} 至 {behavior_time.max()}")


def clean_user_behavior(raw_df: pd.DataFrame) -> pd.DataFrame:
    """执行 DWD 明细层所需的基础清洗和字段标准化。"""
    df = raw_df.copy()

    # 删除完全重复行，避免日志重复采集影响后续指标。
    df = df.drop_duplicates()

    # 删除关键字段缺失记录，这些字段缺失会导致无法识别用户、商品、行为和时间。
    df = df.dropna(subset=KEY_COLUMNS)

    # 只保留合法行为类型，防止脏值进入 DWD 明细层。
    df = df[df["behavior_type"].isin(VALID_BEHAVIOR_TYPES)].copy()

    # 转换行为时间，无法解析的时间会变成 NaT，并在下一步删除。
    df["behavior_time"] = pd.to_datetime(df["behavior_time"], errors="coerce")
    df = df.dropna(subset=["behavior_time"])

    # 重新生成所有时间衍生字段，确保它们和 behavior_time 保持一致。
    df["date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["behavior_date"] = df["behavior_time"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["behavior_time"].dt.hour.astype(int)
    df["weekday"] = df["behavior_time"].dt.weekday.astype(int)

    # 行为标记字段用于后续 DWS 聚合，避免 SQL/Pandas 反复写 CASE。
    df["is_pv"] = (df["behavior_type"] == "pv").astype(int)
    df["is_cart"] = (df["behavior_type"] == "cart").astype(int)
    df["is_fav"] = (df["behavior_type"] == "fav").astype(int)
    df["is_buy"] = (df["behavior_type"] == "buy").astype(int)

    df = df.sort_values("behavior_time").reset_index(drop=True)
    df["behavior_time"] = df["behavior_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df[OUTPUT_COLUMNS]


def validate_cleaned_data(df: pd.DataFrame) -> None:
    """校验清洗后的 DWD 明细数据是否满足本阶段要求。"""
    if df.empty:
        raise ValueError("清洗后数据为空")

    missing_key_count = df[KEY_COLUMNS].isna().sum().sum()
    if missing_key_count > 0:
        raise ValueError(f"关键字段仍存在缺失值: {missing_key_count}")

    invalid_behavior_types = set(df["behavior_type"].unique()) - VALID_BEHAVIOR_TYPES
    if invalid_behavior_types:
        raise ValueError(f"存在非法 behavior_type: {invalid_behavior_types}")

    behavior_time = pd.to_datetime(df["behavior_time"], errors="coerce")
    if behavior_time.isna().any():
        raise ValueError("behavior_time 仍存在无法解析的值")

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
    """脚本入口函数。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        raw_path = project_root / RAW_FILE
        output_path = project_root / OUTPUT_FILE

        if not raw_path.exists():
            raise FileNotFoundError(
                f"输入文件不存在: {RAW_FILE}，请先运行 python src/01_generate_sample_data.py"
            )

        raw_df = pd.read_csv(raw_path, encoding="utf-8-sig")
        print_data_profile(raw_df, "清洗前数据概况")

        cleaned_df = clean_user_behavior(raw_df)
        validate_cleaned_data(cleaned_df)
        print_data_profile(cleaned_df, "清洗后数据概况")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print("\n清洗完成")
        print(f"清洗前行数: {len(raw_df):,}")
        print(f"清洗后行数: {len(cleaned_df):,}")
        print(f"减少行数: {len(raw_df) - len(cleaned_df):,}")
        print(f"输出文件路径: {output_path.relative_to(project_root)}")
    except Exception as exc:
        print(f"数据清洗失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
