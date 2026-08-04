"""
PySpark 离线 ETL：基于 DWD 明细数据生成 DWS 和 ADS 层结果。

如果本地暂未配置 PySpark 或 Java，可以先运行：
python src/04_calculate_metrics.py
完成 Pandas 版本的指标 CSV 和图表输出。
"""

from __future__ import annotations

import sys
from pathlib import Path


INPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"
SPARK_OUTPUT_DIR = Path("data") / "output" / "spark"


def get_spark_session():
    """创建本地 SparkSession，并对缺少 PySpark 的情况给出友好提示。"""
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError(
            "当前环境未安装 PySpark。可以先运行 python src/04_calculate_metrics.py "
            "完成 Pandas 版本指标输出；如需运行 Spark，请执行 pip install pyspark 并确认 Java 环境可用。"
        ) from exc

    return (
        SparkSession.builder.master("local[*]")
        .appName("ecommerce_user_behavior_etl")
        .getOrCreate()
    )


def write_csv(df, output_path: Path, project_root: Path) -> None:
    """写出 Spark CSV 结果，带 header，并打印行数和路径。"""
    row_count = df.count()
    (
        df.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(output_path))
    )
    print(f"输出 Spark 表: {output_path.relative_to(project_root)}，行数: {row_count:,}")


def run_spark_etl(project_root: Path) -> None:
    """执行 Spark ETL 主流程。"""
    input_path = project_root / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入文件不存在: {INPUT_FILE}，请先运行 python src/02_clean_data_pandas.py"
        )

    spark = get_spark_session()
    try:
        print("SparkSession 创建成功，appName=ecommerce_user_behavior_etl")
        df = (
            spark.read.option("header", True)
            .option("inferSchema", True)
            .csv(str(input_path))
        )

        # 统一字段类型，保证日期、小时、标记字段可以稳定聚合。
        df = df.createOrReplaceTempView("raw_view") or spark.sql(
            """
            SELECT
              user_id,
              item_id,
              category_id,
              behavior_type,
              CAST(behavior_time AS TIMESTAMP) AS behavior_time,
              CAST(date AS DATE) AS date,
              CAST(hour AS INT) AS hour,
              CAST(weekday AS INT) AS weekday,
              CAST(behavior_date AS DATE) AS behavior_date,
              CAST(is_pv AS INT) AS is_pv,
              CAST(is_cart AS INT) AS is_cart,
              CAST(is_fav AS INT) AS is_fav,
              CAST(is_buy AS INT) AS is_buy
            FROM raw_view
            """
        )
        df.createOrReplaceTempView("dwd_user_behavior_detail")
        input_count = df.count()
        print(f"读取 DWD 数据成功: {INPUT_FILE}，输入行数: {input_count:,}")

        # DWS 用户每日行为汇总。
        dws_user_daily_behavior = spark.sql(
            """
            SELECT
              behavior_date,
              user_id,
              SUM(is_pv) AS pv_count,
              SUM(is_cart) AS cart_count,
              SUM(is_fav) AS fav_count,
              SUM(is_buy) AS buy_count,
              COUNT(*) AS behavior_count,
              COUNT(DISTINCT hour) AS active_hours,
              COUNT(DISTINCT item_id) AS view_item_count,
              COUNT(DISTINCT category_id) AS view_category_count
            FROM dwd_user_behavior_detail
            GROUP BY behavior_date, user_id
            """
        )
        dws_user_daily_behavior.createOrReplaceTempView("dws_user_daily_behavior")

        # DWS 商品每日行为汇总。
        dws_item_daily_behavior = spark.sql(
            """
            SELECT
              behavior_date,
              item_id,
              MAX(category_id) AS category_id,
              SUM(is_pv) AS pv_count,
              SUM(is_cart) AS cart_count,
              SUM(is_fav) AS fav_count,
              SUM(is_buy) AS buy_count,
              COUNT(DISTINCT user_id) AS uv_count,
              COUNT(DISTINCT CASE WHEN is_buy = 1 THEN user_id END) AS buyer_count,
              CASE WHEN SUM(is_pv) = 0 THEN 0 ELSE ROUND(SUM(is_buy) / SUM(is_pv), 4) END AS click_to_buy_rate
            FROM dwd_user_behavior_detail
            GROUP BY behavior_date, item_id
            """
        )

        # DWS 类目每日行为汇总。
        dws_category_daily_behavior = spark.sql(
            """
            SELECT
              behavior_date,
              category_id,
              SUM(is_pv) AS pv_count,
              SUM(is_cart) AS cart_count,
              SUM(is_fav) AS fav_count,
              SUM(is_buy) AS buy_count,
              COUNT(DISTINCT user_id) AS uv_count,
              COUNT(DISTINCT CASE WHEN is_buy = 1 THEN user_id END) AS buyer_count,
              CASE WHEN SUM(is_pv) = 0 THEN 0 ELSE ROUND(SUM(is_buy) / SUM(is_pv), 4) END AS click_to_buy_rate
            FROM dwd_user_behavior_detail
            GROUP BY behavior_date, category_id
            """
        )

        # ADS 每日核心指标。
        ads_daily_core_metrics = spark.sql(
            """
            SELECT
              behavior_date,
              SUM(pv_count) AS pv,
              COUNT(DISTINCT user_id) AS uv,
              COUNT(DISTINCT user_id) AS dau,
              COUNT(DISTINCT CASE WHEN cart_count > 0 THEN user_id END) AS cart_users,
              COUNT(DISTINCT CASE WHEN fav_count > 0 THEN user_id END) AS fav_users,
              COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) AS buyers,
              SUM(buy_count) AS buy_count,
              CASE WHEN COUNT(DISTINCT user_id) = 0 THEN 0 ELSE ROUND(SUM(pv_count) / COUNT(DISTINCT user_id), 4) END AS avg_pv_per_user,
              CASE
                WHEN COUNT(DISTINCT user_id) = 0 THEN 0
                ELSE ROUND(COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) / COUNT(DISTINCT user_id), 4)
              END AS user_buy_rate
            FROM dws_user_daily_behavior
            GROUP BY behavior_date
            """
        )

        # ADS 转化漏斗。
        ads_conversion_funnel = spark.sql(
            """
            WITH stage_metrics AS (
              SELECT 'pv' AS behavior_stage, COUNT(DISTINCT user_id) AS user_count, COUNT(*) AS behavior_count, 1 AS stage_order
              FROM dwd_user_behavior_detail WHERE behavior_type = 'pv'
              UNION ALL
              SELECT 'cart', COUNT(DISTINCT user_id), COUNT(*), 2
              FROM dwd_user_behavior_detail WHERE behavior_type = 'cart'
              UNION ALL
              SELECT 'fav', COUNT(DISTINCT user_id), COUNT(*), 3
              FROM dwd_user_behavior_detail WHERE behavior_type = 'fav'
              UNION ALL
              SELECT 'buy', COUNT(DISTINCT user_id), COUNT(*), 4
              FROM dwd_user_behavior_detail WHERE behavior_type = 'buy'
            ),
            stage_with_base AS (
              SELECT
                behavior_stage,
                user_count,
                behavior_count,
                stage_order,
                FIRST_VALUE(user_count) OVER (ORDER BY stage_order) AS pv_user_count,
                LAG(user_count) OVER (ORDER BY stage_order) AS previous_user_count
              FROM stage_metrics
            )
            SELECT
              behavior_stage,
              user_count,
              behavior_count,
              stage_order,
              CASE WHEN pv_user_count = 0 THEN 0 ELSE ROUND(user_count / pv_user_count, 4) END AS conversion_rate_from_pv,
              CASE
                WHEN previous_user_count IS NULL THEN 1.0000
                WHEN previous_user_count = 0 THEN 0
                ELSE ROUND(user_count / previous_user_count, 4)
              END AS conversion_rate_from_previous_stage
            FROM stage_with_base
            """
        )

        # ADS 小时级行为趋势。
        ads_hourly_behavior_trend = spark.sql(
            """
            SELECT
              hour,
              SUM(is_pv) AS pv_count,
              SUM(is_cart) AS cart_count,
              SUM(is_fav) AS fav_count,
              SUM(is_buy) AS buy_count,
              COUNT(*) AS behavior_count,
              COUNT(DISTINCT user_id) AS active_users
            FROM dwd_user_behavior_detail
            GROUP BY hour
            ORDER BY hour
            """
        )

        outputs = {
            "dws_user_daily_behavior": dws_user_daily_behavior,
            "dws_item_daily_behavior": dws_item_daily_behavior,
            "dws_category_daily_behavior": dws_category_daily_behavior,
            "ads_daily_core_metrics": ads_daily_core_metrics,
            "ads_conversion_funnel": ads_conversion_funnel,
            "ads_hourly_behavior_trend": ads_hourly_behavior_trend,
        }

        for table_name, result_df in outputs.items():
            write_csv(result_df, project_root / SPARK_OUTPUT_DIR / table_name, project_root)

        print("PySpark ETL 执行完成")
    finally:
        spark.stop()


def main() -> None:
    """脚本入口函数。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        run_spark_etl(project_root)
    except Exception as exc:
        print(f"PySpark ETL 未能完成: {exc}", file=sys.stderr)
        print(
            "提示：如果是 PySpark 或 Java 环境问题，可以先运行 "
            "python src/04_calculate_metrics.py 完成 Pandas 版本指标和图表输出。",
            file=sys.stderr,
        )
        # 环境不可用时给出友好提示即可，不删除脚本，也不影响 Pandas 指标流程。
        return


if __name__ == "__main__":
    main()
