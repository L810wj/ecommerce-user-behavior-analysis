-- ODS 到 DWD 清洗 SQL
-- 作用：将 ods_user_behavior_raw 中的原始日志清洗为标准 DWD 明细表。

USE ecommerce_behavior_warehouse;

-- 重跑时先清空 DWD 明细层，保证结果可重复。
TRUNCATE TABLE dwd_user_behavior_detail;

-- MySQL WEEKDAY(date) 返回 0-6，其中 0 表示周一，6 表示周日。
-- 这与 Python pandas dt.weekday 的映射保持一致。
INSERT INTO dwd_user_behavior_detail (
  user_id,
  item_id,
  category_id,
  behavior_type,
  behavior_time,
  date,
  hour,
  weekday,
  behavior_date,
  is_pv,
  is_cart,
  is_fav,
  is_buy
)
-- 使用 SELECT DISTINCT 增强去重能力，避免 ODS 重复采集记录进入 DWD 明细层。
SELECT DISTINCT
  TRIM(user_id) AS user_id,
  TRIM(item_id) AS item_id,
  TRIM(category_id) AS category_id,
  TRIM(behavior_type) AS behavior_type,
  behavior_time,
  DATE(behavior_time) AS date,
  HOUR(behavior_time) AS hour,
  WEEKDAY(behavior_time) AS weekday,
  DATE(behavior_time) AS behavior_date,
  CASE WHEN TRIM(behavior_type) = 'pv' THEN 1 ELSE 0 END AS is_pv,
  CASE WHEN TRIM(behavior_type) = 'cart' THEN 1 ELSE 0 END AS is_cart,
  CASE WHEN TRIM(behavior_type) = 'fav' THEN 1 ELSE 0 END AS is_fav,
  CASE WHEN TRIM(behavior_type) = 'buy' THEN 1 ELSE 0 END AS is_buy
FROM ods_user_behavior_raw
WHERE user_id IS NOT NULL
  AND item_id IS NOT NULL
  AND category_id IS NOT NULL
  AND behavior_type IS NOT NULL
  AND behavior_time IS NOT NULL
  AND TRIM(user_id) <> ''
  AND TRIM(item_id) <> ''
  AND TRIM(category_id) <> ''
  AND TRIM(behavior_type) IN ('pv', 'cart', 'fav', 'buy');
