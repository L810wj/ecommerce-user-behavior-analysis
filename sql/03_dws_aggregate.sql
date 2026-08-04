-- DWS 聚合 SQL
-- 作用：基于 DWD 明细层生成用户、商品、类目三个常用分析粒度的每日汇总。

USE ecommerce_behavior_warehouse;

-- 用户 + 日期粒度：用于 DAU、用户活跃、留存和 RFM 分析。
TRUNCATE TABLE dws_user_daily_behavior;

INSERT INTO dws_user_daily_behavior (
  behavior_date,
  user_id,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  behavior_count,
  active_hours,
  view_item_count,
  view_category_count
)
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
GROUP BY behavior_date, user_id;

-- 商品 + 日期粒度：用于商品热度、商品转化率和 Top 商品分析。
TRUNCATE TABLE dws_item_daily_behavior;

INSERT INTO dws_item_daily_behavior (
  behavior_date,
  item_id,
  category_id,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  uv_count,
  buyer_count,
  click_to_buy_rate
)
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
  CASE
    WHEN SUM(is_pv) = 0 THEN 0
    ELSE ROUND(SUM(is_buy) / SUM(is_pv), 4)
  END AS click_to_buy_rate
FROM dwd_user_behavior_detail
GROUP BY behavior_date, item_id;

-- 类目 + 日期粒度：用于类目热度、类目转化率和 Top 类目分析。
TRUNCATE TABLE dws_category_daily_behavior;

INSERT INTO dws_category_daily_behavior (
  behavior_date,
  category_id,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  uv_count,
  buyer_count,
  click_to_buy_rate
)
SELECT
  behavior_date,
  category_id,
  SUM(is_pv) AS pv_count,
  SUM(is_cart) AS cart_count,
  SUM(is_fav) AS fav_count,
  SUM(is_buy) AS buy_count,
  COUNT(DISTINCT user_id) AS uv_count,
  COUNT(DISTINCT CASE WHEN is_buy = 1 THEN user_id END) AS buyer_count,
  CASE
    WHEN SUM(is_pv) = 0 THEN 0
    ELSE ROUND(SUM(is_buy) / SUM(is_pv), 4)
  END AS click_to_buy_rate
FROM dwd_user_behavior_detail
GROUP BY behavior_date, category_id;
