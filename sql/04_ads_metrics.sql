-- ADS 指标 SQL
-- 作用：生成面向业务分析、面试讲解和后续看板输出的结果层指标。

USE ecommerce_behavior_warehouse;

-- 1. 每日核心指标：观察每天流量、活跃和购买转化情况。
TRUNCATE TABLE ads_daily_core_metrics;

INSERT INTO ads_daily_core_metrics (
  behavior_date,
  pv,
  uv,
  dau,
  cart_users,
  fav_users,
  buyers,
  buy_count,
  avg_pv_per_user,
  user_buy_rate
)
SELECT
  behavior_date,
  SUM(pv_count) AS pv,
  COUNT(DISTINCT user_id) AS uv,
  COUNT(DISTINCT user_id) AS dau,
  COUNT(DISTINCT CASE WHEN cart_count > 0 THEN user_id END) AS cart_users,
  COUNT(DISTINCT CASE WHEN fav_count > 0 THEN user_id END) AS fav_users,
  COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) AS buyers,
  SUM(buy_count) AS buy_count,
  CASE
    WHEN COUNT(DISTINCT user_id) = 0 THEN 0
    ELSE ROUND(SUM(pv_count) / COUNT(DISTINCT user_id), 4)
  END AS avg_pv_per_user,
  CASE
    WHEN COUNT(DISTINCT user_id) = 0 THEN 0
    ELSE ROUND(COUNT(DISTINCT CASE WHEN buy_count > 0 THEN user_id END) / COUNT(DISTINCT user_id), 4)
  END AS user_buy_rate
FROM dws_user_daily_behavior
GROUP BY behavior_date;

-- 2. 整体转化漏斗：以用户数为主口径，比较 pv、cart、fav、buy 阶段。
TRUNCATE TABLE ads_conversion_funnel;

INSERT INTO ads_conversion_funnel (
  behavior_stage,
  user_count,
  behavior_count,
  stage_order,
  conversion_rate_from_pv,
  conversion_rate_from_previous_stage
)
WITH stage_metrics AS (
  SELECT 'pv' AS behavior_stage, COUNT(DISTINCT user_id) AS user_count, COUNT(*) AS behavior_count, 1 AS stage_order
  FROM dwd_user_behavior_detail
  WHERE behavior_type = 'pv'
  UNION ALL
  SELECT 'cart', COUNT(DISTINCT user_id), COUNT(*), 2
  FROM dwd_user_behavior_detail
  WHERE behavior_type = 'cart'
  UNION ALL
  SELECT 'fav', COUNT(DISTINCT user_id), COUNT(*), 3
  FROM dwd_user_behavior_detail
  WHERE behavior_type = 'fav'
  UNION ALL
  SELECT 'buy', COUNT(DISTINCT user_id), COUNT(*), 4
  FROM dwd_user_behavior_detail
  WHERE behavior_type = 'buy'
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
FROM stage_with_base;

-- 3. 留存分析：模拟数据没有注册时间，因此用“某日活跃用户第 N 天仍活跃”定义留存。
TRUNCATE TABLE ads_user_retention;

INSERT INTO ads_user_retention (
  base_date,
  retention_day,
  new_or_active_users,
  retained_users,
  retention_rate
)
WITH retention_days AS (
  SELECT 1 AS retention_day
  UNION ALL SELECT 3
  UNION ALL SELECT 7
),
base_users AS (
  SELECT DISTINCT behavior_date AS base_date, user_id
  FROM dws_user_daily_behavior
),
retention_result AS (
  SELECT
    b.base_date,
    r.retention_day,
    COUNT(DISTINCT b.user_id) AS new_or_active_users,
    COUNT(DISTINCT future.user_id) AS retained_users
  FROM base_users b
  CROSS JOIN retention_days r
  LEFT JOIN dws_user_daily_behavior future
    ON future.user_id = b.user_id
   AND future.behavior_date = DATE_ADD(b.base_date, INTERVAL r.retention_day DAY)
  GROUP BY b.base_date, r.retention_day
)
SELECT
  base_date,
  retention_day,
  new_or_active_users,
  retained_users,
  CASE
    WHEN new_or_active_users = 0 THEN 0
    ELSE ROUND(retained_users / new_or_active_users, 4)
  END AS retention_rate
FROM retention_result;

-- 4. RFM 用户价值分层：monetary 暂用购买次数近似，观察截止日取数据中最大日期。
-- Recency 口径：最近一次购买日期距观察截止日的天数，而不是最近一次任意行为日期。
-- 没有购买行为的用户没有最近购买日期，recency_days 统一置为 999，并归为“未购买用户”。
TRUNCATE TABLE ads_rfm_user_value;

INSERT INTO ads_rfm_user_value (
  user_id,
  recency_days,
  frequency,
  monetary,
  r_score,
  f_score,
  m_score,
  rfm_score,
  user_value_segment
)
WITH max_date AS (
  SELECT MAX(behavior_date) AS latest_date
  FROM dwd_user_behavior_detail
),
user_buy AS (
  SELECT
    user_id,
    MAX(CASE WHEN is_buy = 1 THEN behavior_date END) AS last_buy_date,
    SUM(is_buy) AS frequency,
    SUM(is_buy) AS monetary
  FROM dwd_user_behavior_detail
  GROUP BY user_id
),
rfm_base AS (
  SELECT
    user_id,
    CASE
      WHEN last_buy_date IS NULL THEN 999
      ELSE DATEDIFF((SELECT latest_date FROM max_date), last_buy_date)
    END AS recency_days,
    frequency,
    monetary
  FROM user_buy
),
score_base AS (
  SELECT
    user_id,
    recency_days,
    frequency,
    monetary,
    CASE
      WHEN recency_days = 999 THEN 1
      WHEN recency_days <= 3 THEN 5
      WHEN recency_days <= 7 THEN 4
      WHEN recency_days <= 14 THEN 3
      WHEN recency_days <= 21 THEN 2
      ELSE 1
    END AS r_score,
    CASE
      WHEN frequency >= 5 THEN 5
      WHEN frequency >= 3 THEN 4
      WHEN frequency >= 2 THEN 3
      WHEN frequency = 1 THEN 2
      ELSE 1
    END AS f_score,
    CASE
      WHEN monetary >= 5 THEN 5
      WHEN monetary >= 3 THEN 4
      WHEN monetary >= 2 THEN 3
      WHEN monetary = 1 THEN 2
      ELSE 1
    END AS m_score
  FROM rfm_base
)
SELECT
  user_id,
  recency_days,
  frequency,
  monetary,
  r_score,
  f_score,
  m_score,
  r_score + f_score + m_score AS rfm_score,
  CASE
    WHEN frequency = 0 THEN '未购买用户'
    WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN '高价值用户'
    WHEN r_score >= 4 AND (f_score >= 3 OR m_score >= 3) THEN '潜力用户'
    WHEN r_score >= 3 AND f_score >= 2 THEN '一般用户'
    WHEN r_score <= 2 AND frequency > 0 THEN '流失风险用户'
    ELSE '低活跃用户'
  END AS user_value_segment
FROM score_base;

-- 5. 热门商品 Top 20：按 pv_count 排名，附带购买和转化指标。
TRUNCATE TABLE ads_top_items;

INSERT INTO ads_top_items (
  item_id,
  category_id,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  buyer_count,
  click_to_buy_rate,
  rank_no
)
WITH item_total AS (
  SELECT
    item_id,
    MAX(category_id) AS category_id,
    SUM(pv_count) AS pv_count,
    SUM(cart_count) AS cart_count,
    SUM(fav_count) AS fav_count,
    SUM(buy_count) AS buy_count,
    SUM(buyer_count) AS buyer_count,
    CASE WHEN SUM(pv_count) = 0 THEN 0 ELSE ROUND(SUM(buy_count) / SUM(pv_count), 4) END AS click_to_buy_rate
  FROM dws_item_daily_behavior
  GROUP BY item_id
),
ranked AS (
  SELECT
    item_total.*,
    ROW_NUMBER() OVER (ORDER BY pv_count DESC, buy_count DESC, item_id) AS rank_no
  FROM item_total
)
SELECT item_id, category_id, pv_count, cart_count, fav_count, buy_count, buyer_count, click_to_buy_rate, rank_no
FROM ranked
WHERE rank_no <= 20;

-- 6. 热门类目 Top 20：按 pv_count 排名，用于类目运营优先级判断。
TRUNCATE TABLE ads_top_categories;

INSERT INTO ads_top_categories (
  category_id,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  buyer_count,
  click_to_buy_rate,
  rank_no
)
WITH category_total AS (
  SELECT
    category_id,
    SUM(pv_count) AS pv_count,
    SUM(cart_count) AS cart_count,
    SUM(fav_count) AS fav_count,
    SUM(buy_count) AS buy_count,
    SUM(buyer_count) AS buyer_count,
    CASE WHEN SUM(pv_count) = 0 THEN 0 ELSE ROUND(SUM(buy_count) / SUM(pv_count), 4) END AS click_to_buy_rate
  FROM dws_category_daily_behavior
  GROUP BY category_id
),
ranked AS (
  SELECT
    category_total.*,
    ROW_NUMBER() OVER (ORDER BY pv_count DESC, buy_count DESC, category_id) AS rank_no
  FROM category_total
)
SELECT category_id, pv_count, cart_count, fav_count, buy_count, buyer_count, click_to_buy_rate, rank_no
FROM ranked
WHERE rank_no <= 20;

-- 7. 小时级行为趋势：观察全天行为节奏，支持后续看板按小时展示。
TRUNCATE TABLE ads_hourly_behavior_trend;

INSERT INTO ads_hourly_behavior_trend (
  hour,
  pv_count,
  cart_count,
  fav_count,
  buy_count,
  behavior_count,
  active_users
)
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
ORDER BY hour;
