-- 电商用户行为数据分析与离线数仓建设项目
-- MySQL 8.0 建库建表脚本

-- 数据库说明：电商用户行为分析离线数仓。
-- MySQL CREATE DATABASE 不写数据库注释子句，避免不同环境执行时报语法错误。
CREATE DATABASE IF NOT EXISTS ecommerce_behavior_warehouse
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE ecommerce_behavior_warehouse;

-- 为了便于本地反复执行，先删除 ADS、DWS、DWD、ODS 各层表。
DROP TABLE IF EXISTS ads_hourly_behavior_trend;
DROP TABLE IF EXISTS ads_top_categories;
DROP TABLE IF EXISTS ads_top_items;
DROP TABLE IF EXISTS ads_rfm_user_value;
DROP TABLE IF EXISTS ads_user_retention;
DROP TABLE IF EXISTS ads_conversion_funnel;
DROP TABLE IF EXISTS ads_daily_core_metrics;
DROP TABLE IF EXISTS dws_category_daily_behavior;
DROP TABLE IF EXISTS dws_item_daily_behavior;
DROP TABLE IF EXISTS dws_user_daily_behavior;
DROP TABLE IF EXISTS dwd_user_behavior_detail;
DROP TABLE IF EXISTS ods_user_behavior_raw;

-- ODS 层：原始用户行为日志，尽量贴近 CSV 原貌，保留原始字段。
CREATE TABLE ods_user_behavior_raw (
  user_id VARCHAR(20) NOT NULL COMMENT '用户 ID',
  item_id VARCHAR(20) NOT NULL COMMENT '商品 ID',
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  behavior_type VARCHAR(10) NOT NULL COMMENT '行为类型：pv 浏览、cart 加购、fav 收藏、buy 购买',
  behavior_time DATETIME NOT NULL COMMENT '行为发生时间，精确到秒',
  date DATE COMMENT '原始行为日期字段',
  hour INT COMMENT '原始小时字段，0 到 23',
  weekday INT COMMENT '原始星期字段，0 表示周一，6 表示周日',
  KEY idx_ods_user_id (user_id),
  KEY idx_ods_item_id (item_id),
  KEY idx_ods_category_id (category_id),
  KEY idx_ods_behavior_type (behavior_type),
  KEY idx_ods_behavior_time (behavior_time),
  KEY idx_ods_date (date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ODS 层原始用户行为日志表';

-- DWD 层：清洗后的用户行为明细，一行代表一次标准化后的用户行为。
CREATE TABLE dwd_user_behavior_detail (
  user_id VARCHAR(20) NOT NULL COMMENT '用户 ID',
  item_id VARCHAR(20) NOT NULL COMMENT '商品 ID',
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  behavior_type VARCHAR(10) NOT NULL COMMENT '行为类型：pv、cart、fav、buy',
  behavior_time DATETIME NOT NULL COMMENT '标准化后的行为发生时间',
  date DATE NOT NULL COMMENT '行为日期，兼容原始字段命名',
  hour INT NOT NULL COMMENT '行为发生小时，0 到 23',
  weekday INT NOT NULL COMMENT '星期几，0 表示周一，6 表示周日',
  behavior_date DATE NOT NULL COMMENT '行为日期，DWD 标准日期字段',
  is_pv TINYINT NOT NULL COMMENT '是否浏览行为，pv 为 1，否则为 0',
  is_cart TINYINT NOT NULL COMMENT '是否加购行为，cart 为 1，否则为 0',
  is_fav TINYINT NOT NULL COMMENT '是否收藏行为，fav 为 1，否则为 0',
  is_buy TINYINT NOT NULL COMMENT '是否购买行为，buy 为 1，否则为 0',
  KEY idx_dwd_user_id (user_id),
  KEY idx_dwd_item_id (item_id),
  KEY idx_dwd_category_id (category_id),
  KEY idx_dwd_behavior_type (behavior_type),
  KEY idx_dwd_behavior_time (behavior_time),
  KEY idx_dwd_behavior_date (behavior_date),
  KEY idx_dwd_user_date (user_id, behavior_date),
  KEY idx_dwd_item_date (item_id, behavior_date),
  KEY idx_dwd_category_date (category_id, behavior_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWD 层用户行为明细表';

-- DWS 层：用户粒度的每日行为汇总，用于用户活跃、购买、留存和 RFM 分析。
CREATE TABLE dws_user_daily_behavior (
  behavior_date DATE NOT NULL COMMENT '行为日期',
  user_id VARCHAR(20) NOT NULL COMMENT '用户 ID',
  pv_count BIGINT NOT NULL COMMENT '浏览次数',
  cart_count BIGINT NOT NULL COMMENT '加购次数',
  fav_count BIGINT NOT NULL COMMENT '收藏次数',
  buy_count BIGINT NOT NULL COMMENT '购买次数',
  behavior_count BIGINT NOT NULL COMMENT '总行为次数',
  active_hours BIGINT NOT NULL COMMENT '当日活跃小时数',
  view_item_count BIGINT NOT NULL COMMENT '当日浏览或互动过的商品数',
  view_category_count BIGINT NOT NULL COMMENT '当日浏览或互动过的类目数',
  PRIMARY KEY (behavior_date, user_id),
  KEY idx_dws_user_daily_user_id (user_id),
  KEY idx_dws_user_daily_date (behavior_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWS 层用户每日行为汇总表';

-- DWS 层：商品粒度的每日行为汇总，用于商品热度和商品转化分析。
CREATE TABLE dws_item_daily_behavior (
  behavior_date DATE NOT NULL COMMENT '行为日期',
  item_id VARCHAR(20) NOT NULL COMMENT '商品 ID',
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  pv_count BIGINT NOT NULL COMMENT '商品浏览次数',
  cart_count BIGINT NOT NULL COMMENT '商品加购次数',
  fav_count BIGINT NOT NULL COMMENT '商品收藏次数',
  buy_count BIGINT NOT NULL COMMENT '商品购买次数',
  uv_count BIGINT NOT NULL COMMENT '商品访问用户数',
  buyer_count BIGINT NOT NULL COMMENT '商品购买用户数',
  click_to_buy_rate DECIMAL(10,4) NOT NULL COMMENT '点击到购买转化率，buy_count / pv_count',
  PRIMARY KEY (behavior_date, item_id),
  KEY idx_dws_item_daily_item_id (item_id),
  KEY idx_dws_item_daily_category_id (category_id),
  KEY idx_dws_item_daily_date (behavior_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWS 层商品每日行为汇总表';

-- DWS 层：类目粒度的每日行为汇总，用于类目热度和类目转化分析。
CREATE TABLE dws_category_daily_behavior (
  behavior_date DATE NOT NULL COMMENT '行为日期',
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  pv_count BIGINT NOT NULL COMMENT '类目浏览次数',
  cart_count BIGINT NOT NULL COMMENT '类目加购次数',
  fav_count BIGINT NOT NULL COMMENT '类目收藏次数',
  buy_count BIGINT NOT NULL COMMENT '类目购买次数',
  uv_count BIGINT NOT NULL COMMENT '类目访问用户数',
  buyer_count BIGINT NOT NULL COMMENT '类目购买用户数',
  click_to_buy_rate DECIMAL(10,4) NOT NULL COMMENT '点击到购买转化率，buy_count / pv_count',
  PRIMARY KEY (behavior_date, category_id),
  KEY idx_dws_category_daily_category_id (category_id),
  KEY idx_dws_category_daily_date (behavior_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWS 层类目每日行为汇总表';

-- ADS 层：每日核心指标，用于看整体业务健康度。
CREATE TABLE ads_daily_core_metrics (
  behavior_date DATE NOT NULL COMMENT '行为日期',
  pv BIGINT NOT NULL COMMENT '浏览次数',
  uv BIGINT NOT NULL COMMENT '访问用户数',
  dau BIGINT NOT NULL COMMENT '日活跃用户数，本项目等同于当日有任意行为的用户数',
  cart_users BIGINT NOT NULL COMMENT '加购用户数',
  fav_users BIGINT NOT NULL COMMENT '收藏用户数',
  buyers BIGINT NOT NULL COMMENT '购买用户数',
  buy_count BIGINT NOT NULL COMMENT '购买行为次数',
  avg_pv_per_user DECIMAL(10,4) NOT NULL COMMENT '人均浏览次数，pv / uv',
  user_buy_rate DECIMAL(10,4) NOT NULL COMMENT '用户购买转化率，buyers / dau',
  PRIMARY KEY (behavior_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层每日核心指标表';

-- ADS 层：整体行为转化漏斗，用于观察 pv、cart、fav、buy 各阶段转化。
CREATE TABLE ads_conversion_funnel (
  behavior_stage VARCHAR(10) NOT NULL COMMENT '行为阶段：pv、cart、fav、buy',
  user_count BIGINT NOT NULL COMMENT '该阶段用户数',
  behavior_count BIGINT NOT NULL COMMENT '该阶段行为次数',
  stage_order INT NOT NULL COMMENT '漏斗阶段顺序',
  conversion_rate_from_pv DECIMAL(10,4) NOT NULL COMMENT '相对 pv 阶段用户数的转化率',
  conversion_rate_from_previous_stage DECIMAL(10,4) NOT NULL COMMENT '相对上一阶段用户数的转化率',
  PRIMARY KEY (behavior_stage)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层整体转化漏斗表';

-- ADS 层：活跃用户留存，用某日活跃用户在第 N 天是否仍活跃定义。
CREATE TABLE ads_user_retention (
  base_date DATE NOT NULL COMMENT '基准活跃日期',
  retention_day INT NOT NULL COMMENT '留存天数，例如 1、3、7',
  new_or_active_users BIGINT NOT NULL COMMENT '基准日活跃用户数',
  retained_users BIGINT NOT NULL COMMENT '第 N 天仍活跃的用户数',
  retention_rate DECIMAL(10,4) NOT NULL COMMENT '留存率，retained_users / new_or_active_users',
  PRIMARY KEY (base_date, retention_day)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层用户留存分析表';

-- ADS 层：RFM 用户价值分层，monetary 用购买次数近似。
CREATE TABLE ads_rfm_user_value (
  user_id VARCHAR(20) NOT NULL COMMENT '用户 ID',
  recency_days INT NOT NULL COMMENT '最近一次购买距观察截止日的天数',
  frequency BIGINT NOT NULL COMMENT '购买频次',
  monetary BIGINT NOT NULL COMMENT '购买价值，本项目用购买次数近似',
  r_score INT NOT NULL COMMENT '最近购买得分',
  f_score INT NOT NULL COMMENT '购买频次得分',
  m_score INT NOT NULL COMMENT '购买价值得分',
  rfm_score INT NOT NULL COMMENT 'RFM 总分',
  user_value_segment VARCHAR(20) NOT NULL COMMENT '用户价值分层',
  PRIMARY KEY (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层 RFM 用户价值分层表';

-- ADS 层：热门商品 Top 20，用于商品运营分析。
CREATE TABLE ads_top_items (
  item_id VARCHAR(20) NOT NULL COMMENT '商品 ID',
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  pv_count BIGINT NOT NULL COMMENT '浏览次数',
  cart_count BIGINT NOT NULL COMMENT '加购次数',
  fav_count BIGINT NOT NULL COMMENT '收藏次数',
  buy_count BIGINT NOT NULL COMMENT '购买次数',
  buyer_count BIGINT NOT NULL COMMENT '购买用户数',
  click_to_buy_rate DECIMAL(10,4) NOT NULL COMMENT '点击到购买转化率',
  rank_no INT NOT NULL COMMENT '排名',
  PRIMARY KEY (rank_no),
  KEY idx_ads_top_items_item_id (item_id),
  KEY idx_ads_top_items_category_id (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层热门商品 Top 20 表';

-- ADS 层：热门类目 Top 20，用于类目运营分析。
CREATE TABLE ads_top_categories (
  category_id VARCHAR(20) NOT NULL COMMENT '类目 ID',
  pv_count BIGINT NOT NULL COMMENT '浏览次数',
  cart_count BIGINT NOT NULL COMMENT '加购次数',
  fav_count BIGINT NOT NULL COMMENT '收藏次数',
  buy_count BIGINT NOT NULL COMMENT '购买次数',
  buyer_count BIGINT NOT NULL COMMENT '购买用户数',
  click_to_buy_rate DECIMAL(10,4) NOT NULL COMMENT '点击到购买转化率',
  rank_no INT NOT NULL COMMENT '排名',
  PRIMARY KEY (rank_no),
  KEY idx_ads_top_categories_category_id (category_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层热门类目 Top 20 表';

-- ADS 层：小时级行为趋势，用于分析全天活跃节奏。
CREATE TABLE ads_hourly_behavior_trend (
  hour INT NOT NULL COMMENT '行为小时，0 到 23',
  pv_count BIGINT NOT NULL COMMENT '浏览次数',
  cart_count BIGINT NOT NULL COMMENT '加购次数',
  fav_count BIGINT NOT NULL COMMENT '收藏次数',
  buy_count BIGINT NOT NULL COMMENT '购买次数',
  behavior_count BIGINT NOT NULL COMMENT '总行为次数',
  active_users BIGINT NOT NULL COMMENT '活跃用户数',
  PRIMARY KEY (hour)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='ADS 层小时级行为趋势表';
