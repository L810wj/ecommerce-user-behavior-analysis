电商用户行为分析与离线数仓项目



本项目基于模拟电商用户行为日志，围绕用户浏览（pv）、加购（cart）、收藏（fav）和购买（buy）行为，完成数据生成与清洗、ODS–DWD–DWS–ADS 四层离线数仓建设、业务指标分析、可视化数据输出，以及用户未来 7 天购买预测。



项目覆盖从原始行为日志到业务指标和模型结果的完整数据链路，可用于展示数据分析、数据开发、数仓与 ETL、基础机器学习等实践能力。



项目亮点



基于 12 万条模拟用户行为日志，覆盖 6000 名用户、2500 件商品和 120 个商品类目。



设计 ODS、DWD、DWS、ADS 四层离线数仓，形成清晰、可追溯的数据加工链路。



沉淀用户日、商品日、类目日 3 张 DWS 汇总表，以及 7 张 ADS 指标表。



输出 PV、UV、DAU、购买用户数、转化漏斗、活跃留存、RFM、热门商品/类目和小时趋势等业务指标。



使用 Pandas 生成可供 Power BI 或 Excel 导入的 CSV 数据，并输出基础分析图表。



按时间切分观察窗口与标签窗口，构建用户未来 7 天购买预测任务，避免未来信息泄漏。



数据概况



项目



内容



数据来源



项目脚本生成的模拟电商行为日志



数据量



120000 条行为记录



用户数



6000



商品数



2500



类目数



120



时间范围



2024-11-01 00:00:01 至 2024-11-30 23:59:58



行为类型



浏览、加购、收藏、购买



模拟数据引入了用户活跃度长尾、商品热度长尾、用户购买倾向差异、商品转化能力差异、昼夜波动、周末波动和活动日峰值，使数据更接近真实电商行为分布。



技术栈



Python：Pandas、NumPy、Matplotlib、Seaborn、scikit-learn



机器学习：Logistic Regression、Random Forest、XGBoost / Gradient Boosting



数据库与 SQL：MySQL、Navicat



大数据处理：PySpark、Spark SQL



数据仓库：ODS、DWD、DWS、ADS



可视化数据应用：Power BI / Excel



版本管理：Git、GitHub



项目流程



flowchart TD

&#x20;   A\[模拟行为日志] --> B\[Pandas 数据清洗]

&#x20;   B --> C\[ODS 原始层]

&#x20;   C --> D\[DWD 明细层]

&#x20;   D --> E\[DWS 汇总层]

&#x20;   E --> F\[ADS 指标层]

&#x20;   F --> G\[CSV 与图表输出]

&#x20;   D --> H\[用户特征工程]

&#x20;   H --> I\[未来 7 天购买预测]



数仓分层设计



层级



作用



主要表



ODS



保留原始行为日志，支持数据追溯



ods\_user\_behavior\_raw



DWD



清洗并标准化行为明细，补充日期及行为标记字段



dwd\_user\_behavior\_detail



DWS



按常用业务粒度进行汇总，提高指标复用性



用户日、商品日、类目日汇总表



ADS



面向业务分析、报表和看板输出最终指标



核心指标、漏斗、留存、RFM、排行、小时趋势



DWS 汇总表



表名



数据粒度



主要用途



dws\_user\_daily\_behavior



用户 + 日期



用户活跃、购买、留存和 RFM 分析



dws\_item\_daily\_behavior



商品 + 日期



商品热度与商品转化分析



dws\_category\_daily\_behavior



类目 + 日期



类目热度与类目转化分析



ADS 指标表



表名



主要内容



ads\_daily\_core\_metrics



每日 PV、UV、DAU、购买次数、购买用户数和转化率



ads\_conversion\_funnel



浏览、加购、收藏、购买转化漏斗



ads\_user\_retention



次日、3 日和 7 日活跃留存



ads\_rfm\_user\_value



RFM 用户价值分层



ads\_top\_items



热门商品排行与商品转化指标



ads\_top\_categories



热门类目排行与类目转化指标



ads\_hourly\_behavior\_trend



0–23 点用户行为趋势



核心指标口径



PV：behavior\_type = 'pv' 的行为记录数。



UV：当日发生行为的去重用户数。



DAU：当日发生任意行为的去重用户数；由于模拟数据没有登录事件，使用行为活跃近似 DAU。



购买用户数：发生过 buy 行为的去重用户数。



点击到购买转化率：buy\_count / pv\_count；当 pv\_count = 0 时记为 0。



转化漏斗：分别统计产生浏览、加购、收藏和购买行为的去重用户数，观察各阶段用户转化情况。



活跃留存率：基准日活跃用户在第 N 天仍活跃的人数 ÷ 基准日活跃用户数。当前为活跃留存，不是新增用户留存。



RFM：R 为最近一次购买距观察截止日的天数，F 为购买频次；由于数据缺少金额字段，M 暂用购买次数近似。无购买用户的 recency\_days 设为 999，并归入未购买用户。



用户未来 7 天购买预测



预测任务使用用户历史行为特征，判断用户在未来 7 天内是否发生购买：



观察窗口：2024-11-01 至 2024-11-23；



标签窗口：2024-11-24 至 2024-11-30；



标签定义：标签窗口内发生 buy 行为记为 1，否则记为 0；



防止数据泄漏：所有特征仅来自观察窗口，标签窗口只用于生成预测标签。



主要特征包括基础行为次数、近 1/3/7 天行为、活跃度、商品和类目兴趣、转化倾向，以及夜间和周末行为偏好。项目比较 Logistic Regression、Random Forest 和 XGBoost；若本地没有安装 XGBoost，则使用 Gradient Boosting 替代。



购买用户属于少数类，因此模型评估除 Accuracy 和 ROC-AUC 外，更关注 Precision、Recall、F1、PR-AUC 和混淆矩阵。具体结果见 outputs/model\_results/model\_metrics.csv。



项目目录



ecommerce-user-behavior-warehouse/

├── README.md

├── requirements.txt

├── data/

│   ├── raw/

│   ├── processed/

│   └── output/

├── docs/

│   ├── 数据库表设计文档.md

│   ├── 项目学习路线.md

│   └── 面试讲解稿.md

├── sql/

│   ├── 01\_create\_tables\_mysql.sql

│   ├── 02\_ods\_to\_dwd.sql

│   ├── 03\_dws\_aggregate.sql

│   └── 04\_ads\_metrics.sql

├── src/

│   ├── 01\_generate\_sample\_data.py

│   ├── 02\_clean\_data\_pandas.py

│   ├── 03\_pyspark\_etl.py

│   ├── 04\_calculate\_metrics.py

│   └── 05\_train\_purchase\_model.py

├── notebooks/

│   ├── 01\_data\_exploration.ipynb

│   └── 02\_purchase\_prediction\_model.ipynb

├── dashboard/

│   ├── dashboard\_data/

│   └── dashboard\_design\_notes.md

└── outputs/

&#x20;   ├── figures/

&#x20;   ├── metrics/

&#x20;   └── model\_results/



运行环境



Python 3.9 及以上版本



MySQL 8.0（运行 SQL 数仓部分时需要）



Java 8/11 及 PySpark（运行 PySpark ETL 时需要）



Power BI Desktop 或 Excel（仅查看和制作交互式报表时需要）



快速运行



1\. 下载项目



git clone https://github.com/你的GitHub用户名/你的仓库名.git

cd 你的仓库名



也可以在 GitHub 页面点击 Code → Download ZIP，解压后进入项目根目录。



2\. 创建并激活虚拟环境



python -m venv .venv



Windows PowerShell：



.\\.venv\\Scripts\\Activate.ps1



3\. 安装依赖



pip install -r requirements.txt



4\. 按顺序运行 Python 脚本



python src/01\_generate\_sample\_data.py

python src/02\_clean\_data\_pandas.py

python src/04\_calculate\_metrics.py

python src/05\_train\_purchase\_model.py



如已配置 Java 和 PySpark，可在清洗后额外运行：



python src/03\_pyspark\_etl.py



5\. 运行 MySQL 数仓 SQL



在 Navicat 或 MySQL 客户端中按照以下顺序执行：



sql/01\_create\_tables\_mysql.sql

sql/02\_ods\_to\_dwd.sql

sql/03\_dws\_aggregate.sql

sql/04\_ads\_metrics.sql



其中，01\_create\_tables\_mysql.sql 用于创建数据库和各层表；导入 ODS 原始数据后，再依次执行 DWD 清洗、DWS 汇总和 ADS 指标脚本。



输出结果



业务指标 CSV



dashboard/dashboard\_data/ 和 outputs/metrics/ 中包含：



daily\_core\_metrics.csv



conversion\_funnel.csv



retention\_analysis.csv



rfm\_user\_value.csv



top\_items.csv



top\_categories.csv



hourly\_behavior\_trend.csv



基础分析图表



outputs/figures/ 中包含：



daily\_pv\_uv\_trend.png



conversion\_funnel.png



retention\_heatmap.png



rfm\_user\_segments.png



hourly\_behavior\_trend.png



roc\_curve.png



precision\_recall\_curve.png



confusion\_matrix.png



feature\_importance.png



模型结果



outputs/model\_results/ 中包含用户特征、购买标签、模型指标、预测结果、特征重要性和最佳模型文件。



可视化示例



如果以下图片已经保存在对应路径，GitHub 会直接展示：



每日 PV 与 UV 趋势







用户行为转化漏斗







活跃用户留存







模型特征重要性







项目价值



从业务角度分析用户活跃、转化、留存、消费倾向与商品热度。



从数据开发角度实现原始日志、清洗明细、业务汇总和应用指标的分层加工。



为 Power BI / Excel 提供标准化 CSV 数据源，支持后续制作交互式报表。



根据用户历史行为输出未来购买概率，为用户分层、召回和精细化运营提供参考。



项目局限与改进方向



当前使用模拟数据，结果用于流程验证和项目实践，不能直接代表真实业务表现。



数据缺少订单金额，RFM 中的 M 暂用购买次数近似，后续可替换为 GMV 或支付金额。



当前留存指标为活跃留存，不是基于注册日期计算的新增用户留存。



当前 Power BI 部分仅输出可导入的 CSV 和设计说明，未在仓库中提供完整 .pbix 文件。



模拟数据规律相对明显，模型离线指标可能偏高；真实业务中应进一步进行滚动时间验证、阈值优化、线上验证和数据漂移监控。



使用说明



本项目主要用于个人学习和求职作品展示。数据均由项目脚本模拟生成，不包含真实用户隐私或商业数据。

