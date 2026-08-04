"""
用户未来 7 天购买预测模型。

任务定义：
使用 2024-11-01 至 2024-11-23 的用户行为特征，预测用户在
2024-11-24 至 2024-11-30 是否发生购买行为。

注意：所有特征只来自观察窗口，标签只来自未来标签窗口，严格避免数据泄漏。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


RANDOM_STATE = 42
INPUT_FILE = Path("data") / "processed" / "dwd_user_behavior_detail.csv"
MODEL_OUTPUT_DIR = Path("outputs") / "model_results"
FIGURES_DIR = Path("outputs") / "figures"

OBS_START = pd.Timestamp("2024-11-01")
OBS_END = pd.Timestamp("2024-11-23")
LABEL_START = pd.Timestamp("2024-11-24")
LABEL_END = pd.Timestamp("2024-11-30")

FEATURE_COLUMNS = [
    "pv_count_total",
    "cart_count_total",
    "fav_count_total",
    "buy_count_total",
    "behavior_count_total",
    "pv_count_1d",
    "cart_count_1d",
    "fav_count_1d",
    "buy_count_1d",
    "pv_count_3d",
    "cart_count_3d",
    "fav_count_3d",
    "buy_count_3d",
    "pv_count_7d",
    "cart_count_7d",
    "fav_count_7d",
    "buy_count_7d",
    "active_days",
    "active_hours",
    "avg_behavior_per_active_day",
    "last_behavior_recency_days",
    "last_buy_recency_days",
    "view_item_count",
    "view_category_count",
    "cart_item_count",
    "fav_item_count",
    "buy_item_count",
    "cart_rate",
    "fav_rate",
    "buy_rate",
    "cart_to_buy_rate",
    "fav_to_buy_rate",
    "night_behavior_count",
    "night_behavior_ratio",
    "weekend_behavior_count",
    "weekend_behavior_ratio",
]


@dataclass
class TrainedModel:
    """保存模型训练和评估需要的信息。"""

    name: str
    model: Any
    metrics: dict[str, float | int]
    y_pred: np.ndarray
    y_proba: np.ndarray
    confusion: np.ndarray


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """安全除法，避免比例特征产生无穷值或除零错误。"""
    denominator = denominator.replace(0, np.nan)
    result = numerator / denominator
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def load_dwd_data(project_root: Path) -> pd.DataFrame:
    """读取 DWD 明细数据并完成基础类型转换。"""
    input_path = project_root / INPUT_FILE
    if not input_path.exists():
        raise FileNotFoundError(
            f"输入文件不存在: {INPUT_FILE}，请先运行 python src/02_clean_data_pandas.py"
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    required_columns = {
        "user_id",
        "item_id",
        "category_id",
        "behavior_type",
        "behavior_time",
        "behavior_date",
        "hour",
        "weekday",
        "is_buy",
        "is_cart",
        "is_fav",
        "is_pv",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"DWD 数据缺少必要字段: {sorted(missing_columns)}")

    df["behavior_time"] = pd.to_datetime(df["behavior_time"], errors="coerce")
    df["behavior_date"] = pd.to_datetime(df["behavior_date"], errors="coerce")
    if df["behavior_time"].isna().any() or df["behavior_date"].isna().any():
        raise ValueError("DWD 数据中存在无法解析的时间字段")

    for column in ["is_buy", "is_cart", "is_fav", "is_pv", "hour", "weekday"]:
        df[column] = df[column].astype(int)

    print(f"读取 DWD 数据成功: {INPUT_FILE}")
    print(f"数据行数: {len(df):,}")
    print(f"时间范围: {df['behavior_date'].min().date()} 至 {df['behavior_date'].max().date()}")
    return df


def split_windows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按时间切分观察窗口和标签窗口，避免未来数据进入特征。"""
    observation_df = df[(df["behavior_date"] >= OBS_START) & (df["behavior_date"] <= OBS_END)].copy()
    label_df = df[(df["behavior_date"] >= LABEL_START) & (df["behavior_date"] <= LABEL_END)].copy()

    if observation_df.empty:
        raise ValueError("观察窗口数据为空")
    if label_df.empty:
        raise ValueError("标签窗口数据为空")
    if observation_df["behavior_date"].max() >= label_df["behavior_date"].min():
        raise ValueError("观察窗口和标签窗口存在重叠，可能导致数据泄漏")

    print(f"观察窗口: {OBS_START.date()} 至 {OBS_END.date()}，行数: {len(observation_df):,}")
    print(f"标签窗口: {LABEL_START.date()} 至 {LABEL_END.date()}，行数: {len(label_df):,}")
    return observation_df, label_df


def count_behavior_by_user(df: pd.DataFrame, user_base: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """按用户聚合 pv/cart/fav/buy 次数。"""
    counts = (
        df.groupby("user_id")
        .agg(
            **{
                f"pv_count_{suffix}": ("is_pv", "sum"),
                f"cart_count_{suffix}": ("is_cart", "sum"),
                f"fav_count_{suffix}": ("is_fav", "sum"),
                f"buy_count_{suffix}": ("is_buy", "sum"),
            }
        )
        .reset_index()
    )
    return user_base.merge(counts, on="user_id", how="left")


def build_user_features(observation_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """基于观察窗口构造 user_id 粒度特征。"""
    user_base = pd.DataFrame({"user_id": sorted(observation_df["user_id"].unique())})
    features = count_behavior_by_user(observation_df, user_base, "total")

    behavior_total = observation_df.groupby("user_id").size().rename("behavior_count_total").reset_index()
    features = features.merge(behavior_total, on="user_id", how="left")

    # 近 N 天窗口以观察窗口结束日为基准，且全部在观察窗口内。
    for days in [1, 3, 7]:
        window_start = OBS_END - pd.Timedelta(days=days - 1)
        recent_df = observation_df[
            (observation_df["behavior_date"] >= window_start)
            & (observation_df["behavior_date"] <= OBS_END)
        ]
        recent_features = count_behavior_by_user(recent_df, user_base, f"{days}d")
        recent_columns = [column for column in recent_features.columns if column != "user_id"]
        features = features.merge(recent_features[["user_id"] + recent_columns], on="user_id", how="left")

    active = (
        observation_df.groupby("user_id")
        .agg(
            active_days=("behavior_date", "nunique"),
            active_hours=("hour", "nunique"),
            last_behavior_date=("behavior_date", "max"),
        )
        .reset_index()
    )
    features = features.merge(active, on="user_id", how="left")

    last_buy = (
        observation_df[observation_df["is_buy"] == 1]
        .groupby("user_id")["behavior_date"]
        .max()
        .rename("last_buy_date")
        .reset_index()
    )
    features = features.merge(last_buy, on="user_id", how="left")

    # 商品和类目兴趣特征。
    interest_frames = [
        observation_df[observation_df["is_pv"] == 1]
        .groupby("user_id")
        .agg(view_item_count=("item_id", "nunique"), view_category_count=("category_id", "nunique"))
        .reset_index(),
        observation_df[observation_df["is_cart"] == 1]
        .groupby("user_id")
        .agg(cart_item_count=("item_id", "nunique"))
        .reset_index(),
        observation_df[observation_df["is_fav"] == 1]
        .groupby("user_id")
        .agg(fav_item_count=("item_id", "nunique"))
        .reset_index(),
        observation_df[observation_df["is_buy"] == 1]
        .groupby("user_id")
        .agg(buy_item_count=("item_id", "nunique"))
        .reset_index(),
    ]
    for frame in interest_frames:
        features = features.merge(frame, on="user_id", how="left")

    night = (
        observation_df[observation_df["hour"].between(19, 23)]
        .groupby("user_id")
        .size()
        .rename("night_behavior_count")
        .reset_index()
    )
    weekend = (
        observation_df[observation_df["weekday"] >= 5]
        .groupby("user_id")
        .size()
        .rename("weekend_behavior_count")
        .reset_index()
    )
    features = features.merge(night, on="user_id", how="left")
    features = features.merge(weekend, on="user_id", how="left")

    features["last_behavior_recency_days"] = (
        OBS_END - pd.to_datetime(features["last_behavior_date"], errors="coerce")
    ).dt.days.fillna(999).astype(int)
    features["last_buy_recency_days"] = (
        OBS_END - pd.to_datetime(features["last_buy_date"], errors="coerce")
    ).dt.days.fillna(999).astype(int)

    features = features.drop(columns=["last_behavior_date", "last_buy_date"], errors="ignore")
    features = features.fillna(0)
    features["avg_behavior_per_active_day"] = safe_divide(
        features["behavior_count_total"],
        features["active_days"],
    )
    features["cart_rate"] = safe_divide(features["cart_count_total"], features["pv_count_total"])
    features["fav_rate"] = safe_divide(features["fav_count_total"], features["pv_count_total"])
    features["buy_rate"] = safe_divide(features["buy_count_total"], features["pv_count_total"])
    features["cart_to_buy_rate"] = safe_divide(features["buy_count_total"], features["cart_count_total"])
    features["fav_to_buy_rate"] = safe_divide(features["buy_count_total"], features["fav_count_total"])
    features["night_behavior_ratio"] = safe_divide(features["night_behavior_count"], features["behavior_count_total"])
    features["weekend_behavior_ratio"] = safe_divide(features["weekend_behavior_count"], features["behavior_count_total"])

    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].fillna(0)
    features = features[["user_id"] + FEATURE_COLUMNS]

    output_path = project_root / MODEL_OUTPUT_DIR / "user_purchase_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"输出特征表: {output_path.relative_to(project_root)}，行数: {len(features):,}")
    return features


def build_labels(features: pd.DataFrame, label_df: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    """基于标签窗口构造未来 7 天是否购买标签。"""
    positive_users = set(label_df.loc[label_df["is_buy"] == 1, "user_id"])
    labels = features[["user_id"]].copy()
    labels["label"] = labels["user_id"].isin(positive_users).astype(int)

    positive_count = int(labels["label"].sum())
    negative_count = int((labels["label"] == 0).sum())
    positive_rate = positive_count / len(labels)
    print("标签分布:")
    print(f"正样本数: {positive_count:,}")
    print(f"负样本数: {negative_count:,}")
    print(f"正样本比例: {positive_rate:.4%}")

    output_path = project_root / MODEL_OUTPUT_DIR / "user_purchase_labels.csv"
    labels.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"输出标签表: {output_path.relative_to(project_root)}，行数: {len(labels):,}")
    return labels


def try_build_xgboost_model(y_train: pd.Series) -> tuple[str, Any] | None:
    """如果本地安装了 xgboost，则加入 XGBoost；否则返回 None。"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    positive_count = max(int(y_train.sum()), 1)
    negative_count = max(int((y_train == 0).sum()), 1)
    scale_pos_weight = negative_count / positive_count
    model = XGBClassifier(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
        scale_pos_weight=scale_pos_weight,
    )
    return "XGBoost", model


def build_model_candidates(y_train: pd.Series) -> list[tuple[str, Any, bool]]:
    """创建候选模型，第三个布尔值表示是否需要 sample_weight。"""
    models: list[tuple[str, Any, bool]] = [
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            False,
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
    ]

    xgb_model = try_build_xgboost_model(y_train)
    if xgb_model is not None:
        models.append((xgb_model[0], xgb_model[1], False))
        print("检测到 xgboost，将训练 XGBoost 模型")
    else:
        models.append(
            (
                "Gradient Boosting",
                GradientBoostingClassifier(random_state=RANDOM_STATE),
                True,
            )
        )
        print("未检测到 xgboost，使用 scikit-learn GradientBoostingClassifier 作为替代")

    return models


def evaluate_model(name: str, model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> TrainedModel:
    """评估模型并返回统一结果。"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return TrainedModel(name=name, model=model, metrics=metrics, y_pred=y_pred, y_proba=y_proba, confusion=cm)


def train_and_evaluate_models(
    dataset: pd.DataFrame,
    project_root: Path,
) -> tuple[list[TrainedModel], TrainedModel, pd.DataFrame, pd.Series, pd.DataFrame]:
    """划分训练测试集，训练多个模型并保存评估结果。"""
    X = dataset[FEATURE_COLUMNS]
    y = dataset["label"].astype(int)
    user_ids = dataset["user_id"]

    X_train, X_test, y_train, y_test, user_train, user_test = train_test_split(
        X,
        y,
        user_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"训练集用户数: {len(X_train):,}，测试集用户数: {len(X_test):,}")
    print(f"训练集正样本比例: {y_train.mean():.4%}，测试集正样本比例: {y_test.mean():.4%}")

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    trained_models: list[TrainedModel] = []
    prediction_frames = []

    for name, model, needs_sample_weight in build_model_candidates(y_train):
        print(f"开始训练模型: {name}")
        if needs_sample_weight:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)

        result = evaluate_model(name, model, X_test, y_test)
        trained_models.append(result)
        prediction_frames.append(
            pd.DataFrame(
                {
                    "user_id": user_test.to_numpy(),
                    "y_true": y_test.to_numpy(),
                    "y_pred": result.y_pred,
                    "y_proba": result.y_proba,
                    "model_name": name,
                }
            )
        )
        print(
            f"{name}: AUC={result.metrics['roc_auc']:.4f}, "
            f"PR-AUC={result.metrics['pr_auc']:.4f}, "
            f"F1={result.metrics['f1']:.4f}, Recall={result.metrics['recall']:.4f}"
        )

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": result.name,
                **result.metrics,
            }
            for result in trained_models
        ]
    ).sort_values(["roc_auc", "pr_auc", "f1"], ascending=False)
    metrics_path = project_root / MODEL_OUTPUT_DIR / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"输出模型评估结果: {metrics_path.relative_to(project_root)}")

    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = project_root / MODEL_OUTPUT_DIR / "purchase_prediction_results.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"输出预测结果: {prediction_path.relative_to(project_root)}")

    best_model_name = metrics_df.iloc[0]["model_name"]
    best_model = next(result for result in trained_models if result.name == best_model_name)
    joblib_path = project_root / MODEL_OUTPUT_DIR / "best_purchase_model.joblib"
    joblib.dump(best_model.model, joblib_path)
    print(f"保存最佳模型: {joblib_path.relative_to(project_root)}，最佳模型: {best_model.name}")

    return trained_models, best_model, X_test, y_test, metrics_df


def get_feature_importance(
    trained_models: list[TrainedModel],
    metrics_df: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    """优先使用最佳树模型的 feature_importances_，否则使用 LR 系数绝对值。"""
    tree_candidates = [
        result
        for result in trained_models
        if hasattr(result.model, "feature_importances_")
    ]

    if tree_candidates:
        candidate_order = metrics_df["model_name"].tolist()
        tree_candidates = sorted(tree_candidates, key=lambda item: candidate_order.index(item.name))
        selected = tree_candidates[0]
        importance = selected.model.feature_importances_
        print(f"使用树模型输出特征重要性: {selected.name}")
    else:
        selected = next(result for result in trained_models if result.name == "Logistic Regression")
        importance = np.abs(selected.model.named_steps["model"].coef_[0])
        print("仅有 Logistic Regression 可用，使用标准化后系数绝对值表示特征重要性")

    importance_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importance})
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = importance_df.index + 1

    output_path = project_root / MODEL_OUTPUT_DIR / "feature_importance.csv"
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"输出特征重要性: {output_path.relative_to(project_root)}")
    return importance_df


def save_roc_curve(best_model: TrainedModel, y_test: pd.Series, output_path: Path) -> None:
    """保存 ROC 曲线。"""
    fpr, tpr, _ = roc_curve(y_test, best_model.y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f"{best_model.name} AUC={best_model.metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_precision_recall_curve(best_model: TrainedModel, y_test: pd.Series, output_path: Path) -> None:
    """保存 Precision-Recall 曲线。"""
    precision, recall, _ = precision_recall_curve(y_test, best_model.y_proba)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, label=f"{best_model.name} PR-AUC={best_model.metrics['pr_auc']:.3f}")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_confusion_matrix_chart(best_model: TrainedModel, output_path: Path) -> None:
    """保存混淆矩阵图。"""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(best_model.confusion, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {best_model.name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Buy", "Buy"])
    ax.set_yticklabels(["No Buy", "Buy"])
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(best_model.confusion[row, col]), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_feature_importance_chart(importance_df: pd.DataFrame, output_path: Path) -> None:
    """保存 Top 20 特征重要性图。"""
    top_features = importance_df.head(20).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top_features["feature"], top_features["importance"], color="#4C78A8")
    ax.set_title("Top 20 Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_evaluation_figures(
    best_model: TrainedModel,
    y_test: pd.Series,
    importance_df: pd.DataFrame,
    project_root: Path,
) -> None:
    """保存模型评估图表。"""
    figure_dir = project_root / FIGURES_DIR
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_jobs = [
        ("roc_curve.png", lambda path: save_roc_curve(best_model, y_test, path)),
        ("precision_recall_curve.png", lambda path: save_precision_recall_curve(best_model, y_test, path)),
        ("confusion_matrix.png", lambda path: save_confusion_matrix_chart(best_model, path)),
        ("feature_importance.png", lambda path: save_feature_importance_chart(importance_df, path)),
    ]

    for file_name, save_func in figure_jobs:
        output_path = figure_dir / file_name
        save_func(output_path)
        print(f"输出模型图表: {output_path.relative_to(project_root)}")


def main() -> None:
    """脚本入口函数。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        (project_root / MODEL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

        df = load_dwd_data(project_root)
        observation_df, label_df = split_windows(df)
        features = build_user_features(observation_df, project_root)
        labels = build_labels(features, label_df, project_root)
        dataset = features.merge(labels, on="user_id", how="inner")

        if dataset["label"].nunique() < 2:
            raise ValueError("标签只有一个类别，无法训练二分类模型")

        trained_models, best_model, _X_test, y_test, metrics_df = train_and_evaluate_models(dataset, project_root)
        importance_df = get_feature_importance(trained_models, metrics_df, project_root)
        save_evaluation_figures(best_model, y_test, importance_df, project_root)

        print("\n模型训练完成")
        print(f"最佳模型: {best_model.name}")
        print("模型评估结果:")
        print(metrics_df.to_string(index=False))
        print("Top 10 重要特征:")
        print(importance_df.head(10).to_string(index=False))
    except Exception as exc:
        print(f"购买预测模型训练失败: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
