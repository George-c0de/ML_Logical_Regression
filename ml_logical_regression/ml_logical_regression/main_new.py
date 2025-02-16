import warnings
warnings.filterwarnings("ignore", message="Found unknown categories in columns.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    balanced_accuracy_score,
    classification_report,
    roc_curve,
    fbeta_score,
    make_scorer
)
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer

import logging
import seaborn as sns
sns.set_theme(style="whitegrid")
# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ ПАРАМЕТРЫ ============
DATA_PATH = "../../Data/dataNew.csv"
RATING_THRESHOLD = 0.1
TEST_SIZE = 0.4
RANDOM_STATE = 1000

# Новые бизнес-метрики
# 1) DESIRED_ACCEPTANCE_RATE  -> хотим, чтобы (TP / (TP+FN)) * 100% >= этого значения
# 2) DESIRED_INTERVIEWS_REDUCTION -> хотим, чтобы (FP / (TP+FP)) * 100% <= этого значения
DESIRED_ACCEPTANCE_RATE = 90.0    # Процент принятых хороших
DESIRED_INTERVIEWS_REDUCTION = 10.0  # Максимально допустимый % лишних собеседований

# Для приоритета «не пропустить хорошего кандидата» используем F2
f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)

LOGREG_PARAMS = {
    "classifier__fit_intercept": [True, False],
    "classifier__penalty": ["l1", "l2"],
    "classifier__solver": ["liblinear", "saga"],
    "classifier__C": [0.01, 0.1, 1, 10, 100],
    "classifier__class_weight": [None, "balanced"]  # Перебираем также балансировку классов
}


# ============ ФУНКЦИИ ЗАГРУЗКИ И ПРЕДОБРАБОТКИ ДАННЫХ ============
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Загружает CSV данные, обрабатывает пропущенные значения и удаляет лишние столбцы.
    """
    try:
        df = pd.read_csv(path, encoding="utf-8")
        logger.info(f"[10%] Данные успешно загружены из {path}.")
    except FileNotFoundError:
        logger.error(f"Файл не найден: {path}")
        raise

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns="Unnamed: 0")
        logger.info("[15%] Удалён столбец 'Unnamed: 0'.")

    # Обработка пропущенных значений: для строковых – режим, для числовых – медиана
    for column in df.columns:
        if df[column].dtype == 'object':
            mode_val = df[column].mode()[0]
            df[column] = df[column].fillna(mode_val)
        else:
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)
    logger.info("[20%] Пропущенные значения обработаны.")

    return df


def visual_data_analysis(df: pd.DataFrame) -> None:
    """
    Выполняет визуальный анализ исходных данных:
      - Распределение целевой переменной (rating)
      - Гистограммы числовых признаков и корреляционная матрица
      - Анализ категориальных признаков (например, education_level, specialty, city)
      - Анализ столбца 'skill_set' (если присутствует)
    """
    # Настройка стиля графиков
    sns.set_theme(style="whitegrid")
    sns.set(font_scale=1.1)

    # 1. Распределение целевой переменной
    # Если рейтинг представлен числовым значением, можно посмотреть его распределение
    plt.figure(figsize=(8, 5))
    if df["rating"].nunique() > 2:
        sns.histplot(df["rating"], bins=20, kde=True, color="skyblue")
        plt.title("Распределение рейтинга (до бинаризации)")
        plt.xlabel("Рейтинг")
    else:
        sns.countplot(x="rating", data=df, palette="viridis")
        plt.title("Распределение бинарного рейтинга")
        plt.xlabel("Рейтинг (0 - неудачный, 1 - успешный)")
    plt.ylabel("Количество записей")
    plt.show()

    # 2. Анализ числовых признаков
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    # Исключаем столбец 'rating' для гистограмм, если он уже отображается отдельно
    if "rating" in numerical_cols:
        numerical_cols.remove("rating")

    if numerical_cols:
        df[numerical_cols].hist(figsize=(15, 10), bins=20, edgecolor='black')
        plt.suptitle("Гистограммы числовых признаков", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # Корреляционная матрица (включая рейтинг, если бинарный)
        corr_cols = numerical_cols + ["rating"]
        corr_matrix = df[corr_cols].corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
        plt.title("Корреляционная матрица")
        plt.show()
    else:
        logger.info("Нет числовых признаков для визуального анализа.")

    # 3. Анализ категориальных признаков
    # Если в данных есть такие признаки, как education_level, specialty, city
    categorical_cols = []
    for col in ['education_level', 'specialty', 'city']:
        if col in df.columns:
            categorical_cols.append(col)

    for col in categorical_cols:
        plt.figure(figsize=(12, 6))
        sns.countplot(y=col, hue="rating", data=df, palette="Set2")
        plt.title(f"Распределение '{col}' с разбивкой по рейтингу")
        plt.xlabel("Количество записей")
        plt.ylabel(col)
        plt.legend(title="Рейтинг", labels=["0", "1"])
        plt.show()

    # 4. Анализ навыков (skill_set)
    if "skill_set" in df.columns:
        df["skill_set"] = df["skill_set"].fillna("")
        # Разбиваем навыки по запятой и очищаем данные
        all_skills = df["skill_set"].str.split(",").explode().str.strip().str.lower()
        # Убираем пустые значения и ссылки
        all_skills = all_skills[(all_skills != "") & (~all_skills.str.startswith("https://"))]
        top_skills = all_skills.value_counts().head(10)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_skills.values, y=top_skills.index, palette="magma")
        plt.title("Топ-10 наиболее распространённых навыков")
        plt.xlabel("Количество упоминаний")
        plt.ylabel("Навык")
        plt.show()


def encode_skill_set(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очищает и one-hot кодирует столбец 'skill_set'.
    """
    df['skill_set'] = df['skill_set'].fillna('').apply(
        lambda x: [skill.strip().lower() for skill in x.split(',')
                   if skill.strip() and not skill.startswith("https://")]
    )
    mlb = MultiLabelBinarizer()
    skill_encoded = pd.DataFrame(
        mlb.fit_transform(df['skill_set']),
        columns=[f'skill_{skill}' for skill in mlb.classes_],
        index=df.index
    )
    df = pd.concat([df, skill_encoded], axis=1)
    df = df.drop(columns='skill_set')
    logger.info("[25%] Столбец 'skill_set' закодирован.")
    return df


def data_processing(df: pd.DataFrame, rating_threshold: float) -> pd.DataFrame:
    """
    Предобрабатывает DataFrame:
      - Удаляет столбец 'rubbish'
      - Удаляет столбец 'gender' (если есть)
      - Обрабатывает текстовые и категориальные признаки
      - Преобразует целевую переменную 'rating' в бинарную метку (0 или 1)
    """
    if 'rubbish' in df.columns:
        df = df.drop(columns='rubbish')
        logger.info("[30%] Удалён столбец 'rubbish'.")

    df = encode_skill_set(df)

    for col_to_drop in ['gender']:
        if col_to_drop in df.columns:
            df.drop(columns=[col_to_drop], inplace=True)
            logger.info(f"Удалён столбец '{col_to_drop}'.")

    # Преобразуем 'rating' в бинарную метку
    df["rating"] = (df["rating"] >= rating_threshold).astype(int)
    logger.info("[35%] 'rating' преобразован в бинарную метку.")

    return df


# ============ ФУНКЦИИ ДЛЯ ПОДБОРА ПОРОГА ============
def find_best_threshold_f2(y_true: np.ndarray, y_proba: np.ndarray, step: float = 0.01) -> float:
    """
    Перебирает значения порога (threshold) от 0 до 1 с заданным шагом (step)
    и выбирает порог, при котором достигается максимальный F2-score.
    Возвращает лучший порог.
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    best_threshold = 0.5
    best_f2 = 0.0

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        current_f2 = fbeta_score(y_true, y_pred_t, beta=2, zero_division=0)
        if current_f2 > best_f2:
            best_f2 = current_f2
            best_threshold = t

    return best_threshold


# ============ ФУНКЦИИ МОДЕЛИРОВАНИЯ И ОЦЕНКИ ============
def build_and_evaluate_model(X_train: pd.DataFrame, y_train: np.ndarray,
                             X_test: pd.DataFrame, y_test: np.ndarray) -> None:
    """
    Обучает модель логистической регрессии (оптимизация F2) и выводит метрики:
      - Accuracy, Precision, Recall, F1, F2, ROC AUC
      - Процент принятых хороших (Good Acceptance Rate = Recall)
      - Процент лишних собеседований (Extra Interviews Rate = FP / (TP+FP))
      - Сравнение с DESIRED_ACCEPTANCE_RATE и DESIRED_INTERVIEWS_REDUCTION
    """

    # Для новых/редких категорий education_level, specialty и city
    def unify_categories(series: pd.Series, known_cats: set, other_label='other') -> pd.Series:
        return series.apply(lambda x: x if x in known_cats else other_label)

    if 'education_level' in X_train.columns:
        known_ed_levels = set(X_train['education_level'].unique())
        X_train['education_level'] = unify_categories(X_train['education_level'], known_ed_levels)
        X_test['education_level'] = unify_categories(X_test['education_level'], known_ed_levels)
    else:
        known_ed_levels = []

    if 'specialty' in X_train.columns:
        known_sp_levels = set(X_train['specialty'].unique())
        X_train['specialty'] = unify_categories(X_train['specialty'], known_sp_levels)
        X_test['specialty'] = unify_categories(X_test['specialty'], known_sp_levels)
    else:
        known_sp_levels = []

    if 'city' in X_train.columns:
        known_city_levels = set(X_train['city'].unique())
        X_train['city'] = unify_categories(X_train['city'], known_city_levels)
        X_test['city'] = unify_categories(X_test['city'], known_city_levels)
    else:
        known_city_levels = []

    # Определяем, какие колонки числовые, какие категориальные
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = []
    if 'education_level' in X_train.columns:
        categorical_features.append('education_level')
    if 'specialty' in X_train.columns:
        categorical_features.append('specialty')
    if 'city' in X_train.columns:
        categorical_features.append('city')

    # Приводим set в list для OneHotEncoder
    cat_ed_levels = sorted(list(known_ed_levels)) if known_ed_levels else []
    cat_sp_levels = sorted(list(known_sp_levels)) if known_sp_levels else []
    cat_city_levels = sorted(list(known_city_levels)) if known_city_levels else []

    # Настройка ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                # Передаём списки категорий для каждого признака
                ('onehot', OneHotEncoder(
                    categories=[cat_ed_levels, cat_sp_levels, cat_city_levels],
                    drop='first',
                    handle_unknown='ignore'
                ))
            ]), categorical_features)
        ],
        remainder='passthrough'
    )

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', BorderlineSMOTE(random_state=RANDOM_STATE)),
        ('classifier', LogisticRegression(max_iter=1000, tol=1e-3, random_state=RANDOM_STATE))
    ])

    # Подбор гиперпараметров по F2
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=LOGREG_PARAMS,
        scoring=f2_scorer,   # <-- F2
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    logger.info("[40%] Начало обучения модели с GridSearchCV (F2)...")
    grid_search.fit(X_train, y_train)
    logger.info("[70%] Обучение модели завершено.")

    # Предсказания (стандартный порог = 0.5)
    y_pred = grid_search.predict(X_test)
    y_proba = grid_search.predict_proba(X_test)[:, 1]

    # Подбираем оптимальный порог для максимизации F2
    best_threshold = find_best_threshold_f2(y_test, y_proba, step=0.01)
    y_pred_best = (y_proba >= best_threshold).astype(int)

    # --- Метрики при пороге 0.5 ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0.0)
    rec = recall_score(y_test, y_pred, zero_division=0.0)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    ll = log_loss(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # --- Метрики при оптимальном пороге ---
    acc_best = accuracy_score(y_test, y_pred_best)
    prec_best = precision_score(y_test, y_pred_best, zero_division=0.0)
    rec_best = recall_score(y_test, y_pred_best, zero_division=0.0)
    f1_best = f1_score(y_test, y_pred_best)
    f2_best = fbeta_score(y_test, y_pred_best, beta=2, zero_division=0)
    balanced_acc_best = balanced_accuracy_score(y_test, y_pred_best)
    cm_best = confusion_matrix(y_test, y_pred_best)

    cls_report = classification_report(y_test, y_pred, zero_division=0.0)
    cls_report_best = classification_report(y_test, y_pred_best, zero_division=0.0)

    # Вычисление новых бизнес-метрик
    TN, FP, FN, TP = cm.ravel()
    TN_b, FP_b, FN_b, TP_b = cm_best.ravel()

    good_acceptance_rate_05 = (TP / (TP + FN) * 100) if (TP + FN) > 0 else 0.0
    good_acceptance_rate_best = (TP_b / (TP_b + FN_b) * 100) if (TP_b + FN_b) > 0 else 0.0

    extra_interviews_rate_05 = (FP / (TP + FP) * 100) if (TP + FP) > 0 else 0.0
    extra_interviews_rate_best = (FP_b / (TP_b + FP_b) * 100) if (TP_b + FP_b) > 0 else 0.0

    def compare_metric(metric_value: float, desired_value: float, metric_name: str, higher_is_better=True):
        if higher_is_better:
            if metric_value >= desired_value:
                return f"{metric_name} {metric_value:.2f}% (Достигнута/Превышена цель {desired_value:.2f}%)"
            else:
                return f"{metric_name} {metric_value:.2f}% (Ниже желаемой цели {desired_value:.2f}%)"
        else:
            if metric_value <= desired_value:
                return f"{metric_name} {metric_value:.2f}% (Достигнута/Превышена цель {desired_value:.2f}%)"
            else:
                return f"{metric_name} {metric_value:.2f}% (Выше допустимой цели {desired_value:.2f}%)"

    results = (
        "=== Итоговая оценка модели (стандартный порог = 0.5) ===\n"
        f"Лучшие параметры: {grid_search.best_params_}\n"
        f"Accuracy:           {acc:.3f}\n"
        f"Precision:          {prec:.3f}\n"
        f"Recall:             {rec:.3f}\n"
        f"F1-score:           {f1:.3f}\n"
        f"F2-score:           {f2:.3f}\n"
        f"Balanced Accuracy:  {balanced_acc:.3f}\n"
        f"ROC AUC:            {roc_auc:.3f}\n"
        f"Log-loss:           {ll:.3f}\n"
        f"Confusion Matrix:\n{cm}\n"
        f"Classification Report:\n{cls_report}\n"
        "--- Новые бизнес-метрики (порог 0.5) ---\n"
        f"Good Acceptance Rate (Recall):  {good_acceptance_rate_05:.2f}%\n"
        f"Extra Interviews Rate:          {extra_interviews_rate_05:.2f}%\n"
        f"{compare_metric(good_acceptance_rate_05, DESIRED_ACCEPTANCE_RATE, 'Good Acceptance Rate', higher_is_better=True)}.\n"
        f"{compare_metric(extra_interviews_rate_05, DESIRED_INTERVIEWS_REDUCTION, 'Extra Interviews Rate', higher_is_better=False)}.\n"
        "----------------------------------------------------------\n"
        f"=== Оценка модели при оптимальном пороге (threshold = {best_threshold:.2f}) ===\n"
        f"Accuracy:           {acc_best:.3f}\n"
        f"Precision:          {prec_best:.3f}\n"
        f"Recall:             {rec_best:.3f}\n"
        f"F1-score:           {f1_best:.3f}\n"
        f"F2-score:           {f2_best:.3f}\n"
        f"Balanced Accuracy:  {balanced_acc_best:.3f}\n"
        f"Confusion Matrix:\n{cm_best}\n"
        f"Classification Report:\n{cls_report_best}\n"
        "--- Новые бизнес-метрики (оптимальный порог) ---\n"
        f"Good Acceptance Rate (Recall):  {good_acceptance_rate_best:.2f}%\n"
        f"Extra Interviews Rate:          {extra_interviews_rate_best:.2f}%\n"
        f"{compare_metric(good_acceptance_rate_best, DESIRED_ACCEPTANCE_RATE, 'Good Acceptance Rate', higher_is_better=True)}.\n"
        f"{compare_metric(extra_interviews_rate_best, DESIRED_INTERVIEWS_REDUCTION, 'Extra Interviews Rate', higher_is_better=False)}.\n"
    )

    print(results)
    logger.info("[100%] Оценка модели завершена.")

    # ROC-кривая
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def main():
    logger.info("Программа запущена.")

    # 1. Загрузка данных
    df_raw = load_data()

    # 2. Визуальный анализ исходных данных (до преобразований)
    visual_data_analysis(df_raw.copy())

    # 3. Предобработка данных
    df = data_processing(df_raw, rating_threshold=RATING_THRESHOLD)
    logger.info("[37%] Признаки и целевая переменная подготовлены.")

    # 4. Разделение на признаки и целевую переменную
    y = df["rating"].values
    X = df.drop(columns="rating")
    logger.info("[38%] Данные разделены на обучающую и тестовую выборки.")

    # 5. Разделение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 6. Обучение и оценка модели
    build_and_evaluate_model(X_train, y_train, X_test, y_test)
    logger.info("Программа завершена.")


if __name__ == "__main__":
    main()