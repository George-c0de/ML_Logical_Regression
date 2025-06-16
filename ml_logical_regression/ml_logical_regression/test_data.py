import os
import joblib
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

warnings.filterwarnings("ignore", message="Found unknown categories in columns.*")

# Пути к артефактам
MODEL_ARTIFACT_PATH = "models/logreg_model.pkl"  # ожидаем dict с ключами 'model' и 'threshold'
MLB_PATH            = "models/mlb.pkl"       # сохранённый MultiLabelBinarizer

def load_data(path: str) -> pd.DataFrame:
    """
    Загружает CSV, удаляет лишний столбец и обрабатывает пропуски:
     - для object: наиболее частое значение
     - для числовых: медиана
    """
    df = pd.read_csv(path, encoding="utf-8")
    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def encode_skill_set(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> pd.DataFrame:
    """
    Трансформирует столбец 'skill_set' через загруженный MultiLabelBinarizer.
    """
    df = df.copy()
    df['skill_set'] = (
        df.get('skill_set', "")
          .fillna("")
          .apply(lambda s: [skill.strip().lower() for skill in s.split(",") if skill.strip()])
    )
    skill_matrix = mlb.transform(df['skill_set'])
    skill_cols   = [f"skill_{s}" for s in mlb.classes_]
    skills_df    = pd.DataFrame(skill_matrix, columns=skill_cols, index=df.index)

    df.drop(columns=['skill_set'], inplace=True)
    return pd.concat([df, skills_df], axis=1)

def prepare_features(df: pd.DataFrame, mlb: MultiLabelBinarizer) -> pd.DataFrame:
    """
    Полная предобработка:
     - удаляет ненужные столбцы: 'rubbish', 'gender', 'rating'
     - кодирует 'skill_set'
    """
    df = df.copy()
    for unwanted in ['rubbish', 'gender', 'rating']:
        if unwanted in df.columns:
            df.drop(columns=[unwanted], inplace=True)

    return encode_skill_set(df, mlb)

def main():
    os.makedirs("models", exist_ok=True)

    # 1. Загрузка артефактов
    artefact = joblib.load(MODEL_ARTIFACT_PATH)
    model     = artefact['model']
    threshold = artefact['threshold']

    mlb = joblib.load(MLB_PATH)

    # 2. Загрузка новых данных
    csv_path = "../../Data/test_dataset.csv"  # замените на нужный путь
    df_raw   = load_data(csv_path)

    # 3. Предобработка
    X_new = prepare_features(df_raw, mlb)

    # 4. Прогнозирование
    proba = model.predict_proba(X_new)[:, 1]
    preds = (proba >= threshold).astype(int)

    # 5. Сохранение результатов
    output = df_raw.copy()
    output['prediction'] = preds
    output['probability'] = proba
    output.to_csv("predictions.csv", index=False)

    print(f"Predictions saved to predictions.csv (threshold={threshold:.2f})")

if __name__ == "__main__":
    main()