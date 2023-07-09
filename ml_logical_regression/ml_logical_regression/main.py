from sklearn.metrics import confusion_matrix, log_loss

from ml_logical_regression.ml_logical_regression.CustomLogisticRegression import LogicalRegression

test = LogicalRegression('../../Data/dataNew.csv')  # Инициализация класса

test.load_data()  # Загрузка данных

test.conversion_categorical_to_binary()  # Преобразование категориальных признаков в бинарные

test.skills_to_categorical_volume()
# test.delete_columes(columns_to_drop=['skill_set'])  # Удаление столбцов

test.transformation_y()  # Преобразование рейтинга в 0 и 1

test.scaler_columes()  # нормализация данных

test.create_data_for_learning()  # разделение данных на

test.create_train_split()  # Создание выборок (тестовая, обучающая)

test.model_fit()  # Обучение модели

test.result()