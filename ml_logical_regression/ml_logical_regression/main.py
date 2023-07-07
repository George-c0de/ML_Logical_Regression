import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml_logical_regression.ml_logical_regression.decorator.decorators import measure_execution_time


class LogicalRegression:
    def __init__(self, file_name: str):
        self.__file_name = file_name
        self.data_raw: DataFrame | None = None
        self.__tuple_names_to_binary = [
            'city',
            'specialty',
            'gender',
            'education_level',
            'education_level',
            'specialty',
        ]
        self._list_normalization_columes = [
            'age',
            'total_experience',
        ]
        self.model = LogisticRegression(fit_intercept=False, penalty='none')
    @property
    def return_columes(self) -> list[str]:
        """
        возвращает список столбцов
        Returns: Список стоназваний столбцов

        """
        return self.data_raw.columns.to_list()

    @property
    def tuple_names_to_binary(self) -> list[str]:
        """
        Getter for tuple_names_to_binar
        Returns:
            Список элементов дял преобразования в бинарные признаки
        """
        return self.__tuple_names_to_binary

    @tuple_names_to_binary.setter
    def tuple_names_to_binary(self, add_list: list[str], update: bool = True):
        """
        Обновляет или заменяет список
        Args:
            add_list: Список для обновления значения
            update: True - Обновляет новым списком, False - заменяет на новый список
        Returns:

        """
        if update:
            self.__tuple_names_to_binary += add_list
        else:
            self.__tuple_names_to_binary = add_list

    @property
    def data_raw(self) -> DataFrame:
        """
        Getter для получения необработанных данных
        Returns:
            Необработанные данные
        """
        return self._data_raw

    @data_raw.setter
    def data_raw(self, value: DataFrame):
        """
        Setter для изменения необработанных данных
        Args:
            value: Данные для изменения

        Returns:

        """
        self._data_raw = value

    def load_data(self):
        """
        Загрузка данных и сохранение в переменной
        Returns:
            None
        """
        self.data_raw = pd.read_csv(self.__file_name, encoding='utf-8')
        self.data_raw.drop(columns='Unnamed: 0', inplace=True)

    @measure_execution_time  # 0.055
    def process_city_column(self):
        """
        Преобразование категориального признака в бинарный
        Returns:
            Возвращает данные с бинарным признаком - ГОРОД
        """

        self.data_raw = pd.get_dummies(
            self.data_raw,
            columns=['city'],
            dtype=int,  # Преобразование в 0 и 1
            prefix='',
            prefix_sep='',  # prefix = '',prefix_sep='', чтобы убрать переименования столбца
        )

    def delete_columes(self, columes_for_delete: tuple):
        """
        Удаление столбцов
        Args:
            columes_for_delete: Кортеж названий столбцов
        Returns:
            None
        """
        self.data_raw.drop(columes_for_delete, inplace=True, axis=1)

    def process_education_column(self):
        """
        Преобразования столбца из категориального в бинарный признак - Образование
        Returns:
            Необработанные данные с бинарным признаком образования
        """
        self.data_raw = pd.get_dummies(
            self.data_raw,
            columns=['education_level'],
            dtype=int,  # Преобразование в 0 и 1
            prefix='',
            prefix_sep='',  # prefix = '',prefix_sep='', чтобы убрать переименования столбца
        )

    def conversion_categorical_to_binary(self):
        """
        Преобразование категориальных признаков в бинарные
        prefix = '',prefix_sep='', чтобы убрать переименования столбца
        dtype = Преобразование в 0 и 1
        Returns:

        """
        self.data_raw = pd.get_dummies(
            self.data_raw,
            columns=self.__tuple_names_to_binary,
            dtype=int,
            prefix='',
            prefix_sep='',
        )

    def scaler_columes(self):
        scaler = MinMaxScaler()
        features = self.data_raw[self._list_normalization_columes]
        normalized_features = scaler.fit_transform(features)
        normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
        self.data_raw.update(normalized_data)

    def create_data_for_learning(self):
        self.y_raw = self.data_raw.rating
        self.x_raw = self.data_raw.loc[:, test.data_raw.columns != 'rating']

    def create_train_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_raw, self.y_raw, test_size=0.3,
                                                                                random_state=1000)
    def model_train(self):
        model.fit(self.x_train, self.y_train)

test = LogicalRegression('../../Data/dataNew.csv')
test.load_data()

test.conversion_categorical_to_binary()
test.scaler_columes()
print(test.data_raw)

test.create_data_for_learning()
lab = preprocessing.LabelEncoder()
test.create_train_split()

model.fit(X_train, Y_train)
y_pred = model.predict(X_raw)
print(confusion_matrix(Y_raw, y_pred))
print(log_loss(y_pred, Y_raw))
print('Б. Всей обучающей выборки')
y_pred = model.predict(X_train)
print(confusion_matrix(Y_train, y_pred))
print(log_loss(y_pred, Y_train))
print('В. Отобранных на собеседование кандидатов для обучающей выборки')
print(confusion_matrix(Y_train, y_pred)[1][1])
print('Г.  Всей тестовой выборки')
y_pred = model.predict(X_test)
print(confusion_matrix(Y_test, y_pred))
print(log_loss(y_pred, Y_test))
print('Д. Отобранных на собеседование кандидатов для тестовой выборки')
print(confusion_matrix(Y_test, y_pred)[1][1])
