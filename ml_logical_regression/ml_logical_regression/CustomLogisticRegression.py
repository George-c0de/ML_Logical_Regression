import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from ml_logical_regression.ml_logical_regression.decorator.decorators import measure_execution_time
from collections import Counter, defaultdict


class LogicalRegression:
    def __init__(self, file_name: str):
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.x_raw = None
        self.y_raw = None
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
        self.threshold = 0.1
        self.skills_for_save = [
            'Python',
            'Java',
            'SQLite',
            'Flask',
            'Socket',
            'async.io',
            'JavaFX',
        ]

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
        """
        Нормализация данных
        Returns: Возвращает необработанные данные с нормализацией

        """
        scaler = MinMaxScaler()
        features = self.data_raw[self._list_normalization_columes]
        normalized_features = scaler.fit_transform(features)
        normalized_data = pd.DataFrame(normalized_features, columns=features.columns)
        self.data_raw.update(normalized_data)

    def create_data_for_learning(self):
        """
        Создание данных для разбивки на выборки
        Returns:
            Сохраняет данные для выборок
        """
        self.y_raw = self.data_raw.rating
        self.x_raw = self.data_raw.loc[:, self.data_raw.columns != 'rating']

    def use_cluster_centroids(self) -> tuple:
        smote = SMOTE()
        X, y = make_classification(
            n_classes=2,
            class_sep=2,
            weights=[0.1, 0.9],
            n_informative=3,
            n_redundant=1,
            flip_y=0,
            n_features=20,
            n_clusters_per_class=1,
            n_samples=1000,
            random_state=10,
        )
        print(f'Original dataset shape {Counter(y)}')
        X_res, y_res = smote.fit_resample(X, y)
        print(f'Resampled dataset shape {Counter(y_res)}')
        return smote.fit_resample(self.x_train, self.y_train)

    def create_train_split(self):
        """
        Проводит разбивку данных и сохраняет в переменные
        Returns:
            Сохраняет выборки в переменные класса
        """

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_raw, self.y_raw, test_size=0.3, random_state=1000
        )
        # self.x_train, self.y_train = self.use_cluster_centroids()
        # print(self.use_cluster_centroids())

    def model_train(self):
        """
        Обучение модели
        Returns:
            Обученная модель сохраняется в переменную класса
        """
        self.model.fit(self.x_train, self.y_train)

    def get_ru_columes(self) -> list:
        # TODO
        result = []
        result = [
            skill
            for el in list(self.data_raw['skill_set'])
            if not isinstance(el, str)
            for skill in el.split(',')
            if any(ord(c) > 127 for c in skill) and skill
        ]
        print()
        result = []
        for el in list(self.data_raw['skill_set']):
            if not isinstance(el, str):
                continue
            for skill in el.split(','):
                if any(ord(c) > 127 for c in skill) and skill:
                    result.append(skill)
        print(result)
        return result

    def skills_to_categorical_volume(self):  # TODO Проверить метод
        # Разделение столбца skill_set и создание нового столбца для каждого навыка
        # Разбиение столбца skill_set на отдельные столбцы для каждого навыка
        skills = self.data_raw['skill_set'].str.get_dummies(sep=',')
        result = defaultdict(int)
        for el in self.data_raw.skill_set:
            if el is not None and isinstance(el, str):
                el = el.split(',')
                for e in el:
                    result[e.upper()] += 1

        sorted_result = dict(filter(lambda item: item[1] >20, result.items()))

        print(sorted_result)
        # Объединение полученных столбцов с исходным DataFrame
        df = pd.concat([self.data_raw, skills], axis=1)

        # Удаление столбца skill_set

        self.data_raw = df
        # print(self.get_ru_columes())
        # self.delete_columes(columns_to_drop=self.get_ru_columes())
        df.drop('skill_set', axis=1, inplace=True)

        return
        skills.columns = skills.columns.map(lambda x: f'skill_{x + 1}')

        # Заполнение столбцов новым столбцом и преобразование в 0 и 1
        for col in skills.columns:
            self.data_raw[col] = skills[col].apply(lambda x: 1 if pd.notnull(x) else 0)

        # Заполнение отсутствующих навыков нулями
        self.data_raw.fillna(0, inplace=True)

        # Вывод DataFrame
        return
        skills_encoded = pd.get_dummies(self.data_raw['skill_set'], prefix='', prefix_sep='')

        # Join the encoded skills with the original DataFrame
        new_df = pd.concat([self.data_raw.drop('skill_set', axis=1), skills_encoded], axis=1)

        # Print the new DataFrame
        columes = new_df.columns
        self.data_raw = pd.concat([self.data_raw, new_df], axis=1)
        self.data_raw = pd.get_dummies(
            self.data_raw,
            columns=columes,
            dtype=int,  # Преобразование в 0 и 1
            prefix='',
            prefix_sep='',  # prefix = '',prefix_sep='', чтобы убрать переименования столбца
        )
        return
        result = []
        for el in list(self.data_raw['skill_set']):
            if not isinstance(el, str):
                continue
            for skill in el.split(','):
                if skill.upper() in result and all(ord(c) <= 127 for c in skill) and skill:
                    result.append(skill.upper())

        skills = self.data_raw["skill_set"].str.split(",").explode()

        # Преобразование категориальных данных в бинарные столбцы
        skills_encoded = pd.get_dummies(
            skills,
            dtype=int,
            prefix='',
            prefix_sep='',
        )

        # Отфильтровать преобразованные данные по списку result
        skills_encoded = skills_encoded[skills_encoded.columns.intersection(result)]

        # Сбросить индексы преобразованного датафрейма
        skills_encoded = skills_encoded.reset_index(drop=True)

        # Объединение преобразованных данных с исходным датафреймом
        data_encoded = pd.concat([self.data_raw, skills_encoded], axis=1)

        # Удаление исходного столбца "навыки"
        data_encoded.drop("skill_set", axis=1, inplace=True)

        # Вывод преобразованных данных
        self.data_raw = data_encoded

    def delete_columes(self, columns_to_drop: list[str]):
        """

        Args:
            columns_to_drop:

        Returns:

        """
        self.data_raw.drop(columns_to_drop, axis=1, inplace=True)

    def transformation_y(self):
        """
        Проводит разбивку рейтинга, превращая данные в 0 и 1
        Returns:
            Обработанные данные рейтинга
        """
        self.data_raw['rating'] = self.data_raw['rating'].apply(lambda x: 1 if x > self.threshold else 0)

    def model_fit(self):
        """
        Обучение модели
        Returns:
            Сохраняет обученную модель
        """
        self.model.fit(self.x_train, self.y_train)

    def result(self):
        y_pred = self.model.predict(self.x_raw)  # Результаты модели
        print(confusion_matrix(self.y_raw, y_pred))
        print(log_loss(y_pred, self.y_raw))
        print('Б. Всей обучающей выборки')
        y_pred = self.model.predict(self.x_train)
        print(confusion_matrix(self.y_train, y_pred))
        print(log_loss(y_pred, self.y_train))
        print('В. Отобранных на собеседование кандидатов для обучающей выборки')
        print(confusion_matrix(self.y_train, y_pred)[1][1])
        print('Г.  Всей тестовой выборки')
        y_pred = self.model.predict(self.x_test)
        print(confusion_matrix(self.y_test, y_pred))
        print(log_loss(y_pred, self.y_test))
        print('Д. Отобранных на собеседование кандидатов для тестовой выборки')
        print(confusion_matrix(self.y_test, y_pred)[1][1])
