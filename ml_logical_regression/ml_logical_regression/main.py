import pandas as pd
from pandas import DataFrame

from ml_logical_regression.ml_logical_regression.decorator.decorators import measure_execution_time


class LogicalRegression:
    def __init__(self, file_name: str):
        self.__file_name = file_name
        self.data_raw = None

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
            drop_first=True,
            dtype=int,
            sparse=True,
            prefix='',
            prefix_sep='',
        )


test = LogicalRegression('../../Data/dataNew.csv')
test.load_data()
test.process_city_column()

print(test.data_raw)
