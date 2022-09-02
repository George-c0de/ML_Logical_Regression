from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from CustomLogisticRegression import CustomLogisticRegression

import numpy as np
import pandas as pd

from main_old import data_processing


def load_data(path_of_file='Data/dataNew.csv'):
    data_csv = pd.read_csv(path_of_file, encoding='utf-8')
    data_csv.drop(columns='Unnamed: 0', inplace=True)
    return data_csv


intercept_flag = True
n_epoch, learning_rate = 1000, 0.01
train_size, random_state = 0.8, 43
cut_off = 0.5
data = load_data()
data = data_processing(data, 0.5)
Y = data.rating
X = data.loc[:, data.columns != 'rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=27)
print(1)
model = CustomLogisticRegression(fit_intercept=intercept_flag, l_rate=learning_rate, n_epoch=n_epoch)
model_2 = CustomLogisticRegression(fit_intercept=intercept_flag, l_rate=learning_rate, n_epoch=n_epoch)
model.fit_mse(X_train, Y_train)
model_2.fit_log_loss(X_train, Y_train)
y_predicted = model.predict(X_test, cut_off=cut_off)
y_predicted_2 = model_2.predict(X_test, cut_off=cut_off)
acc_score = accuracy_score(np.array(Y_test), np.array(y_predicted))
acc_score_2 = accuracy_score(np.array(Y_test), np.array(y_predicted_2))
print(acc_score)
print(acc_score_2)
