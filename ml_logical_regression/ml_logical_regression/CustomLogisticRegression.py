import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = np.array([[]])

    def sigmoid(self, t):
        return [1 / (1 + math.exp(-t[i])) for i in range(len(t))]

    def predict_proba(self, row, coef_):
        t = row @ coef_
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        n_row = np.shape(X_train)[0]
        n_coefficient = np.shape(X_train)[1]
        self.coef_ = np.zeros((n_coefficient, 1))
        X_train = np.array(X_train)  # (n_row) x (n_feature)
        y_train = np.array([y_train])  # (n_row) x (n_target)

        for iter_ in range(1, self.n_epoch + 1):
            # if iter_ % 100 == 0:
            #     print(f"\nIteration: {iter_}\n------------\n")
            #     print(f"\nX_train: {X_train}\n")
            #     print(f"\nCoefficients: {self.coef_}\n")
            for i, row in enumerate(X_train):
                row = np.array([row])  # 1 x (n_feature)
                y_hat = self.predict_proba(row, self.coef_)  # 1x1
                y_hat = np.array([y_hat])
                # update all weights
                self.coef_ = self.coef_ - self.l_rate * (y_hat - y_train[0, i]) * y_hat * (1 - y_hat) * row.T

    def predict(self, X_test, cut_off=0.5):
        n_row = np.shape(X_test)[0]
        predictions = [0 for _ in range(n_row)]
        X_test = np.array(X_test)
        for i, row in enumerate(X_test):
            y_hat = self.predict_proba(row, self.coef_)
            y_hat = 1 if y_hat[0] >= cut_off else 0
            predictions[i] = y_hat
        return predictions  # predictions are binary values - 0 or 1

    def fit_log_loss(self, X_train, y_train):
        n_row = np.shape(X_train)[0]
        n_coefficient = np.shape(X_train)[1]
        self.coef_ = np.zeros((n_coefficient, 1))  # initialized weights and/or bias 1 x (n_coefficient)
        X_train = np.array(X_train)  # (n_row) x (n_feature)
        y_train = np.array([y_train])  # (n_row) x (n_target)

        for iter_ in range(1, self.n_epoch + 1):
            # if iter_ % 100 == 0:
            #     print(f"\nIteration: {iter_}\n------------\n")
            #     print(f"\nX_train: {X_train}\n")
            #     print(f"\nCoefficients: {self.coef_}\n")
            for i, row in enumerate(X_train):
                row = np.array([row])  # 1 x (n_feature)
                y_hat = self.predict_proba(row, self.coef_)  # 1x1
                y_hat = np.array([y_hat])
                # update all weights
                self.coef_ = self.coef_ - self.l_rate * ((y_hat - y_train[0, i]) / n_row) * row.T


def standardize_z(x):
    """
    Take a nx1-vector, then perform z-standardization using:
    "z[i] = (x[i] - mu) / sigma" where:

    - z[i]  --> i-th sample standard score
    - x[i]  --> i-th sample value
    - mu    --> mean of x
    - sigma --> standard deviation of x

    :param x: feature data in dataframe (569 x 1 columns)
              --> i.e. 'worst concave points', 'worst perimeter'
    :return: standardized feature data in numpy array (569 x 1 columns)
             --> i.e. 'std worst concave points', 'std worst perimeter'
    """
    z = list(map(lambda x_: (x_ - np.mean(x)) / np.std(x), x))
    return z
