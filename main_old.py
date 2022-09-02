import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from warnings import simplefilter
import numpy as np
from imblearn.over_sampling import SMOTE

from CustomLogisticRegression import CustomLogisticRegression

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

""" Функция для создания модели и обучения, выводит выводит название модели и оценку """


def fit_predict_eval(model, features_train, target_train):
    model.fit(features_train, target_train)
    return model


""" Загрузка данных """


def load_data(path_of_file='Data/dataNew.csv'):
    data_csv = pd.read_csv(path_of_file, encoding='utf-8')
    data_csv.drop(columns='Unnamed: 0', inplace=True)
    return data_csv


""" Оценка """


def score_model(model, features_train, features_test, target_train, target_test, data_raw, numb, min_max, X):
    # Оценка
    y_pred = model.predict(features_test)
    # precision = precision_score(target_test, y_pred, average='macro')
    # print(f'Model: {model}\nAccuracy: {precision}\n')

    score_f1 = f1_score(target_test, y_pred, average='macro')
    # print(f'Accuracy: {score}\n')
    # y_prob = model.predict_proba(features_test)
    # score_f = precision_recall_fscore_support(target_test, y_pred, average='macro')
    # print(f'Accuracy: {score_f}\n')

    # Precision & Recall
    # print(f'Accuracy: {precision_score(target_test, y_pred)}')
    # print(f'Accuracy: {recall_score(target_test, y_pred)}')

    # ошибки
    # # Mean Absolute Error
    # print(f'Mean Absolute Error: {mean_absolute_error(target_test, y_pred)}')
    #
    # # Root Mean Squared Deviation
    # print(f'Root Mean Squared Deviation: {mean_squared_error(target_test, y_pred)}')
    #
    # # R2 score
    # print(f'R2 score: {r2_score(target_test, y_pred)}')
    #
    # # Mean Squared Error
    # print(f'Mean Squared Error: {mean_squared_error(target_test, y_pred)}')
    #
    # # Accuracy
    # print(f'Accuracy: {accuracy_score(target_test, y_pred)}')
    #
    # print('coefficient of determination:', model.score(features_train, target_train))
    # print('slope:', model.coef_[0])
    i = 0
    score_new = []
    columns = data_raw.columns
    new_columns = []
    for el in model.coef_[0]:
        if el > 0:
            new_columns.append(columns[i])
            score_new.append(el)
            # score[columns[i]] = [el, ]
        i += 1

    score_result = pd.DataFrame(score_new, index=new_columns, columns=['score'])
    score_result.sort_values(by='score', ascending=False, inplace=True)
    score_result.to_csv('Data/score.csv')
    matrix = confusion_matrix(target_test, y_pred)
    # print(matrix)
    # cm_display = ConfusionMatrixDisplay(confusion_matrix(target_test, y_pred)).plot()
    if min_max:
        plt.savefig('Image/min_max/filename{}.svg'.format(numb))
    else:
        plt.savefig('Image/Дисперсия/filename{}.svg'.format(numb))
    # print(precision_recall_curve(target_test, y_pred))
    # score_result.plot()

    # plt.show()
    # print('intercept:', model.intercept_)
    return_score = {
        'f1': score_f1,
        'matrix': matrix
    }
    return return_score


""" Преобразование категориального признака - ГОРОД"""


def data_city(data_raw):
    cities = data_raw.city.unique()
    city = {

    }
    i = 1
    for el in cities:
        city[el] = i
        i += 1
    for el in cities:
        data_raw.insert(len(data_raw.columns), el, 0)
    for el in cities:
        data_raw.loc[(data_raw.city == el), ('city', el)] = 1
    data_raw.drop(
        ['city'],
        inplace=True,
        axis=1
    )
    return data_raw


""" Преобразование данных """


def change_age(data_raw):
    import matplotlib.pyplot as plt
    group_age = ['0_21', '21_25', '25_31', '31_']
    for el in group_age:
        data_raw.insert(len(data_raw.columns), el, 0)

    data_raw.loc[(data_raw.age <= 25) & (data_raw.age > 21), ('age', group_age[1])] = 1

    data_raw.loc[data_raw.age <= 21, ('age', group_age[0])] = 1

    data_raw.loc[(data_raw.age > 25) & (data_raw.age < 31), ('age', group_age[2])] = 1

    data_raw.loc[data_raw.age >= 31, ('age', group_age[3])] = 1
    data_raw.drop(
        ['age'],
        inplace=True,
        axis=1
    )

    return data_raw


def data_processing(data_raw, first):
    data_raw.drop(
        ['gender'],
        inplace=True,
        axis=1
    )
    data_raw = data_city(data_raw)

    # Замена рейтинга 0 и 1
    a = data_raw.loc[data_raw['rating'] >= 0.1]
    data_raw.loc[(data_raw.rating <= first), 'rating'] = 0
    data_raw.loc[(data_raw.rating > first), 'rating'] = 1

    # Сохраним кандидатов с 1 рейтинга

    a.to_csv('Data/score_one_1.csv')
    # Категориальные признаки в столбцы
    data_raw = data_to_category(data_raw)
    data_raw = change_skill_set(data_raw.skill_set, data_raw)
    data_raw = change_age(data_raw)
    return data_raw


""" Категориальные признаки """


def data_to_category(data_raw):
    # work_experience = [
    #     'candidate',
    #     'master',
    #     'higher',
    #     'special_secondary',
    #     'secondary',
    #     'unfinished_higher',
    #     'bachelor',
    # ]
    work_experience = pd.unique(data_raw.education_level)
    # specialty_list = [
    #     'developer',
    #     'javadeveloper',
    #     'javajun',
    #     'javaintern',
    #     'other',
    # ]
    specialty_list = pd.unique(data_raw.specialty)
    # add specialty and education_level
    for el in work_experience:
        data_raw.insert(len(data_raw.columns), el, 0)
    for el in specialty_list:
        data_raw.insert(len(data_raw.columns), el, 0)

    # Изменяем значения
    for el in specialty_list:
        data_raw.loc[(data_raw.specialty == el), ('specialty', el)] = 1

    for el in work_experience:
        data_raw.loc[(data_raw.education_level == el), ('education_level', el)] = 1
    # Удаление 'specialty', 'education_level'
    data_raw.drop(
        ['specialty', 'education_level'],
        inplace=True,
        axis=1
    )
    return data_raw


""" Преобразовываем skill_set """


def change_skill_set(skill_set, data_raw):
    already_columns = list(data_raw.columns)
    all_skills = []
    for el in skill_set:
        if type(el) == float:
            continue
        skill = el.split(',')
        for sk in skill:
            sk = sk.strip()
            sk = sk.lower()
            if sk not in all_skills and sk not in already_columns:
                if sk is not None and sk != '':
                    if not sk.startswith('https://'):
                        all_skills.append(sk)

    for el in all_skills:
        data_raw.insert(len(data_raw.columns), el, 0)
    error_list = [None, '']
    for i in range(len(data_raw.skill_set)):
        if type(data_raw.loc[i, 'skill_set']) == float:
            continue
        for el in data_raw.loc[i, 'skill_set'].split(','):
            if el in error_list or el.startswith('https://'):
                continue
            el = el.strip()
            el = el.lower()
            data_raw.loc[i, el] = 1
    pf = pd.DataFrame(data_raw)
    pf.to_csv('Data/mylist.csv')
    data_raw.drop(
        ['skill_set'],
        inplace=True,
        axis=1
    )
    return data_raw


""" Стандартизация """


def standardization(data_raw):
    X = data_raw.drop(['rating'], axis=1)
    X_scaled = scale(X)
    return X_scaled


""" Нормализация """


def min_max_scaler(data_raw):
    scaler = preprocessing.MinMaxScaler()
    names = data_raw.columns
    d = scaler.fit_transform(data_raw)
    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df


""" Дисперсия """


def variance(data_raw):
    scaler = preprocessing.StandardScaler()
    names = data_raw.columns
    scaler.fit(data_raw)
    d = scaler.transform(data_raw)
    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df


""" Сохранить коэффициенты для модели """


def save_score(data_raw, model):
    columns = data_raw.columns
    new_columns = []
    for el in columns:
        if el != 'rating':
            new_columns.append(el)
    # array = []
    # for el in model.coef_:
    #     array.append(el[0])
    # score_result = pd.DataFrame(array, index=new_columns, columns=['score'])
    score_result = pd.DataFrame(model.coef_[0], index=new_columns, columns=['score'])
    score_result.sort_values(by='score', ascending=False, inplace=True)
    score_result.to_csv('Data/score.csv')


"""Результат для модели"""


def get_result(model, X, Y, data_raw, X_test, Y_test):
    i = 0
    score_new = []
    columns = data_raw.columns
    new_columns = []
    for el in columns:
        if el != 'rating':
            new_columns.append(el)

    # for el in model.coef_[0]:
    #     if el > 0:
    #         new_columns.append(columns[i])
    #         score_new.append(el)
    #         # score[columns[i]] = [el, ]
    #     i += 1
    # старая
    # score_result = pd.DataFrame(model.coef_[0], index=new_columns, columns=['score'])
    # score_result.sort_values(by='score', ascending=False, inplace=True)
    # score_result.to_csv('Data/score.csv')
    # y_pred = model.predict(X_test)
    # all_0 = 0
    # all_1 = 0
    # true_1 = 0
    # true_0 = 0
    # successful_interviews = 0
    # print(log_loss(Y_test, y_pred))
    # for x, y in zip(y_pred, Y_test):
    #     if x == y:
    #         if y == 0:
    #             all_0 += 1
    #             true_0 += 1
    #         else:
    #             successful_interviews += 1
    #             all_1 += 1
    #             true_1 += 1
    #     elif y == 1:
    #         all_1 += 1
    #     else:
    #         all_0 += 1
    # if successful_interviews != 0:
    #     result_end = {
    #         'score': (len(y_pred) * 25000 / successful_interviews),
    #         'percent_0': true_0 / all_0,
    #         'percent_1': true_1 / all_1,
    #         'all_0': all_0,
    #         'all_1': all_1,
    #         'true_0': true_0,
    #         'true_1': true_1
    #     }
    # else:
    #     result_end = {
    #         'score': 0,
    #         'percent_0': true_0 / all_0,
    #         'percent_1': true_1 / all_1,
    #         'all_0': all_0,
    #         'all_1': all_1,
    #         'true_0': true_0,
    #         'true_1': true_1
    #     }
    # Новая старая
    save_score(data_raw, model)
    result = model.predict(X)
    i = 0
    res_end = 0
    all_0 = 0
    all_1 = 0
    true_1 = 0
    true_0 = 0
    for lio in result:
        if lio == Y[i]:
            if lio == 1:
                res_end += 1
                true_1 += 1
            else:
                true_0 += 1
        i += 1
    for o in Y:
        if o == 0:
            all_0 += 1
        else:
            all_1 += 1
    if res_end != 0:
        result_end = {
            'score': (len(result) * 25000 / res_end),
            'percent_0': true_0 / all_0,
            'percent_1': true_1 / all_1,
            'all_0': all_0,
            'all_1': all_1,
            'true_0': true_0,
            'true_1': true_1
        }
    else:
        result_end = {
            'score': 0,
            'percent_0': true_0 / all_0,
            'percent_1': true_1 / all_1,
            'all_0': all_0,
            'all_1': all_1,
            'true_0': true_0,
            'true_1': true_1
        }

    return result_end


""" Создание модели (подготовка) """


def model_create(X, Y, data_raw, numb, model, min_max=True):
    lab = preprocessing.LabelEncoder()
    lab.fit(Y)
    y_new = lab.fit_transform(Y)
    #
    # from sklearn.utils import resample
    # x_resampled = resample(X[y_new == 1], n_samples=X[y_new == 0].shape[0], random_state=1000)
    #
    # X_ = np.concatenate((X[y_new == 0], x_resampled))
    # Y_ = np.concatenate((y_new[y_new == 0], np.ones(shape=(X[y_new == 0].shape[0],), dtype=np.int32)))

    X_train, X_test, Y_train, Y_test = train_test_split(X, y_new, test_size=0.3, random_state=27)
    # smote = SMOTE(k_neighbors=3)
    # print(X_train.shape)
    # print(len(Y_train))
    # X_, Y_ = smote.fit_resample(X_train, Y_train)

    # print(X_.shape)
    # print(Y_.shape)
    y_1 = 0
    y_0 = 0
    for el in Y_train:
        if el == 1:
            y_1 += 1
        else:
            y_0 += 1
    # print(y_0)
    # print(y_1)
    model_fit = fit_predict_eval(
        model=model,
        features_train=X_train,
        target_train=Y_train,
    )
    result_end = get_result(
        model=model_fit,
        X_test=X_test,
        Y_test=Y_test,
        X=X,
        Y=Y,
        data_raw=data_raw
    )

    # score_of_model = score_model(
    #     model=model_fit,
    #     features_train=X_train,
    #     target_train=Y_train,
    #     features_test=X_test,
    #     target_test=Y_test,
    #     data_raw=data_raw,
    #     numb=numb,
    #     min_max=min_max,
    #     X=X
    # )

    return result_end


""" Обработка данных """


def main():
    max_score = {
        'f1': 0,
        'matrix': None,
        'el': 0.0
    }
    best_score = 1000000000
    res = []
    ni = [0.7]
    # np.arange(0.1, 0.9, 0.1)
    for e in np.arange(0.1, 0.9, 0.1):
        print('Загрузка...')
        data = load_data()
        data = data_processing(data, e)
        Y_raw = data.rating
        # Y = data.drop(['rating'], axis=1)
        # X = standardization(data)

        # Дисперсия
        # data = variance(data)

        # min_max
        data = min_max_scaler(data)

        X_raw = data.loc[:, data.columns != 'rating']
        model = LogisticRegression(fit_intercept=False, multi_class='ovr')
        print('Данные обработались, начинается обучение')
        # model = CustomLogisticRegression(fit_intercept=False, l_rate=0.01, n_epoch=3000)
        score = model_create(X_raw, Y_raw, data, e, model=model,
                             min_max=False)
        print('Обучение закончилось, сохранение результатов')
        if score['score'] < best_score and score['score'] != 0:
            best_score = score['score']
        score['el'] = e
        res.append(score)
    print('best_score: {}\n'.format(best_score))
    for b in res:
        print(round(b['score']))
        print('{} - {}/{}'.format(round(b['percent_0'], 2), b['true_0'], b['all_0']))
        print('{} - {}/{}'.format(round(b['percent_1'], 2), b['true_1'], b['all_1']))
        print('el: {}'.format(b['el']))
        print()


if __name__ == "__main__":
    main()
