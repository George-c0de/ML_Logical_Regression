import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from warnings import simplefilter
import numpy as np
from sklearn.utils import resample

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def fit_predict_eval(model, features_train, target_train):
    """ Обучение модели """
    model.fit(features_train, target_train)
    # model.fit(features_train, target_train)
    # model = model.best_estimator_
    return model


def load_data(path_of_file='Data/dataNew.csv'):
    """ Загрузка данных """
    data_csv = pd.read_csv(path_of_file, encoding='utf-8')
    data_csv.drop(columns='Unnamed: 0', inplace=True)
    return data_csv


def score_model(model, features_train, features_test, target_train, target_test, data_raw, numb, min_max, X):
    """ Оценка
    Метод не используется """
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


def data_city(data_raw):
    """ Преобразование категориального признака - city"""
    cities = data_raw.city.unique()
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


def change_age(data_raw, group=2):
    """ Разбиваем весь возраст на группы бинарных
    По умолчанию групп 1
    Всего получается 4 группы:
    1. [0,21]
    2. (21, 25]
    3. (25,31)
    4. [31, +∞]
    """
    a = pd.qcut(data_raw['age'], q=group)
    values = a.unique()
    # group_age = ['0_21', '21_25', '25_31', '31_']
    group_age = []
    for el in values:
        name = str(el.left) + '_' + str(el.right)
        group_age.append(name)

    for el in group_age:
        data_raw.insert(len(data_raw.columns), el, 0)

    for el, i in zip(values, range(len(values))):
        if el.closed == 'right':
            data_raw.loc[(data_raw.age > el.left) & (data_raw.age <= el.right), ('age', group_age[i])] = 1
        elif el.closed == 'left':
            data_raw.loc[(data_raw.age >= el.left) & (data_raw.age < el.right), ('age', group_age[i])] = 1
        elif el.closed == 'both':
            data_raw.loc[(data_raw.age >= el.left) & (data_raw.age <= el.right), ('age', group_age[i])] = 1
        else:
            data_raw.loc[(data_raw.age > el.left) & (data_raw.age < el.right), ('age', group_age[i])] = 1
    # data_raw.loc[data_raw.age <= 21, ('age', group_age[0])] = 1
    # data_raw.loc[(data_raw.age <= 25) & (data_raw.age > 21), ('age', group_age[1])] = 1
    # data_raw.loc[(data_raw.age > 25) & (data_raw.age < 31), ('age', group_age[2])] = 1
    # data_raw.loc[data_raw.age >= 31, ('age', group_age[3])] = 1

    data_raw.drop(
        ['age'],
        inplace=True,
        axis=1
    )

    return data_raw


def change_total_experience(data_raw, group=1):
    a = pd.qcut(data_raw['total_experience'], q=group)
    values = a.unique()
    # group_age = ['0_21', '21_25', '25_31', '31_']
    group_age = []
    for el in values:
        name = str(el.left) + '_' + str(el.right)
        group_age.append(name)

    for el in group_age:
        data_raw.insert(len(data_raw.columns), el, 0)

    for el, i in zip(values, range(len(values))):
        if el.closed == 'right':
            data_raw.loc[(data_raw.total_experience > el.left) & (data_raw.total_experience <= el.right), (
                'total_experience', group_age[i])] = 1
        elif el.closed == 'left':
            data_raw.loc[(data_raw.total_experience >= el.left) & (data_raw.total_experience < el.right), (
                'total_experience', group_age[i])] = 1
        elif el.closed == 'both':
            data_raw.loc[(data_raw.total_experience >= el.left) & (data_raw.total_experience <= el.right), (
                'total_experience', group_age[i])] = 1
        else:
            data_raw.loc[(data_raw.total_experience > el.left) & (data_raw.total_experience < el.right), (
                'total_experience', group_age[i])] = 1
    # data_raw.loc[data_raw.age <= 21, ('age', group_age[0])] = 1
    # data_raw.loc[(data_raw.age <= 25) & (data_raw.age > 21), ('age', group_age[1])] = 1
    # data_raw.loc[(data_raw.age > 25) & (data_raw.age < 31), ('age', group_age[2])] = 1
    # data_raw.loc[data_raw.age >= 31, ('age', group_age[3])] = 1

    data_raw.drop(
        ['total_experience'],
        inplace=True,
        axis=1
    )
    return data_raw


def change_education_level(data_raw):
    work_experience = pd.unique(data_raw.education_level)

    # add specialty and education_level
    for el in work_experience:
        data_raw.insert(len(data_raw.columns), el, 0)

    # Изменяем значения
    for el in work_experience:
        data_raw.loc[(data_raw.education_level == el), ('education_level', el)] = 1
    # Удаление 'specialty', 'education_level'
    data_raw.drop(
        ['education_level'],
        inplace=True,
        axis=1
    )
    return data_raw


def data_processing(data_raw, first):
    """ Обработка данных """
    data_raw.drop(
        ['gender', 'specialty'],
        inplace=True,
        axis=1
    )
    a = data_raw.loc[data_raw['rating'] >= first]
    data_raw = data_city(data_raw)

    # Замена рейтинга 0 и 1
    data_raw.loc[(data_raw.rating < first), 'rating'] = 0
    data_raw.loc[(data_raw.rating >= first), 'rating'] = 1

    # Сохраним кандидатов с 1 рейтинга
    a.to_csv('Data/кандидаты_1.csv')
    # Категориальные признаки в столбцы
    data_raw = change_education_level(data_raw)
    # data_raw = change_specialty(data_raw)
    data_raw = change_skill_set(data_raw)
    data_raw = change_age(data_raw, group=4)
    data_raw = change_total_experience(data_raw, group=3)
    return data_raw


def change_specialty(data_raw):
    specialty_list = pd.unique(data_raw.specialty)
    for el in specialty_list:
        data_raw.insert(len(data_raw.columns), el, 0)
    for el in specialty_list:
        data_raw.loc[(data_raw.specialty == el), ('specialty', el)] = 1
    data_raw.drop(
        ['specialty'],
        inplace=True,
        axis=1
    )
    return data_raw


def change_skill_set(data_raw):
    """ Преобразовываем skill_set """
    error_list = [None, '', 'инвентаризация', 'аналитический склад ума', 'решительность', 'поиск информации в интернет',
                  'немного php', 'адаптивность']
    already_columns = list(data_raw.columns)
    all_skills = []
    for el in data_raw.skill_set:
        if type(el) == float:
            continue
        skill = el.split(',')
        for sk in skill:
            sk = sk.strip()
            sk = sk.lower()
            if sk not in all_skills and sk not in already_columns:
                if sk.lower() not in error_list:
                    if not sk.startswith('https://'):
                        all_skills.append(sk)

    for el in all_skills:
        data_raw.insert(len(data_raw.columns), el, 0)

    for i in range(len(data_raw.skill_set)):
        if type(data_raw.loc[i, 'skill_set']) == float:
            continue
        for el in data_raw.loc[i, 'skill_set'].split(','):
            if el.lower() in error_list or el.startswith('https://'):
                continue
            el = el.strip()
            el = el.lower()
            data_raw.loc[i, el] = 1
    for el in all_skills:
        if data_raw[el].sum() == 1:
            data_raw.drop(columns=el, axis=1, inplace=True)
    data_raw.drop(
        ['skill_set'],
        inplace=True,
        axis=1
    )
    return data_raw


def standardization(data_raw):
    """ Стандартизация """
    X = data_raw.drop(['rating'], axis=1)
    X_scaled = scale(X)
    return X_scaled


def min_max_scaler(data_raw):
    """ Нормализация через минимальный и максимальный """
    scaler = preprocessing.MinMaxScaler()
    names = data_raw.columns
    d = scaler.fit_transform(data_raw)
    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df


def variance(data_raw):
    """ Нормализация через дисперсию """
    scaler = preprocessing.StandardScaler()
    names = data_raw.columns
    scaler.fit(data_raw)
    d = scaler.transform(data_raw)
    scaled_df = pd.DataFrame(d, columns=names)
    return scaled_df


def save_score(data_raw, model):
    """ Сохранить коэффициенты для модели
    Метод не используется
    """
    columns = data_raw.columns
    new_columns = []
    for el in columns:
        if el != 'rating':
            new_columns.append(el)
    # array = []
    # for el in model.coef_:
    #     array.append(el[0])
    # score_result = pd.DataFrame(array, index=new_columns, columns=['score'])
    score_result = pd.DataFrame(model.best_estimator_.coef_[0], index=new_columns, columns=['score'])
    score_result.sort_values(by='score', ascending=False, inplace=True)
    score_result.to_csv('Data/score.csv')


def get_result(model, data_raw, X_test, Y_test):
    """Результат для модели"""
    columns = data_raw.columns
    columns = columns.values.tolist()
    columns.remove('rating')
    y_pred = model.predict(X_test)
    # print(precision_score(Y, y_pred, average='macro'))
    # старая версия подсчета
    model.densify()
    coef = model.coef_.reshape(-1, 1)
    score_result = pd.DataFrame(coef, index=columns, columns=['score'])

    score_result.sort_values(by='score', ascending=False, inplace=True)
    score_result.to_csv('Data/score.csv')
    all_0, all_1, true_1, true_1, true_0, successful_interviews = 0, 0, 0, 0, 0, 0
    print(log_loss(Y_test, y_pred))
    for x, y in zip(y_pred, Y_test):
        if x == y:
            if y == 0:
                all_0 += 1
                true_0 += 1
            else:
                successful_interviews += 1
                all_1 += 1
                true_1 += 1
        elif y == 1:
            all_1 += 1
        else:
            all_0 += 1
    if successful_interviews != 0:
        result_end = {
            'score': (len(y_pred) * 25000 / successful_interviews),
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
    # Новая старая
    # save_score(data_raw, model)
    # result = model.predict(X)
    # i = 0
    # res_end = 0
    # all_0 = 0
    # all_1 = 0
    # true_1 = 0
    # true_0 = 0
    # for lio in result:
    #     if lio == Y[i]:
    #         if lio == 1:
    #             res_end += 1
    #             true_1 += 1
    #         else:
    #             true_0 += 1
    #     i += 1
    # for o in Y:
    #     if o == 0:
    #         all_0 += 1
    #     else:
    #         all_1 += 1
    # if res_end != 0:
    #     result_end = {
    #         'score': (len(result) * 25000 / res_end),
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

    return result_end


def model_create(X, Y, data_raw, model):
    """ Создание модели (подготовка) """
    lab = preprocessing.LabelEncoder()
    lab.fit(Y)
    y_new = lab.fit_transform(Y)

    # X_train_, X_test_, Y_train_, Y_test_ = train_test_split(X, y_new, test_size=0.3, random_state=50)

    x_resampled = resample(X[Y == 1], n_samples=X[Y == 0].shape[0], random_state=1000)

    X_ = np.concatenate((X[Y == 0], x_resampled))
    Y_ = np.concatenate((Y[Y == 0], np.ones(shape=(X[Y == 0].shape[0],), dtype=np.int32)))

    X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.3, random_state=27)

    # smote = SMOTE(k_neighbors=3)
    # print(X_train.shape)
    # print(len(Y_train))
    # X_, Y_ = smote.fit_resample(X_train, Y_train)

    model_fit = fit_predict_eval(
        model=model,
        features_train=X_train,
        target_train=Y_train,
    )
    print('А. Всех данных')
    y_pred = model.predict(X)
    print(confusion_matrix(Y, y_pred))
    print(log_loss(y_pred, Y))
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

    result_end = get_result(
        model=model_fit,
        X_test=X_test,
        Y_test=Y_test,
        data_raw=data_raw
    )

    return result_end


def main():
    """ Главная функция """
    best_score = 1000000000
    res = []
    # np.arange(0.1, 0.9, 0.1)
    for e in np.arange(0.1, 0.2, 0.1):
        print('Загрузка...')
        data = load_data()
        data = data_processing(
            data_raw=data,
            first=e
        )
        Y_raw = data.rating

        # Стандартизация
        # X = standardization(data)
        # X_raw = scale(X_raw)

        # Дисперсия
        # data = variance(data)

        # min_max
        data = min_max_scaler(data)

        X_raw = data.loc[:, data.columns != 'rating']

        param = {
            'fit_intercept': [True, False],
            'multi_class': ['ovr'],
            'penalty': ['l2'],
            'C': [0.001, 1, 10, 100],
            'solver': ['liblinear'],
            'class_weight': [{
                1: 1,
                0: 0
            }],
            'max_iter': [100, 1000]
        }
        # model = GridSearchCV(LogisticRegression(), param_grid=param)

        """
        Собственная модель CustomLogisticRegression() имеет 2 метода обучения: fit_mse, fit_log_loss
        model = CustomLogisticRegression()
        """
        # model = CustomLogisticRegression()
        """
        Логическая регрессия от sklearn
        # best LogisticRegression(C=0.001, fit_intercept=False, penalty='none')
        """
        # best
        model = LogisticRegression(fit_intercept=False, penalty='none')
        # model = LogisticRegression(C=0.001, class_weight={
        #         1: 0.6,
        #         0: 0.4
        #     })
        # model = LogisticRegression(C=0.001, penalty='none')
        # model = LogisticRegression(C=1)
        # model = LogisticRegression(fit_intercept=False, multi_class='ovr')
        # model = LogisticRegression()
        # Для 0.1
        # model = LogisticRegression(C=1000, multi_class='multinomial')

        # model = GridSearchCV(SVR(), param_grid={
        #     'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        #         'degree':[1,3,5],
        #             'gamma': ['scale', 'auto'],
        #
        #     'C': [0.01, 1, 10, 100,1000],

        # })
        # model = SVR(kernel='rbf', C=100)
        # model = LogisticRegression()

        # model = LogisticRegression(C=10, class_weight='balanced')
        # model = LogisticRegression(C=0.001,
        # fit_intercept=False,max_iter=500,multi_class='multinomial', penalty='none')
        # model = CustomLogisticRegression()
        # best
        # model = RandomForestRegressor(criterion='absolute_error', n_estimators=1000)
        # model = GridSearchCV(RandomForestRegressor(), param_grid={
        #     'n_estimators': [10, 100, 1000],
        #     'criterion': ['squared_error', 'absolute_error', 'poisson'],
        #
        # })
        print('Данные обработались, начинается обучение')
        # model = GridSearchCV(SVR(), param_grid={
        #     'kernel': ['linear'],
        #     'C': [0.01, 1, 1000],
        #
        # })
        # model = SVR(C=1000, kernel='poly')
        # model = CustomLogisticRegression(fit_intercept=False, l_rate=0.01, n_epoch=3000)
        score = model_create(
            X=X_raw,
            Y=Y_raw,
            data_raw=data,
            model=model)
        # Проверка на наших данных
        # error_df = pd.DataFrame()
        # for el, i, y in zip(y_pred, range(len(y_pred)), Y_raw):
        #     if el != y:
        #         data_raw = load_data()
        #         error_df = error_df.append(data_raw.iloc[i])
        # error_df.to_csv('error.csv')

        print('Обучение закончилось, сохранение результатов')
        if score['score'] < best_score and score['score'] != 0:
            best_score = score['score']
        score['el'] = e
        res.append(score)
    print('best_score: {}\n'.format(best_score))
    df = pd.DataFrame(columns=['money', 'score_1', 'score_0', 'true_1', 'true_0', 'all_0', 'all_1', 'el'])
    for b in res:
        df.loc[len(df.index)] = [b['score'], round(b['percent_1'], 2), round(b['percent_0'], 2), b['true_1'],
                                 b['true_0'], b['all_0'], b['all_1'], b['el']]
        print(round(b['score']))
        print('{} - {}/{}'.format(round(b['percent_0'], 2), b['true_0'], b['all_0']))
        print('{} - {}/{}'.format(round(b['percent_1'], 2), b['true_1'], b['all_1']))
        print('el: {}'.format(b['el']))
        print()
    df.to_csv('End_score.csv')


if __name__ == "__main__":
    main()
