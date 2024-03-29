# Предсказание рейтинга кандидата с помощью Логистической регрессии

Программа предназначена для предсказания рейтинга кандидата с использованием Логистической регрессии.

## Описание

- Функция **load_data** загружает данные, путь указывается в переменной `path_of_file`, по умолчанию **Data/dataNew.csv
  **.
- По умолчанию разграничение между нулевым и единичным рейтингом происходит при значении параметра 0.1 (строгая
  классификация, то есть кандидат с рейтингом больше 0 будет считаться единицей).
  Этот параметр можно изменить в функции `main`, указав одно или несколько значений для сравнения.
- В результате программа выводит следующую информацию:
    + Матрицу путаницы для:
        + А. Всех данных
        + Б. Всей обучающей выборки
        + Г. Всей тестовой выборки
    + В. Отобранных на собеседование кандидатов для обучающей выборки
    + Д. Отобранных на собеседование кандидатов для тестовой выборки
    + Лучшую стоимость одного собеседования, рассчитанную по формуле:
        + (Количество всех кандидатов * стоимость одного собеседования) / Количество успешных собеседований
    + **Если указано более одного разграничителя**, выводится:
        + Стоимость одного собеседования
        + Матрица путаницы
        + Разграничитель

## Структура программы

```plaintext
├── Data/                          # Папка для добавления файлов
├── .env                           # Файл с секретными ключами, паролями, хостами и портами
├── .gitignore                     # Файл для игнорирования файлов при выгрузке в систему отслеживания версий
├── ml_logical_regression/         # Директория с внутренней структурой программы
│   ├── CustomLogisticRegression.py    # Логистическая регрессия без использования библиотек
│   ├── main_old.py        # Логистическая регрессия с использованием библиотеки sklearn
│   ├── settings.py        # Настройка зависимостей и подключений
│   ├── postgres_work.py   # Настройка базы данных PostgreSQL
│   └── tests/             # Папка для создания и запуска тестов
├── pyproject.toml        # Файл зависимостей
└── README.md             # Инструкции по запуску программы
```
## Зависимости

Для работы программы необходимы следующие зависимости:

- Python 3.x
- Библиотеки Python, перечисленные в файле `pyproject.toml`

## Инструкции по установке и запуску

1. Установите Poetry, если вы еще не установили его на своей системе. Инструкции по установке Poetry можно найти на официальном сайте: [https://python-poetry.org/docs/](https://python-poetry.org/docs/).
2. Склонируйте репозиторий с программой на свой компьютер.
3. Перейдите в корневую директорию проекта.
4. Запустите команду `poetry install` для установки всех зависимостей, указанных в файле pyproject.toml.
5. Запустите программу, выполните команду `poetry run python ml_logical_regression/main_old.py`.

## Лицензия

Эта программа распространяется под лицензией [MIT](LICENSE).
