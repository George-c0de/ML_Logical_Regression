# Предсказание рейтинга кандидата с помощью Логистической регрессии
***
- Функция ***load_data*** загружает данные, путь указывается в переменной path_of_fileпо умолчанию ***Data/dataNew.csv***
- По умолчанию разграничение нулевого и единичного рейтинга происходит по параметру 0.1 (строгая классификация, то есть кандидат с рейтингом больше 0 будет считаться как единица).
Этот параметр можно изменить в функции main указав или один или несколько значений для сравнения их.
```python
for e in np.arange(0.1, 0.2, 0.1):

```
- В результате программа выводит информацию 
  + Матрицу путанности для :
    + А. Всех данных
    + Б. Всей обучающей выборки
    + Г.  Всей тестовой выборки
  + В. Отобранных на собеседование кандидатов для обучающей выборки
  + Д. Отобранных на собеседование кандидатов для тестовой выборки
  + Лучшую стоимость одного собеседования, расчитанную по формуле:
    + (Кол-во всех кандидатов * стоимость одного собеседования)/Кол-во успешных собеседований
  + ***Если указано больше однго разграничителя*** выводиться
    + Стоимость одного собеседования 
    + Матрица путанности 
    + Разграничитель