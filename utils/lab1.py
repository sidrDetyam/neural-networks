import pandas as pd
from typing import Union, Callable


def get_stats(df: pd.DataFrame,
              *stats: Union[str, Callable[[pd.Series], float]]) -> pd.DataFrame:
    stats = df.agg(stats, axis=0)
    return stats


def misses_stat(s: pd.Series) -> float:
    return s.count() / len(s)


def stre(s: pd.Series) -> float:
    return s.count() / len(s)


# C=len(df.columns)
# L=len(df.index)
# CN=df.count() #количество
# NP=((L-CN)/L)*100 #процент пропущенных значений
# MN=df.min() #минимум
# Q1=df.quantile(q=0.25) #первый квартиль
# MA=df.mean() #среднее значение
# ME=df.median() #медиана
# Q3=df.quantile(q=0.75) #третий квартиль
# MX=df.max() #максимум
# ST=df.std() #стандартное отклонение
# P=df.nunique() #мощность
# IQ=Q3-Q1 #интерквартильный размах
