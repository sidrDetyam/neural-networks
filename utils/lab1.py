import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List
import seaborn as sns


def get_stats(df: pd.DataFrame,
              *stats: str | Callable[[pd.Series], float],
              **kwargs: List[str]) -> pd.DataFrame:
    stats = df.agg(stats, axis=0)
    if "names" in kwargs:
        stats.set_index(pd.Index(data=kwargs["names"]), inplace=True)

    return stats


def misses_stat(s: pd.Series) -> float:
    return s.count() / len(s)


def quantile_stat_decorator(q: float) -> Callable[[pd.Series], float]:
    return lambda s: s.quantile(q)


def interquantile_stat_decorator(q_min: float,
                                 q_max: float) -> Callable[[pd.Series], float]:
    qmin_stat = quantile_stat_decorator(q_min)
    qmax_stat = quantile_stat_decorator(q_max)
    return lambda s: qmax_stat(s) - qmin_stat(s)


def default_stats(df: pd.DataFrame) -> pd.DataFrame:
    return get_stats(df,
                     "count", misses_stat, "nunique",
                     "min", "max",
                     quantile_stat_decorator(0.25),
                     quantile_stat_decorator(0.5),
                     quantile_stat_decorator(0.75),
                     interquantile_stat_decorator(0.25, 0.75),
                     "mean",
                     "median",
                     "std",
                     names=["Не null", "Заполнено", "Мощность",
                            "Min", "Max",
                            "q=0.25", "q=0.5", "q=0.75", "Размах",
                            "Среднее", "Медиана", "Стд"])


def remove_deviations(df: pd.DataFrame,
                      columns: List[Tuple[str, float, float]],
                      target_columns: List[str]) -> pd.DataFrame:
    for name, mn, mx in columns:
        mask = (df[name] < mn) | (df[name] > mx)

        for target_column in target_columns:
            mask &= df[target_column].notnull()

        df = df.loc[~mask]

    return df


def show_density_with_deviations(df: pd.DataFrame) -> None:
    stats = default_stats(df)
    for i in df.columns:
        inter = stats.loc["Размах"][i]
        left = stats.loc["q=0.25"][i] - 1.5 * inter
        right = stats.loc["q=0.75"][i] + 1.5 * inter

        plt.figure(i)
        sns.histplot(df[i], kde=True, stat="density")
        plt.axvline(left, color="red", ls='--')
        plt.axvline(right, color="red", ls='--')
        plt.show()


