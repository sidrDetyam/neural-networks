import numpy as np
import pandas as pd
import math
from typing import List, Tuple


def slice_sterges(df: pd.DataFrame, col: str) -> List[pd.DataFrame]:
    df = df.dropna(subset=[col])
    if df.shape[0] == 0:
        return []

    num_bins = int(math.ceil(math.log(df.shape[0], 2))) + 1
    mn = df[col].min()
    mx = df[col].max()
    width = (mx - mn) / num_bins

    bins = []
    for i in range(num_bins):
        bins.append(df.loc[(df[col] >= mn + width * i)
                           & (df[col] < mn + width * (i + 1))])

    return bins


def entropy(df: pd.DataFrame, target: str) -> float:
    bins = slice_sterges(df, target)
    cnt = sum(map(lambda x: x.shape[0], bins))
    if cnt == 0:
        return 0

    s = 0.0
    for i in bins:
        p = i.shape[0] / cnt
        if p != 0.0:
            s += p * math.log(p, 2)

    return -s


def entropy_a(df: pd.DataFrame, col: str, target: str) -> float:
    bins = slice_sterges(df, col)
    cnt = sum(map(lambda x: x.shape[0], bins))
    if not bins:
        return 0

    s = 0.0
    for i in range(len(bins)):
        s += entropy(bins[i], target) * bins[i].shape[0] / cnt

    return -s


def split_info(df: pd.DataFrame, col: str) -> float:
    return entropy(df, col)


def gain(df: pd.DataFrame, col: str, target: str) -> float:
    return entropy(df, target) - entropy_a(df, col, target)


def gr(df: pd.DataFrame, attr_name: str, target_name: str) -> float:
    return gain(df, attr_name, target_name) / split_info(df, attr_name)

