"""Общие импорты проекта.

Модули src импортируют нужные имена явно: ``from src.config import pd``.
Функции проекта остаются в своих модулях, чтобы избежать циклических импортов.
Здесь собраны стандартная библиотека, внешние зависимости и общие типы.
"""

# Стандартная библиотека и типы.
import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, TypeAlias

# Таблицы и численные расчёты.
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

# Модель и классификационные метрики. Ранжирование реализовано в ranking_metrics.
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

# Общие типы данных и параметров. Здесь нет логики подготовки или обучения.
FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]
TemporalFold: TypeAlias = tuple[IndexArray, IndexArray]
ModelParams: TypeAlias = dict[str, Any]
