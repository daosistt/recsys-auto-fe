"""Метрики ранжирования, реализованные без sklearn и других metric-библиотек."""

from __future__ import annotations

from src.config import Any, ArrayLike, FloatArray, IndexArray, NDArray, np


def _as_arrays(y_true: ArrayLike, y_score: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Привести метки и оценки к одномерным вещественным массивам.

    Args:
        y_true: Метки релевантности.
        y_score: Предсказанные оценки в том же порядке.

    Returns:
        Два массива float64 одинаковой длины.

    Raises:
        ValueError: Массивы не одномерны или имеют разные длины.
    """
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError('y_true и y_score должны быть одномерными')
    if len(labels) != len(scores):
        raise ValueError('y_true и y_score должны иметь одинаковую длину')
    return labels, scores


def reciprocal_rank(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Посчитать обратный ранг первого релевантного объекта в одной выдаче.

    Args:
        y_true: Метки; положительная метка означает релевантность.
        y_score: Оценки для сортировки по убыванию. При равенстве сохраняется
            исходный порядок объектов.

    Returns:
        1 / ранг первого релевантного объекта либо 0, если таких объектов нет.

    Raises:
        ValueError: Метки и оценки имеют несовместимые размеры.
    """
    labels, scores = _as_arrays(y_true, y_score)
    order = np.argsort(-scores, kind='stable')
    relevant_positions = np.flatnonzero(labels[order] > 0)
    if len(relevant_positions) == 0:
        return 0.0
    return 1.0 / (int(relevant_positions[0]) + 1)


def dcg_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Посчитать DCG с логарифмическим дисконтированием первых k позиций.

    Args:
        y_true: Неотрицательные метки релевантности объектов одной выдачи.
        y_score: Оценки для сортировки объектов по убыванию.
        k: Число верхних позиций ранжирования.

    Returns:
        Сумма (2 ** label - 1) / log2(rank + 1) для первых k объектов.
        Если объектов меньше k, используются все доступные объекты.

    Raises:
        ValueError: k меньше 1 либо метки и оценки несовместимы по размеру.
    """
    if k < 1:
        raise ValueError('k должен быть положительным')
    labels, scores = _as_arrays(y_true, y_score)
    order = np.argsort(-scores, kind='stable')[:k]
    ranked_labels = labels[order]
    discounts = np.log2(np.arange(2, len(ranked_labels) + 2))
    gains = np.power(2.0, ranked_labels) - 1.0
    return float(np.sum(gains / discounts))


def ndcg_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Нормировать DCG одной выдачи на качество идеального ранжирования.

    Args:
        y_true: Неотрицательные метки релевантности объектов одной выдачи.
        y_score: Предсказанные оценки объектов.
        k: Число верхних позиций ранжирования.

    Returns:
        DCG@k / ideal_DCG@k. Если идеальный DCG равен нулю, возвращается 0.

    Raises:
        ValueError: k меньше 1 либо метки и оценки несовместимы по размеру.
    """
    labels, scores = _as_arrays(y_true, y_score)
    actual_dcg = dcg_at_k(labels, scores, k)
    ideal_dcg = dcg_at_k(labels, labels, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def _group_positions(group_ids: NDArray[Any]) -> list[IndexArray]:
    """Собрать позиции строк каждой выдачи без перестановки исходных данных.

    Args:
        group_ids: Одномерный массив хешируемых идентификаторов выдач.

    Returns:
        Список массивов позиционных индексов, по одному на каждую выдачу.
        Порядок выдач соответствует их первому появлению.
    """
    groups: dict[Any, list[int]] = {}
    for position, group_id in enumerate(group_ids):
        groups.setdefault(group_id, []).append(position)
    return [np.asarray(positions, dtype=int) for positions in groups.values()]


def ranking_metrics(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_ids: ArrayLike,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, float]:
    """Усреднить MRR и nDCG@k по выдачам, содержащим хотя бы один клик.

    Args:
        y_true: Метки релевантности всех строк.
        y_score: Предсказанные оценки в том же порядке.
        group_ids: Идентификаторы выдач; одинаковые значения объединяют строки.
        ks: Глубины ранжирования для расчёта nDCG.

    Returns:
        Словарь с mrr и ndcg@k для каждого k. Выдачи без положительных меток
        исключаются из среднего. Если все выдачи такие, значения равны нулю.

    Raises:
        ValueError: Массивы имеют несовместимые размеры или одно из k меньше 1.
    """
    labels, scores = _as_arrays(y_true, y_score)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or len(groups) != len(labels):
        raise ValueError('group_ids должен быть одномерным и совпадать по длине')
    if any(k < 1 for k in ks):
        raise ValueError('Все значения k должны быть положительными')

    values: dict[str, list[float]] = {
        'mrr': [],
        **{f'ndcg@{k}': [] for k in ks},
    }
    for positions in _group_positions(groups):
        group_labels = labels[positions]
        if not np.any(group_labels > 0):
            continue
        group_scores = scores[positions]
        values['mrr'].append(reciprocal_rank(group_labels, group_scores))
        for k in ks:
            values[f'ndcg@{k}'].append(
                ndcg_at_k(group_labels, group_scores, k)
            )

    if not values['mrr']:
        return {name: 0.0 for name in values}
    return {
        name: float(np.mean(metric_values))
        for name, metric_values in values.items()
    }


def mean_ndcg_at_k(
    y_true: ArrayLike,
    y_score: ArrayLike,
    group_ids: ArrayLike,
    k: int,
) -> float:
    """Посчитать средний nDCG@k для выбора модели и признаков.

    Args:
        y_true: Метки релевантности всех строк.
        y_score: Предсказанные оценки в том же порядке.
        group_ids: Идентификаторы выдач для группировки строк.
        k: Число верхних позиций ранжирования.

    Returns:
        Средний nDCG@k по выдачам с положительными метками либо 0, если их нет.

    Raises:
        ValueError: Размеры массивов несовместимы или k меньше 1.
    """
    return ranking_metrics(y_true, y_score, group_ids, ks=(k,))[f'ndcg@{k}']
