"""Оценка вероятностей клика и ранжирования на фиксированной выборке."""

from src.config import ArrayLike, average_precision_score, log_loss, np, pd, roc_auc_score

from src.ranking_metrics import ranking_metrics


def evaluate_quality(
    valid: pd.DataFrame,
    click_probability: ArrayLike,
    train_click_rate: float,
) -> dict[str, float]:
    """Посчитать качество модели, вывести метрики и случайные ориентиры.

    Args:
        valid: Проверочная таблица с label и impression_id. Порядок строк
            должен совпадать с порядком переданных вероятностей.
        click_probability: Вероятности положительного класса, по одной на строку.
        train_click_rate: Доля кликов в обучающих данных для константного
            baseline LogLoss. Проверочные метки для её выбора не используются.

    Returns:
        Словарь метрик: ROC-AUC, average precision (ключ pr_auc), MRR,
        nDCG@5/10, LogLoss, средняя вероятность и реальные доли кликов.
        Также включает LogLoss константы и средние метрики 20 случайных
        ранжирований. MRR/nDCG усредняются по выдачам хотя бы с одним кликом.

    Raises:
        ValueError: Размеры или значения входов несовместимы с метриками;
            для ROC-AUC в проверочной выборке должны присутствовать оба класса.
    """
    y_true = valid['label'].to_numpy()
    scores = np.asarray(click_probability)

    # 1. Основные метрики классификации
    roc_auc = float(roc_auc_score(y_true, scores))
    pr_auc = float(average_precision_score(y_true, scores))

    ranking = ranking_metrics(
        y_true=y_true,
        y_score=scores,
        group_ids=valid['impression_id'].to_numpy(),
        ks=(5, 10),
    )

    # Случайный baseline для MRR и nDCG: усредняем 20 перемешиваний.
    rng = np.random.default_rng(42)
    random_ranking_values: list[dict[str, float]] = []
    for _ in range(20):
        random_ranking_values.append(
            ranking_metrics(
                y_true=y_true,
                y_score=rng.random(len(y_true)),
                group_ids=valid['impression_id'].to_numpy(),
                ks=(5, 10),
            )
        )
    random_ranking = {
        name: float(np.mean([result[name] for result in random_ranking_values]))
        for name in ('mrr', 'ndcg@5', 'ndcg@10')
    }

    # 2. Средняя вероятность и реальная доля кликов
    mean_prediction = float(scores.mean())
    real_click_rate = float(y_true.mean())

    # 3. LogLoss модели и константного baseline
    model_logloss = float(log_loss(y_true, scores))
    baseline_scores = np.full(len(y_true), train_click_rate)
    baseline_logloss = float(log_loss(y_true, baseline_scores))

    print('1. Основные метрики')
    print(f'ROC-AUC:  {roc_auc:.6f} (baseline: 0.500000)')
    print(f'PR-AUC:   {pr_auc:.6f} (baseline: {real_click_rate:.6f})')
    print(f'MRR:      {ranking["mrr"]:.6f} (baseline: {random_ranking["mrr"]:.6f})')
    print(f'nDCG@5:   {ranking["ndcg@5"]:.6f} (baseline: {random_ranking["ndcg@5"]:.6f})')
    print(f'nDCG@10:  {ranking["ndcg@10"]:.6f} (baseline: {random_ranking["ndcg@10"]:.6f})')

    print('\n2. Средняя вероятность клика')
    print(f'Предсказание: {mean_prediction:.6f} (реальная доля: {real_click_rate:.6f})')

    print('\n3. LogLoss')
    print(f'Модель: {model_logloss:.6f} (baseline: {baseline_logloss:.6f})\n')

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'mrr': ranking['mrr'],
        'ndcg@5': ranking['ndcg@5'],
        'ndcg@10': ranking['ndcg@10'],
        'mean_prediction': mean_prediction,
        'real_click_rate': real_click_rate,
        'logloss': model_logloss,
        'baseline_logloss': baseline_logloss,
        'baseline_mrr': random_ranking['mrr'],
        'baseline_ndcg@5': random_ranking['ndcg@5'],
        'baseline_ndcg@10': random_ranking['ndcg@10'],
    }
