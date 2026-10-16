"""Временные фолды и подбор гиперпараметров ручного baseline."""

from src.config import Any, CatBoostClassifier, TemporalFold, dataclass, np, pd, product

from src.ranking_metrics import mean_ndcg_at_k


@dataclass
class TemporalSearchResult:
    """Результат временного подбора параметров CatBoost.

    Attributes:
        best_estimator_: Модель с лучшими параметрами на всех переданных строках.
        best_params_: Параметры, максимизирующие средний CV nDCG@10.
        best_score_: Средний nDCG@10 лучшей конфигурации на поисковых фолдах.
        cv_results_: Параметры, оценки отдельных фолдов и их среднее для
            каждой проверенной конфигурации.
    """

    best_estimator_: CatBoostClassifier
    best_params_: dict[str, int | float]
    best_score_: float
    cv_results_: list[dict[str, Any]]


def make_temporal_splits(
    impression_ids: pd.Series,
    timestamps: pd.Series,
    n_splits: int = 3,
    initial_train_fraction: float = 0.4,
) -> list[TemporalFold]:
    """Построить expanding-window фолды по целым выдачам.

    При трёх фолдах и initial_train_fraction=0.4 схема примерно такая:
    train 0–40% -> valid 40–60%, train 0–60% -> valid 60–80%,
    train 0–80% -> valid 80–100%.

    Args:
        impression_ids: Идентификатор выдачи для каждой строки.
        timestamps: Время каждой строки в том же порядке.
        n_splits: Количество последовательных проверочных интервалов.
        initial_train_fraction: Доля первых выдач для обучения первого фолда.

    Returns:
        Пары позиционных индексов train/validation относительно входных
        строк. Выдачи и одинаковые timestamp не разделяются между частями;
        максимальное время train строго меньше минимального времени validation.

    Raises:
        ValueError: Входы разной длины, неверны параметры, одна выдача имеет
            несколько timestamp или данных недостаточно для непустых фолдов.
    """
    if len(impression_ids) != len(timestamps):
        raise ValueError('impression_ids и timestamps должны иметь одну длину')
    if n_splits < 1:
        raise ValueError('n_splits должен быть положительным')
    if not 0 < initial_train_fraction < 1:
        raise ValueError('initial_train_fraction должен находиться между 0 и 1')

    rows = pd.DataFrame({
        'impression_id': impression_ids.to_numpy(),
        'timestamp': pd.to_datetime(timestamps).to_numpy(),
    })
    timestamps_per_impression = rows.groupby(
        'impression_id', observed=True
    )['timestamp'].nunique()
    if (timestamps_per_impression != 1).any():
        raise ValueError('Одна выдача не может иметь несколько timestamp')

    impressions = (
        rows.drop_duplicates('impression_id')
        .sort_values(['timestamp', 'impression_id'])
        .reset_index(drop=True)
    )
    if len(impressions) < n_splits + 1:
        raise ValueError('Слишком мало выдач для временной кросс-валидации')

    validation_fraction = (1 - initial_train_fraction) / n_splits
    boundary_fractions = [
        initial_train_fraction + fold * validation_fraction
        for fold in range(n_splits)
    ]
    boundary_positions = [
        min(int(len(impressions) * fraction), len(impressions) - 1)
        for fraction in boundary_fractions
    ]
    boundary_times = [
        impressions.iloc[position]['timestamp']
        for position in boundary_positions
    ]

    row_times = rows['timestamp']
    splits: list[TemporalFold] = []
    for fold, valid_start in enumerate(boundary_times):
        train_mask = row_times < valid_start
        if fold + 1 < len(boundary_times):
            valid_mask = (
                (row_times >= valid_start)
                & (row_times < boundary_times[fold + 1])
            )
        else:
            valid_mask = row_times >= valid_start

        train_indices = np.flatnonzero(train_mask.to_numpy())
        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        if len(train_indices) == 0 or len(valid_indices) == 0:
            raise ValueError('Получился пустой временной train или validation')
        splits.append((train_indices, valid_indices))

    return splits


def cross_validation(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    impression_ids: pd.Series,
    timestamps: pd.Series,
    categorical_columns: list[str],
) -> TemporalSearchResult:
    """Подобрать CatBoost по среднему nDCG@10 на временных фолдах.

    Args:
        X_train: Готовые признаки; порядок строк совпадает с остальными входами.
        y_train: Бинарные метки клика для этих строк.
        impression_ids: Идентификаторы выдач для вычисления nDCG@10.
        timestamps: Временные метки для построения последовательных фолдов.
        categorical_columns: Категориальные колонки матрицы X_train.

    Returns:
        Лучшие параметры, CV-оценки и модель, обученная на всём X_train.
        Отдельный dev не используется. Заранее рассчитанные признаки должны
        соблюдать временную причинность, включая проверочные интервалы.

    Raises:
        ValueError: Невозможно построить временные фолды или обучить модель
            на переданных данных.
    """
    X_cv = X_train.copy()
    for column in categorical_columns:
        X_cv[column] = X_cv[column].fillna('<missing>').astype(str)

    parameter_sets = [
        {
            'depth': depth,
            'learning_rate': learning_rate,
            'iterations': iterations,
        }
        for depth, learning_rate, iterations in product(
            [6, 8],
            [0.05, 0.1],
            [300],
        )
    ]

    temporal_splits = make_temporal_splits(
        impression_ids=impression_ids,
        timestamps=timestamps,
    )

    def build_model(params: dict[str, int | float]) -> CatBoostClassifier:
        """Создать модель с одинаковыми общими настройками для всех фолдов.

        Args:
            params: Проверяемые depth, learning_rate и iterations.

        Returns:
            Необученный классификатор CatBoost.
        """
        return CatBoostClassifier(
            **params,
            loss_function='Logloss',
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )

    cv_results: list[dict[str, Any]] = []
    for params in parameter_sets:
        fold_scores: list[float] = []
        for fold_number, (train_indices, valid_indices) in enumerate(
            temporal_splits,
            start=1,
        ):
            model = build_model(params)
            model.fit(
                X_cv.iloc[train_indices],
                y_train.iloc[train_indices],
                cat_features=categorical_columns,
            )
            probabilities = model.predict_proba(
                X_cv.iloc[valid_indices]
            )[:, 1]
            score = mean_ndcg_at_k(
                y_true=y_train.iloc[valid_indices].to_numpy(),
                y_score=probabilities,
                group_ids=impression_ids.iloc[valid_indices].to_numpy(),
                k=10,
            )
            fold_scores.append(score)
            print(
                f'params={params}, fold={fold_number}, '
                f'nDCG@10={score:.6f}'
            )

        cv_results.append({
            'params': params,
            'fold_scores': fold_scores,
            'mean_ndcg@10': float(np.mean(fold_scores)),
        })

    best_result = max(cv_results, key=lambda result: result['mean_ndcg@10'])
    best_model = build_model(best_result['params'])
    best_model.fit(
        X_cv,
        y_train,
        cat_features=categorical_columns,
    )

    return TemporalSearchResult(
        best_estimator_=best_model,
        best_params_=best_result['params'],
        best_score_=best_result['mean_ndcg@10'],
        cv_results_=cv_results,
    )
