"""Ручные признаки baseline и подготовка матриц для CatBoost."""

from src.config import Sequence, np, pd


CATEGORICAL_COLUMNS: tuple[str, ...] = ('category', 'subcategory')


def temporal_train_valid_split(
    data: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Разделить отсортированные выдачи на обучение и проверку.

    Args:
        data: Таблица с timestamp, impression_id и position.
        train_fraction: Доля первых выдач, отводимая под обучение.

    Returns:
        Копии train и validation, отсортированные по времени, выдаче и позиции.
        Одна выдача не разделяется. Разные выдачи с одинаковым timestamp
        могут попасть в разные части; для строгих временных границ используется
        make_temporal_splits из src.cv.

    Raises:
        ValueError: Доля не находится между 0 и 1 или выдач меньше двух.
    """
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction должен находиться между 0 и 1')

    df = data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(
        ['timestamp', 'impression_id', 'position']
    ).reset_index(drop=True)

    impressions = (
        df[['impression_id', 'timestamp']]
        .drop_duplicates('impression_id')
        .sort_values(['timestamp', 'impression_id'])
    )
    if len(impressions) < 2:
        raise ValueError('Для разбиения нужны как минимум две выдачи')

    split_index = int(len(impressions) * train_fraction)
    split_index = max(1, min(split_index, len(impressions) - 1))
    train_ids = set(impressions.iloc[:split_index]['impression_id'])

    train = df[df['impression_id'].isin(train_ids)].copy()
    valid = df[~df['impression_id'].isin(train_ids)].copy()
    return train, valid


def make_catboost_matrix(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] = CATEGORICAL_COLUMNS,
) -> pd.DataFrame:
    """Выбрать колонки и привести категории к строковому формату CatBoost.

    Args:
        data: Таблица с рассчитанными признаками.
        feature_columns: Колонки в том порядке, в котором их ожидает модель.
        categorical_columns: Категории; пропуски в них заменяются на <missing>.
            Категориальные колонки, не входящие в feature_columns, пропускаются.

    Returns:
        Копия матрицы признаков с исходным порядком строк и индексом.

    Raises:
        KeyError: В data отсутствует один из запрошенных признаков.
    """
    matrix = data[feature_columns].copy()
    for column in categorical_columns:
        if column in matrix.columns:
            matrix[column] = matrix[column].fillna('<missing>').astype(str)
    return matrix


def prepare_features(
    data: pd.DataFrame,
    train_fraction: float = 0.8,
    include_one_day: bool = False,
    include_user_item: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Добавить ручные исторические агрегаты и выделить train/validation.

    Args:
        data: Исходные взаимодействия с user_id, item_id, timestamp,
            impression_id, position, label, категориями и базовыми признаками.
            Готовые history-признаки из MIND используются, если присутствуют.
        train_fraction: Доля первых выдач для обучения.
        include_one_day: Добавлять показы, клики и CTR за последние 24 часа.
        include_user_item: Добавлять группировку user_id/item_id в дополнение
            к user, item, user/category и user/subcategory.

    Returns:
        Train, validation, список признаков и список категориальных колонок.
        Агрегаты исключают все события с текущим timestamp. Поздние строки
        validation используют более ранние validation-label в rolling-протоколе.

    Raises:
        ValueError: Невозможно преобразовать время или выполнить разбиение.
        KeyError: Отсутствует колонка, необходимая для вычисления признаков.
    """
    df = data.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'impression_id', 'position']).reset_index(drop=True)

    def add_previous_stats(
        df: pd.DataFrame,
        keys: list[str],
        prefix: str,
    ) -> pd.DataFrame:
        """Добавить показы, клики и CTR за всю прошлую историю.

        Args:
            df: Таблица взаимодействий с label и timestamp.
            keys: Колонки, по которым объединяются события.
            prefix: Префикс имён новых колонок.

        Returns:
            Таблица с тремя агрегатами строго до текущего timestamp.
        """
        # Сначала собираем статистику в момент времени целиком. Поэтому label
        # других кандидатов из текущей выдачи не попадёт в её признаки.
        stats = (
            df.groupby(keys + ['timestamp'], as_index=False, observed=True)
              .agg(current_impressions=('label', 'size'),
                   current_clicks=('label', 'sum'))
              .sort_values(keys + ['timestamp'])
        )

        grouped = stats.groupby(keys, observed=True, sort=False)
        impressions = f'{prefix}_previous_impressions'
        clicks = f'{prefix}_previous_clicks'
        ctr = f'{prefix}_previous_ctr'

        stats[impressions] = (
            grouped['current_impressions'].cumsum() - stats['current_impressions']
        )
        stats[clicks] = (
            grouped['current_clicks'].cumsum() - stats['current_clicks']
        )
        stats[ctr] = np.where(
            stats[impressions] > 0,
            stats[clicks] / stats[impressions],
            0.0,
        )

        columns = keys + ['timestamp', impressions, clicks, ctr]
        return df.merge(stats[columns], on=keys + ['timestamp'], how='left')

    def add_window_stats(
        df: pd.DataFrame,
        keys: list[str],
        prefix: str,
        window: str = '1d',
    ) -> pd.DataFrame:
        """Добавить показы, клики и CTR за временное окно перед событием.

        Args:
            df: Таблица взаимодействий с label и timestamp.
            keys: Колонки, по которым объединяются события.
            prefix: Префикс имён новых колонок.
            window: Длина окна, например 1d; интервал равен [t - window, t).

        Returns:
            Таблица с тремя оконными агрегатами и нулями при пустой истории.
        """
        stats = (
            df.groupby(keys + ['timestamp'], as_index=False, observed=True)
              .agg(current_impressions=('label', 'size'),
                   current_clicks=('label', 'sum'))
              .sort_values(keys + ['timestamp'])
              .reset_index(drop=True)
        )

        impressions = f'{prefix}_previous_impressions_{window}'
        clicks = f'{prefix}_previous_clicks_{window}'
        ctr = f'{prefix}_previous_ctr_{window}'

        # closed='left' исключает текущий timestamp и включает левую границу:
        # для окна 1d используются события из [t - 24 часа, t).
        window_stats = (
            stats.set_index('timestamp')
            .groupby(keys, observed=True)[
                ['current_impressions', 'current_clicks']
            ]
            .rolling(window.upper(), closed='left')
            .sum()
            .fillna(0)
            .reset_index()
            .rename(
                columns={
                    'current_impressions': impressions,
                    'current_clicks': clicks,
                }
            )
        )
        window_stats[ctr] = np.where(
            window_stats[impressions] > 0,
            window_stats[clicks] / window_stats[impressions],
            0.0,
        )

        columns = keys + ['timestamp', impressions, clicks, ctr]
        result = df.merge(
            window_stats[columns],
            on=keys + ['timestamp'],
            how='left',
        )
        result[[impressions, clicks, ctr]] = result[
            [impressions, clicks, ctr]
        ].fillna(0)
        return result

    # Набор признаков выбран вручную — это наш baseline.
    groups: list[tuple[list[str], str]] = [
        (['user_id'], 'user'),
        (['item_id'], 'item'),
        (['user_id', 'category'], 'user_category'),
        (['user_id', 'subcategory'], 'user_subcategory'),
    ]
    if include_user_item:
        groups.append((['user_id', 'item_id'], 'user_item'))

    for keys, prefix in groups:
        df = add_previous_stats(df, keys, prefix)
        if include_one_day:
            df = add_window_stats(df, keys, prefix, window='1d')

    df = df.sort_values(['timestamp', 'impression_id', 'position']).reset_index(drop=True)

    train, valid = temporal_train_valid_split(df, train_fraction)

    feature_columns = [
        'category',
        'subcategory',
        'history_length',
        'title_length',
        'abstract_length',
    ]

    # History-признаки входят в baseline v1. Проверка оставляет функцию
    # применимой и к таблицам без этих дополнительных колонок.
    history_features = [
        'history_contains_item',
        'history_unique_categories',
        'history_unique_subcategories',
        'history_category_clicks',
        'history_category_share',
        'history_subcategory_clicks',
        'history_subcategory_share',
    ]
    feature_columns += [
        feature for feature in history_features if feature in df.columns
    ]
    for _, prefix in groups:
        feature_columns += [
            f'{prefix}_previous_impressions',
            f'{prefix}_previous_clicks',
            f'{prefix}_previous_ctr',
        ]
        if include_one_day:
            feature_columns += [
                f'{prefix}_previous_impressions_1d',
                f'{prefix}_previous_clicks_1d',
                f'{prefix}_previous_ctr_1d',
            ]

    return train, valid, feature_columns, list(CATEGORICAL_COLUMNS)
