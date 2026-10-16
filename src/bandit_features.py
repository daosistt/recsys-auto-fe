"""Поиск временных признаков с последовательным отсевом рук.

Рука — рецепт признака. Награда — прирост nDCG@10 на временном фолде.
Дешёвый скрининг выбирает финалистов для проверки на нескольких фолдах и seed.
Поздний контрольный участок train выбирает лучший шаг поиска.
Это bandit-inspired эвристика, без теоретических гарантий MAB.

Example:
    data = pd.read_csv('data/train_v1.csv', parse_dates=['timestamp'])
    result = search_features(data, budget=300, max_features=15)
    print(result.selected_arms)
    print(result.history)

Для оценки dev выбранные признаки можно рассчитать через build_feature
на объединённых train и dev, затем взять строки dev. Это rolling-протокол:
прошлые dev-label доступны будущим dev-строкам. В поиск dev не передаётся.
"""

from src.config import (
    Any,
    CatBoostClassifier,
    FloatArray,
    Literal,
    ModelParams,
    Sequence,
    TemporalFold,
    dataclass,
    np,
    pd,
    perf_counter,
    product,
)

from src.cv import make_temporal_splits
from src.manual_feature_selection import make_catboost_matrix
from src.ranking_metrics import mean_ndcg_at_k


# ---------------------------------------------------------------------------
# 1. Руки, параметры и результат поиска
# ---------------------------------------------------------------------------

Statistic = Literal['impressions', 'clicks', 'ctr']

GROUPS: dict[str, tuple[str, ...]] = {
    'user': ('user_id',),
    'item': ('item_id',),
    'user_item': ('user_id', 'item_id'),
    'user_category': ('user_id', 'category'),
    'user_subcategory': ('user_id', 'subcategory'),
}
STATISTICS: tuple[Statistic, ...] = ('impressions', 'clicks', 'ctr')
DEFAULT_WINDOWS: tuple[str, ...] = ('6h', '1d', '3d', '7d', 'all')
DEFAULT_BASE_FEATURES: tuple[str, ...] = ('category', 'subcategory')

# Эти колонки получены при подготовке MIND из полей history/title/abstract.
# Они доступны генератору как кандидаты, но НЕ включаются в модель заранее.
SOURCE_FEATURES: tuple[str, ...] = (
    'history_length',
    'history_contains_item',
    'history_unique_categories',
    'history_unique_subcategories',
    'history_category_clicks',
    'history_category_share',
    'history_subcategory_clicks',
    'history_subcategory_share',
    'title_length',
    'abstract_length',
)
DEFAULT_MODEL_PARAMS: ModelParams = {
    'iterations': 300,
    'depth': 8,
    'learning_rate': 0.05,
    'loss_function': 'Logloss',
    'random_seed': 42,
    'verbose': False,
    'allow_writing_files': False,
    'thread_count': -1,
}


@dataclass(frozen=True)
class FeatureArm:
    """Рука бандита: рецепт одного кандидатного признака.

    Attributes:
        group: Группировка из GROUPS, например user_category. Для готовой
            колонки не используется.
        statistic: Число показов, число кликов или их отношение (CTR).
        window: Временное окно, например '1d', либо 'all' для всей истории.
        column: Готовая колонка из SOURCE_FEATURES. Если задана, рука проверяет
            эту колонку вместо расчёта агрегата. Например,
            FeatureArm(column='history_length').
    """

    group: str = 'column'
    statistic: Statistic = 'impressions'
    window: str = 'all'
    column: str | None = None

    @property
    def name(self) -> str:
        """Вернуть название колонки, совместимое с ручными признаками.

        Returns:
            Имя вида user_category_previous_clicks_1d.
        """
        if self.column is not None:
            return self.column
        suffix = '' if self.window == 'all' else f'_{self.window}'
        return f'{self.group}_previous_{self.statistic}{suffix}'


@dataclass
class FeatureSearchResult:
    """Результат отбора и финального обучения.

    Attributes:
        model: CatBoost, обученный на всём переданном train.
        selected_arms: Рецепты добавленных признаков в порядке выбора.
        feature_columns: Исходные и выбранные признаки модели.
        cv_score: Средний nDCG@10 на поисковых фолдах выбранного набора.
        history: Журнал проверок: этап, рука, seed, фолд, число деревьев,
            качество, соответствующий baseline и парная награда.
        evaluations: Число поисковых и контрольных обучений, без финального fit.
        control_score: Качество выбранного шага на позднем контрольном участке.
            Участок используется для выбора шага и не является тестом.
        checkpoints: Оценки начального набора и каждого добавленного признака.
        discarded_arms: Константные и дублирующие кандидаты, удалённые до поиска.
        trees_built: Число деревьев во всех проверках, без финального fit.
        elapsed_seconds: Время генерации, отбора и финального обучения в секундах.
    """

    model: CatBoostClassifier
    selected_arms: list[FeatureArm]
    feature_columns: list[str]
    cv_score: float
    history: pd.DataFrame
    evaluations: int
    control_score: float
    checkpoints: pd.DataFrame
    discarded_arms: pd.DataFrame
    trees_built: int
    elapsed_seconds: float


# ---------------------------------------------------------------------------
# 2. Генерация рецептов и расчёт признаков по прошлым событиям
# ---------------------------------------------------------------------------

def generate_arms(windows: Sequence[str] = DEFAULT_WINDOWS) -> list[FeatureArm]:
    """Создать кандидатов из временных агрегатов и исходных полей MIND.

    Args:
        windows: Допустимые временные окна. Повторы удаляются.

    Returns:
        Все комбинации группировок, статистик и окон, затем десять колонок
        из SOURCE_FEATURES. С окнами по умолчанию получается 85 рук.
        All-time признаки v1 участвуют на общих основаниях.
    """
    combinations = product(GROUPS, STATISTICS, dict.fromkeys(windows))
    temporal_arms = [
        FeatureArm(group, statistic, window)
        for group, statistic, window in combinations
    ]
    column_arms = [FeatureArm(column=column) for column in SOURCE_FEATURES]
    return temporal_arms + column_arms


def build_feature(data: pd.DataFrame, arm: FeatureArm) -> pd.Series:
    """Рассчитать временной признак или взять готовую колонку MIND.

    Args:
        data: События с timestamp, бинарным label и ключами группировки.
            Для руки column нужна соответствующая колонка из SOURCE_FEATURES,
            рассчитанная при подготовке данных. Каждая строка — показ;
            label=1 означает клик. Временные агрегаты используют [t - window, t).
        arm: Рецепт рассчитываемого признака.

    Returns:
        Колонка с именем arm.name, в исходном порядке и с индексом data.

    Raises:
        ValueError: Неизвестен рецепт, окно неположительно или данные некорректны.
    """
    # Готовая колонка тоже является рукой: она попадёт в модель только
    # после отбора. Допустимы лишь явно перечисленные исходные признаки.
    if arm.column is not None:
        if arm.column not in SOURCE_FEATURES:
            raise ValueError(f'Недопустимая колонка кандидата: {arm.column}')
        return pd.to_numeric(data[arm.column], errors='raise').rename(arm.name)

    # Агрегаты всегда пересчитываем из событий, даже если в CSV есть старая
    # одноимённая колонка от ручного baseline.
    if arm.group not in GROUPS or arm.statistic not in STATISTICS:
        raise ValueError(f'Неизвестный рецепт: {arm}')

    # pandas 3 рекомендует D для дней; имена признаков сохраняют привычное 1d.
    window = None if arm.window == 'all' else pd.Timedelta(arm.window.replace('d', 'D'))
    if window is not None and (pd.isna(window) or window <= pd.Timedelta(0)):
        raise ValueError('Окно должно быть положительным или all')

    keys = list(GROUPS[arm.group])
    frame = data[keys + ['timestamp', 'label']].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    if frame['timestamp'].isna().any() or not frame['label'].isin([0, 1]).all():
        raise ValueError('Нужны заполненные timestamp и бинарный label')

    frame['_row'] = np.arange(len(frame))
    frame = frame.sort_values('timestamp', kind='stable')
    values = np.zeros(len(frame), dtype=float)

    for _, group in frame.groupby(keys, observed=True, dropna=False, sort=False):
        times = group['timestamp'].to_numpy(dtype='datetime64[ns]')

        # side='left' исключает ВСЕ события в момент t, включая другие
        # строки текущей выдачи. Левая граница окна при этом включена.
        right = np.searchsorted(times, times, side='left')
        if window is None:
            left = np.zeros(len(group), dtype=int)
        else:
            window_start = times - window.to_timedelta64()
            left = np.searchsorted(times, window_start, side='left')

        # Разность накопительных сумм даёт число кликов внутри окна.
        clicks_prefix = np.r_[0.0, group['label'].to_numpy(dtype=float).cumsum()]
        impressions = right - left
        clicks = clicks_prefix[right] - clicks_prefix[left]

        if arm.statistic == 'impressions':
            feature = impressions
        elif arm.statistic == 'clicks':
            feature = clicks
        else:
            feature = np.divide(
                clicks,
                impressions,
                out=np.zeros(len(group)),
                where=impressions > 0,
            )

        # Восстанавливаем исходный порядок, даже при повторяющихся индексах.
        values[group['_row'].to_numpy()] = feature

    return pd.Series(values, index=data.index, name=arm.name)


# ---------------------------------------------------------------------------
# 3. Обучение, оценка одной попытки и награда
# ---------------------------------------------------------------------------

def train_model(
    data: pd.DataFrame,
    features: list[str],
    model_params: ModelParams | None = None,
) -> CatBoostClassifier:
    """Обучить CatBoost на переданных строках и признаках.

    Args:
        data: Обучающие строки с label и уже рассчитанными признаками.
        features: Названия колонок для обучения.
        model_params: Переопределения стандартных параметров CatBoost.

    Returns:
        Обученная модель.
    """
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    matrix = make_catboost_matrix(data, features)
    categories = [c for c in ('category', 'subcategory') if c in features]

    model = CatBoostClassifier(**params)
    model.fit(matrix, data['label'], cat_features=categories)
    return model


def evaluate_features(
    data: pd.DataFrame,
    features: list[str],
    fold: TemporalFold,
    model_params: ModelParams | None = None,
) -> float:
    """Оценить набор признаков на одном временном фолде.

    Args:
        data: Таблица событий с уже рассчитанными признаками.
        features: Колонки для обучения и предсказания.
        fold: Позиционные индексы обучающих и проверочных строк.
        model_params: Параметры CatBoost, одинаковые для всех кандидатов.

    Returns:
        nDCG@10, усреднённый по выдачам проверочной части.
    """
    train_indices, valid_indices = fold
    train = data.iloc[train_indices]
    valid = data.iloc[valid_indices]

    model = train_model(train, features, model_params)
    matrix = make_catboost_matrix(valid, features)
    probabilities = model.predict_proba(matrix)[:, 1]

    return mean_ndcg_at_k(
        y_true=valid['label'].to_numpy(),
        y_score=probabilities,
        group_ids=valid['impression_id'].to_numpy(),
        k=10,
    )


# ---------------------------------------------------------------------------
# 4. Подготовка поиска и распределение бюджета
# ---------------------------------------------------------------------------

def _prepare_search(
    data: pd.DataFrame,
    base_features: list[str],
    control_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[TemporalFold], TemporalFold]:
    """Разделить train на ранний поиск и поздний контроль по timestamp.

    Args:
        data: Весь train с категориями, label, impression_id и timestamp.
        base_features: Стартовые признаки без label и служебных колонок.
        control_fraction: Доля последних выдач для контроля остановки.

    Returns:
        Полный train, ранняя поисковая таблица, её CV-фолды и контрольный фолд.
        Все строки с одинаковым timestamp остаются по одну сторону границы.

    Raises:
        ValueError: Некорректны входные поля, доля или состав классов.
    """
    forbidden = {'label', 'timestamp', 'impression_id', 'user_id', 'item_id', 'position'}
    if not base_features or forbidden.intersection(base_features):
        raise ValueError('Нужны стартовые признаки без label и служебных колонок')
    if len(set(base_features)) != len(base_features):
        raise ValueError('В base_features есть дубликаты')
    if not 0 < control_fraction < 1:
        raise ValueError('control_fraction должна находиться между 0 и 1')

    frame = data.copy().reset_index(drop=True)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    if frame['timestamp'].isna().any() or not frame['label'].isin([0, 1]).all():
        raise ValueError('Нужны заполненные timestamp и бинарный label')

    impressions = (
        frame[['impression_id', 'timestamp']]
        .drop_duplicates()
        .sort_values(['timestamp', 'impression_id'])
    )
    if impressions['impression_id'].duplicated().any():
        raise ValueError('Одна выдача не может иметь несколько timestamp')
    if len(impressions) < 5:
        raise ValueError('Недостаточно выдач для поиска и контроля')
    boundary = int(len(impressions) * (1 - control_fraction))
    cutoff = impressions.iloc[boundary]['timestamp']
    search_indices = np.flatnonzero((frame['timestamp'] < cutoff).to_numpy())
    control_indices = np.flatnonzero((frame['timestamp'] >= cutoff).to_numpy())
    search_data = frame.iloc[search_indices].copy().reset_index(drop=True)
    folds = make_temporal_splits(search_data['impression_id'], search_data['timestamp'])

    for train_indices, valid_indices in folds:
        if search_data.iloc[train_indices]['label'].nunique() < 2:
            raise ValueError('В каждом поисковом train нужны оба класса')
        if not search_data.iloc[valid_indices]['label'].any():
            raise ValueError('В каждом поисковом validation нужны клики')
    if not len(control_indices) or not frame.iloc[control_indices]['label'].any():
        raise ValueError('В контрольной части нужны выдачи с кликами')

    return frame, search_data, folds, (search_indices, control_indices)


def prepare_candidates(
    data: pd.DataFrame,
    arms: list[FeatureArm],
    base_features: list[str],
) -> tuple[pd.DataFrame, list[FeatureArm], pd.DataFrame]:
    """Рассчитать кандидатов и убрать константы и точные дубликаты.

    Args:
        data: Только ранние поисковые данные. Контроль сюда не передаётся.
        arms: Рецепты кандидатов.
        base_features: Уже включённые колонки.

    Returns:
        Поисковая таблица с признаками, оставшиеся руки и журнал исключений.
        Дубликаты сравниваются только на поисковых строках, включая пропуски.
    """
    columns: dict[str, pd.Series] = {}
    kept: list[FeatureArm] = []
    discarded: list[dict[str, str]] = []
    seen_names = set(base_features)

    for index, arm in enumerate(arms, start=1):
        if index == 1 or index % 10 == 0:
            print(f'Подготовка кандидатов: {index}/{len(arms)}', flush=True)
        if arm.name in seen_names:
            continue
        seen_names.add(arm.name)
        values = build_feature(data, arm).astype(float)
        if values.nunique(dropna=False) <= 1:
            discarded.append({'arm': arm.name, 'reason': 'constant'})
            continue
        duplicate = next(
            (name for name, other in columns.items() if values.equals(other)),
            None,
        )
        if duplicate is not None:
            discarded.append({'arm': arm.name, 'reason': f'duplicate:{duplicate}'})
            continue
        columns[arm.name] = values
        kept.append(arm)

    # Одно присоединение вместо десятков вставок колонок в DataFrame.
    frame = data.drop(columns=list(columns), errors='ignore')
    frame = pd.concat([frame, pd.DataFrame(columns, index=data.index)], axis=1)
    return frame, kept, pd.DataFrame(discarded, columns=['arm', 'reason'])


def round_cost(
    n_arms: int,
    n_folds: int,
    n_seeds: int = 2,
    n_finalists: int = 3,
) -> int:
    """Посчитать верхнюю границу числа обучений одного нового раунда.

    Args:
        n_arms: Число оставшихся рук после удаления дубликатов и констант.
        n_folds: Число поисковых временных фолдов.
        n_seeds: Число seed для полной проверки.
        n_finalists: Максимальное число финалистов скрининга.

    Returns:
        Дешёвый baseline + скрининг рук + полная проверка финалистов +
        одна контрольная оценка победителя. Финальный fit сюда не входит.
    """
    return 1 + n_arms + min(n_arms, n_finalists) * n_folds * n_seeds + 1


def search_features(
    data: pd.DataFrame,
    base_features: list[str] | None = None,
    arms: list[FeatureArm] | None = None,
    budget: int = 300,
    max_features: int = 15,
    min_gain: float = 0.001,
    model_params: ModelParams | None = None,
    control_fraction: float = 0.2,
    screen_iterations: int = 50,
    n_finalists: int = 3,
    seeds: Sequence[int] = (42, 0),
    patience: int = 2,
) -> FeatureSearchResult:
    """Искать признаки дёшево и выбирать лучший шаг по позднему контролю.

    Руки ранжируются только по ранним данным. На последних control_fraction
    выдачах оценивается один победитель каждого шага. При отсутствии улучшений
    patience шагов возвращается лучший сохранённый набор, а не последний.

    Args:
        data: Весь train. Dev сюда не передаётся. Исторические исходные
            колонки MIND должны быть подготовлены без использования будущего.
        base_features: Стартовые колонки; по умолчанию category/subcategory.
        arms: Кандидаты; по умолчанию generate_arms() с доступными колонками.
        budget: Лимит всех поисковых и контрольных fit, кроме финального.
        max_features: Максимальная длина пути последовательного добавления.
        min_gain: Минимальный средний прирост поискового nDCG@10 и минимальное
            улучшение для обновления лучшего контрольного шага.
        model_params: Общие параметры полной модели. По умолчанию 300 деревьев.
        control_fraction: Доля последних train-выдач, скрытая от поиска рук.
        screen_iterations: Число деревьев дешёвого скрининга на последнем
            поисковом фолде. Baseline скрининга обучается с тем же числом деревьев.
        n_finalists: Сколько рук проходят полную проверку после скрининга.
        seeds: Seed полной проверки; награда усредняется по фолдам и seed.
            Контроль использует первый seed, чтобы ограничить его стоимость.
        patience: Число шагов без улучшения контроля до остановки.

    Returns:
        Модель лучшего контрольного набора, переобученная на всём train,
        выбранные руки, журнал проверок и контрольные чекпоинты. Ни поисковая,
        ни контрольная метрика не является независимым тестом.

    Raises:
        ValueError: Некорректны параметры поиска, данные или бюджет.
    """
    started = perf_counter()
    params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
    seed_values = list(dict.fromkeys(seeds))
    if not seed_values or n_finalists < 1 or patience < 1:
        raise ValueError('Нужны seed, n_finalists >= 1 и patience >= 1')
    if not 1 <= screen_iterations <= params['iterations']:
        raise ValueError('Нужны 1 <= screen_iterations <= iterations полной модели')
    if max_features < 0 or min_gain < 0:
        raise ValueError('Нужны max_features >= 0 и min_gain >= 0')
    initial_cost = 3 * len(seed_values) + 1
    if budget < initial_cost:
        raise ValueError(f'На исходные CV и контроль нужно хотя бы {initial_cost} обучений')

    # 1. Контрольные метки и распределения не участвуют в создании библиотеки.
    features = list(DEFAULT_BASE_FEATURES if base_features is None else base_features)
    frame, search_data, folds, control_fold = _prepare_search(
        data, features, control_fraction,
    )
    candidates = generate_arms() if arms is None else arms
    if arms is None:
        candidates = [
            arm for arm in candidates
            if arm.column is None or arm.column in search_data.columns
        ]
    search_data, remaining, discarded = prepare_candidates(
        search_data, candidates, features,
    )
    selected: list[FeatureArm] = []
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    evaluations = 0
    trees_built = 0

    def score(
        table: pd.DataFrame,
        columns: list[str],
        fold: TemporalFold,
        settings: ModelParams,
    ) -> float:
        """Выполнить одну оценку и учесть её вычислительную стоимость.

        Args:
            table: Поисковые данные либо полный train для контрольной оценки.
            columns: Проверяемый набор признаков.
            fold: Позиционные индексы train/validation внутри table.
            settings: Параметры модели, включая число деревьев и seed.

        Returns:
            nDCG@10 проверочной части. Счётчики обучений и деревьев
            обновляются в состоянии текущего запуска поиска.
        """
        nonlocal evaluations, trees_built
        value = evaluate_features(table, columns, fold, settings)
        evaluations += 1
        trees_built += settings['iterations']
        return value

    def full_scores(columns: list[str]) -> FloatArray:
        """Оценить один набор на одинаковых фолдах и всех заданных seed.

        Args:
            columns: Признаки, уже рассчитанные на поисковой таблице.

        Returns:
            Матрица nDCG@10 размера (число seed, число поисковых фолдов).
            Контрольные строки в эти оценки не входят.
        """
        return np.array([
            [
                score(search_data, columns, fold, {**params, 'random_seed': seed})
                for fold in folds
            ]
            for seed in seed_values
        ])

    # 2. Исходный набор оценивается полноценно. Эти результаты используются
    # для парного сравнения; дешёвые и полные скоры между собой не вычитаются.
    current_scores = full_scores(features)
    control_params = {**params, 'random_seed': seed_values[0]}
    best_control = score(frame, features, control_fold, control_params)
    best_scores = current_scores.copy()
    best_selected: list[FeatureArm] = []
    best_features = features.copy()
    stale_steps = 0
    checkpoints.append({
        'step': 0, 'added_arm': None, 'search_score': float(current_scores.mean()),
        'control_score': best_control, 'is_best': True,
        'features': features.copy(),
    })
    print(f'Старт: {features}; рук после фильтрации: {len(remaining)}')
    print(f'Исключено: {len(discarded)}; бюджет: {budget}; seed: {seed_values}')
    print(f'Поиск CV: {current_scores.mean():.6f}; контроль: {best_control:.6f}')

    for step in range(1, max_features + 1):
        if not remaining:
            break
        required = round_cost(len(remaining), len(folds), len(seed_values), n_finalists)
        if evaluations + required > budget:
            print(f'Остановка: нужно до {required} оценок, осталось {budget - evaluations}.')
            break

        # 3. ДЕШЁВЫЕ ПОПЫТКИ РУК: один поисковый фолд, 50 деревьев.
        # Следующий этап получат лишь три лучшие руки.
        cheap_params = {
            **params, 'iterations': screen_iterations, 'random_seed': seed_values[0],
        }
        cheap_baseline = score(search_data, features, folds[-1], cheap_params)
        screen_rewards: dict[FeatureArm, float] = {}
        for arm_index, arm in enumerate(remaining, start=1):
            cheap_score = score(search_data, features + [arm.name], folds[-1], cheap_params)
            reward = cheap_score - cheap_baseline
            screen_rewards[arm] = reward
            history.append({
                'step': step, 'stage': 'screen', 'arm': arm.name,
                'seed': seed_values[0], 'fold': len(folds),
                'iterations': screen_iterations, 'score': cheap_score,
                'baseline_score': cheap_baseline, 'reward': reward,
            })
            if arm_index % 10 == 0 or arm_index == len(remaining):
                print(f'Step {step}: скрининг {arm_index}/{len(remaining)}', flush=True)
        finalists = sorted(
            remaining, key=lambda arm: screen_rewards[arm], reverse=True,
        )[:n_finalists]
        print(f'Step {step}: скрининг завершён; финалисты: {[a.name for a in finalists]}')

        # 4. ПОЛНАЯ НАГРАДА: средняя разница на одинаковых seed и фолдах.
        # Требуем также положительного среднего эффекта на большинстве фолдов.
        winner: FeatureArm | None = None
        winner_scores = current_scores
        winner_gain = min_gain
        for arm in finalists:
            candidate_scores = full_scores(features + [arm.name])
            rewards = candidate_scores - current_scores
            gain = float(rewards.mean())
            positive_folds = int((rewards.mean(axis=0) > 0).sum())
            for seed_index, seed in enumerate(seed_values):
                for fold_index in range(len(folds)):
                    history.append({
                        'step': step, 'stage': 'full', 'arm': arm.name,
                        'seed': seed, 'fold': fold_index + 1,
                        'iterations': params['iterations'],
                        'score': float(candidate_scores[seed_index, fold_index]),
                        'baseline_score': float(current_scores[seed_index, fold_index]),
                        'reward': float(rewards[seed_index, fold_index]),
                    })
            print(f'  {arm.name}: награда={gain:+.6f}; положительных фолдов={positive_folds}/3')
            if gain > winner_gain and positive_folds > len(folds) / 2:
                winner, winner_scores, winner_gain = arm, candidate_scores, gain

        if winner is None:
            print('Остановка: у финалистов нет достаточного устойчивого прироста.')
            break

        selected.append(winner)
        features.append(winner.name)
        remaining.remove(winner)
        current_scores = winner_scores

        # 5. КОНТРОЛЬ: только теперь считаем победителя на полном временном
        # потоке и проверяем один выбранный набор на поздних 20% train.
        # Контрольные метки не попадают в обучение; ранние control-label
        # доступны более поздним control-строкам только в rolling-признаках.
        frame[winner.name] = build_feature(frame, winner)
        control_score = score(frame, features, control_fold, control_params)
        improved = control_score > best_control + min_gain
        checkpoints.append({
            'step': step, 'added_arm': winner.name,
            'search_score': float(current_scores.mean()),
            'control_score': control_score, 'is_best': improved,
            'features': features.copy(),
        })
        if improved:
            best_control = control_score
            best_scores = current_scores.copy()
            best_selected = selected.copy()
            best_features = features.copy()
            stale_steps = 0
        else:
            stale_steps += 1
        print(
            f'  Контроль={control_score:.6f}; лучший={best_control:.6f}; '
            f'без улучшения={stale_steps}'
        )
        if stale_steps >= patience:
            print('Остановка по контролю: возвращаем лучший сохранённый шаг.')
            break

    # 6. Обучаем на всём train именно лучший контрольный набор.
    # Поздние добавления могут быть отброшены, даже если они повышали CV.
    final_model = train_model(frame, best_features, control_params)
    checkpoint_table = pd.DataFrame(checkpoints)
    checkpoint_table['selected'] = False
    best_step = len(best_selected)
    checkpoint_table.loc[checkpoint_table['step'] == best_step, 'selected'] = True
    print(f'Выбран шаг {best_step}, признаков={len(best_features)}, оценок={evaluations}')
    return FeatureSearchResult(
        model=final_model,
        selected_arms=best_selected,
        feature_columns=best_features,
        cv_score=float(best_scores.mean()),
        history=pd.DataFrame(history, columns=[
            'step', 'stage', 'arm', 'seed', 'fold', 'iterations',
            'score', 'baseline_score', 'reward',
        ]),
        evaluations=evaluations,
        control_score=best_control,
        checkpoints=checkpoint_table,
        discarded_arms=discarded,
        trees_built=trees_built,
        elapsed_seconds=perf_counter() - started,
    )
