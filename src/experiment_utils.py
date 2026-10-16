"""Общая подготовка признаков и оценка моделей для экспериментального ноутбука."""

from src.config import CatBoostClassifier, Path, Sequence, json, pd

from src.bandit_features import FeatureArm, build_feature
from src.manual_feature_selection import make_catboost_matrix
from src.ranking_metrics import mean_ndcg_at_k


def load_candidate_pool(run_dir: str | Path) -> list[FeatureArm]:
    """Восстановить ту же библиотеку рук, которая участвовала в отборе.

    Args:
        run_dir: Каталог с summary.json и discarded_arms.csv выбранного запуска.

    Returns:
        Кандидаты в исходном порядке без исключённых констант, дубликатов
        и фиксированных стартовых колонок. Все временные окна сохраняются.

    Raises:
        FileNotFoundError: Не сохранена конфигурация или журнал фильтрации.
    """
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    excluded = set(pd.read_csv(run_dir / 'discarded_arms.csv')['arm'])
    excluded.update(summary['base_features'])
    arms = [FeatureArm(**recipe) for recipe in summary['candidate_arms']]
    return [arm for arm in arms if arm.name not in excluded]


def prepare_experiment_data(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    arms: Sequence[FeatureArm],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Один раз посчитать выбранные признаки для train и dev в rolling-протоколе.

    Args:
        train: Обучающие события с базовыми колонками MIND.
        dev: Более поздние события с теми же ключами и бинарным label.
        arms: Рецепты рассчитываемых признаков.

    Returns:
        Копии train/dev с новыми признаками и последовательными индексами.
        Порядок строк сохраняется; исходные таблицы не меняются. Текущий
        timestamp исключается. Ранние dev-label доступны поздним dev-строкам.

    Raises:
        ValueError: Время train и dev пересекается либо некорректен рецепт.
        KeyError: Отсутствуют колонки, необходимые для расчёта признака.
    """
    if pd.to_datetime(train['timestamp']).max() >= pd.to_datetime(dev['timestamp']).min():
        raise ValueError('Все train-события должны предшествовать dev')

    n_train = len(train)
    history = pd.concat([train, dev], ignore_index=True)
    columns: dict[str, pd.Series] = {}
    for index, arm in enumerate(arms, start=1):
        columns[arm.name] = build_feature(history, arm)
        if index % 10 == 0 or index == len(arms):
            print(f'Признаки: {index}/{len(arms)}', flush=True)

    history = pd.concat([
        history.drop(columns=list(columns), errors='ignore'),
        pd.DataFrame(columns, index=history.index),
    ], axis=1)
    return history.iloc[:n_train].copy(), history.iloc[n_train:].copy()


def evaluate_ndcg(model: CatBoostClassifier, data: pd.DataFrame) -> float:
    """Оценить готовую модель по nDCG@10 без повторного обучения.

    Args:
        model: Обученный CatBoost с сохранённым порядком признаков.
        data: Таблица с признаками, label и impression_id.

    Returns:
        Средний nDCG@10 по выдачам, содержащим хотя бы один клик.
    """
    matrix = make_catboost_matrix(data, model.feature_names_)
    return mean_ndcg_at_k(
        y_true=data['label'].to_numpy(),
        y_score=model.predict_proba(matrix)[:, 1],
        group_ids=data['impression_id'].to_numpy(),
        k=10,
    )
