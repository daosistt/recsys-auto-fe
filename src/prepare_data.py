"""Подготовка MINDsmall к компактному табличному виду.

Одна строка результата соответствует одной новости из одной выдачи:
пользователь увидел новость и либо кликнул по ней (label=1), либо нет (label=0).
"""

from __future__ import annotations

# Поддерживаем и python -m src.prepare_data, и python src/prepare_data.py.
if __package__:
    from src.config import Path, argparse, pd
else:
    from config import Path, argparse, pd


BEHAVIOR_COLUMNS: list[str] = [
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
]

NEWS_COLUMNS: list[str] = [
    "item_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

RESULT_COLUMNS: list[str] = [
    "impression_id",
    "user_id",
    "timestamp",
    "item_id",
    "position",
    "label",
    "category",
    "subcategory",
    "history_length",
    "history_contains_item",
    "history_unique_categories",
    "history_unique_subcategories",
    "history_category_clicks",
    "history_category_share",
    "history_subcategory_clicks",
    "history_subcategory_share",
    "title_length",
    "abstract_length",
]


def read_news(data_dir: str | Path) -> pd.DataFrame:
    """Прочитать метаданные новостей и посчитать длины текстов в словах.

    Args:
        data_dir: Папка split MIND с файлом news.tsv.

    Returns:
        Таблица item_id, category, subcategory, title_length, abstract_length.
        Исходные тексты не сохраняются.

    Raises:
        FileNotFoundError: Файл news.tsv отсутствует.
    """
    data_dir = Path(data_dir)
    news = pd.read_csv(
        data_dir / "news.tsv",
        sep="\t",
        names=NEWS_COLUMNS,
        usecols=["item_id", "category", "subcategory", "title", "abstract"],
        dtype="string",
        keep_default_na=False,
    )
    news["title_length"] = news["title"].str.split().str.len().astype("int16")
    news["abstract_length"] = news["abstract"].str.split().str.len().astype("int16")
    return news.drop(columns=["title", "abstract"])


def prepare_behavior_chunk(
    behaviors: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    """Развернуть выдачи в строки кандидатов и присоединить историю кликов.

    Args:
        behaviors: Порция behaviors.tsv с колонками BEHAVIOR_COLUMNS.
            Функция добавляет в переданную таблицу history_length.
        news: Метаданные новостей, полученные через read_news.

    Returns:
        Таблица с колонками RESULT_COLUMNS: одна строка на кандидата в выдаче.
        Label берётся из записи кандидата вида N123-1; history-признаки —
        из предоставленного MIND списка кликов до выдачи, без текущих меток.

    Raises:
        KeyError: Входные таблицы не содержат необходимых колонок.
        ValueError: Некорректны время, метки кандидатов или уникальность news.
    """
    behaviors["history_length"] = behaviors["history"].str.count(" ")
    behaviors.loc[behaviors["history"].ne(""), "history_length"] += 1
    behaviors["history_length"] = behaviors["history_length"].astype("int16")

    # В одной строке behaviors лежит несколько кандидатов вида
    # "N55689-1 N35729-0". Превращаем их в отдельные строки.
    interactions = behaviors.assign(
        candidate=behaviors["impressions"].str.split()
    ).explode("candidate", ignore_index=True)

    item_and_label = interactions["candidate"].str.rsplit("-", n=1, expand=True)
    interactions["item_id"] = item_and_label[0]
    interactions["label"] = item_and_label[1].astype("int8")

    # Позиция новости внутри исходной выдачи.
    interactions["position"] = (
        interactions.groupby("impression_id", sort=False).cumcount().astype("int16")
    )

    interactions["timestamp"] = pd.to_datetime(
        interactions["timestamp"],
        format="%m/%d/%Y %I:%M:%S %p",
    )

    result = interactions.merge(
        news,
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    # Поле history содержит новости, по которым пользователь кликал до
    # текущей выдачи. Разворачиваем его отдельно, чтобы не хранить длинную
    # строку history в каждой строке итогового датасета.
    history = behaviors[["impression_id", "history"]].copy()
    history["history_item_id"] = history["history"].str.split()
    history = history.drop(columns="history").explode(
        "history_item_id", ignore_index=True
    )
    history = history[history["history_item_id"].notna()]
    history = history[history["history_item_id"].ne("")]

    history_news = news[["item_id", "category", "subcategory"]].rename(
        columns={
            "item_id": "history_item_id",
            "category": "history_category",
            "subcategory": "history_subcategory",
        }
    )
    history = history.merge(
        history_news,
        on="history_item_id",
        how="left",
        validate="many_to_one",
    )

    # Прямой user-item признак: встречалась ли новость среди прошлых кликов.
    seen_items = history[["impression_id", "history_item_id"]].drop_duplicates()
    seen_items["history_contains_item"] = 1
    result = result.merge(
        seen_items,
        left_on=["impression_id", "item_id"],
        right_on=["impression_id", "history_item_id"],
        how="left",
    ).drop(columns="history_item_id")

    # Общая разнообразность пользовательской истории.
    history_summary = (
        history.groupby("impression_id", as_index=False)
        .agg(
            history_unique_categories=("history_category", "nunique"),
            history_unique_subcategories=("history_subcategory", "nunique"),
        )
    )
    result = result.merge(history_summary, on="impression_id", how="left")

    # Сколько прошлых кликов пользователя относится к категории кандидата.
    category_counts = (
        history.groupby(
            ["impression_id", "history_category"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "history_category": "category",
                "size": "history_category_clicks",
            }
        )
    )
    result = result.merge(
        category_counts,
        on=["impression_id", "category"],
        how="left",
    )

    # Аналогичный признак для более узкой подкатегории.
    subcategory_counts = (
        history.groupby(
            ["impression_id", "history_subcategory"],
            as_index=False,
        )
        .size()
        .rename(
            columns={
                "history_subcategory": "subcategory",
                "size": "history_subcategory_clicks",
            }
        )
    )
    result = result.merge(
        subcategory_counts,
        on=["impression_id", "subcategory"],
        how="left",
    )

    count_columns = [
        "history_contains_item",
        "history_unique_categories",
        "history_unique_subcategories",
        "history_category_clicks",
        "history_subcategory_clicks",
    ]
    result[count_columns] = result[count_columns].fillna(0).astype("int16")

    denominator = result["history_length"].replace(0, pd.NA)
    result["history_category_share"] = (
        result["history_category_clicks"] / denominator
    ).fillna(0).astype("float32")
    result["history_subcategory_share"] = (
        result["history_subcategory_clicks"] / denominator
    ).fillna(0).astype("float32")

    return result[RESULT_COLUMNS]


def load_mind_dataframe(
    data_dir: str | Path,
    max_impressions: int | None = None,
) -> pd.DataFrame:
    """Прочитать один split MINDsmall в память и вернуть плоский DataFrame.

    Для полного train лучше использовать :func:`export_mind_csv`: после
    разворачивания полный набор содержит около 5.84 млн строк.

    Args:
        data_dir: Папка split с behaviors.tsv и news.tsv.
        max_impressions: Максимум исходных выдач; None означает весь split.

    Returns:
        Подготовленная таблица кандидатов с колонками RESULT_COLUMNS.

    Raises:
        FileNotFoundError: Один из исходных TSV-файлов отсутствует.
        ValueError: Не удалось преобразовать исходные данные.
    """
    data_dir = Path(data_dir)
    behaviors = pd.read_csv(
        data_dir / "behaviors.tsv",
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        dtype="string",
        keep_default_na=False,
        nrows=max_impressions,
    )
    return prepare_behavior_chunk(behaviors, read_news(data_dir))


def export_mind_csv(
    data_dir: str | Path,
    output: str | Path,
    max_impressions: int | None = None,
    chunk_size: int = 5_000,
) -> int:
    """Порциями преобразовать MINDsmall и сохранить плоскую таблицу в CSV.

    Args:
        data_dir: Папка split с behaviors.tsv и news.tsv.
        output: Путь CSV. Родительская папка создаётся при необходимости;
            существующий файл перезаписывается при записи первой порции.
        max_impressions: Максимум выдач; None означает весь split.
        chunk_size: Число исходных выдач в одной порции чтения.

    Returns:
        Число сохранённых строк кандидатов, а не исходных выдач.

    Raises:
        FileNotFoundError: Один из исходных TSV-файлов отсутствует.
        ValueError: Некорректны исходные данные или параметры чтения.
        OSError: Не удалось создать каталог или записать CSV.
    """
    data_dir = Path(data_dir)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    news = read_news(data_dir)
    written_rows = 0

    chunks = pd.read_csv(
        data_dir / "behaviors.tsv",
        sep="\t",
        names=BEHAVIOR_COLUMNS,
        dtype="string",
        keep_default_na=False,
        nrows=max_impressions,
        chunksize=chunk_size,
    )
    for chunk_number, behaviors in enumerate(chunks):
        frame = prepare_behavior_chunk(behaviors, news)
        frame.to_csv(
            output,
            mode="w" if chunk_number == 0 else "a",
            header=chunk_number == 0,
            index=False,
        )
        written_rows += len(frame)
        print(f"Записано строк: {written_rows:,}", end="\r", flush=True)

    print()
    return written_rows


def main() -> None:
    """Прочитать параметры CLI и показать таблицу либо сохранить CSV.

    Returns:
        None. Результат печатается в консоль или записывается в указанный CSV.

    Raises:
        SystemExit: Параметры CLI некорректны либо запрошена загрузка всего
            split в память без указания выходного CSV.
    """
    parser = argparse.ArgumentParser(description="Собрать MINDsmall в DataFrame")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("MINDsmall_train"),
        help="Папка одного split MINDsmall",
    )
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=1_000,
        help="Число выдач для проверки; передайте 0, чтобы прочитать весь split",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Необязательный путь для сохранения результата в CSV",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Число выдач в одной порции при сохранении CSV",
    )
    args = parser.parse_args()

    limit = args.max_impressions or None
    if args.output is not None:
        rows = export_mind_csv(
            args.data_dir,
            args.output,
            max_impressions=limit,
            chunk_size=args.chunk_size,
        )
        print(f"Сохранено {rows:,} строк в {args.output}")
        return

    if limit is None:
        raise SystemExit(
            "Полный split не следует собирать в RAM. Добавьте --output или "
            "задайте --max-impressions."
        )

    dataframe = load_mind_dataframe(args.data_dir, max_impressions=limit)
    print(dataframe.head())
    print(f"\nСтрок: {len(dataframe):,}")
    print(f"Колонок: {len(dataframe.columns)}")
    print(f"\nКолонки: {list(dataframe.columns)}")


if __name__ == "__main__":
    main()
