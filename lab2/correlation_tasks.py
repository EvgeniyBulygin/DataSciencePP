"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]
TARGET = "MRI_Count"


def load_brain_data(csv_path: str) -> pd.DataFrame:
    """Загружает brainsize.txt: разделитель — tab, 'NA' — пропуск."""
    return pd.read_csv(csv_path, sep="\t", na_values=["NA"])


def mri_correlations(group: pd.DataFrame) -> dict[str, float]:
    """Корреляция Пирсона каждого признака с MRI_Count.

    Пропуски отбрасываются попарно: для каждого признака используются
    только строки, где известны и он, и MRI_Count.
    """
    return {
        feature: float(group[feature].corr(group[TARGET], method="pearson"))
        for feature in FEATURES
    }


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    df = load_brain_data(data.csv_path)

    men = df[df["Gender"] == "Male"]
    women = df[df["Gender"] == "Female"]

    men_corr = mri_correlations(men)
    women_corr = mri_correlations(women)

    # ищем максимум |r| среди результатов обеих групп; MRI_Count не входит в FEATURES
    strongest = max(
        [*men_corr.items(), *women_corr.items()],
        key=lambda item: abs(item[1]),
    )[0]

    return BrainCorrelationSummary(
        men_count=int(len(men)),
        women_count=int(len(women)),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=strongest,
    )
