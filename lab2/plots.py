"""Графики для отчёта по части 2: scatter-графики и heatmap корреляций."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from correlation_tasks import FEATURES, TARGET, load_brain_data


def plot_scatter(df, show: bool = True):
    """Scatter-графики признаков против MRI_Count, цвет — пол."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, feature in zip(axes.ravel(), FEATURES):
        sns.scatterplot(data=df, x=feature, y=TARGET, hue="Gender", ax=ax)
        ax.set_title(f"{feature} vs {TARGET}")
    axes.ravel()[-1].axis("off")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_heatmaps(df, show: bool = True):
    """Heatmap корреляций отдельно для женщин и мужчин."""
    columns = FEATURES + [TARGET]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (gender, title) in zip(axes, [("Female", "Женщины"), ("Male", "Мужчины")]):
        corr = df[df["Gender"] == gender][columns].corr(method="pearson")
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    data = load_brain_data("brainsize.txt")
    plot_scatter(data)
    plot_heatmaps(data)
