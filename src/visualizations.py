"""
visualizations.py
=================

Utilidades de graficado reutilizables para el análisis OULAD.

Diseño consistente para todas las figuras:
    - figsize mínimo (12, 6)
    - paleta 'Set2' / 'tab10' de seaborn
    - títulos descriptivos en español y ejes etiquetados
    - función `save_fig()` que guarda en outputs/figures/ a 150 dpi

Las funciones devuelven el objeto Figure de matplotlib para permitir
composición o personalización adicional desde el notebook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .load_data import get_figures_dir

# --------------------------------------------------------------------------- #
# Estilo global
# --------------------------------------------------------------------------- #
PALETTE = "Set2"
DEFAULT_FIGSIZE = (12, 6)


def set_style() -> None:
    """Aplica un estilo visual consistente a todas las figuras."""
    sns.set_theme(style="whitegrid", palette=PALETTE)
    plt.rcParams.update({
        "figure.figsize": DEFAULT_FIGSIZE,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "font.size": 11,
    })


def save_fig(fig: plt.Figure, filename: str) -> Path:
    """
    Guarda la figura en outputs/figures/<filename> y devuelve la ruta.

    Acepta nombres con o sin extensión .png.
    """
    if not filename.lower().endswith(".png"):
        filename += ".png"
    path = get_figures_dir() / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[figura guardada] {path}")
    return path


# --------------------------------------------------------------------------- #
# Helpers de graficado
# --------------------------------------------------------------------------- #
def annotate_bars(ax: plt.Axes, fmt: str = "{:.0f}", horizontal: bool = False,
                  fontsize: int = 10) -> None:
    """Anota cada barra de un eje con su valor."""
    for p in ax.patches:
        if horizontal:
            width = p.get_width()
            if np.isnan(width):
                continue
            ax.annotate(fmt.format(width),
                        (width, p.get_y() + p.get_height() / 2),
                        ha="left", va="center", fontsize=fontsize,
                        xytext=(4, 0), textcoords="offset points")
        else:
            height = p.get_height()
            if np.isnan(height):
                continue
            ax.annotate(fmt.format(height),
                        (p.get_x() + p.get_width() / 2, height),
                        ha="center", va="bottom", fontsize=fontsize,
                        xytext=(0, 3), textcoords="offset points")


def barh_counts_pct(series: pd.Series, title: str, xlabel: str,
                    order: Iterable | None = None) -> plt.Figure:
    """
    Barras horizontales con conteo y porcentaje de cada categoría.
    """
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex([c for c in order if c in counts.index])
    total = counts.sum()
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    colors = sns.color_palette(PALETTE, len(counts))
    ax.barh(counts.index.astype(str), counts.values, color=colors)
    for i, (cat, val) in enumerate(counts.items()):
        ax.text(val, i, f"  {val:,} ({val / total * 100:.1f}%)",
                va="center", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def pie_chart(series: pd.Series, title: str, order: Iterable | None = None) -> plt.Figure:
    """Gráfico de torta de una serie categórica."""
    counts = series.value_counts()
    if order is not None:
        counts = counts.reindex([c for c in order if c in counts.index])
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    colors = sns.color_palette(PALETTE, len(counts))
    ax.pie(counts.values, labels=counts.index.astype(str), autopct="%1.1f%%",
           startangle=90, colors=colors, wedgeprops={"edgecolor": "white"},
           textprops={"fontsize": 11})
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()
    return fig


def stacked_100(ct: pd.DataFrame, title: str, xlabel: str, ylabel: str = "Percentage (%)",
                col_order: Iterable | None = None) -> plt.Figure:
    """
    Barras apiladas al 100% a partir de una tabla de contingencia
    (filas = categoría del eje X, columnas = categorías a apilar).
    """
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    if col_order is not None:
        pct = pct[[c for c in col_order if c in pct.columns]]
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    pct.plot(kind="bar", stacked=True, ax=ax,
             color=sns.color_palette(PALETTE, pct.shape[1]))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def grouped_pct(ct: pd.DataFrame, title: str, xlabel: str,
                col_order: Iterable | None = None) -> plt.Figure:
    """
    Barras agrupadas mostrando el % de cada categoría apilada dentro de cada
    grupo del eje X (normalizado por fila).
    """
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    if col_order is not None:
        pct = pct[[c for c in col_order if c in pct.columns]]
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    pct.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, pct.shape[1]))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=0)
    fig.tight_layout()
    return fig


def heatmap_pct(ct: pd.DataFrame, title: str, xlabel: str, ylabel: str,
                col_order: Iterable | None = None, fmt: str = ".1f") -> plt.Figure:
    """
    Heatmap de porcentajes normalizados por fila a partir de una tabla de
    contingencia.
    """
    pct = ct.div(ct.sum(axis=1), axis=0) * 100
    if col_order is not None:
        pct = pct[[c for c in col_order if c in pct.columns]]
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.heatmap(pct, annot=True, fmt=fmt, cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "Percentage (%)"})
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig
