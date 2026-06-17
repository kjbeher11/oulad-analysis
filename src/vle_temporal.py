"""
vle_temporal.py
===============

Análisis temporal del clic diario por `activity_type` (la dimensión más rica
del dataset OULAD, cruzando `studentVle` con `vle`).

Responde a tres ángulos:
  1. Patrones semanales  -> qué tipos de actividad se usan en cada semana.
  2. Ventana predictora  -> con cuántas semanas iniciales de actividad ya se
                            puede anticipar el resultado (curva de AUC).
  3. Composición/secuencia -> mezcla de tipos de actividad por grupo de
                            resultado (Pass / Fail / Withdrawn / Distinction).

Funciones principales:
  - enrich_vle():               añade activity_type, week y resultado a studentVle.
  - weekly_by_activity():       clics por (semana, activity_type), opcional por resultado.
  - mean_weekly_per_student():  media de clics por estudante/semana/activity_type y grupo.
  - composition_by_result():    % de clics por activity_type dentro de cada resultado.
  - predictive_window():        AUC de un modelo logístico según la semana de corte.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

JOIN_KEYS = ["id_student", "code_module", "code_presentation"]

# Resultados considerados "éxito" para la tarea predictiva binaria.
SUCCESS = {"Pass", "Distinction"}


def enrich_vle(
    df_vle: pd.DataFrame,
    df_vle_meta: pd.DataFrame,
    df_info: pd.DataFrame,
    max_week: int | None = 30,
) -> pd.DataFrame:
    """
    Enriquece studentVle con:
      - `activity_type` (vía id_site -> vle.activity_type)
      - `week` = date // 7
      - `resultado` (vía join con studentInfo por la clave estudante-curso)

    Si `max_week` no es None, recorta a semanas [0, max_week] para los análisis
    temporales (descarta actividad pre-curso y la cola larga poco poblada).
    """
    df = df_vle.copy()
    site2act = df_vle_meta.set_index("id_site")["activity_type"]
    df["activity_type"] = df["id_site"].map(site2act).astype("category")
    df["week"] = (df["date"].astype("int32") // 7).astype("int16")

    res = df_info[JOIN_KEYS + ["resultado"]].copy()
    res["resultado"] = res["resultado"].astype(str)
    df = df.merge(res, on=JOIN_KEYS, how="left")

    if max_week is not None:
        df = df[(df["week"] >= 0) & (df["week"] <= max_week)]
    return df


def top_activities(df_enriched: pd.DataFrame, n: int = 8) -> List[str]:
    """Devuelve los `n` activity_type con más clics totales."""
    return (
        df_enriched.groupby("activity_type", observed=True)["sum_click"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index.tolist()
    )


def weekly_by_activity(df_enriched: pd.DataFrame, by_result: bool = False) -> pd.DataFrame:
    """
    Suma de clics por (week, activity_type). Si by_result=True, añade resultado
    como nivel adicional del groupby.
    """
    keys = (["resultado"] if by_result else []) + ["week", "activity_type"]
    return (
        df_enriched.groupby(keys, observed=True)["sum_click"].sum().reset_index()
    )


def mean_weekly_per_student(
    df_enriched: pd.DataFrame, df_info: pd.DataFrame, activities: List[str]
) -> pd.DataFrame:
    """
    Media de clics por estudante para cada (resultado, week, activity_type),
    limitada a la lista `activities`. Normaliza por el nº de estudantes de cada
    grupo de resultado (no por los que tuvieron actividad ese día), de modo que
    los grupos sean comparables.

    Devuelve un DataFrame largo: [resultado, week, activity_type, mean_clicks].
    """
    n_por_result = (
        df_info.assign(resultado=df_info["resultado"].astype(str))
        .groupby("resultado", observed=True)["id_student"].size()
    )
    sub = df_enriched[df_enriched["activity_type"].isin(activities)]
    tot = (
        sub.groupby(["resultado", "week", "activity_type"], observed=True)["sum_click"]
        .sum()
        .reset_index()
    )
    tot["mean_clicks"] = tot.apply(
        lambda r: r["sum_click"] / n_por_result.get(r["resultado"], np.nan), axis=1
    )
    return tot


def composition_by_result(
    df_enriched: pd.DataFrame, activities: List[str]
) -> pd.DataFrame:
    """
    % de clics que cada activity_type representa dentro de cada grupo de
    resultado. Filas = resultado, columnas = activity_type (de `activities`).
    """
    comp = (
        df_enriched.groupby(["resultado", "activity_type"], observed=True)["sum_click"]
        .sum()
        .unstack(fill_value=0)
    )
    comp = comp[[a for a in activities if a in comp.columns]]
    pct = comp.div(comp.sum(axis=1), axis=0) * 100
    return pct


def predictive_window(
    df_enriched: pd.DataFrame,
    df_info: pd.DataFrame,
    cutoffs: List[int],
    activities: List[str],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Para cada semana de corte k en `cutoffs`, construye features = clics
    acumulados por activity_type en las semanas [0, k] por estudante, y entrena
    una regresión logística para predecir el ÉXITO (Pass/Distinction = 1 vs
    Fail/Withdrawn = 0). Devuelve el AUC medio (validación cruzada 4-fold).

    Devuelve DataFrame: [semana_corte, auc, n_estudiantes, tasa_exito].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # Target por estudante-curso.
    target = df_info[JOIN_KEYS + ["resultado"]].copy()
    target["resultado"] = target["resultado"].astype(str)
    target["y"] = target["resultado"].isin(SUCCESS).astype(int)
    target = target.set_index(JOIN_KEYS)["y"]

    # Pre-agregado compacto: clics por (clave, activity_type, week).
    sub = df_enriched[df_enriched["activity_type"].isin(activities)]
    g = (
        sub.groupby(JOIN_KEYS + ["activity_type", "week"], observed=True)["sum_click"]
        .sum()
        .reset_index()
    )

    rows = []
    for k in cutoffs:
        gk = g[g["week"] <= k]
        feats = (
            gk.groupby(JOIN_KEYS + ["activity_type"], observed=True)["sum_click"]
            .sum()
            .unstack(fill_value=0)
        )
        feats = feats.reindex(columns=activities, fill_value=0)
        # Estudantes sin actividad hasta k: features en 0.
        feats = feats.reindex(target.index, fill_value=0)
        y = target.loc[feats.index]
        X = feats.values

        clf = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)
        )
        auc = cross_val_score(clf, X, y, cv=4, scoring="roc_auc").mean()
        rows.append({
            "semana_corte": k,
            "auc": round(float(auc), 4),
            "n_estudiantes": int(len(y)),
            "tasa_exito": round(float(y.mean()) * 100, 2),
        })
    return pd.DataFrame(rows)
