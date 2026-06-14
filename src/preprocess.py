"""
preprocess.py
=============

Limpieza, transformación y construcción del DataFrame maestro del dataset OULAD.

Funciones principales:
    - clean_student_info():          mapea final_result, convierte a Categorical, imputa imd_band.
    - clean_student_assessment():    score numérico + marca de no entregados.
    - aggregate_clicks():            clics totales por (estudiante, módulo, presentación).
    - clean_student_registration():  fechas numéricas + días hasta retiro.
    - mean_score_per_student():      score promedio por estudiante/módulo/presentación.
    - build_master():                construye df_master mediante left joins.
    - encode_demographics():         codificación numérica de variables para modelos.

Convenciones:
    - Las claves de unión son siempre (id_student, code_module, code_presentation).
    - Se devuelven copias para no mutar los DataFrames originales.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constantes de mapeo
# --------------------------------------------------------------------------- #

# final_result -> etiqueta para los gráficos (inglés, términos originales OULAD).
# Se mantiene la columna `resultado` para conservar la estructura del pipeline.
RESULT_MAP = {
    "Pass": "Pass",
    "Fail": "Fail",
    "Withdrawn": "Withdrawn",
    "Distinction": "Distinction",
}

# Orden lógico de las categorías de resultado (para gráficos consistentes).
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]

# Orden lógico del nivel educativo (de menor a mayor).
EDUCATION_ORDER = [
    "No Formal quals",
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Post Graduate Qualification",
]

# Orden lógico de las bandas de edad.
AGE_ORDER = ["0-35", "35-55", "55<="]

# Orden lógico de las bandas IMD (índice de privación; menor = más pobre).
IMD_ORDER = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]

# Claves de unión canónicas.
JOIN_KEYS = ["id_student", "code_module", "code_presentation"]


# --------------------------------------------------------------------------- #
# 1. studentInfo
# --------------------------------------------------------------------------- #
def clean_student_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la tabla studentInfo:
      - Mapea final_result a etiquetas claras en español (columna `resultado`).
      - Convierte age_band, gender, disability, highest_education a Categorical
        (ordenadas donde tiene sentido).
      - Normaliza e imputa imd_band con la moda.
    """
    df = df.copy()

    # Mapeo de resultado a español, conservando el original.
    df["resultado"] = df["final_result"].map(RESULT_MAP)
    df["resultado"] = pd.Categorical(df["resultado"], categories=RESULT_ORDER, ordered=False)

    # Normalización de imd_band: '10-20' -> '10-20%'.
    df["imd_band"] = df["imd_band"].replace({"10-20": "10-20%"})
    # Imputación de nulos con la moda.
    if df["imd_band"].isna().any():
        moda_imd = df["imd_band"].mode(dropna=True).iloc[0]
        n_nulos = int(df["imd_band"].isna().sum())
        df["imd_band"] = df["imd_band"].fillna(moda_imd)
        print(f"[imd_band] Imputados {n_nulos:,} nulos con la moda = '{moda_imd}'")
    df["imd_band"] = pd.Categorical(df["imd_band"], categories=IMD_ORDER, ordered=True)

    # Conversión a Categorical (ordenadas donde aplica).
    df["age_band"] = pd.Categorical(df["age_band"], categories=AGE_ORDER, ordered=True)
    df["gender"] = pd.Categorical(df["gender"], categories=["F", "M"], ordered=False)
    df["disability"] = pd.Categorical(df["disability"], categories=["N", "Y"], ordered=False)
    df["highest_education"] = pd.Categorical(
        df["highest_education"], categories=EDUCATION_ORDER, ordered=True
    )

    return df


# --------------------------------------------------------------------------- #
# 2. studentAssessment
# --------------------------------------------------------------------------- #
def clean_student_assessment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la tabla studentAssessment:
      - Convierte score a numérico (errors='coerce').
      - Marca como 'no_submitted' (True) los registros con is_banked == 1
        o score nulo.
    """
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["no_submitted"] = (df["is_banked"] == 1) | (df["score"].isna())
    n = int(df["no_submitted"].sum())
    print(f"[studentAssessment] Marcados {n:,} registros como 'no_submitted' "
          f"({n / len(df) * 100:.2f}%)")
    return df


# --------------------------------------------------------------------------- #
# 3. studentVle  ->  clics totales
# --------------------------------------------------------------------------- #
def aggregate_clicks(df_vle: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula los clics totales por estudiante:
    agrupa por (id_student, code_module, code_presentation) y suma sum_click.

    Devuelve un DataFrame con la columna `clicks_totales` y también
    `dias_activos` (número de días distintos con actividad), útil para K-Means.
    """
    g = df_vle.groupby(JOIN_KEYS, observed=True)
    out = g.agg(
        clicks_totales=("sum_click", "sum"),
        dias_activos=("date", "nunique"),
    ).reset_index()
    print(f"[studentVle] Clics agregados para {len(out):,} combinaciones "
          f"estudiante-módulo-presentación")
    return out


# --------------------------------------------------------------------------- #
# 4. studentRegistration
# --------------------------------------------------------------------------- #
def clean_student_registration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia la tabla studentRegistration:
      - Convierte date_registration y date_unregistration a numérico.
      - Calcula 'dias_hasta_retiro' = date_unregistration - date_registration.
    """
    df = df.copy()
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_unregistration"] = pd.to_numeric(df["date_unregistration"], errors="coerce")
    df["dias_hasta_retiro"] = df["date_unregistration"] - df["date_registration"]
    return df


# --------------------------------------------------------------------------- #
# 5. score promedio por estudiante
# --------------------------------------------------------------------------- #
def mean_score_per_student(
    df_assessment: pd.DataFrame, df_assessments_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Calcula el score promedio por (id_student, code_module, code_presentation).

    Une studentAssessment con assessments (metadatos) para recuperar
    code_module y code_presentation, y promedia el score de las entregas
    válidas (no marcadas como no_submitted).
    """
    meta = df_assessments_meta[["id_assessment", "code_module", "code_presentation"]]
    merged = df_assessment.merge(meta, on="id_assessment", how="left")
    valid = merged.loc[~merged["no_submitted"]]
    out = (
        valid.groupby(JOIN_KEYS, observed=True)["score"]
        .mean()
        .reset_index()
        .rename(columns={"score": "score_promedio"})
    )
    print(f"[scores] Score promedio calculado para {len(out):,} estudiantes")
    return out


# --------------------------------------------------------------------------- #
# 6. DataFrame maestro
# --------------------------------------------------------------------------- #
def build_master(
    df_info_clean: pd.DataFrame,
    df_reg_clean: pd.DataFrame,
    df_clicks: pd.DataFrame,
    df_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Construye df_master mediante left joins por (id_student, code_module,
    code_presentation):

        studentInfo + studentRegistration + clicks_totales [+ score_promedio]

    El score promedio es opcional pero recomendado para las preguntas 2 y 7.
    """
    master = df_info_clean.merge(
        df_reg_clean[JOIN_KEYS + ["date_registration", "date_unregistration", "dias_hasta_retiro"]],
        on=JOIN_KEYS,
        how="left",
    )
    master = master.merge(df_clicks, on=JOIN_KEYS, how="left")

    if df_scores is not None:
        master = master.merge(df_scores, on=JOIN_KEYS, how="left")

    # Estudiantes sin registros VLE -> 0 clics / 0 días activos.
    master["clicks_totales"] = master["clicks_totales"].fillna(0).astype("int64")
    master["dias_activos"] = master["dias_activos"].fillna(0).astype("int64")

    print(f"[df_master] Construido con shape={master.shape}")
    return master


# --------------------------------------------------------------------------- #
# 7. Codificación numérica para modelos
# --------------------------------------------------------------------------- #
def encode_demographics(df_master: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas codificadas numéricamente para análisis de correlación y
    clustering:
      - highest_education_encoded  (orden ordinal)
      - imd_band_encoded           (orden ordinal)
      - age_band_encoded           (orden ordinal)
      - gender_encoded             (F=0, M=1)
      - disability_encoded         (N=0, Y=1)
      - resultado_encoded          (Retirado=0, Reprobado=1, Aprobado=2, Distinción=3)
    """
    df = df_master.copy()

    edu_map = {v: i for i, v in enumerate(EDUCATION_ORDER)}
    imd_map = {v: i for i, v in enumerate(IMD_ORDER)}
    age_map = {v: i for i, v in enumerate(AGE_ORDER)}
    resultado_map = {"Withdrawn": 0, "Fail": 1, "Pass": 2, "Distinction": 3}

    df["highest_education_encoded"] = df["highest_education"].astype(str).map(edu_map)
    df["imd_band_encoded"] = df["imd_band"].astype(str).map(imd_map)
    df["age_band_encoded"] = df["age_band"].astype(str).map(age_map)
    df["gender_encoded"] = df["gender"].astype(str).map({"F": 0, "M": 1})
    df["disability_encoded"] = df["disability"].astype(str).map({"N": 0, "Y": 1})
    df["resultado_encoded"] = df["resultado"].astype(str).map(resultado_map)

    return df


def run_full_pipeline(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Ejecuta el pipeline completo de preprocesamiento sobre el diccionario de
    DataFrames crudos y devuelve un diccionario con los productos limpios:
        'info', 'assessment', 'registration', 'clicks', 'scores', 'master'.
    """
    info = clean_student_info(data["studentInfo"])
    assessment = clean_student_assessment(data["studentAssessment"])
    registration = clean_student_registration(data["studentRegistration"])
    clicks = aggregate_clicks(data["studentVle"])
    scores = mean_score_per_student(assessment, data["assessments"])
    master = build_master(info, registration, clicks, scores)
    master = encode_demographics(master)

    return {
        "info": info,
        "assessment": assessment,
        "registration": registration,
        "clicks": clicks,
        "scores": scores,
        "master": master,
    }
