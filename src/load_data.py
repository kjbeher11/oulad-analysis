"""
load_data.py
============

Carga e inspección de los 7 archivos CSV del dataset OULAD.

Funciones principales:
    - get_raw_dir():   devuelve la ruta a `data/raw` de forma robusta.
    - check_files():   verifica que los 7 CSV existan y lanza un error claro si falta alguno.
    - load_all():      carga los 7 CSV en un diccionario de DataFrames.
    - inspect_all():   imprime shape, dtypes y primeras filas de cada DataFrame.

Todos los paths se resuelven con `pathlib.Path` de forma relativa a la raíz
del proyecto, de modo que el código funciona tanto desde `notebooks/` como
desde la raíz del repositorio.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# Nombres canónicos de los 7 CSV del dataset OULAD.
CSV_FILES = [
    "courses.csv",
    "assessments.csv",
    "vle.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
]

# Clave amigable (sin extensión) -> nombre de archivo.
TABLE_NAMES = {Path(f).stem: f for f in CSV_FILES}


def get_project_root() -> Path:
    """
    Devuelve la raíz del proyecto `oulad-analysis`.

    Busca hacia arriba desde este archivo hasta encontrar la carpeta que
    contiene `data/raw`. Esto permite ejecutar el código desde `notebooks/`,
    desde `src/` o desde la raíz indistintamente.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "data" / "raw").exists():
            return parent
    # Fallback: dos niveles arriba de src/ (raíz del proyecto).
    return here.parent.parent


def get_raw_dir() -> Path:
    """Ruta absoluta a la carpeta `data/raw`."""
    return get_project_root() / "data" / "raw"


def get_outputs_dir() -> Path:
    """Ruta absoluta a la carpeta `outputs` (la crea si no existe)."""
    out = get_project_root() / "outputs"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "report").mkdir(parents=True, exist_ok=True)
    return out


def get_figures_dir() -> Path:
    """Ruta absoluta a `outputs/figures` (la crea si no existe)."""
    fig = get_outputs_dir() / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    return fig


def check_files(raw_dir: Path | None = None) -> None:
    """
    Verifica que los 7 CSV existan en `data/raw`.

    Lanza FileNotFoundError con un mensaje claro indicando qué archivos faltan.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else get_raw_dir()
    missing = [f for f in CSV_FILES if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan los siguientes archivos CSV en "
            f"'{raw_dir}':\n  - " + "\n  - ".join(missing) +
            "\n\nCopia los 7 CSV del dataset OULAD en data/raw/ antes de continuar."
        )
    print(f"[OK] Los 7 CSV están presentes en: {raw_dir}")


def load_all(raw_dir: Path | None = None, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Carga los 7 CSV del dataset OULAD en un diccionario de DataFrames.

    Las claves del diccionario son los nombres de tabla sin extensión, p. ej.
    'courses', 'assessments', 'vle', 'studentInfo', 'studentRegistration',
    'studentAssessment', 'studentVle'.

    studentVle.csv es grande (~10.6M filas); se cargan dtypes optimizados para
    reducir el uso de memoria.
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else get_raw_dir()
    check_files(raw_dir)

    # dtypes optimizados para el archivo grande de interacciones VLE.
    studentvle_dtypes = {
        "code_module": "category",
        "code_presentation": "category",
        "id_student": "int32",
        "id_site": "int32",
        "date": "int16",
        "sum_click": "int16",
    }

    dtype_map = {"studentVle.csv": studentvle_dtypes}

    data: Dict[str, pd.DataFrame] = {}
    for fname in CSV_FILES:
        key = Path(fname).stem
        path = raw_dir / fname
        df = pd.read_csv(path, dtype=dtype_map.get(fname))
        data[key] = df
        if verbose:
            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            print(f"[cargado] {key:<22} shape={str(df.shape):<18} mem={mem_mb:7.1f} MB")
    return data


def inspect_all(data: Dict[str, pd.DataFrame], n_rows: int = 5) -> None:
    """
    Imprime shape, dtypes y primeras filas de cada DataFrame del diccionario.
    """
    for key, df in data.items():
        print("=" * 80)
        print(f"TABLA: {key}")
        print("=" * 80)
        print(f"Shape: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")
        print("Dtypes:")
        print(df.dtypes.to_string())
        print(f"\nPrimeras {n_rows} filas:")
        with pd.option_context("display.max_columns", None, "display.width", 160):
            print(df.head(n_rows).to_string())
        print()


if __name__ == "__main__":
    d = load_all()
    inspect_all(d)
