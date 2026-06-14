"""
Paquete `src` para el análisis del dataset OULAD.

Módulos:
    - load_data:      carga e inspección de los 7 CSV.
    - preprocess:     limpieza, transformación y construcción del DataFrame maestro.
    - visualizations: utilidades de graficado reutilizables.
"""

from . import load_data
from . import preprocess
from . import visualizations

__all__ = ["load_data", "preprocess", "visualizations"]
