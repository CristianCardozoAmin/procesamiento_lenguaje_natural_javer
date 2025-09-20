import json
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import pandas as pd


#############################################################################
# Función de limpieza
#############################################################################
def limpiar_texto(texto: str) -> str:
    """
    Limpia y normaliza textos de entrevistas con encabezados tipo 001-VI-00011,
    marcas de rol (ENT/TEST), bloques censurados, anotaciones entre corchetes
    y timestamps. Devuelve solo el contenido legible.
    """

    if texto is None:
        return ""
    if not isinstance(texto, str):
        texto = str(texto)

    # --- Normalización Unicode y espacios ---
    t = unicodedata.normalize("NFKC", texto)
    t = t.replace("\u00A0", " ")                # NBSP -> espacio
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\t", " ")

    # --- Eliminar encabezados/códigos por línea ---
    # 001-VI-00011
    # 001-VI-00011_(48057): 01:30:51
    t = re.sub(
        r"(?mi)^\s*\d{2,5}-[A-ZÁÉÍÓÚÑÜ]{2,}-\d{3,}(?:_\(\d+\))?(?::\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*$",
        " ",
        t,
    )
    # También códigos incrustados, por si quedaron
    t = re.sub(r"\b\d{2,5}-[A-ZÁÉÍÓÚÑÜ]{2,}-\d{3,}\b", " ", t)

    # --- Eliminar timestamps sueltos HH:MM(:SS) ---
    t = re.sub(r"\b\d{1,2}:\d{2}(?::\d{2})?\b", " ", t)

    # --- Anotaciones entre corchetes (INC, INTERRUP, etc.) ---
    t = re.sub(r"\[(?:[^\]]*?)\]", " ", t)  # genérico y seguro

    # --- Quitar marcas de rol (ENT, TEST con/ sin número y espacios) ---
    t = re.sub(r"\b(?:ENT|TEST)\s*\d*\s*[:\-]\s*", " ", t, flags=re.I)

    # --- Eliminar bloques de guiones/underscores (censura) ---
    t = re.sub(r"[-_]{2,}", " ", t)

    # --- Unificar saltos de línea múltiples en espacio ---
    t = re.sub(r"\s*\n\s*", " ", t)

    # --- Normalización de puntuación ---
    t = t.replace("…", "...")                     # elipsis unicode
    t = re.sub(r"\.{3,}", "...", t)               # 3+ puntos -> '...'
    t = re.sub(r"([,;:?!¡¿])\1{1,}", r"\1", t)    # '!!' '??' ',,' -> 1

    # Mantener solo letras (con tildes), dígitos, espacios y puntuación clave + guion interno
    t = re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑüÜ0-9\s,.;:?!¡¿\-.]", " ", t)

    # Proteger elipsis para espaciado
    t = t.replace("...", " <ELIPSIS> ")
    # Espacio consistente alrededor de signos (incluye '.')
    t = re.sub(r"\s*([,;:?!¡¿\.])\s*", r" \1 ", t)
    # Restaurar elipsis y compactar
    t = re.sub(r"\s*<ELIPSIS>\s*", " ... ", t)

    # Quitar dígitos pegados a palabras (ORG1 -> ORG)
    t = re.sub(r"\b([A-Za-záéíóúÁÉÍÓÚñÑüÜ]+)\d+\b", r"\1", t)

    # Colapsar espacios y recortar
    t = re.sub(r"\s{2,}", " ", t).strip()

    # Minúsculas 
    t = t.lower()

    return t

#############################################################################
# Helper para paralelizar
#############################################################################
def _limpiar_chunk(chunk: pd.Series) -> pd.Series:
    """Aplica limpiar_texto a un chunk (Serie) y devuelve la Serie resultante."""
    return chunk.map(limpiar_texto)


def _chunks_from_series(s: pd.Series, n_chunks: int) -> List[pd.Series]:
    """
    Divide una Serie en n_chunks preservando los índices,
    evitando el FutureWarning de Series.swapaxes.
    """
    n_chunks = max(1, min(n_chunks, len(s) or 1))

    # Convertimos valores e índices a ndarray antes de array_split
    arrays = np.array_split(s.to_numpy(), n_chunks)
    index_chunks = np.array_split(s.index.to_numpy(), n_chunks)

    return [pd.Series(a, index=i) for a, i in zip(arrays, index_chunks)]



#############################################################################
# Función principal
#############################################################################
def procesar_json(
    path_json: str,
    columna: str = "text",
    n_workers: int = None,
) -> pd.DataFrame:
    """
    Carga un JSON de entrevistas, limpia la columna indicada en paralelo y
    retorna un DataFrame con una nueva columna 'text_clean'.
    """
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"No se encontró el archivo: {path_json}")

    with open(path_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    if columna not in df.columns:
        raise KeyError(
            f"La columna '{columna}' no existe en el DataFrame. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # 1) Eliminar vacíos y NaN
    s = df[columna]
    vacios_exactos = s.eq("   ")
    vacios_nan = s.isna()
    total_vacios = (vacios_exactos | vacios_nan).sum()
    print(f"Entrevistas vacías detectadas: {total_vacios}")
    df = df.loc[~(vacios_exactos | vacios_nan)].reset_index(drop=True)

    # 2) Eliminar duplicados
    duplicados = df[columna].duplicated().sum()
    print(f"Se encontraron {duplicados} entrevistas duplicadas.")
    df = df.drop_duplicates(subset=columna).reset_index(drop=True)

    # 3) Paralelizar limpieza
    n_workers = n_workers or os.cpu_count() or 1
    series_obj = df[columna]

    chunks = _chunks_from_series(series_obj, n_workers)

    cleaned_parts: List[pd.Series] = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_limpiar_chunk, c) for c in chunks]
        for fut in as_completed(futures):
            cleaned_parts.append(fut.result())

    cleaned = pd.concat(cleaned_parts).sort_index()

    df = df.loc[cleaned.index].copy()
    df["text_clean"] = cleaned.values

    return df
