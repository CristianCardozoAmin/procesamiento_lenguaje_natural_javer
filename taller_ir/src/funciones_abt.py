import pandas as pd
import ast
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from PIL import Image
from math import sqrt

def frecuencia_terminos_columna_str(df: pd.DataFrame, col: str, min_len: int = 1):
    """
    Devuelve una lista de (término, frecuencia) a partir de una columna
    que contiene strings con formato de lista de listas.
    """
    tokens_doc = []
    for fila in df[col]:
        try:
            listas = ast.literal_eval(fila)
            for sub in listas:
                for tok in sub:
                    if len(tok) >= min_len:
                        tokens_doc.append(tok)
        except (ValueError, SyntaxError):
            continue
    fdist = FreqDist(tokens_doc)
    # Convertir a lista ordenada de tuplas (término, frecuencia)
    return sorted(fdist.items(), key=lambda kv: (-kv[1], kv[0]))

def graficar_frecuencias(df: pd.DataFrame, col: str, top_n: int = 20, min_len: int = 1):
    """
    Grafica las top_n palabras más frecuentes de la columna indicada.
    """
    frecuencias = frecuencia_terminos_columna_str(df, col, min_len)
    if not frecuencias:
        print("No hay datos para graficar.")
        return

    # Ahora sí se puede hacer slicing
    palabras, counts = zip(*frecuencias[:top_n])

    plt.figure(figsize=(20, 6))
    plt.bar(range(len(palabras)), counts)
    plt.xticks(range(len(palabras)), palabras, rotation=90, ha='right')
    plt.xlabel("Término")
    plt.ylabel("Frecuencia")
    plt.title(f"Top {top_n} términos más frecuentes")
    plt.tight_layout()
    plt.show()



def nube_palabras(df: pd.DataFrame, col: str, min_len: int = 1, titulo="Nube de Palabras") -> Image.Image:
    """
    Genera una nube de palabras tal cual el texto recibido, sin filtrado ni stopwords.
    Retorna un objeto PIL.Image.
    """
    frecuencias = frecuencia_terminos_columna_str(df, col, min_len)

    if not frecuencias:
        print("No hay datos para graficar.")
        return
    
    frecuencias_dict = dict(frecuencias)
    
    # Crear la nube de palabras
    nube = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis',   # puedes cambiar el esquema de colores
        max_words=200
    ).generate_from_frequencies(frecuencias_dict)

    plt.figure(figsize=(15, 7))
    plt.imshow(nube, interpolation='bilinear')
    plt.axis('off')
    plt.title(titulo, fontsize=16)
    plt.show()

def aplanar_tokens(df: pd.DataFrame, col: str, min_len: int = 1):
    """
    Devuelve una lista de (término, frecuencia) a partir de una columna
    que contiene strings con formato de lista de listas.
    """
    tokens_doc = []
    for fila in df[col]:
        try:
            listas = ast.literal_eval(fila)
            for sub in listas:
                for tok in sub:
                    if len(tok) >= min_len:
                        tokens_doc.append(tok)
        except (ValueError, SyntaxError):
            continue
    return tokens_doc

def diversidad_lexica(df: pd.DataFrame, col: str , min_len: int = 1) -> None:
    """
    Calcula e imprime las métricas de diversidad léxica (TTR y RTTR)
    para la columna indicada del DataFrame.
    """

    # Aplanar los tokens
    tokens_doc = aplanar_tokens(df, col, min_len)

    # Calcular métricas
    total_tokens = len(tokens_doc)
    tipos_unicos = len(set(tokens_doc))

    # Evitar división por cero
    if total_tokens == 0:
        ttr = 0.0
        rttr = 0.0
    else:
        ttr  = tipos_unicos / total_tokens
        rttr = tipos_unicos / sqrt(total_tokens)
    
     # Informe por print
    print("=== Informe de Diversidad Léxica (Corpus) ===")
    print(f"Tokens totales:       {total_tokens}")
    print(f"Tipos únicos:         {tipos_unicos}")
    print(f"TTR (Type-Token Ratio): {ttr:.4f}")
    print(f"RTTR (Root TTR):        {rttr:.4f}")