# =========================
# Dependencias
# =========================
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast
import numpy as np
import math

def preparar_texto(df, col_entrada="contenido_preprocesado", col_salida="contenido_texto"):
    """
    Convierte la columna preprocesada (lista/lista de listas o string de lista) en texto plano.
    Mantiene la lógica original: join de tokens.
    """
    df = df.copy()
    # si viene como string de lista, volver a lista
    df[col_entrada] = df[col_entrada].apply(ast.literal_eval if df[col_entrada].dtype == object else (lambda x: x))
    # convertir a texto plano (mismo enfoque de tu lambda)
    df[col_salida] = df[col_entrada].apply(lambda row: " ".join([word for phrase in row for word in phrase]))
    return df

def construir_tfidf(df_libro_texto, col_texto="contenido_texto"):
    """
    Crea el vectorizador TF-IDF y la matriz tfs sobre df_libro_texto[col_texto].
    Devuelve también feature_names y dimensiones
    """
    tfidf = TfidfVectorizer()
    tfs = tfidf.fit_transform(df_libro_texto[col_texto])
    secc_num, feature_num = tfs.shape
    feature_names = tfidf.get_feature_names_out()
    return tfidf, tfs, feature_names, secc_num, feature_num

def agregar_similitudes_y_topn(df_query_texto, tfidf, tfs, col_texto="contenido_texto", top_n=5):
    """
    Calcula cosine_similarity por filas
    y agrega columnas 'cosine_similarities' y 'top_matches' como en el original.
    """
    df_out = df_query_texto.copy()

    def compute_similarities(doc):
        response = tfidf.transform([doc])
        sims = cosine_similarity(response, tfs)[0]
        return sims

    df_out["cosine_similarities"] = df_out[col_texto].apply(compute_similarities)

    df_out["top_matches"] = df_out["cosine_similarities"].apply(
        lambda sims: sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_n]
    )
    return df_out

def generar_modelo_ir(df_libro, df_testimonios, col_preprocesada="contenido_preprocesado", top_n=5):
    """
    1) literal_eval -> 2) a texto plano -> 3) TF-IDF sobre libro -> 4) similitud para testimonios -> 5) top_n
    Retorna los dos DataFrames transformados (libro con texto, testimonios con similitudes/top_matches).
    """
    # Paso 1-2: literal_eval + join
    df_libro_texto = preparar_texto(df_libro, col_entrada=col_preprocesada, col_salida="contenido_texto")
    df_testimonios_texto = preparar_texto(df_testimonios, col_entrada=col_preprocesada, col_salida="contenido_texto")

    # Paso 3: TF-IDF sobre libro
    tfidf, tfs, feature_names, secc_num, feature_num = construir_tfidf(df_libro_texto, col_texto="contenido_texto")

    print(len(feature_names))
    print(feature_names)
    print("# secciones: %d, n_features: %d" % tfs.shape)
    print("###### Calculo de Feature Names ######")
    for x in range(0, feature_num):
         print(" # ", x , " - ", feature_names[x], " \t - ", [tfs[n,x] for n in range(0, secc_num)])

    # Paso 4-5: similitudes y top_n en testimonios
    df_testimonios_out = agregar_similitudes_y_topn(
        df_testimonios_texto, tfidf, tfs, col_texto="contenido_texto", top_n=top_n
    )

    # Devuelvo los dos DataFrames transformados
    return df_libro_texto, df_testimonios_out, tfidf
