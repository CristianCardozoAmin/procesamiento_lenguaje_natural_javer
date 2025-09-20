from typing import List, Dict, Any, Iterable, Tuple
from collections import Counter
from nltk.probability import FreqDist
import pandas as pd


def flatten(list_of_lists: Iterable[Iterable[str]]) -> List[str]:
    """Aplana [[t1,t2],[t3]] -> [t1,t2,t3]."""
    return [tok for sent in list_of_lists for tok in sent]

def frecuencia_terminos(contenido_preprocesado: List[List[str]],
                        min_len: int = 1) -> List[Tuple[str, int]]:
    """
    Calcula frecuencia de términos con NLTK (FreqDist) a partir de listas de tokens por oración.
    Devuelve una lista de (término, frecuencia), ordenada descendentemente.
    """
    tokens_doc = [t for t in flatten(contenido_preprocesado) if len(t) >= min_len]
    fdist = FreqDist(tokens_doc)
    # Ordenar por frecuencia desc, luego término asc para estabilidad
    return sorted(fdist.items(), key=lambda kv: (-kv[1], kv[0]))
