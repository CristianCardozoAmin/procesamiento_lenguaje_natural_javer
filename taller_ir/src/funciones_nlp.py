#############################################################################
# Morfología del texto
#############################################################################
def morfologia_texto(text: str):
    
    # Importar librerías
    from statistics import mean
    import spacy

    # Cargar el modelo de spaCy para español
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")

    # Aumentar el límite de longitud del texto (si es necesario)
    if len(text) > nlp.max_length:
        nlp.max_length = 200_000_000

    # Procesar el texto
    doc = nlp(text)

    # Obtener las oraciones
    sentences = list(doc.sents)
    tokens_por_oracion = [[t for t in sent if t.is_alpha] for sent in sentences]

    # Función para calcular el promedio de una lista, manejando listas vacías
    prom = lambda lst: float(mean(lst)) if lst else 0.0

    # -------- ORIGINAL --------
    palabras_totales = sum(len(t) for t in tokens_por_oracion)
    oraciones = len(tokens_por_oracion)
    palabras_por_oracion = [len(toks) for toks in tokens_por_oracion]

    verbos_list  = [sum(1 for t in toks if t.pos_ in {"VERB", "AUX"})  for toks in tokens_por_oracion]
    sustant_list = [sum(1 for t in toks if t.pos_ in {"NOUN", "PROPN"}) for toks in tokens_por_oracion]
    adjetiv_list = [sum(1 for t in toks if t.pos_ == "ADJ")             for toks in tokens_por_oracion]
    pronomb_list = [sum(1 for t in toks if t.pos_ == "PRON")            for toks in tokens_por_oracion]
    adverb_list  = [sum(1 for t in toks if t.pos_ == "ADV")             for toks in tokens_por_oracion]
    stopw_list   = [sum(1 for t in toks if t.is_stop)                   for toks in tokens_por_oracion]


    original = {
        "oraciones": oraciones,
        "cantidad_palabras": sum(palabras_por_oracion),
        "cantidad_palabras_sin_stopwords": sum(1 for t in doc if t.is_alpha and not t.is_stop),

        # Promedios por oración
        "promedio_palabras_por_oracion": prom(palabras_por_oracion),
        "promedio_verbos_por_oracion": prom(verbos_list),
        "promedio_sustantivos_por_oracion": prom(sustant_list),
        "promedio_adjetivos_por_oracion": prom(adjetiv_list),
        "promedio_pronombres_por_oracion": prom(pronomb_list),
        "promedio_adverbios_por_oracion": prom(adverb_list),
        "promedio_stopwords_por_oracion": prom(stopw_list),

        # Totales en todo el texto
        "total_verbos": sum(verbos_list),
        "total_sustantivos": sum(sustant_list),
        "total_adjetivos": sum(adjetiv_list),
        "total_pronombres": sum(pronomb_list),
        "total_adverbios": sum(adverb_list),
        "total_stopwords": sum(stopw_list)
    }

    # -------- SIN STOPWORDS --------
    tokens_sin_sw = [[t for t in toks if not t.is_stop] for toks in tokens_por_oracion]
    palabras_por_oracion_sw = [len(toks) for toks in tokens_sin_sw]

    verbos_sw_list  = [sum(1 for t in toks if t.pos_ in {"VERB", "AUX"})  for toks in tokens_sin_sw]
    sustant_sw_list = [sum(1 for t in toks if t.pos_ in {"NOUN", "PROPN"}) for toks in tokens_sin_sw]
    adjetiv_sw_list = [sum(1 for t in toks if t.pos_ == "ADJ")             for toks in tokens_sin_sw]
    pronomb_sw_list = [sum(1 for t in toks if t.pos_ == "PRON")            for toks in tokens_sin_sw]
    adverb_sw_list  = [sum(1 for t in toks if t.pos_ == "ADV")             for toks in tokens_sin_sw]

    sin_stopwords = {
        "oraciones": oraciones,
        "cantidad_palabras": sum(palabras_por_oracion_sw),
        "cantidad_palabras_sin_stopwords": sum(palabras_por_oracion_sw),

        # Promedios por oración
        "promedio_palabras_por_oracion": prom(palabras_por_oracion_sw),
        "promedio_verbos_por_oracion": prom(verbos_sw_list),
        "promedio_sustantivos_por_oracion": prom(sustant_sw_list),
        "promedio_adjetivos_por_oracion": prom(adjetiv_sw_list),
        "promedio_pronombres_por_oracion": prom(pronomb_sw_list),
        "promedio_adverbios_por_oracion": prom(adverb_sw_list),
        "promedio_stopwords_por_oracion": 0.0,

        # Totales en todo el texto (sin stopwords)
        "total_verbos": sum(verbos_sw_list),
        "total_sustantivos": sum(sustant_sw_list),
        "total_adjetivos": sum(adjetiv_sw_list),
        "total_pronombres": sum(pronomb_sw_list),
        "total_adverbios": sum(adverb_sw_list),
        "total_stopwords": 0
    }


    return {
        "original": original,
        "sin_stopwords": sin_stopwords,
    }

#############################################################################
# Morfología del texto - High Volume
#############################################################################
def morfologia_texto_high_volume(text: str, 
                     n_process: int = 4, 
                     batch_size: int = 1000, 
                     chunk_chars: int = 200_000):
    """
    - n_process: nº de procesos para spaCy (>=2 si tienes cores suficientes).
    - batch_size: nº de docs (chunks) por batch en pipe.
    - chunk_chars: tamaño aproximado de cada chunk de texto (ajústalo).
    """

    import spacy
    from statistics import mean
    try:
        # tqdm.auto se adapta a notebook/terminal
        from tqdm.auto import tqdm
    except Exception:
        tqdm = None  # por si no está instalado

    # ---- Carga del modelo con componentes ligeros ----
    try:
        nlp = spacy.load("es_core_news_sm", exclude=["parser", "ner", "lemmatizer"])
    except OSError:
        from spacy.cli import download
        download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm", exclude=["parser", "ner", "lemmatizer"])

    # Segmentación de oraciones barata
    if "senter" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    # Límite justo
    nlp.max_length = max(nlp.max_length, len(text) + 1)

    # POS ids
    POS = nlp.vocab.strings
    VERB = POS["VERB"]; AUX = POS["AUX"]; NOUN = POS["NOUN"]; PROPN = POS["PROPN"]
    ADJ = POS["ADJ"]; PRON = POS["PRON"]; ADV = POS["ADV"]

    # ---- Split en chunks ----
    def split_text(txt, max_chars):
        parts, start, n = [], 0, len(txt)
        while start < n:
            end = min(start + max_chars, n)
            cut = txt.rfind("\n\n", start, end)
            if cut == -1 or cut <= start + max_chars * 0.5:
                cut = end
            parts.append(txt[start:cut]); start = cut
        return parts

    chunks = split_text(text, chunk_chars) if len(text) > chunk_chars else [text]

    # ---- Acumuladores ----
    oraciones = 0
    palabras_por_oracion = []; verbos_list=[]; sustant_list=[]; adjetiv_list=[]
    pronomb_list=[]; adverb_list=[]; stopw_list=[]
    palabras_por_oracion_sw=[]; verbos_sw_list=[]; sustant_sw_list=[]
    adjetiv_sw_list=[]; pronomb_sw_list=[]; adverb_sw_list=[]

    # ---- Progreso ----
    pbar = None
    if tqdm is not None:
        pbar = tqdm(total=len(chunks), desc="Procesando chunks", unit="chunk")

    # ---- Pipe (streaming) ----
    try:
        for doc in nlp.pipe(chunks, n_process=n_process, batch_size=batch_size):
            for sent in doc.sents:
                w_total = 0; w_sw = 0
                v=s=a=p=r = 0
                v_sw=s_sw=a_sw=p_sw=r_sw = 0

                for t in sent:
                    if not t.is_alpha:
                        continue
                    w_total += 1
                    is_sw = t.is_stop
                    pos = t.pos

                    if pos == VERB or pos == AUX:
                        v += 1;        v_sw += (0 if is_sw else 1)
                    elif pos == NOUN or pos == PROPN:
                        s += 1;        s_sw += (0 if is_sw else 1)
                    elif pos == ADJ:
                        a += 1;        a_sw += (0 if is_sw else 1)
                    elif pos == PRON:
                        p += 1;        p_sw += (0 if is_sw else 1)
                    elif pos == ADV:
                        r += 1;        r_sw += (0 if is_sw else 1)

                    if is_sw:
                        w_sw += 1

                if w_total > 0 or w_sw > 0:
                    oraciones += 1
                    palabras_por_oracion.append(w_total)
                    verbos_list.append(v); sustant_list.append(s); adjetiv_list.append(a)
                    pronomb_list.append(p); adverb_list.append(r); stopw_list.append(w_sw)

                    palabras_por_oracion_sw.append(w_total - w_sw)
                    verbos_sw_list.append(v_sw); sustant_sw_list.append(s_sw); adjetiv_sw_list.append(a_sw)
                    pronomb_sw_list.append(p_sw); adverb_sw_list.append(r_sw)

            # update barra por doc
            if pbar is not None:
                # opcional: muestra métricas en vivo
                pbar.update(1)
                pbar.set_postfix(oraciones=oraciones)
            del doc
    finally:
        if pbar is not None:
            pbar.close()

    prom = lambda lst: float(mean(lst)) if lst else 0.0

    original = {
        "oraciones": oraciones,
        "cantidad_palabras": sum(palabras_por_oracion),
        "cantidad_palabras_sin_stopwords": sum(palabras_por_oracion) - sum(stopw_list),
        "promedio_palabras_por_oracion": prom(palabras_por_oracion),
        "promedio_verbos_por_oracion": prom(verbos_list),
        "promedio_sustantivos_por_oracion": prom(sustant_list),
        "promedio_adjetivos_por_oracion": prom(adjetiv_list),
        "promedio_pronombres_por_oracion": prom(pronomb_list),
        "promedio_adverbios_por_oracion": prom(adverb_list),
        "promedio_stopwords_por_oracion": prom(stopw_list),
        "total_verbos": sum(verbos_list),
        "total_sustantivos": sum(sustant_list),
        "total_adjetivos": sum(adjetiv_list),
        "total_pronombres": sum(pronomb_list),
        "total_adverbios": sum(adverb_list),
        "total_stopwords": sum(stopw_list),
    }

    sin_stopwords = {
        "oraciones": oraciones,
        "cantidad_palabras": sum(palabras_por_oracion_sw),
        "cantidad_palabras_sin_stopwords": sum(palabras_por_oracion_sw),
        "promedio_palabras_por_oracion": prom(palabras_por_oracion_sw),
        "promedio_verbos_por_oracion": prom(verbos_sw_list),
        "promedio_sustantivos_por_oracion": prom(sustant_sw_list),
        "promedio_adjetivos_por_oracion": prom(adjetiv_sw_list),
        "promedio_pronombres_por_oracion": prom(pronomb_sw_list),
        "promedio_adverbios_por_oracion": prom(adverb_sw_list),
        "promedio_stopwords_por_oracion": 0.0,
        "total_verbos": sum(verbos_sw_list),
        "total_sustantivos": sum(sustant_sw_list),
        "total_adjetivos": sum(adjetiv_sw_list),
        "total_pronombres": sum(pronomb_sw_list),
        "total_adverbios": sum(adverb_sw_list),
        "total_stopwords": 0,
    }

    return {"original": original, "sin_stopwords": sin_stopwords}


#############################################################################
# Preprocesamiento del texto
#############################################################################
def preprocesar_texto(texto: str) -> list:
    """
    Preprocesa un texto en español: 
    - Tokeniza
    - Elimina stop words y signos de puntuación
    - Lematiza
    
    Parámetros:
        texto (str): Texto de entrada.
    
    Retorna:
        list: Lista de tokens lematizados limpios.
    """
    import spacy
    from spacy.lang.es.stop_words import STOP_WORDS

    # Cargar el modelo de spaCy para español
    try:
        nlp = spacy.load("es_core_news_sm")
    except OSError:
        from spacy.cli import download
        download("es_core_news_sm")
        nlp = spacy.load("es_core_news_sm")
    
    # Procesar el texto
    doc = nlp(texto.lower())  # pasamos todo a minúsculas
    
    # Filtrar tokens: sin stopwords, sin puntuación y con longitud > 1
    oraciones_procesadas = []
    for sent in doc.sents:
        tokens_limpios = [
            token.lemma_
            for token in sent
            if token.is_alpha and token.lemma_ not in STOP_WORDS
        ]
        oraciones_procesadas.append(tokens_limpios)
    
    return oraciones_procesadas
