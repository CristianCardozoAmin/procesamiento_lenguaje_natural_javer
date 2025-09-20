import re
from typing import Dict, List, Pattern, Tuple, Optional
import pandas as pd

#############################################################################
# Lectura de PDF
#############################################################################
def _read_pdf_text_pymupdf(path: str, start_after_page_1based: int) -> Optional[str]:
    """
    Intenta leer con PyMuPDF (fitz) si está disponible (suele extraer mejor).
    Devuelve None si no está instalado o si falla.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    try:
        with fitz.open(path) as doc:
            start_idx0 = max(0, min(start_after_page_1based, len(doc)))  # 1-based -> 0-based ya 'después'
            parts: List[str] = []
            for i in range(start_idx0, len(doc)):
                try:
                    parts.append(doc[i].get_text() or "")
                except Exception:
                    parts.append("")
            return "\n".join(parts)
    except Exception:
        return None
    
def _read_pdf_text_pypdf2(path: str, start_after_page_1based: int) -> str:
    """
    Lectura de respaldo con PyPDF2.
    """
    import PyPDF2  # lazy import para no requerirlo si se usa solo fitz
    text_pages: List[str] = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        # 1-based "después de" => arrancar en ese índice mismo (0-based)
        start_idx0 = max(0, min(start_after_page_1based, len(reader.pages)))
        for i in range(start_idx0, len(reader.pages)):
            try:
                t = reader.pages[i].extract_text() or ""
            except Exception:
                t = ""
            text_pages.append(t)
    return "\n".join(text_pages)

def read_pdf_text(path: str, start_after_page_1based: int) -> str:
    """
    Lee el PDF a partir de la página *siguiente* a start_after_page_1based (1-based).
    Prioriza PyMuPDF si está disponible; si no, usa PyPDF2.
    """
    txt = _read_pdf_text_pymupdf(path, start_after_page_1based)
    if txt is not None:
        return txt
    return _read_pdf_text_pypdf2(path, start_after_page_1based)

#############################################################################
# Normalización y utilidades
#############################################################################
def normalize_text(txt: str) -> str:
    """
    - Repara palabras cortadas por guion al final de línea (e.g., 'muje-\\nres' -> 'mujeres').
    - Homogeneiza saltos de línea y espacios.
    """
    # Unir palabras cortadas por guion al final de línea
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    # Normalizar saltos y espacios
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()


def make_pattern(title: str) -> Pattern:
    """
    Crea un patrón regex robusto para encontrar el título:
    - Insensible a mayúsculas/minúsculas
    - Tolera múltiples espacios/saltos entre palabras
    - Tolera espacios alrededor de ':'
    """
    parts: List[str] = []
    for ch in title.strip():
        if ch.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(ch))
    pat = "".join(parts).replace(r"\:", r"\s*:\s*")
    return re.compile(pat, flags=re.IGNORECASE | re.DOTALL)


def find_once(txt: str, title: str) -> Optional[int]:
    m = make_pattern(title).search(txt)
    return m.start() if m else None


def slice_after_heading(txt: str, start_pos: int, end_pos: int, heading_title: str) -> str:
    """
    Devuelve el contenido de la sección SIN el rótulo del encabezado.
    Si el encabezado aparece al inicio (±5 chars), lo recorta.
    """
    chunk = txt[start_pos:end_pos]
    m = make_pattern(heading_title).search(chunk)
    if m and m.start() <= 5:
        return chunk[m.end():].lstrip()
    return chunk.strip()

#############################################################################
# Lógica principal de extracción por schema
#############################################################################
def _collect_anchor_titles(schema: Dict) -> List[str]:
    """
    Recolecta anclas (títulos) en orden estable a partir del schema.
    Respeta keys especiales: '_end_before' y '_boundaries'.
    """
    ordered: List[str] = []
    # orden de las secciones superiores tal como vienen en el dict
    for top in schema.keys():
        # límites de cierre y boundaries
        end_before = schema[top].get("_end_before")
        if end_before:
            ordered.append(end_before)
        for b in schema[top].get("_boundaries", []):
            ordered.append(b)
        # segundo nivel y tercer nivel
        for lvl2, lvl3_list in schema[top].items():
            if str(lvl2).startswith("_"):
                continue
            ordered.append(lvl2)
            ordered.extend(lvl3_list)

    # quitar duplicados preservando orden
    seen, out = set(), []
    for t in ordered:
        if t not in seen and isinstance(t, str):
            seen.add(t)
            out.append(t)
    return out


def _build_positions_index(text: str, titles: List[str]) -> Tuple[dict, List[int], List[str]]:
    title2pos: Dict[str, int] = {}
    missing: List[str] = []
    for t in titles:
        p = find_once(text, t)
        if p is None:
            missing.append(t)
        else:
            title2pos[t] = p
    all_positions_sorted = sorted(title2pos.values())
    return title2pos, all_positions_sorted, missing


def _next_end_pos(all_positions_sorted: List[int], start_pos: int, text_len: int) -> int:
    for p in all_positions_sorted:
        if p > start_pos:
            return p
    return text_len


def extract_dataframe_from_pdf(
    pdf_path: str,
    start_after_page: int,
    schema: Dict,
    drop_empty: bool = False,
) -> pd.DataFrame:
    """
    Extrae un DataFrame limpio según el schema dado.

    Parámetros
    ----------
    pdf_path : str
        Ruta del PDF.
    start_after_page : int
        Página 1-based después de la cual iniciar la lectura (p.ej., 37 => inicia en 38).
    schema : Dict
        Estructura jerárquica:
        {
          "Mujeres": {
            "¿Qué pasó?...": ["Sub1", "Sub2", ...],
            "_end_before": "Epílogo",
            "_boundaries": ["Otro límite 1", "Otro límite 2"]
          },
          "LGBTIQ+": { ... }
        }
    drop_empty : bool
        Si True, elimina filas cuyo 'contenido' esté vacío.

    Returns
    -------
    pd.DataFrame
        Columnas: ['seccion', 'seccion_principal', 'subseccion', 'contenido']
        Mantiene el orden del schema.
    """
    # 0) Leer y normalizar texto
    raw_text = read_pdf_text(pdf_path, start_after_page)
    text = normalize_text(raw_text)

    # 1) Colectar anclas e indexar posiciones
    anchor_titles = _collect_anchor_titles(schema)
    title2pos, all_positions_sorted, missing_anchors = _build_positions_index(text, anchor_titles)

    # 2) Recorrer el schema y cortar bloques
    rows: List[dict] = []
    text_len = len(text)

    for top in schema.keys():
        # Orden natural de lvl2 respetando el dict
        lvl2_order = [k for k in schema[top].keys() if not str(k).startswith("_")]
        for lvl2 in lvl2_order:
            for lvl3 in schema[top][lvl2]:
                start = title2pos.get(lvl3)
                if start is None:
                    content = ""
                else:
                    end = _next_end_pos(all_positions_sorted, start, text_len)
                    content = slice_after_heading(text, start, end, lvl3)

                    # Protección extra: cortar si aparece el "_end_before" dentro del bloque
                    end_before = schema[top].get("_end_before")
                    if isinstance(end_before, str):
                        cut_inside = find_once(content, end_before)
                        if cut_inside is not None:
                            content = content[:cut_inside].rstrip()

                rows.append({
                    "seccion": str(top),
                    "seccion_principal": str(lvl2),
                    "subseccion": str(lvl3),
                    "contenido": (content or "").strip()
                })

    df = pd.DataFrame(rows, columns=["seccion", "seccion_principal", "subseccion", "contenido"])

    if drop_empty:
        df = df[df["contenido"].str.len() > 0].reset_index(drop=True)

    df.attrs["missing_anchors"] = missing_anchors
    return df