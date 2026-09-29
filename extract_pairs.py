"""
Paso 2 — Construcción del dataset de entrenamiento.

Lee los JSON crudos en data/raw/, y para cada resolución:
  1. Detecta los artículos del Código Penal citados en el texto
     (ej. "artículo 273 del Código Penal", "Art. 20 CP", "arts. 251-252 CP",
     con soporte para sufijos "bis"/"ter"/"quater"), con cuidado de NO
     confundirlos con artículos del CPP (Código de Procedimiento Penal).
  2. Para CADA cita encontrada, extrae una VENTANA de texto alrededor de
     esa cita puntual (no el documento completo, y no un único fragmento
     compartido por todas las citas del documento) — el mismo defecto que
     ya se corrigió en el chunking de casos de producción, pero del lado
     del dataset: reusar 8,000+ caracteres como "hecho" para hasta 32
     artículos distintos citados en un mismo documento le enseña al
     modelo asociaciones sin relación real caso-artículo.
  3. Clasifica cada ventana como "factico", "mixto" o "procesal" según
     predomine narración de hechos o debate de admisibilidad del recurso
     — la clasificación es POR CITA, no por documento, porque el mismo
     documento puede tener secciones fácticas y secciones puramente
     procesales.
  4. Genera un par (ventana_de_hecho, articulo_citado) por cada cita.

Salidas:
  - data/processed/pares_entrenamiento.jsonl
      Pares con contenido fáctico o mixto — este es el dataset que se usa
      para el fine-tuning.
  - data/processed/pares_descartados_procesales.jsonl
      Pares donde la ventana es predominantemente debate procesal de
      admisibilidad (no hechos reales) — se guardan aparte por
      trazabilidad, pero NO se usan para entrenar.
  - data/processed/pares_revision.csv
      Mismo contenido que pares_entrenamiento.jsonl, en CSV con BOM
      (utf-8-sig) para que Excel muestre bien los acentos.

Uso:
    python extract_pairs.py
"""

import glob
import json
import os
import re
import csv

RAW_DIR = os.path.join("data", "raw")
OUT_DIR = os.path.join("data", "processed")

# --- Ventana de contexto alrededor de cada cita de artículo ---
# Más texto ANTES que DESPUÉS: en la redacción legal boliviana, el hecho
# o el delito que motiva la cita casi siempre se describe justo antes de
# ella ("...por el delito de Estafa, previsto y sancionado por el art.
# 335 del CP"), mientras que lo que sigue después de la cita suele ser
# ya otro tema (otro artículo, u otra parte del trámite).
VENTANA_ANTES = 600
VENTANA_DESPUES = 150

# --- Regex para detectar citas de artículos del Código Penal ---
PATRON_ARTICULO = re.compile(
    r"""
    (?:art[íi]culos?|arts?\.?)\s+          # "artículo", "artículos", "art.", "arts."
    (?P<numeros>\d+\s*(?:bis|ter|quater|quinquies)?
        (?:\s*[ºo°]?\s*)?
        (?:\s*(?:,|y|al|-)\s*\d+\s*(?:bis|ter|quater|quinquies)?)*)
    \s*(?:del|de\s+la)?\s*
    (?P<norma>
        c[óo]digo\s+penal
        | c\.?\s*p\.?
    )
    (?!\s*[A-Za-zÁÉÍÓÚáéíóúñÑ])          # evita matchear "CPP" (Cód. de
                                          # Procedimiento Penal) como si
                                          # fuera "CP" (Código Penal):
                                          # exige que NO siga otra letra
                                          # pegada (ej. la segunda "P")
    """,
    re.IGNORECASE | re.VERBOSE,
)

# --- Marcadores de corte: dónde empieza el trámite del recurso ---
# Ya no se usan para recortar el "hecho" global (eso lo reemplaza la
# ventana por cita), pero se mantienen para PATRON_ARTICULO no confunda
# nada — se dejan solo por si en el futuro hace falta descartar de raíz
# resoluciones enteras sin ninguna cita útil.
MARCADORES_INICIO_RECURSO = [
    r"I\.\s*DEL\s+RECURSO",
    r"DEL\s+RECURSO\s+DE\s+CASACI[ÓO]N",
    r"MOTIVOS?\s+DEL\s+RECURSO",
]
PATRON_INICIO_RECURSO = re.compile("|".join(MARCADORES_INICIO_RECURSO), re.IGNORECASE)

MARCADORES_FIN_HECHOS = [
    r"CONSIDERANDO",
    r"FUNDAMENTOS?\s+(DE\s+)?DERECHO",
    r"POR\s+TANTO",
    r"DOCTRINA\s+LEGAL",
]
PATRON_FIN_HECHOS = re.compile("|".join(MARCADORES_FIN_HECHOS), re.IGNORECASE)

PATRON_INICIO_VISTOS = re.compile(r"VISTOS\s*:?", re.IGNORECASE)

PATRON_HTML = re.compile(r"<[^>]+>")

# --- Filtro de calidad: hechos reales vs. debate procesal ---
#
# "recurso de casación" y "auto de vista" NO están en esta lista a
# propósito: son frases de apertura estándar de CUALQUIER Auto Supremo de
# este corpus (todos resuelven un recurso de casación contra un Auto de
# Vista), así que no distinguen nada — solo inflaban artificialmente el
# conteo procesal y hacían que casi todo se clasificara como "procesal".
PALABRAS_PROCESALES = [
    "apelación restringida", "apelacion restringida",
    "precedente contradictorio",
    "admisibilidad", "admisible", "inadmisible",
    "art. 416", "art. 417", "arts. 416", "arts. 417",
    "requisitos de admisión", "requisitos de admision",
    "tribunal de alzada", "tribunal de casación", "tribunal de casacion",
    "fundamentación de la sentencia", "fundamentacion de la sentencia",
    "defecto absoluto", "defectos absolutos",
    "nulidad", "reposición del juicio", "reposicion del juicio",
]

PALABRAS_FACTICAS = [
    "sustrajo", "sustrajeron", "se apoderó", "se apodero",
    "golpeó", "golpeo", "agredió", "agredio",
    "amenazó", "amenazo", "disparó", "disparo",
    "ingresó a", "ingreso a", "ingresó al", "ingreso al",
    "portaba", "utilizando un arma", "con violencia",
    "falleció", "fallecio", "murió", "murio", "causó la muerte", "causo la muerte",
    "se hizo entrega", "entregó dinero", "entrego dinero",
    "el día", "el dia", "aproximadamente a horas", "en la vía pública",
    "en la via publica", "en su domicilio",
]


def _normalizar_numero_articulo(numero: str) -> str:
    """
    '308bis', '308 Bis', '308 BIS' -> '308 bis'. Uniforma mayúsculas y
    asegura un solo espacio antes del sufijo, para que todas las
    variantes de un mismo artículo con "bis"/"ter"/"quater"/"quinquies"
    se traten como IGUALES tanto al contar pares como al unir con el
    catálogo de Django.
    """
    numero = numero.strip()
    match = re.match(r"(\d+)\s*(bis|ter|quater|quinquies)?", numero, re.IGNORECASE)
    if not match:
        return numero
    base, sufijo = match.groups()
    return f"{base} {sufijo.lower()}" if sufijo else base


def extraer_articulos_citados(texto_completo: str):
    """
    Devuelve una lista de dicts {"numero", "norma", "inicio", "fin"} con
    cada artículo del Código Penal citado en el texto, sin duplicados de
    (número normalizado). "inicio"/"fin" son las posiciones del match en
    texto_completo — se usan para recortar la ventana de contexto de esa
    cita puntual en extraer_ventana_hecho().
    """
    encontrados = []
    vistos = set()

    for match in PATRON_ARTICULO.finditer(texto_completo):
        numeros_raw = match.group("numeros")
        for numero_match in re.finditer(r"\d+\s*(?:bis|ter|quater|quinquies)?", numeros_raw, re.IGNORECASE):
            numero = _normalizar_numero_articulo(numero_match.group(0))
            if numero in vistos:
                continue
            vistos.add(numero)
            encontrados.append({
                "numero": numero,
                "norma": "Codigo Penal",
                "inicio": match.start(),
                "fin": match.end(),
            })

    return encontrados


def extraer_ventana_hecho(texto_completo: str, inicio: int, fin: int) -> str:
    """
    Recorta VENTANA_ANTES caracteres antes de la cita y VENTANA_DESPUES
    después, en vez de reusar el documento entero (o un único fragmento
    compartido) como "hecho" para cada artículo citado. Después le saca
    cualquier cita explícita a artículo que haya quedado adentro de la
    ventana — tanto la cita que motivó esta ventana como cualquier OTRA
    cita cercana — para que el modelo no aprenda a copiar el número en
    vez de entender el contexto.
    """
    desde = max(0, inicio - VENTANA_ANTES)
    hasta = min(len(texto_completo), fin + VENTANA_DESPUES)
    ventana = texto_completo[desde:hasta]

    ventana = PATRON_ARTICULO.sub("", ventana)
    ventana = re.sub(r"\s+", " ", ventana).strip()
    return ventana


def clasificar_contenido_hechos(texto_hechos: str) -> dict:
    """
    Cuenta ocurrencias de vocabulario procesal vs. fáctico en la ventana
    de "hecho" ya recortada, y devuelve una clasificación orientativa
    junto con los conteos.
    """
    texto_lower = texto_hechos.lower()

    conteo_procesal = sum(texto_lower.count(p) for p in PALABRAS_PROCESALES)
    conteo_factico = sum(texto_lower.count(p) for p in PALABRAS_FACTICAS)

    if conteo_procesal == 0 and conteo_factico == 0:
        tipo = "indeterminado"
    elif conteo_procesal >= 2 and conteo_procesal > conteo_factico * 2:
        tipo = "procesal"
    elif conteo_factico >= conteo_procesal:
        tipo = "factico"
    else:
        tipo = "mixto"

    return {
        "tipo_contenido": tipo,
        "señales_procesales": conteo_procesal,
        "señales_facticas": conteo_factico,
    }


def procesar_archivo(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texto_completo = data.get("contenido", "")
    if not texto_completo:
        return []

    # Una porción del corpus (resoluciones más recientes) trae el
    # contenido en HTML crudo (<p style="...">, <span>, etc.) en vez de
    # texto plano — hay que limpiarlo antes de cualquier otro
    # procesamiento, si no, el ruido de markup contamina tanto la
    # detección de citas como las ventanas de contexto.
    texto_completo = PATRON_HTML.sub(" ", texto_completo)
    texto_completo = re.sub(r"\s+", " ", texto_completo).strip()

    articulos = extraer_articulos_citados(texto_completo)
    tipo_proceso = data.get("procesos", "")

    pares = []
    for art in articulos:
        hecho_ventana = extraer_ventana_hecho(texto_completo, art["inicio"], art["fin"])
        if not hecho_ventana:
            continue

        clasificacion = clasificar_contenido_hechos(hecho_ventana)

        pares.append({
            "hechos": hecho_ventana,
            "articulo": art["numero"],
            "norma": art["norma"],
            "tipo_proceso": tipo_proceso,
            "fuente_id": data.get("id"),
            "nro_resolucion": data.get("nro_resolucion"),
            "sala": data.get("sala"),
            "materia": data.get("materia"),
            **clasificacion,
        })
    return pares


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    archivos = glob.glob(os.path.join(RAW_DIR, "*.json"))
    print(f"Archivos crudos encontrados: {len(archivos)}")

    if not archivos:
        print("No hay archivos en data/raw/. Corré primero scraper_api.py.")
        return

    todos_los_pares = []
    archivos_sin_articulos = 0

    for path in archivos:
        pares = procesar_archivo(path)
        if not pares:
            archivos_sin_articulos += 1
        todos_los_pares.extend(pares)

    print(f"Resoluciones sin ningún artículo detectado: {archivos_sin_articulos}")
    print(f"Pares (hechos, artículo) generados en total: {len(todos_los_pares)}")

    # Métrica de validación del fix de dilución: promedio de caracteres
    # por ventana de "hecho". Antes de este cambio rondaba 8,611 (todo el
    # documento reusado); con la ventana acotada, debería estar cerca de
    # VENTANA_ANTES + VENTANA_DESPUES (unos 750), salvo los casos donde
    # el recorte queda más corto por estar cerca del inicio/fin del texto.
    if todos_los_pares:
        promedio_caracteres = sum(len(p["hechos"]) for p in todos_los_pares) / len(todos_los_pares)
        print(f"Longitud promedio de la ventana de hecho: {promedio_caracteres:.0f} caracteres")

    pares_buenos = [p for p in todos_los_pares if p["tipo_contenido"] in ("factico", "mixto", "indeterminado")]
    pares_procesales = [p for p in todos_los_pares if p["tipo_contenido"] == "procesal"]

    print(f"  - Con contenido fáctico/mixto/indeterminado (van al dataset): {len(pares_buenos)}")
    print(f"  - Predominantemente procesales (separados aparte): {len(pares_procesales)}")

    campos = [
        "hechos", "articulo", "norma", "tipo_proceso", "fuente_id",
        "nro_resolucion", "sala", "materia",
        "tipo_contenido", "señales_procesales", "señales_facticas",
    ]

    jsonl_path = os.path.join(OUT_DIR, "pares_entrenamiento.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for par in pares_buenos:
            f.write(json.dumps(par, ensure_ascii=False) + "\n")

    jsonl_procesales_path = os.path.join(OUT_DIR, "pares_descartados_procesales.jsonl")
    with open(jsonl_procesales_path, "w", encoding="utf-8") as f:
        for par in pares_procesales:
            f.write(json.dumps(par, ensure_ascii=False) + "\n")

    csv_path = os.path.join(OUT_DIR, "pares_revision.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        for par in pares_buenos:
            writer.writerow(par)

    print(f"\nGuardado (dataset de entrenamiento): {jsonl_path}")
    print(f"Guardado (descartados, solo referencia): {jsonl_procesales_path}")
    print(f"Guardado: {csv_path}")
    print(
        "\nRecomendación: abrí pares_revision.csv y revisá al menos 30-50 "
        "filas al azar antes de usar el dataset para entrenar. Las ventanas "
        "ahora deberían ser mucho más cortas y específicas que antes — si "
        "ves fragmentos que siguen pareciendo el documento completo, avisá."
    )


if __name__ == "__main__":
    main()