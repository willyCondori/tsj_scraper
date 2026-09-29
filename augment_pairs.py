"""
Paso 3 — Aumentación del dataset (entre extract_pairs.py y el fine-tuning).

Hace tres cosas:
1. BALANCE POR ARTÍCULO: antes de parafrasear, recorta los artículos
   sobrerrepresentados (undersampling) y decide cuánto parafraseo le toca
   a cada uno según cuántos ejemplos ORIGINALES tiene — a más escaso, más
   parafraseo; a más frecuente, menos o ninguno. Antes era un número fijo
   para todos, lo que hacía crecer más al artículo ya sobrerrepresentado
   (335: 4,195 ejemplos) mientras los raros (90 artículos con 1 solo
   ejemplo) apenas se beneficiaban.
2. PARAFRASEO POR SINÓNIMOS: genera variantes de cada "hechos" reemplazando
   términos por sinónimos de data/reference/sinonimos_juridicos.json,
   agregadas como positivos adicionales del mismo artículo.
3. NEGATIVOS DIFÍCILES: agrega, por referencia, 1-2 artículos de categorías
   de delito "cercanas pero distintas" (ROBO/HURTO, HOMICIDIO/LESIONES).

Uso:
    python augment_pairs.py
"""

import json
import os
import random
import re
from collections import defaultdict

PARES_PATH = os.path.join("data", "processed", "pares_entrenamiento.jsonl")
SINONIMOS_PATH = os.path.join("data", "reference", "sinonimos_juridicos.json")
OUT_PATH = os.path.join("data", "processed", "pares_aumentados.jsonl")

random.seed(42)  # reproducible: el undersampling y la elección de variantes/negativos no cambian entre corridas

# --- Objetivo de balance por artículo ---
# Un artículo con pocos ejemplos originales recibe MÁS parafraseo; uno
# con muchos recibe MENOS (o ninguno). Los umbrales son ajustables —
# calibrados para este dataset (art. más frecuente: 4,195 ejemplos;
# 90 artículos con 1 solo ejemplo).
OBJETIVO_MINIMO_POR_ARTICULO = 15    # se intenta acercar a esto vía parafraseo, para artículos escasos
MAXIMO_VARIANTES_POR_EJEMPLO = 8     # techo absoluto, para no generar variantes casi idénticas en exceso
MAXIMO_ORIGINALES_POR_ARTICULO = 300  # artículos con más originales que esto se recortan (undersampling)

CATEGORIAS_DELITO = {
    "ROBO": ["robo", "asalto", "atraco"],
    "HURTO": ["hurto", "sustracción", "sustrajo sin violencia"],
    "HOMICIDIO": ["homicidio", "muerte", "occiso", "falleció"],
    "LESIONES": ["lesiones", "heridas", "daño corporal"],
    "VIOLACION": ["violación", "acceso carnal violento", "abuso sexual"],
    "ESTAFA": ["estafa", "engaño", "ardid"],
    "AMENAZAS": ["amenaza", "intimidación", "coacción"],
    "SECUESTRO": ["secuestro", "privación de libertad"],
}


def cargar_sinonimos(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_comentario", None)
    return data


def cargar_pares(path: str) -> list:
    pares = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            pares.append(json.loads(linea))
    return pares


def parafrasear(texto: str, sinonimos: dict, n_variantes: int) -> list:
    texto_lower = texto.lower()
    terminos_presentes = [t for t in sinonimos if t.replace("_", " ") in texto_lower]
    if not terminos_presentes:
        return []

    variantes = []
    for _ in range(n_variantes):
        termino = random.choice(terminos_presentes)
        reemplazo = random.choice(sinonimos[termino])
        patron = re.compile(re.escape(termino.replace("_", " ")), re.IGNORECASE)
        variante = patron.sub(reemplazo, texto, count=1)
        if variante != texto:
            variantes.append(variante)
    return list(set(variantes))


def variantes_necesarias(cantidad_original: int) -> int:
    """
    Cuántas variantes parafraseadas generar POR CADA ejemplo original de
    un artículo, según cuántos ejemplos tiene ese artículo en total
    (antes del undersampling — la decisión se toma sobre la frecuencia
    real del corpus, no sobre lo que sobrevive al recorte).
    """
    if cantidad_original >= 50:
        return 0  # ya tiene de sobra, no lo infles más
    if cantidad_original >= 20:
        return 1
    if cantidad_original >= 5:
        return 3
    # Muy escaso: tratar de acercarse a OBJETIVO_MINIMO_POR_ARTICULO,
    # sin pasarse de MAXIMO_VARIANTES_POR_EJEMPLO por ejemplo individual
    # (generar 15 parafraseos casi idénticos de UN solo texto original
    # no agrega variedad real, solo repite el mismo patrón de sinónimos).
    faltante = OBJETIVO_MINIMO_POR_ARTICULO - cantidad_original
    return max(1, min(MAXIMO_VARIANTES_POR_EJEMPLO, -(-faltante // max(cantidad_original, 1))))


def categoria_de(par: dict) -> str:
    texto_tipo_proceso = (par.get("tipo_proceso") or "").lower()
    texto_hechos = par.get("hechos", "").lower()
    texto_busqueda = f"{texto_tipo_proceso} {texto_hechos}"
    for categoria, palabras_clave in CATEGORIAS_DELITO.items():
        if any(p in texto_busqueda for p in palabras_clave):
            return categoria
    return "OTRO"


def construir_indice_por_categoria(pares: list) -> dict:
    indice = {}
    for par in pares:
        cat = categoria_de(par)
        indice.setdefault(cat, set()).add(par["articulo"])
    return indice


def elegir_negativo_dificil(par: dict, indice_categorias: dict) -> list:
    cat_actual = categoria_de(par)
    cercanas = {
        "ROBO": ["HURTO"], "HURTO": ["ROBO"],
        "HOMICIDIO": ["LESIONES"], "LESIONES": ["HOMICIDIO"],
        "VIOLACION": ["AMENAZAS"], "AMENAZAS": ["VIOLACION", "SECUESTRO"],
        "SECUESTRO": ["AMENAZAS"],
        "ESTAFA": ["HURTO"],
    }.get(cat_actual, [])

    negativos = []
    for cat_cercana in cercanas:
        candidatos = indice_categorias.get(cat_cercana, set()) - {par["articulo"]}
        if candidatos:
            negativos.append(random.choice(list(candidatos)))
    return negativos


def balancear_originales(pares: list) -> tuple:
    """
    Undersampling: recorta los artículos con más de
    MAXIMO_ORIGINALES_POR_ARTICULO ejemplos originales a ese techo, elegidos
    al azar (con semilla fija). Devuelve (pares_recortados, conteo_original)
    — el conteo ORIGINAL (antes del recorte) es el que se usa después para
    decidir cuánto parafraseo aplicar, porque refleja la frecuencia real
    del corpus, no lo que sobrevivió al undersampling.
    """
    por_articulo = defaultdict(list)
    for par in pares:
        por_articulo[par["articulo"]].append(par)

    conteo_original = {articulo: len(lista) for articulo, lista in por_articulo.items()}

    recortados = []
    for articulo, lista in por_articulo.items():
        if len(lista) > MAXIMO_ORIGINALES_POR_ARTICULO:
            recortados.extend(random.sample(lista, MAXIMO_ORIGINALES_POR_ARTICULO))
        else:
            recortados.extend(lista)

    return recortados, conteo_original


def main():
    if not os.path.exists(PARES_PATH):
        print(f"No encontré {PARES_PATH}. Corré primero extract_pairs.py.")
        return

    sinonimos = cargar_sinonimos(SINONIMOS_PATH)
    pares = cargar_pares(PARES_PATH)
    indice_categorias = construir_indice_por_categoria(pares)

    pares_balanceados, conteo_original = balancear_originales(pares)
    removidos_por_undersampling = len(pares) - len(pares_balanceados)

    dataset_final = []
    total_parafraseos = 0

    for par in pares_balanceados:
        registro = dict(par)
        registro["tipo"] = "original"
        registro["negativos_dificiles"] = elegir_negativo_dificil(par, indice_categorias)
        dataset_final.append(registro)

        n_variantes = variantes_necesarias(conteo_original[par["articulo"]])
        if n_variantes > 0:
            variantes = parafrasear(par["hechos"], sinonimos, n_variantes)
            for variante in variantes:
                nuevo = dict(par)
                nuevo["hechos"] = variante
                nuevo["tipo"] = "parafraseado"
                nuevo["negativos_dificiles"] = registro["negativos_dificiles"]
                dataset_final.append(nuevo)
                total_parafraseos += 1

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for registro in dataset_final:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    # --- Reporte de balance, antes y después ---
    conteo_final = defaultdict(int)
    for registro in dataset_final:
        conteo_final[registro["articulo"]] += 1

    articulos_con_1_ejemplo_antes = sum(1 for c in conteo_original.values() if c == 1)
    articulos_con_1_ejemplo_despues = sum(1 for c in conteo_final.values() if c == 1)
    max_articulo, max_cantidad = max(conteo_final.items(), key=lambda x: x[1])

    print(f"Pares originales:                    {len(pares)}")
    print(f"  - Removidos por undersampling:      {removidos_por_undersampling}")
    print(f"Variantes parafraseadas:             {total_parafraseos}")
    print(f"Total en dataset aumentado:          {len(dataset_final)}")
    print(f"\nArtículo más frecuente ahora:        {max_articulo} ({max_cantidad} ejemplos, "
          f"{max_cantidad/len(dataset_final)*100:.2f}% del total)")
    print(f"Artículos con 1 solo ejemplo — antes: {articulos_con_1_ejemplo_antes}  "
          f"después: {articulos_con_1_ejemplo_despues}")
    print(f"\nGuardado en: {OUT_PATH}")


if __name__ == "__main__":
    main()