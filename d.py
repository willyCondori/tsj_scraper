import os
import re
import random
import json

ARTICULOS_CATALOGO_PATH = os.path.join("data", "reference", "articulos_codigo_penal.json")
VENTANA_ARTICULOS_CERCANOS = 8  # cuántos artículos antes/después, en la numeración del CP, se consideran "vecinos"


def _numero_ordenable(numero: str) -> tuple:
    """
    '308' -> (308, 0); '308 bis' -> (308, 1); '308 ter' -> (308, 2).
    Permite ordenar los artículos del Código Penal en el mismo orden en
    que aparecen en el texto real (el 308 bis va justo después del 308,
    no al final de la lista).
    """
    orden_sufijo = {"": 0, "bis": 1, "ter": 2, "quater": 3, "quinquies": 4, "sexies": 5, "septies": 6}
    match = re.match(r"(\d+)\s*(bis|ter|quater|quinquies|sexies|septies)?", numero.strip(), re.IGNORECASE)
    if not match:
        return (999999, 0)
    base, sufijo = match.groups()
    return (int(base), orden_sufijo.get((sufijo or "").lower(), 0))


def construir_orden_catalogo_cp(path: str) -> list:
    """Lista de números de artículo del CP, en el orden real del Código."""
    with open(path, "r", encoding="utf-8") as f:
        articulos = json.load(f)
    numeros_cp = [a["numero_articulo"] for a in articulos if a.get("norma_sigla", "").upper() == "CP"]
    return sorted(set(numeros_cp), key=_numero_ordenable)


def construir_vecinos_por_articulo(orden_catalogo: list, ventana: int) -> dict:
    """{'335': ['330','331',...,'340'], ...} — los artículos numéricamente
    cercanos a cada uno, dentro de VENTANA_ARTICULOS_CERCANOS posiciones."""
    vecinos = {}
    for i, numero in enumerate(orden_catalogo):
        desde = max(0, i - ventana)
        hasta = min(len(orden_catalogo), i + ventana + 1)
        vecinos[numero] = [orden_catalogo[j] for j in range(desde, hasta) if j != i]
    return vecinos


def elegir_negativo_dificil(par: dict, vecinos_por_articulo: dict) -> list:
    """Negativos difíciles = artículos numéricamente vecinos en el CP —
    misma zona del Código, probablemente mismo capítulo/título, texto
    con vocabulario jurídico parecido. Cubre el 100% de los artículos,
    a diferencia de las categorías por palabra clave (57% caía en
    'OTRO' y se quedaba sin ningún negativo)."""
    candidatos = vecinos_por_articulo.get(par["articulo"], [])
    if not candidatos:
        return []
    return random.sample(candidatos, min(2, len(candidatos)))


