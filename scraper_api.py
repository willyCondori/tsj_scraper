"""
Scraper directo contra la API de Genesis (TSJ Bolivia) — SIN Playwright.

Endpoints reales (confirmados inspeccionando el tráfico de red del sitio):
  - Listado:  POST https://apigenesis.tsj.bo/api/v1/resoluciones/busqueda_avanzada
  - Detalle:  GET  https://apigenesis.tsj.bo/api/v1/resoluciones/{id}

El listado se filtra con idMateria="1" (Materia "Penal", string no int —
los <select> de HTML mandan el valor como texto). La API exige además que
searchData tenga al menos un campo no vacío (error 422 si no), por eso
usamos "penal" como palabra de búsqueda por defecto.

Uso:
    python scraper_api.py
"""

import json
import os
import time
import argparse
from pathlib import Path

import requests

OUT_DIR = str(Path(__file__).resolve().parent / "data" / "raw")

BASE_URL = "https://apigenesis.tsj.bo/api/v1"
URL_BUSQUEDA = f"{BASE_URL}/resoluciones/busqueda_avanzada"
URL_DETALLE = f"{BASE_URL}/resoluciones/{{id}}"

# Materia "Penal" según /api/v1/catalogos/materias (id: 1, nombre: "Penal")
# OJO: se manda como STRING, no como número.
ID_MATERIA_PENAL = "1"

RESULTADOS_POR_PAGINA = 50
DELAY_ENTRE_REQUESTS = 1.5  # segundos, para no saturar el servidor público

HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "es-ES,es;q=0.9",
    "apikey": "CiAYFxnN4GwYgtDv+0jo8MSm1VuTZ53ah8aJ2L8GkgI=",
    "content-type": "application/json",
    "origin": "https://genesis.tsj.bo",
    "username": "buscadorgenesis",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36"
    ),
}


def cuerpo_busqueda(page: int, todas_estas_palabras: str = None) -> dict:
    return {
        "searchData": {
            "todasEstasPalabras": todas_estas_palabras or "penal",
            "estaPalabraOFraseExacta": None,
            "cualquieraEstasPalabras": None,
            "ningunaEstasPalabras": None,
        },
        "filter": {
            "idSala": None,
            "gestion": None,
            "idTipoResolucion": None,
            "idMagistrado": None,
            "idDepartamento": None,
            "idMateria": ID_MATERIA_PENAL,
            "idTipoProceso": None,
            "idFormaResolucion": None,
        },
        "paginate": {"page": page, "limit": RESULTADOS_POR_PAGINA},
    }


def ya_descargado(id_resolucion) -> bool:
    path = Path(OUT_DIR) / f"{id_resolucion}.json"
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("id")) == str(id_resolucion) and bool(data.get("contenido"))
    except (ValueError, OSError):
        return False


def guardar_resolucion(id_resolucion, data: dict):
    path = os.path.join(OUT_DIR, f"{id_resolucion}.json")
    if str(data.get("id")) != str(id_resolucion) or not data.get("contenido"):
        raise ValueError(f"Detalle inválido: {id_resolucion}")
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp, path)


def _post_con_reintentos(url, headers, json_body, intentos=4):
    """
    POST con reintentos y backoff exponencial para errores TRANSITORIOS de
    red (cortes de DNS/wifi, timeouts, 5xx). Los 4xx (ej. el 422 de
    validación) NO se reintentan porque son permanentes.
    """
    espera = 3
    resp = None
    for intento in range(1, intentos + 1):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=30)
            if (resp.status_code >= 500 or resp.status_code == 429) and intento < intentos:
                print(f"  [!] Error {resp.status_code} del servidor, reintentando en {espera}s...")
                time.sleep(espera)
                espera *= 2
                continue
            return resp
        except requests.exceptions.RequestException as e:
            if intento == intentos:
                raise
            print(f"  [!] Error de red ({type(e).__name__}), reintentando en {espera}s... (intento {intento}/{intentos})")
            time.sleep(espera)
            espera *= 2
    return resp


def _get_con_reintentos(url, headers, intentos=4):
    espera = 3
    resp = None
    for intento in range(1, intentos + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if (resp.status_code >= 500 or resp.status_code == 429) and intento < intentos:
                print(f"  [!] Error {resp.status_code} del servidor, reintentando en {espera}s...")
                time.sleep(espera)
                espera *= 2
                continue
            return resp
        except requests.exceptions.RequestException as e:
            if intento == intentos:
                raise
            print(f"  [!] Error de red ({type(e).__name__}), reintentando en {espera}s... (intento {intento}/{intentos})")
            time.sleep(espera)
            espera *= 2
    return resp


def obtener_pagina_listado(page: int) -> dict:
    resp = _post_con_reintentos(URL_BUSQUEDA, HEADERS, cuerpo_busqueda(page))
    if not resp.ok:
        print(f"\n[!] Error {resp.status_code} del servidor. Respuesta completa:")
        print(resp.text[:2000])
        print()
    resp.raise_for_status()
    return resp.json()


def obtener_detalle(id_resolucion) -> dict:
    resp = _get_con_reintentos(URL_DETALLE.format(id=id_resolucion), HEADERS)
    if not resp.ok:
        print(f"\n[!] Error {resp.status_code} del servidor en detalle {id_resolucion}:")
        print(resp.text[:2000])
        print()
    resp.raise_for_status()
    return resp.json()


def main(max_paginas: int = None):
    os.makedirs(OUT_DIR, exist_ok=True)

    primera = obtener_pagina_listado(1)
    meta = primera["data"]["meta"]
    total_paginas = meta["totalPages"]
    total_resultados = meta["count"]

    print(f"Total de resoluciones con Materia Penal: {total_resultados}")
    print(f"Total de páginas ({RESULTADOS_POR_PAGINA} por página): {total_paginas}")

    if max_paginas:
        total_paginas = min(total_paginas, max_paginas)
        print(f"Limitando a {total_paginas} páginas para esta corrida.")

    total_descargadas = 0
    total_saltadas = 0
    total_errores = 0

    for page in range(1, total_paginas + 1):
        print(f"\n--- Página {page}/{total_paginas} ---")

        if page == 1:
            resultado = primera
        else:
            time.sleep(DELAY_ENTRE_REQUESTS)
            try:
                resultado = obtener_pagina_listado(page)
            except (requests.RequestException, ValueError, KeyError) as e:
                total_errores += 1
                print(f"  [!] No se pudo obtener la página {page} tras varios reintentos: {e}")
                print(f"  [!] Se salta esta página. Podés volver a correr el script luego para reintentarla.")
                continue

        filas = resultado["data"]["data"]
        print(f"Resoluciones en esta página: {len(filas)}")

        for fila in filas:
            id_resolucion = fila["id"]

            if ya_descargado(id_resolucion):
                total_saltadas += 1
                continue

            try:
                time.sleep(DELAY_ENTRE_REQUESTS)
                detalle = obtener_detalle(id_resolucion)
                data_resolucion = detalle["data"]

                guardar_resolucion(id_resolucion, data_resolucion)
                total_descargadas += 1
                print(f"  [OK] {id_resolucion} - {data_resolucion.get('nro_resolucion')}")

            except (requests.RequestException, ValueError, KeyError) as e:
                total_errores += 1
                print(f"  [!] Error en {id_resolucion}: {e}")

    print("\n=== Resumen ===")
    print(f"Descargadas nuevas: {total_descargadas}")
    print(f"Ya existían (saltadas): {total_saltadas}")
    print(f"Errores: {total_errores}")
    print(f"Archivos en: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--max-pages", type=int)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    main(max_paginas=args.max_pages)
