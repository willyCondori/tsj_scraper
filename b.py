# ¿busqueda_avanzada acepta filtrar por nodo del árbol?
import requests, json
from scraper_api import HEADERS

body = {"idMateria": 1, "idArbol": 1791}  # ajusta el nombre del parámetro según lo que aceptes probar
r = requests.post('https://apigenesis.tsj.bo/api/v1/resoluciones/busqueda_avanzada', headers=HEADERS, json=body)
print(r.status_code, r.text[:500])