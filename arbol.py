from scraper_api import HEADERS  # o como se llame la constante en tu script
import requests, json

# 1. Trae el árbol completo
r = requests.get('https://apigenesis.tsj.bo/api/v1/catalogos/arbol-jurisprudencia', headers=HEADERS)
print(json.dumps(r.json(), indent=2, ensure_ascii=False)[:2000])

# 2. Mira el detalle de una resolución que ya tengas en data/raw/, buscando cualquier campo
#    que parezca un id o nombre de categoría del árbol
import glob
archivo = glob.glob('data/raw/*.json')[0]
with open(archivo, encoding='utf-8') as f:
    data = json.load(f)
print(list(data.keys()))