import json
from collections import Counter

conteo = Counter()
with open("data/processed/pares_entrenamiento.jsonl", encoding="utf-8") as f:
    for linea in f:
        p = json.loads(linea)
        if p["articulo"].startswith("308"):
            conteo[p["articulo"]] += 1

print(conteo)