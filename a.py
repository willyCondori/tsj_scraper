import glob, json, re
from collections import Counter

patron_bis = re.compile(r"art[íi]culo?s?\.?\s*(\d+)\s*bis", re.IGNORECASE)

archivos = glob.glob("data/raw/*.json")
resoluciones_con_bis = 0
articulos_afectados = Counter()

for path in archivos:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    texto = data.get("contenido", "")
    matches = patron_bis.findall(texto)
    if matches:
        resoluciones_con_bis += 1
        for numero in set(matches):
            articulos_afectados[numero] += 1

print(f"Archivos totales: {len(archivos)}")
print(f"Resoluciones que citan al menos un artículo 'bis': {resoluciones_con_bis}")
print("Artículos 'bis' más citados (número base):", articulos_afectados.most_common(15))