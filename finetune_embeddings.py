"""
Paso 4 — Fine-tuning del modelo de embeddings.

Necesita, además de pares_aumentados.jsonl, un archivo
articulos_codigo_penal.json con formato {"326": "texto completo del
articulo 326...", ...}, exportado desde tu catálogo Articulo de Django.

Uso:
    python finetune_embeddings.py
"""

import json
import os
import random

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

PARES_PATH = os.path.join("data", "processed", "pares_aumentados.jsonl")
ARTICULOS_PATH = "articulos_codigo_penal.json"
MODELO_BASE = "paraphrase-multilingual-mpnet-base-v2"
MODELO_SALIDA = "modelo_finetuneado_penal_bolivia"


def cargar_articulos(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No encontré {path}. Armá un JSON {{'numero_articulo': 'texto...'}} "
            "a partir del catálogo de Articulo que ya tenés en el backend Django."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_pares(path: str) -> list:
    pares = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            pares.append(json.loads(linea))
    return pares


def main():
    articulos = cargar_articulos(ARTICULOS_PATH)
    pares = cargar_pares(PARES_PATH)

    ejemplos = []
    descartados = 0
    for par in pares:
        texto_articulo = articulos.get(par["articulo"])
        if not texto_articulo:
            descartados += 1
            continue
        ejemplos.append(InputExample(texts=[par["hechos"], texto_articulo]))

    print(f"Ejemplos de entrenamiento válidos: {len(ejemplos)}")
    print(f"Descartados (artículo no encontrado en catálogo): {descartados}")

    if len(ejemplos) < 50:
        print(
            "\n[!] Tenés muy pocos ejemplos para un fine-tuning estable "
            "(recomendado: al menos algunos cientos)."
        )

    random.shuffle(ejemplos)

    modelo = SentenceTransformer(MODELO_BASE)
    dataloader = DataLoader(ejemplos, shuffle=True, batch_size=16)
    loss = losses.MultipleNegativesRankingLoss(modelo)

    modelo.fit(
        train_objectives=[(dataloader, loss)],
        epochs=3,
        warmup_steps=int(len(dataloader) * 0.1),
        output_path=MODELO_SALIDA,
        show_progress_bar=True,
    )

    print(f"\nModelo afinado guardado en: {MODELO_SALIDA}")


if __name__ == "__main__":
    main()