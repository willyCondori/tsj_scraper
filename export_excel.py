"""
Paso 4 — Exportar el dataset a Excel para revisión humana.

Un .jsonl es lo que usa el código para entrenar, pero para que vos (o un
abogado que te ayude a validar) revisen el dataset a ojo, Excel es mucho
más cómodo. Este script arma un único .xlsx con varias hojas:

  - "Pares_originales"   -> los pares (hechos, artículo) tal cual salieron
                            del scraping, sin aumentar.
  - "Pares_aumentados"   -> incluye las variantes parafraseadas y marca
                            con la columna "tipo" cuál es cuál.
  - "Sinonimos"          -> el diccionario de sinónimos jurídicos usado,
                            en formato tabla (para que sea fácil de
                            ampliar/corregir sin tocar JSON a mano).
  - "Estadisticas"       -> conteo de pares por artículo (para detectar
                            desbalance: artículos con 1 solo ejemplo vs.
                            artículos con cientos).

Requiere: pandas, openpyxl (ver requirements.txt)

Uso:
    python export_excel.py
"""

import json
import os

import pandas as pd

PARES_ORIGINALES_PATH = os.path.join("data", "processed", "pares_entrenamiento.jsonl")
PARES_AUMENTADOS_PATH = os.path.join("data", "processed", "pares_aumentados.jsonl")
SINONIMOS_PATH = os.path.join("data", "reference", "sinonimos_juridicos.json")
OUT_XLSX = os.path.join("data", "processed", "dataset_completo.xlsx")


def cargar_jsonl_como_df(path: str) -> pd.DataFrame:
    filas = []
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            filas.append(json.loads(linea))
    df = pd.DataFrame(filas)
    if "negativos_dificiles" in df.columns:
        df["negativos_dificiles"] = df["negativos_dificiles"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )
    return df


def cargar_sinonimos_como_df(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_comentario", None)

    filas = []
    for termino_canonico, variantes in data.items():
        filas.append({
            "termino_canonico": termino_canonico,
            "variantes": ", ".join(variantes),
            "cantidad_variantes": len(variantes),
        })
    return pd.DataFrame(filas)


def calcular_estadisticas(df_aumentado: pd.DataFrame) -> pd.DataFrame:
    conteo = (
        df_aumentado.groupby("articulo")
        .size()
        .reset_index(name="cantidad_ejemplos")
        .sort_values("cantidad_ejemplos", ascending=False)
    )
    total = conteo["cantidad_ejemplos"].sum()
    conteo["porcentaje_del_total"] = (conteo["cantidad_ejemplos"] / total * 100).round(2)

    # marca artículos con muy pocos ejemplos, que van a rendir peor en el
    # fine-tuning y conviene reforzar con más scraping o más parafraseo
    conteo["alerta_pocos_ejemplos"] = conteo["cantidad_ejemplos"] < 5

    return conteo


def main():
    faltantes = [p for p in (PARES_ORIGINALES_PATH, PARES_AUMENTADOS_PATH, SINONIMOS_PATH) if not os.path.exists(p)]
    if faltantes:
        print("Faltan archivos previos, corré antes:")
        for f in faltantes:
            print(f"  - {f}")
        print("(extract_pairs.py y augment_pairs.py, en ese orden)")
        return

    df_originales = cargar_jsonl_como_df(PARES_ORIGINALES_PATH)
    df_aumentados = cargar_jsonl_como_df(PARES_AUMENTADOS_PATH)
    df_sinonimos = cargar_sinonimos_como_df(SINONIMOS_PATH)
    df_estadisticas = calcular_estadisticas(df_aumentados)

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df_originales.to_excel(writer, sheet_name="Pares_originales", index=False)
        df_aumentados.to_excel(writer, sheet_name="Pares_aumentados", index=False)
        df_sinonimos.to_excel(writer, sheet_name="Sinonimos", index=False)
        df_estadisticas.to_excel(writer, sheet_name="Estadisticas", index=False)

    print(f"Dataset exportado a: {os.path.abspath(OUT_XLSX)}")
    print(f"  - Pares_originales:  {len(df_originales)} filas")
    print(f"  - Pares_aumentados:  {len(df_aumentados)} filas")
    print(f"  - Sinonimos:         {len(df_sinonimos)} términos")
    print(f"  - Estadisticas:      {len(df_estadisticas)} artículos distintos")

    articulos_con_pocos = df_estadisticas["alerta_pocos_ejemplos"].sum()
    if articulos_con_pocos:
        print(
            f"\n[!] {articulos_con_pocos} artículos tienen menos de 5 ejemplos. "
            "Revisá la hoja 'Estadisticas' — esos artículos van a rendir peor "
            "en el fine-tuning. Opciones: scrapear más casos que los mencionen, "
            "o aumentar VARIANTES_POR_PAR en augment_pairs.py solo para esos."
        )


if __name__ == "__main__":
    main()
