# TSJ scraper corregido

Esta carpeta es independiente del proyecto original. No incluye los 21.697 JSON ni los datasets antiguos. Conserva el catálogo recibido; artículos sin entrada CP pasan a revisión y no se reasignan a otra norma.

## Error del checkpoint

El código original devuelve `(numero, ruta)` pero lo lee como `(ruta, numero)`. La copia `Modelo_fin_checkpoint_corregido.ipynb` devuelve siempre `(ruta, numero)`, incluido `(None, 0)`. Conserva tu entrenamiento anterior. Los pesos guardados por época no contienen todo el estado del optimizador.

`Modelo_fin_corregido.ipynb` usa el pipeline nuevo y Trainer. Necesita subir esta carpeta y el run a Google Drive. Ajusta PROJECT_DIR y MI_RUN en su celda de configuración. Usa una carpeta de modelo nueva; opcionalmente INITIAL_MODEL puede apuntar a pesos antiguos para una nueva corrida, no para reanudar épocas anteriores.

## Preparar datos en Windows

Desde esta carpeta:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py --raw-dir "C:\Users\kiro\Downloads\tsj_scraper (1)\tsj_scraper\data\raw" --run-name dataset_v3
```

La salida queda en `data/processed/runs/dataset_v3`. Un nombre existente produce error para evitar sobrescribirlo.

Para generar el Excel de revisión desde ese mismo run:

```powershell
python export_excel.py --run-dir data/processed/runs/dataset_v3
```

El Excel incluye todos los pares elegibles, `split`, evidencia de cita, offsets, estado de revisión y estadísticas por artículo/split. `Muestra_revision` contiene hasta 1.000 descartados, distribuidos por motivo, para inspección; todos los descartados se conservan en `pares_revision.jsonl`. Si revisas etiquetas en Excel, debes trasladarlas al JSONL antes de entrenar: el entrenador consume JSONL. El notebook compatible antiguo sí consume el Excel antiguo.

Se genera:

- `pares_candidatos.jsonl`: todas las citas extraídas, con evidencia y offsets del texto limpio.
- `pares_entrenamiento.jsonl`: candidatos fácticos/mixtos con artículo en catálogo, deduplicados.
- `pares_aumentados.jsonl`: originales con split; sin paráfrasis automáticas por defecto.
- `pares_revision.jsonl`: contenido dudoso o sin catálogo CP; preservado para revisión.
- `pares_duplicados.jsonl` y `errores_extraccion.jsonl`: trazabilidad de exclusiones y fallos.
- `catalogo_cp.json`, `manifest.json`: catálogo usado, hashes, configuración y recuentos.

Las etiquetas iniciales siguen siendo supervisión débil. El filtro no demuestra que la cita corresponda al hecho ni distingue automáticamente acusación, condena, absolución y precedente. Revisar antes de usarlo para conclusiones jurídicas.

Los grupos unen transitivamente fuentes, documentos y textos idénticos antes de deduplicar; 80/10/10 por hash estable. Esto no detecta todas las paráfrasis o expedientes relacionados. Validación/test conservan originales. Los hechos con varias etiquetas conocidas se conservan para evaluación y se excluyen de entrenamiento de un único positivo con MNRL.

También se agrupa por `nro_expediente` cuando tiene año, y se conservan fecha de emisión y forma de resolución. Los expedientes con nombres incompatibles aún requieren revisión. `data/reference/politica_dataset.json` separa 24 etiquetas de reglas generales de esta tarea y las guarda en Revision. No modifica el catálogo ni elimina todos los artículos de numeración baja.

Para añadir variantes, edita `data/reference/sinonimos_aprobados.json` con sustituciones semánticas revisadas y marca las filas aprobadas en `pares_entrenamiento.jsonl`. Después:

```powershell
python augment_pairs.py --run-dir data/processed/runs/dataset_v3 --synonyms data/reference/sinonimos_aprobados.json
```

Solo aumenta filas aprobadas de train. Nunca vuelve a dividir después del aumento.

## Entrenar localmente o en Colab

Se recomienda Colab con GPU y Python 3.11/3.12. En un entorno nuevo:

```powershell
python -m pip install -r requirements-training.txt
python finetune_embeddings.py --run-dir data/processed/runs/dataset_v3 --output-dir modelos/dataset_v3_256
```

Añade `--approved-only` para entrenar/evaluar únicamente ejemplos revisados. Si un split queda vacío, detiene el entrenamiento. Añade `--initial-model RUTA` solo para comenzar una corrida nueva desde pesos anteriores. Cambiar datos o configuración requiere otra carpeta de salida.

El entrenamiento usa batches sin duplicados de anclas/positivos, una corrida de cuatro épocas, checkpoints con optimizador/scheduler y evaluación de recuperación contra todo el catálogo CP disponible. Guarda métricas y distribución de tokens de hechos y textos de artículos. No modifica el catálogo ni elimina nombres de delitos automáticamente.

## Descargar resoluciones nuevas, opcional

```powershell
python scraper_api.py --max-pages 2 --out-dir data/raw
```

La consulta y headers se conservan del scraper recibido. La disponibilidad actual de la API no se ha comprobado. Se añadió validación de JSON existentes, guardado atómico y reintentos de 429.

## Verificación

```powershell
python -m unittest discover -s tests -v
```

Fuentes de API: https://www.sbert.net/docs/sentence_transformer/training_overview.html y https://www.sbert.net/docs/package_reference/base/sampler.html
