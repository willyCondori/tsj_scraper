# TSJ scraper — dataset jurídico boliviano y entrenamiento final E5 base

El proyecto descarga resoluciones penales del Tribunal Supremo de Justicia de Bolivia y construye pares **hechos → artículo del Código Penal** para recuperación semántica. El resultado permite introducir el relato de un caso y recuperar artículos candidatos. No genera sentencias ni determina automáticamente culpabilidad.

La versión elegida para la presentación es **multilingual E5 base + dataset final v6**, con la receta que obtuvo el mejor F1 macro sobre la validación original. El notebook principal es [Entrenamiento_final_E5_base.ipynb](Entrenamiento_final_E5_base.ipynb). Sentence Transformers es la biblioteca; el modelo utilizado es `intfloat/multilingual-e5-base`, no MPNet.

## 1. Qué datos se obtuvieron y de dónde

La colección local utilizada contiene **21.697 resoluciones JSON** descargadas del buscador Genesis del TSJ. Cada resolución es un documento completo; no equivale a un ejemplo de entrenamiento.

`scraper_api.py` realiza una búsqueda paginada mediante `POST /resoluciones/busqueda_avanzada` y descarga el detalle mediante `GET /resoluciones/{id}` bajo `https://apigenesis.tsj.bo/api/v1`. La consulta recibida filtra materia Penal (`idMateria="1"`) y usa por defecto la palabra `penal`. No se afirma que incluya todas las resoluciones del TSJ o toda la jurisprudencia boliviana. La disponibilidad actual del servicio debe comprobarse al ejecutar.

Los JSON crudos conservan campos como `id`, `nro_resolucion`, `nro_expediente`, `fecha_emision`, partes, departamento, sala, magistrado, materia, procesos, forma de resolución, `contenido`, `tiene_html` y `url_pdf_escaneado`. El extractor usa principalmente contenido y metadatos de procedencia. `url_pdf_escaneado` es una referencia: esta receta no realiza OCR automático de todos los PDF.

El scraper comprueba JSON ya descargados, guarda mediante archivo temporal y reemplazo atómico, añade pausas y reintentos para fallos transitorios. No vuelve a descargar automáticamente documentos válidos. Descargar nuevas resoluciones no cambia por sí solo las particiones del dataset final existente.

```powershell
python -m pip install -r requirements.txt
python scraper_api.py --max-pages 2 --out-dir data/raw
```

## 2. Cómo se extraen los pares

El flujo básico es:

1. Leer la resolución y limpiar HTML, entidades y espacios; conservar párrafos y texto.
2. Detectar citas explícitas del Código Penal: `art.`, `arts.`, artículos múltiples y sufijos como `bis` o `ter`. No asumir que un artículo sin norma pertenece al CP.
3. Guardar cita literal y posiciones en el **texto limpio**, no en el HTML original.
4. Extraer una ventana alrededor de la cita: aproximadamente 650 caracteres anteriores y 250 posteriores, con expansión limitada hacia fronteras de oración/párrafo.
5. Enmascarar citas en esa ventana para reducir que el número del artículo revele directamente la etiqueta. Esto no elimina todas las pistas, nombres de delitos ni argumentos judiciales.
6. Clasificar el fragmento por señales fácticas y procesales. Es PLN basado en reglas y expresiones regulares, no un resumidor generativo.
7. Comprobar que el artículo tenga texto en el catálogo CP, aplicar la política de la tarea, separar propuestas dudosas y deduplicar texto–artículo.
8. Agrupar fuentes/documentos/textos y expedientes antes de dividir en train, validation y test.

`extract_pairs.py` realiza la extracción inicial; `run_pipeline.py` crea una carpeta de ejecución nueva. `extract_sections.py` y `build_v5.py` buscan bloques adicionales y evidencias para revisión. `build_v6.py` selecciona propuestas de train con reglas de acciones pasadas, longitud, cercanía de cita, contenido procesal y controles de solapamiento.

**Una cita no demuestra aplicabilidad.** Puede pertenecer a la acusación, absolución, condena, un precedente o un argumento rechazado. Por ello las etiquetas del v6 son **supervisión débil** y tienen revisión pendiente. El entrenamiento aprende estas asociaciones; no las convierte en verdades jurídicas.

## 3. Dataset usado en el entrenamiento final

El notebook consume la carpeta de Drive `Mi unidad/datasetFinalDerecho`, concretamente `pares_aumentados.jsonl` y `catalogo_cp.json`.

| Elemento | Cantidad |
|---|---:|
| Pares totales | 368 |
| Pares train | 327 |
| Consultas diferentes en train | 267 |
| Artículos con ejemplos train | 68 |
| Pares validation | 18 |
| Consultas validation | 16 |
| Expedientes de validation | 12 |
| Pares test | 23 |
| Consultas test | 21 |
| Artículos candidatos del catálogo | 444 |

V4 aportaba 300 pares: 259 train, 18 validation y 23 test, con catálogo de 376 artículos. V6 añadió 68 pares débiles de train y 68 entradas de catálogo obtenidas de una fuente oficial. Validation/test se conservaron. El dataset final utiliza la variante ampliada v6; la carpeta `comparacion_conservadora` conserva otra variante y no es el input final. La ampliación del catálogo cambia la dificultad de recuperación: no comparar sus métricas directamente con un catálogo menor.

Los 368 registros están marcados `tipo="original"` en el archivo final. **Original significa texto procedente de documentos**, no etiqueta aprobada por un experto. `pares_aumentados.jsonl` es el nombre de entrada del entrenador; en esta versión no implica que existan 368 paráfrasis artificiales. Las variantes sintéticas solo se generan por separado bajo reglas y revisiones explícitas.

Hay pocos pares frente a las 21.697 resoluciones porque se necesita una asociación utilizable hechos–artículo, texto suficiente, norma reconocida, entrada de catálogo y controles de duplicados/contenido. Gran parte de las resoluciones trata recursos, admisibilidad o doctrina, y no aporta un relato fáctico autónomo. No sería correcto convertir automáticamente cada documento o cada cita en un ejemplo positivo.

### Qué es un par y dónde está la etiqueta

Un par contiene `hechos` como consulta y `articulo` como etiqueta de recuperación. `articulo_texto` es el texto que se representa como documento candidato. No hay una etiqueta binaria universal 0/1 ni una columna que certifique culpabilidad.

Ejemplo esquemático: un relato sobre una conducta documental puede tener positivos conocidos `200` y `203`. Son dos filas del mismo hecho. Para evaluar se agrupan y se conserva el conjunto de positivos. Un artículo no anotado es desconocido; no necesariamente constituye un negativo jurídico.

### Diccionario de columnas de los pares

| Campo | Significado |
|---|---|
| `hechos` | Texto utilizado como consulta; puede conservar ruido procesal en v6. |
| `articulo` | Clave de artículo CP, por ejemplo `251` o `252 bis`; etiqueta conocida débil. |
| `norma` | Norma de la etiqueta; el catálogo final de entrenamiento es CP. |
| `articulo_texto` | Texto del artículo presente en el catálogo exacto de esa corrida. |
| `fuente_id` | Identificador de la resolución del TSJ. |
| `fuente_url` | Enlace de procedencia, cuando está disponible. |
| `nro_resolucion`, `nro_expediente` | Resolución y expediente; ayudan a rastrear y agrupar casos. |
| `fecha_emision`, `sala`, `materia`, `tipo_proceso` | Metadatos del documento; algunos pueden ser nulos. |
| `resultado_resolucion` | Forma/resultado reportado por la fuente; no sustituye verificar cada conducta. |
| `documento_hash` | Huella del documento normalizado para trazabilidad y duplicados. |
| `cita_original`, `cita_inicio`, `cita_fin` | Evidencia literal y offsets de la cita en el documento limpio. |
| `hechos_inicio`, `hechos_fin`, `seccion` | Posiciones/sección de propuestas por bloques, cuando existen. |
| `evidencia_cita`, `papel_cita`, `distancia_cita` | Contexto, rol heurístico y distancia entre hecho y cita en propuestas. |
| `tipo_contenido`, señales fácticas/procesales | Indicadores de reglas para clasificación de texto. |
| `estado_revision` | Pendiente/aprobado; pendiente no se convierte en aprobado al entrenar. |
| `grupo_id`, `split` | Grupo del caso y partición guardada; no se recalculan al entrenar. |
| `tipo`, `parent_id`, `metodo`, `transformacion` | Procedencia y relación con el original, según la etapa. |
| `candidate_id`, `nivel_seleccion`, `supervision` | Identidad y trazabilidad de propuestas incorporadas en v6. |

No todas las filas tienen todas las columnas: el esquema conserva la procedencia de originales y adiciones. Una ausencia no debe rellenarse inventando evidencia.

### Archivos y formatos

| Archivo/carpeta | Función |
|---|---|
| `pares_entrenamiento.jsonl` | Pares base/elegibles de la versión; referencia de construcción. |
| `pares_aumentados.jsonl` | Input final exacto consumido por el entrenador. |
| `catalogo_cp.json` | Mapa artículo → texto; los 444 candidatos se evalúan completos. |
| `manifest.json` | Cantidades, configuración, procedencia y hashes del dataset/catálogo. |
| `propuestas_secciones.jsonl` | Bloques candidatos extraídos; no aprobados automáticamente. |
| `nuevos_pares_pendientes.jsonl` | Posibles ampliaciones que requieren revisión. |
| `cola_revision_prioritaria.jsonl` | Propuestas priorizadas para inspección. |
| `plantilla_revisiones.jsonl` | Campos para aprobar/rechazar con revisor y justificación. |
| `errores_extraccion.jsonl` | Fallos de lectura/extracción; no es entrenamiento. |
| `catalogo_adiciones_fuente.jsonl` | Procedencia documental/páginas de entradas nuevas. |
| `auditoria_v6.json` | Recuentos, exclusiones y controles de la construcción. |
| `estadisticas_articulos.csv` | Distribución y soporte por artículo/split. |
| `revision_muestra.html` | Vista legible de propuestas y evidencias. |
| Excel `.xlsx`/`.xlsm` | Vista de inspección/presentación; el notebook final no lo consume. |
| `fuentes/` | Documentos de procedencia; incluye el PDF del CP usado. |
| `comparacion_conservadora/` | Variante de comparación, no el dataset final entrenado. |
| `codigos/` | Copia de scripts vinculados a una entrega. |

JSONL contiene un objeto JSON por línea; JSON puede contener un mapa, lista o manifiesto completo. Editar Excel no cambia los JSONL: las revisiones deben incorporarse con un procedimiento trazable.

La ampliación de catálogo utilizó el [Código Penal publicado por la Fiscalía](https://fiscalia.gob.bo/marco-legal/leyes/codigo-penal). Sus textos y los anteriores se conservaron para reproducir las pruebas. No se certificó exhaustivamente la vigencia temporal para cada resolución histórica; verificarla antes de declarar etiquetas jurídicamente validadas.

## 4. Particiones, revisión y v7

La partición inicial utiliza grupos y un hash estable con seed 42, aproximadamente 80/10/10. Une transitivamente fuente, documento, texto y expediente identificable. No garantiza detectar todas las paráfrasis o casos relacionados. Los tamaños reales no deben forzarse a porcentajes exactos. El entrenamiento final utiliza los `split` ya guardados.

`apply_reviews.py` incorpora o rechaza propuestas únicamente de train, exige revisor y justificación y bloquea cambios/fugas a heldout. Una aprobación reemplaza las etiquetas débiles de esa propuesta; no agrega silenciosamente etiquetas incompatibles. `mejorar_dataset.py` ofrece un importador más estricto para revisiones documentales completas, incluyendo evidencia y papel confirmado. No aprobar automáticamente la predicción del modelo.

V7 es una extracción estructurada **para revisión**, separada del entrenamiento ganador:

- `hechos.jsonl`: oraciones fácticas candidatas, texto limpio, offsets y procedencia.
- `decisiones.jsonl`: fragmentos de decisiones, rol propuesto y metadatos.
- `relaciones.jsonl`: citas con norma/artículo, evidencia, decisión candidata y `usable_entrenamiento=False`.
- `citas_sin_norma.jsonl`: citas cuyo documento/norma no se pudo resolver.
- `muestra_revision.json`, plantilla, manifiesto y errores: revisión y trazabilidad.

Se distinguen `CP:251`, `CPP:251` y `LEY:348:5`. En v7 proximidad no prueba el vínculo persona–hecho–artículo; `hecho_id` y `persona_id` pueden quedar sin confirmar. Las leyes y el CPP detectados no se añaden automáticamente al catálogo CP ni a los positivos del entrenamiento final.

La entrega v7 final separada procesó 21.697 documentos: 6.123 hechos candidatos, 90.151 fragmentos de decisión, 280.693 relaciones/citas explícitas y 205.352 citas sin norma resuelta. Son recuentos de fragmentos/citas, **no de pares aprobados**. No se incorporaron íntegramente a la receta ganadora. Una variante extractiva v7 y otra de limpieza mínima se probaron y no mejoraron el F1 original.

`exportar_revision_fuentes.py` prepara evidencia de train/validation, sin abrir fuentes test: en la revisión local se obtuvieron 204 documentos, 283 consultas y 5.265 citas verificadas literalmente. Estas verificaciones prueban procedencia textual, no aplicabilidad jurídica.

## 5. Proceso de entrenamiento final

1. Montar Drive y comprobar hashes, 368 pares, particiones y 444 textos de catálogo.
2. Cargar `intfloat/multilingual-e5-base` al repetir el entrenamiento, o el checkpoint ganador para mostrar/evaluar el modelo existente.
3. Configurar longitud máxima 512 y comprobar capacidad de arquitectura. Medir tokens sobre los hechos y los textos de artículos, no sobre identificadores numéricos.
4. Anteponer `query: ` a los hechos y `passage: ` a los artículos, como corresponde a E5.
5. Construir ejemplos con el texto consulta, texto positivo y máscara de todos los artículos positivos conocidos de esa consulta. También se enmascaran entradas con texto de catálogo idéntico.
6. Utilizar batches sin duplicados de textos. La pérdida contrastiva compara consultas y artículos del batch, tratando los positivos conocidos como positivos y evitando usarlos como negativos falsos.
7. Entrenar con la receta fija siguiente. No se agregan negativos difíciles no revisados ni se vuelve a buscar tasas de aprendizaje en el notebook final.
8. Evaluar cada época contra **los 444 artículos**, seleccionar por F1 macro top1 de validation y detener tras tres épocas sin mejora.
9. Recargar el mejor checkpoint y exportar rankings/métricas. Test y guardado de pesos finales en Drive permanecen desactivados por defecto.

| Parámetro | Valor |
|---|---|
| Modelo | `intfloat/multilingual-e5-base` |
| Batch size | 16 |
| Learning rate | `2e-5` |
| Longitud máxima | 512 tokens |
| Seed | 42 |
| Épocas máximas | 12 |
| Paciencia de early stopping | 3 |
| Escala de pérdida | 20 |
| Weight decay | 0.01 |
| Warmup | 10 % |
| Selección | F1 macro top1, validation original |
| Precisión | FP16 en GPU; FP32 sin CUDA |
| Gradient checkpointing | Activado en esta receta |

La pérdida utiliza embeddings normalizados y log-softmax de similitudes escaladas. Distribuye el objetivo entre positivos conocidos presentes entre los candidatos del batch. Los otros candidatos funcionan como negativos de entrenamiento bajo supervisión débil; pueden existir positivos jurídicos aún no anotados.

Los checkpoints completos guardan pesos, estado del entrenador, optimizador y scheduler; la reanudación comprueba los archivos y la configuración. Cambiar datos o parámetros exige otra salida. `save_total_limit=1` puede conservar mejor y último checkpoint según el Trainer. Una carpeta de pesos antiguos por época sin optimizador no equivale a reanudación completa.

### Ejecutar el notebook final

Subir únicamente `Entrenamiento_final_E5_base.ipynb` a Colab. Incluye los módulos necesarios y no requiere subir ZIP ni nuevos datos. La carpeta de Drive debe contener el v6 original.

- `REENTRENAR=False`: carga el ganador en `/content/tsj_e5_base_v6_mejoras/lr_2e-05_seed_42/checkpoint-126`.
- Si se eliminó la sesión, ajustar la ruta a los archivos recuperados o activar `REENTRENAR=True` para repetir la receta. No confundir repetición con recuperación exacta de pesos.
- `REENTRENAR=True`: inicia/reanuda en una carpeta final local nueva, con la receta fija.
- `EVALUAR_TEST=False`: no consulta el test. Cambiarlo solo al realizar una evaluación final fijada; no usar test para elegir parámetros.
- La instrucción `modelo.save(...)` permanece comentada. Los reportes sí se guardan en `datasetFinalDerecho/reportes_validation/entrenamiento_final_e5_base`.

El notebook comprueba que Colab importa su código y no un módulo antiguo guardado en memoria. Registra versiones de bibliotecas en cada ejecución. Dependencias recomendadas están fijadas, pero el entorno histórico exacto de todas las corridas no fue registrado; nuevas ejecuciones pueden diferir por GPU, versiones y operaciones no deterministas.

`entrenamiento_final/train_e5_base.py` permite también ejecución por CLI:

```powershell
python -m pip install -r requirements-training.txt
python entrenamiento_final/train_e5_base.py --run-dir RUTA_DATASET_V6 --output-dir modelos/e5_base_final --epochs 12 --batch-size 16 --max-seq-length 512 --learning-rate 2e-5 --seed 42 --selection-metric f1_macro
```

## 6. Evaluación y pruebas presentables

La recuperación calcula embeddings normalizados para consultas/artículos y los ordena por producto escalar, equivalente a similitud coseno. La similitud no es una probabilidad de aplicabilidad jurídica.

| Métrica | Interpretación |
|---|---|
| Accuracy@1 | Fracción de consultas cuyo primer artículo pertenece a los positivos conocidos. |
| Recall@k | Proporción de positivos conocidos recuperados entre los primeros k, promediada por consulta. |
| Precision@k | Fracción de resultados recuperados que están anotados como positivos conocidos. |
| MRR | Premia que el primer positivo aparezca pronto en el ranking. |
| NDCG@10 | Premia posiciones tempranas de positivos dentro de los diez primeros. |
| F1 micro top1 | Combina errores y aciertos de etiquetas acumulados; predice un artículo por consulta. |
| F1 macro top1 | Promedio por etiqueta sobre la unión de etiquetas verdaderas y predichas activas. |
| F1 weighted top1 | Promedio por etiqueta ponderado por soporte verdadero. |
| Métricas por expediente | Ponderan cada consulta por 1/número de consultas de su grupo, dando igual peso a los casos. |

Con consultas multietiqueta, acertar el primer artículo puede dar accuracy=1 para esa consulta pero no recall=1, porque quedan otros positivos sin devolver. Por eso accuracy y F1 no son intercambiables. Cuatro fragmentos de validation provienen del mismo expediente de cheques; por expediente se evita que dominen el promedio.

### Resultados reportados por el usuario en Colab

| Prueba sobre validation original | F1 macro top1 | Accuracy@1 | Recall@3 | Recall@5 |
|---|---:|---:|---:|---:|
| **E5 base elegido, checkpoint-126** | **0.655678** | **0.875** | 0.875 | 0.875 |
| Selección por expediente, checkpoint-84 | 0.619048 | 0.875 | **0.9375** | 0.9375 |
| Limpieza mínima de encabezados | 0.511111 | 0.8125 | 0.875 | 0.9375 |

El ganador tuvo F1 micro=0.823529, F1 weighted=0.772487, MRR=0.885802 y NDCG@10=0.887228. Accuracy ponderada por expediente=0.833333, F1 macro ponderado por expediente=0.651282. El checkpoint ganador corresponde a la época 6; la búsqueda previa se detuvo en la 9. Estos datos son de **validación**, no de test.

La evaluación de sensibilidad apartó dos consultas problemáticas y dio F1 macro=0.727273 sobre 14 consultas. No demuestra mejora del modelo ni sustituye la referencia original. El reordenador multilingüe sin adaptación empeoró: accuracy top1 0 en la referencia anterior y 0.0625 con el checkpoint por expediente. RRF tampoco mejoró; ambas políticas están fuera de la receta final. No se alcanzó un F1 macro original de 75–80 %.

`entrenamiento_final/resultados_validation_reportados.json` conserva la comparación completa del candidato de limpieza y la decisión de volver a checkpoint-126, procedente de la salida compartida por el usuario. No se presenta como una nueva ejecución local. El notebook calcula y registra métricas frescas al ejecutarse.

El aviso de secuencias >512 surge al contar tokens sin truncarlos; al codificar se aplica el límite. Aproximadamente un 1.35 % del catálogo supera 512 tokens. Esa pérdida de texto puede afectar recuperación y requiere estudiar fragmentación por separado, sin afirmar que ya mejoró la receta final.

## 7. Construcción y revisión reproducible

El repositorio publica código y resultados agregados, no los 21.697 JSON, Excel ni pesos. `data/` está ignorado. Para reconstruir se necesitan localmente los JSON crudos, el catálogo, política de referencias y, para v6, sus versiones previas y PDF oficial utilizado. No inventar ni descargar otra versión del catálogo y afirmar que reproduce los mismos hashes.

```powershell
python run_pipeline.py --raw-dir data/raw --run-name dataset_v4_nuevo
python export_excel.py --run-dir data/processed/runs/dataset_v4_nuevo
python build_v5.py --raw-dir data/raw --v4-dir data/processed/runs/dataset_v4 --output-dir data/processed/runs/dataset_v5_nuevo
python build_v6.py --v4-dir data/processed/runs/dataset_v4 --v5-dir data/processed/runs/dataset_v5_nuevo --output-dir data/processed/runs/dataset_v6_nuevo --official-pages RUTA_PAGINAS_JSON --official-pdf RUTA_PDF --raw-dir data/raw
python extract_structured_v7.py --raw-dir data/raw --baseline RUTA_CARPETA_BASELINE --out-dir data/processed/runs/v7_revision_nueva
```

Usar nombres nuevos: no sobrescribir runs de referencia. `build_v6.py` genera variantes conservadora/ampliada y auditorías; la carpeta entregada en Drive usa la ampliada como raíz final. Los hashes fijados en el notebook identifican esa entrega exacta, no cualquier reconstrucción denominada v6.

```powershell
python apply_reviews.py --run-dir RUTA_RUN --reviews revisiones.jsonl --output-dir RUTA_RUN_REVISADO
```

Para generar el expediente documental:

```powershell
python exportar_revision_fuentes.py --source RUTA_DATASET_V6 --raw-dir data/raw --output RUTA_REVISION_NUEVA
```

`mejorar_dataset.py`, `seleccionar_modelo.py` y `probar_reranking.py` conservan las herramientas de experimentos controlados. No forman parte de la limpieza ni recuperación del notebook final. Los notebooks MPNet/v5 antiguos se conservan como historial; sus README no describen la receta E5 final.

## 8. Limitaciones y trabajo pendiente

La evaluación es pequeña y se consultó repetidamente durante la búsqueda, por lo que puede sobreajustarse. Hay fragmentos procesales, nombres de delitos, decisiones y etiquetas inconsistentes. Debe revisarse la fuente 49506 (referencias 135/335/337), recuperar hechos subyacentes de 49019 y comprobar el caso 251 de la fuente 36794. No cambiar etiquetas solo porque otra predicción parezca mejor.

Falta una revisión jurídica completa de persona, conducta, artículo aplicado, etapa y versión normativa; ampliar expedientes independientes y artículos escasos con datos verificados; evaluar generalización en un benchmark independiente. El modelo final aquí significa **versión elegida para presentar las pruebas actuales**, no aprobación para decisiones jurídicas ni desempeño final demostrado en test.

La vuelta atrás de la prueba controlada funcionó: al bajar F1 macro y accuracy se cargó checkpoint-126. No se sobrescribió v6 ni se aprobó la limpieza fallida. Los pesos en `/content` se pierden si Colab elimina el entorno; el Git de este proyecto no los contiene.

## 9. Verificar código

```powershell
python -m unittest discover -s tests -v
python -m compileall -q entrenamiento_final
```

Las pruebas cubren extracción, separación de normas, offsets, particiones, reemplazo/rechazo de revisiones y estructura/configuración del notebook final. No sustituyen revisión jurídica ni una corrida de GPU. Consultar [documentación de Sentence Transformers](https://www.sbert.net/docs/sentence_transformer/training_overview.html) y la [ficha de multilingual E5 base](https://huggingface.co/intfloat/multilingual-e5-base) para el modelo/biblioteca.
