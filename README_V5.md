# V5: extracción y entrenamiento con el mismo modelo

La construcción conserva los 300 pares débiles y la evaluación del v4. Genera nuevas propuestas por párrafos, prioriza etiquetas escasas y separa evidencia factual de cita. No aprueba jurídicamente las propuestas. El catálogo permanece sin cambios: vigencia normativa requiere verificación.

`experimental_debil/` es una variante opcional: incorpora únicamente nuevas propuestas train que cumplen reglas conservadoras de proximidad, señales factuales y rol de condena heurístico. Sus etiquetas siguen pendientes, sin garantía de relevancia jurídica. Comparar esta variante con el baseline y conservar validación/test idénticos; no atribuir una eventual mejora a supervisión humana.

## Archivos

- `extract_sections.py`: hechos candidatos, offsets, evidencia, papel de cita heurístico, distancia y URL.
- `build_v5.py`: ejecución nueva, grupos v4 congelados, cola de revisión train y bloqueo de textos compartidos.
- `apply_reviews.py`: incorporación trazable únicamente de candidatos train revisados; rechazo de fugas.
- `prepare_v5_experiment.py`: variante débil opcional, auditoría exacta v4/v5 y muestra HTML para revisar.
- `train_v5.py`: mismo MPNet multilingual, 512 tokens, múltiples positivos y negativos revisados opcionales; LR, escala, seed y paciencia configurables. Resume optimizador/scheduler; guardado final comentado. Rechaza output en `/drive/`.

La selección del checkpoint ahora puede priorizar F1 macro (predeterminado), MRR o NDCG. Exporta errores/ranking de validación, NDCG@10, soporte por artículo y cohortes con/sin positivos vistos en train. Test solo se consulta con `--evaluate-test`.
- `experiment_v5.py`: tres tasas de aprendizaje; compara exclusivamente validation. No ejecuta test automáticamente.
- `retrieval_tools.py`: evaluación BM25/embeddings/RRF y propuestas de negativos solo en train. Las métricas híbridas se reportan aparte del encoder.
- `Modelo_fin_v5.ipynb`: notebook autónomo con módulos incorporados; usa carpeta v5 en Drive y checkpoints locales.

## Revisiones

Cada registro de `revisiones.jsonl`: candidate_id, decision (`aprobar`/`rechazar`), articulos_aprobados (lista de claves del catálogo), revisor, justificacion y opcional hechos_corregidos. No rellenar automáticamente aprobaciones. Mantener todos los positivos jurídicamente aplicables, no solamente uno.

Para negativos revisados: query_id (fingerprint de hechos), estado_revision=`aprobado`, negativos_aprobados (lista), revisor, justificacion. Mismo número de negativos para todas las anclas train. La herramienta de minado propone candidatos, no garantiza que sean negativos. No usar ID numéricamente cercano como criterio jurídico.

## Orden de ejecución

1. Ejecutar build_v5 con raw/v4 y carpeta nueva.
2. Revisar propuestas y ejecutar apply_reviews hacia otra carpeta nueva.
3. Ejecutar experiment_v5 con la versión elegida y outputs locales. Sin revisiones, el dataset de entrenamiento sigue siendo el baseline v4; cambia únicamente la búsqueda de hiperparámetros.
4. Elegir por validation y repetir finalistas con seeds nuevos. Ejecutar test una vez para la selección final; ampliar con evaluación humana independiente.
5. Recuperación híbrida y adaptación de dominio son experimentos independientes, no mejoras demostradas.

No se afirma un aumento de accuracy/F1 antes de medirlo. No se cambia a E5.
