# Validación de dataset_v4

Se procesaron los 21.697 archivos originales y se obtuvieron 80.884 candidatos. No hubo errores de lectura/extracción. Se preserva evidencia de cita y metadatos de cada resolución.

- Pares que pasan el filtro automático: 300, de 181 resoluciones y 69 artículos.
- Clasificación orientativa: 192 fácticos y 108 mixtos.
- Particiones guardadas: train 259, validation 18 y test 23.
- Pares excluidos para revisión: 80.583. Motivos: 60.150 contenido sin señales suficientes, 16.462 etiquetas generales separadas por política y 3.971 sin texto CP en catálogo.
- Duplicados exactos excedentes de texto/artículo entre candidatos elegibles: 1.
- No se generaron paráfrasis: el diccionario aprobado empieza vacío.
- 240 pares tienen número de expediente.

Las comprobaciones no detectaron cruce de textos idénticos, resoluciones ni grupos entre particiones. Los grupos también vinculan expedientes con año. Las paráfrasis no idénticas y expedientes con nombres inconsistentes todavía requieren revisión.

Se detectaron 33 anclas con varios artículos conocidos. Se conservan en el dataset de revisión/evaluación. El entrenador de un único positivo las excluye de train, que queda en 182 pares. Esto no convierte en definitivas las etiquetas únicas restantes.

54 de los 69 artículos tienen menos de cinco pares. Estos datos no permiten afirmar cobertura suficiente ni buen rendimiento del modelo. La reducción refleja la falta de narración factual cerca de las citas en muchas resoluciones, y las limitaciones de la heurística léxica. Los excluidos no están declarados jurídicamente incorrectos.

Todos los estados de revisión empiezan como `pendiente`. Antes de usarlo como dataset validado, revisar que cada pasaje describe hechos del caso y que el artículo corresponde a esos hechos; distinguir acusación, condena, absolución, precedente y comentario general. Las filas multietiqueta pueden ser legítimas y requieren revisión, no eliminación indiscriminada.

Los scripts pasan diez pruebas de regresión: citas CP/CPP y rangos, sufijos, limpieza, conteos sin solapamiento, agrupación transitiva y por expediente, ausencia de aumento en held-out, unión de catálogo por sigla, detección de fugas y orden correcto del checkpoint del notebook compatible.

Se comprobó entrenamiento y lectura de checkpoints completos con un BERT diminuto y datos sintéticos en CPU. No se entrenó el modelo real ni se evaluó calidad jurídica con esos resultados.
