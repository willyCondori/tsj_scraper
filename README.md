# Scraper de jurisprudencia penal — TSJ Bolivia (vía API directa)

Construye un dataset propio de pares **(hechos del caso → artículo del
Código Penal citado)** a partir de resoluciones públicas de Materia Penal
del Tribunal Supremo de Justicia, consultando **directamente la API**
que usa `genesis.tsj.bo` (encontrada inspeccionando el tráfico de red del
sitio con DevTools).

Es standalone: no toca el backend Django ni el modelo `Hecho`.

## Por qué API directa y no Playwright

La primera versión de este proyecto scrapeaba el sitio con un navegador
headless (Playwright), porque `genesis.tsj.bo` es una SPA. Inspeccionando
la pestaña Network del navegador encontramos que el sitio en realidad
llama a una API REST (`apigenesis.tsj.bo`) que devuelve JSON limpio.
Consultar esa API directo con `requests` es mucho más simple, rápido y
estable — sin navegador, sin selectores CSS frágiles, sin esperas de
renderizado.

## Instalación

```bash
cd tsj_scraper
python -m venv env
source env/bin/activate     # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Mucho más liviano que antes: ya no hace falta `playwright install chromium`.

## Uso — en orden

### 1. Scrapear resoluciones de Materia Penal

```bash
python scraper_api.py
```

Qué hace:
- Llama a `POST /api/v1/resoluciones/busqueda_avanzada` filtrando por
  `idMateria: 1` (Penal), que trae todas las salas penales juntas (Sala
  Penal, Sala Penal 1, Sala Penal 2, Sala Penal Liquidadora).
- Pagina automáticamente por todos los resultados.
- Para cada resolución del listado, llama a
  `GET /api/v1/resoluciones/{id}` para bajar el texto completo del fallo.
- Guarda un JSON por resolución en `data/raw/`.
- Es reanudable: si lo cortás y lo corrés de nuevo, no vuelve a bajar lo
  que ya tiene guardado.

**Recomendación para la primera corrida**: al final del archivo
`scraper_api.py` hay una línea comentada `# main(max_paginas=2)`.
Descomentala (y comentá el `main()` de abajo) para tu primera prueba —
así bajás solo ~100 resoluciones y confirmás que todo funciona antes de
lanzar la corrida completa (puede haber varios cientos de páginas).

El delay entre requests (`DELAY_ENTRE_REQUESTS = 1.5` segundos en el
script) es a propósito — es un servidor público de gobierno, no hay que
saturarlo.

### 2. Generar los pares de entrenamiento

```bash
python extract_pairs.py
```

Extrae de cada resolución:
- El fragmento de **hechos** (heurística basada en marcadores típicos:
  "CONSIDERANDO", "POR TANTO", etc. — revisable/ajustable).
- Los **artículos del Código Penal citados** (regex).
- De regalo, el campo `procesos` que ya trae la propia API (ej. "Robo
  Agravado y Hurto") — esto es más confiable que adivinar el tipo de
  delito por palabras clave, y se guarda como `tipo_proceso` en cada par.

Salidas en `data/processed/`:
- `pares_entrenamiento.jsonl`
- `pares_revision.csv` — **revisá una muestra a mano** antes de confiar
  en el dataset.

### 3. Aumentar con sinónimos jurídicos

```bash
python augment_pairs.py
```

Genera variantes parafraseadas (intento↔tentativa, hurto↔sustracción,
etc.) usando `data/reference/sinonimos_juridicos.json`, y agrega
negativos difíciles por categoría de delito (usando el `tipo_proceso`
real cuando está disponible). Ver comentarios en el propio script para
más detalle.

Salida: `data/processed/pares_aumentados.jsonl`

### 4. Exportar a Excel para revisión

```bash
python export_excel.py
```

Salida: `data/processed/dataset_completo.xlsx` con hojas
`Pares_originales`, `Pares_aumentados`, `Sinonimos`, `Estadisticas`.

### 5. Fine-tuning (cuando el dataset esté validado)

```bash
python finetune_embeddings.py
```

Necesita que armes `articulos_codigo_penal.json` (mapeo número de
artículo → texto) desde tu catálogo `Articulo` existente en Django.

## Sobre la API

- **Base**: `https://apigenesis.tsj.bo/api/v1`
- **Listado**: `POST /resoluciones/busqueda_avanzada`
- **Detalle**: `GET /resoluciones/{id}`
- **Catálogos útiles** (por si querés filtrar distinto):
  `GET /catalogos/materias` (Penal = id 1),
  `GET /catalogos/salas` (Sala Penal = id 24, Sala Penal 1 = id 2,
  Sala Penal 2 = id 5012, Sala Penal Liquidadora = id 4)
- Headers `apikey` y `username` son valores públicos usados por el propio
  frontend del sitio (visibles en su JS), no son credenciales privadas,
  pero **podrían cambiar en el futuro** si el TSJ actualiza el sitio —
  si algún día el scraper empieza a devolver 401/403, hay que volver a
  inspeccionar el tráfico de red y actualizar `HEADERS` en
  `scraper_api.py`.

## Consideraciones

- **Volumen**: la búsqueda por Materia Penal devolvió miles de
  resultados en las pruebas — bastante más que suficiente para un
  dataset de tesis. No hace falta bajar todo; podés limitar con
  `max_paginas` según cuánto tiempo/volumen quieras.
- **Calidad > cantidad**: seguí revisando `pares_revision.csv` con
  atención — la heurística de "hechos" y la regex de artículos van a
  tener ruido en fallos con estructura distinta a la esperada.
- **Ética/legal**: información pública, uso académico. Respetá el
  rate-limit y no satures el servidor del Órgano Judicial.
