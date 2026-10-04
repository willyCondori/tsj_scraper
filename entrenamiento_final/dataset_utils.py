"""Normalización, catálogo y partición compartidos por pipeline y entrenamiento."""
import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

SUFFIX = r'(?:bis|ter|quater|quinquies|sexies|septies)'
NUMBER = rf'\d+(?:\s*{SUFFIX})?'

def normalize_text(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(value or ''))).strip()

def article_number(value):
    text = normalize_text(value).lower()
    if re.fullmatch(r'\d+\.0', text): text = text[:-2]
    match = re.fullmatch(rf'(\d+)\s*({SUFFIX})?', text)
    if not match: raise ValueError(f'Número inválido: {value!r}')
    return str(int(match[1])) + (f' {match[2]}' if match[2] else '')

def fingerprint(value):
    return hashlib.sha256(normalize_text(value).casefold().encode('utf-8')).hexdigest()

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    temp.replace(path)

def load_catalog(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if isinstance(data, dict):
        data = [{'numero_articulo': k, 'contenido': v, 'norma_sigla': 'CP'} for k,v in data.items()]
    if not isinstance(data, list): raise ValueError('Catálogo: se espera lista de objetos o mapa CP número→texto.')
    catalog = {}
    for row in data:
        if normalize_text(row.get('norma_sigla')).upper() != 'CP': continue
        number = article_number(row['numero_articulo'])
        text = normalize_text(html.unescape(re.sub(r'<[^>]+>', ' ', str(row.get('contenido') or ''))))
        if not text: raise ValueError(f'CP {number}: contenido vacío')
        if number in catalog and catalog[number] != text: raise ValueError(f'CP {number}: textos incompatibles')
        catalog[number] = text
    if not catalog: raise ValueError('No hay artículos con norma_sigla=CP')
    return catalog

def grouped_split(rows, seed=42):
    """Unión transitiva por resolución, documento y texto; se divide antes de aumentar."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    seen = {}
    for i,row in enumerate(rows):
        source = row.get('fuente_id')
        if source is None or normalize_text(source) == '': raise ValueError('Falta fuente_id')
        keys = [('source', str(source)), ('text', fingerprint(row['hechos']))]
        if row.get('documento_hash'): keys.append(('document', row['documento_hash']))
        case = normalize_text(row.get('nro_expediente')).casefold()
        if case and re.search(r'\d{4}',case): keys.append(('case',case))
        for key in keys:
            if key in seen: parent[find(i)] = find(seen[key])
            else: seen[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)): groups[find(i)].append(i)
    result = []
    for indices in groups.values():
        # Contenido estable: independencia del orden de archivos/filas.
        identity = sorted({str(rows[i]['fuente_id']) for i in indices})
        group = fingerprint(json.dumps(identity, ensure_ascii=False))
        value = int(fingerprint(f'{seed}:{group}')[:16],16) / 2**64
        split = 'train' if value < .8 else ('validation' if value < .9 else 'test')
        for i in indices: result.append({**rows[i], 'grupo_id': group, 'split': split})
    return sorted(result, key=lambda r:(str(r['fuente_id']),r['articulo'],r['hechos']))
