"""Extracción auditable v7. No convierte citas en etiquetas ni genera hechos nuevos."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from dataset_utils import NUMBER, article_number, fingerprint, file_hash, normalize_text, read_jsonl
from extract_pairs import clean_document, GENERIC_CITATION
from extract_sections import plain
from build_v6 import ACTIONS, PROCEDURE, THEORY

NORMS = r'(?:c[óo]digo\s+de\s+procedimiento\s+penal|c[óo]digo\s+procesal\s+penal|c[óo]digo\s+penal|C\.?\s*P\.?\s*P\.?\b|C\.?\s*P\.?\b|ley\s*(?:n[°ºo.]?\s*)?\d+)'
CITES = re.compile(rf'\b(?:art[íi]culos?|arts?)\.?\s*(?P<nums>{NUMBER}(?:\s*(?:,|y|e|al|a|[-–])\s*{NUMBER})*)(?:\s*[º°])?(?:\s*(?:inciso|inc\.)\s*(?P<inc>\d+|[a-z]))?\s*,?\s*(?:del|de\s+la|de)?\s*(?P<norm>{NORMS})', re.I)
DECISION = re.compile(r'\b(?:conden\w*|absolvi\w*|absuelv\w*|absoluc\w*|declar\w*\s+(?:infundado|inadmisible|improcedente|procedente|culpable|autor)|anul\w*|cas[oó]\s+el|dejo\s+sin\s+efecto)\b', re.I)
HEADING = re.compile(r'^\s*(?:(?:[IVXLCDM]+|\d+)(?:[.\-:]|\)\s*)?\s*)?(?P<h>hechos\s+(?:probados|acreditados|alegados)|relaci[oó]n\s+(?:f[aá]ctica|de\s+(?:los\s+)?hechos)|antecedentes(?:\s+procesales)?|fundamentos\s+(?:jur[ií]dicos|legales)|doctrina\s+legal|por\s+tanto)\s*[:.\-]?\s*', re.I)
ANSWER = re.compile(r'\b(?:conden\w*|absolvi\w*|absuelv\w*|acusad\w*\s+por|delito\s+de|tipific\w*|previsto\s+(?:y\s+sancionado\s+)?en)\b', re.I)

def sentences(text):
    """Offsets exactos en texto limpio; no separar art., inc. ni iniciales."""
    start = 0
    for m in re.finditer(r'\n+|(?<=[!?;])\s+|(?<=\.)\s+(?=[A-ZÁÉÍÓÚÑ])', text):
        if m[0].startswith('\n') or not re.search(r'\b(?:art|arts|inc|nro|dr|dra|sr|sra)\.$', text[start:m.start()], re.I):
            if text[start:m.start()].strip(): yield start, m.start()
            start = m.end()
    if text[start:].strip(): yield start, len(text)

def norm_id(text):
    p = plain(text).replace('.', '').replace(' ', '')
    if p.startswith('ley'): return 'LEY:' + str(int(re.search(r'\d+', p)[0]))
    return 'CPP' if 'procedimiento' in p or 'procesal' in p or p == 'cpp' else 'CP'

def citations(text):
    for m in CITES.finditer(text):
        tokens = list(re.finditer(NUMBER, m['nums'], re.I)); numbers = []
        for i, t in enumerate(tokens):
            n = article_number(t[0]); numbers.append(n)
            if i:
                prev = article_number(tokens[i-1][0]); sep = m['nums'][tokens[i-1].end():t.start()].strip()
                if sep in ('al', 'a', '-', '–') and prev.isdigit() and n.isdigit() and 0 < int(n)-int(prev) <= 50:
                    numbers.extend(map(str, range(int(prev)+1, int(n))))
        for n in dict.fromkeys(numbers):
            yield {'norma_id': norm_id(m['norm']), 'articulo': n, 'inciso': m['inc'], 'inicio': m.start(), 'fin': m.end(), 'texto': m[0]}

def role(text):
    p = plain(text)
    if re.search(r'precedente|auto supremo\s+\d|sentencia constitucional\s+\d', p): return 'precedente_o_documento_citado'
    matches = []
    for name, pattern in [('absolucion', r'absolvi\w*|absuelv\w*|absoluc\w*'), ('condena', r'conden\w*|declar\w*\s+(?:autor|culpable)'), ('alegado', r'acus\w*|imput\w*|denunci\w*'), ('rechazado', r'no\s+(?:es\s+)?aplica\w*|inaplicable'), ('fundamento_procesal', r'declar\w*\s+(?:infundado|inadmisible)|anul\w*|dejo\s+sin\s+efecto')]:
        if re.search(pattern, p): matches.append(name)
    return matches[0] if len(matches) == 1 else ('ambiguo' if matches else 'cita_sin_aplicacion_demostrada')

def extract_document(data, split='revision_only'):
    text = clean_document(data.get('contenido')); source = str(data['id'])
    base = {'fuente_id': source, 'documento_hash': fingerprint(text), 'split': split, 'estado_revision': 'pendiente', 'offsets_sobre': 'clean_document(contenido)'}
    output = {k: [] for k in ('hechos', 'decisiones', 'relaciones', 'citas_sin_norma')}
    section = 'no_identificada'; facts = []
    for start, end in sentences(text):
        raw = text[start:end]; h = HEADING.match(raw)
        if h:
            section = plain(h['h']); start += h.end(); raw = text[start:end]
        if not raw.strip(): continue
        p = plain(raw); decision = bool(DECISION.search(p))
        local_cites = list(citations(raw))
        if decision:
            did = fingerprint(f'{source}|decision|{start}|{end}')
            output['decisiones'].append({**base, 'decision_id': did, 'texto': raw, 'inicio': start, 'fin': end, 'seccion': section, 'papel': role(raw), 'organo_etapa': 'por_revisar', 'persona_id': None})
        else: did = None
        # No usar condenas, nombres de delitos ni artículos como entrada del modelo.
        actions = [m[0] for m in ACTIONS.finditer(p) if m[0] not in ('abuso', 'sujeto', 'deposito', 'impacto')]
        if actions and not decision and not ANSWER.search(p) and not PROCEDURE.search(p) and not THEORY.search(p) and not re.search(r'debido proceso|violo\s+(?:el|la|los|las)\s+(?:derecho|garantia|norma|legalidad)|doctrina|bien juridico|tipo penal|jurisprudencia|precedente', p):
            cleaned = CITES.sub('', raw); cleaned = GENERIC_CITATION.sub('', cleaned)
            cleaned = normalize_text(cleaned)
            if len(cleaned) >= 60:
                fid = fingerprint(f'{source}|hecho|{start}|{end}')
                f = {**base, 'hecho_id': fid, 'hechos_originales': raw, 'hechos_limpios': cleaned, 'hechos_resumidos': cleaned, 'metodo_resumen': 'extractivo_oracion_completa', 'inicio': start, 'fin': end, 'seccion': section, 'estatus_hecho': 'probado_reportado' if 'probados' in section or 'acreditados' in section else 'por_revisar', 'persona_id': None, 'evidencia_resumen': [{'inicio': start, 'fin': end, 'texto': raw}]}
                output['hechos'].append(f); facts.append(f)
        for c in local_cites:
            cid = fingerprint(f'{source}|{start+c["inicio"]}|{c["norma_id"]}|{c["articulo"]}')
            output['relaciones'].append({**base, 'relacion_id': cid, 'norma_id': c['norma_id'], 'articulo': c['articulo'], 'norma_articulo_id': c['norma_id']+':'+c['articulo'], 'inciso': c['inciso'], 'cita_original': c['texto'], 'cita_inicio': start+c['inicio'], 'cita_fin': start+c['fin'], 'evidencia': raw, 'evidencia_inicio': start, 'evidencia_fin': end, 'papel_propuesto': role(raw), 'decision_id': did, 'hecho_id': None, 'persona_id': None, 'usable_entrenamiento': False, 'version_norma': 'por_verificar', 'motivo': 'vinculo_hecho_persona_y_aplicacion_requieren_revision'})
        for m in GENERIC_CITATION.finditer(raw):
            if not any(c['inicio'] <= m.start() < c['fin'] for c in local_cites):
                output['citas_sin_norma'].append({**base, 'texto': m[0], 'inicio': start+m.start(), 'fin': start+m.end(), 'evidencia': raw, 'motivo': 'norma_no_resuelta_no_inferir_CP'})
    # Cercanía es una sugerencia para revisión, nunca vínculo confirmado.
    for r in output['relaciones']:
        nearest = sorted(facts, key=lambda f: min(abs(f['inicio']-r['cita_inicio']), abs(f['fin']-r['cita_inicio'])))[:3]
        r['hechos_candidatos'] = [f['hecho_id'] for f in nearest]
    return output

def build(raw_dir, baseline, out):
    out = Path(out)
    if out.exists(): raise ValueError('Usar una carpeta nueva; no sobrescribir resultados')
    out.mkdir(parents=True)
    known = {}; conflicting = set()
    for r in read_jsonl(Path(baseline)/'pares_entrenamiento.jsonl'):
        source = str(r['fuente_id'])
        if source in known and known[source] != r['split']: conflicting.add(source)
        known[source] = r['split']
    if conflicting: raise ValueError('Fuentes con particiones incompatibles')
    counts = Counter(); roles = Counter(); norms = Counter(); sample = []; errors = []; sources = []
    names = ['hechos', 'decisiones', 'relaciones', 'citas_sin_norma']
    streams = {k: (out/(k+'.jsonl')).open('w', encoding='utf-8') for k in names}
    try:
        paths = sorted(Path(raw_dir).glob('*.json'))
        if not paths: raise ValueError('Sin documentos fuente')
        for i, path in enumerate(paths, 1):
            try:
                data = json.loads(path.read_text(encoding='utf-8-sig'))
                result = extract_document(data, known.get(str(data['id']), 'revision_only'))
                sources.append({'archivo': path.name, 'sha256': file_hash(path)})
                for k, rows in result.items():
                    for r in rows: streams[k].write(json.dumps(r, ensure_ascii=False)+'\n')
                    counts[k] += len(rows)
                for r in result['relaciones']:
                    roles[r['papel_propuesto']] += 1; norms[r['norma_id']] += 1
                if len(sample) < 40 and result['hechos'] and result['decisiones'] and result['relaciones']:
                    sample.append({'fuente_id': data['id'], **result})
            except (ValueError, KeyError, TypeError) as e: errors.append({'archivo': path.name, 'error': str(e)})
            if i % 1000 == 0: print(f'v7: {i}/{len(paths)}', flush=True)
    finally:
        for stream in streams.values(): stream.close()
    for name, value in [('muestra_revision.json', sample), ('errores.json', errors), ('fuentes_manifest.json', sources)]:
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    report = {'version': 'v7_estructurado_revision', 'documentos': len(sources), 'errores': len(errors), 'filas': dict(counts), 'papeles_propuestos': dict(roles), 'normas': dict(norms), 'pares_aprobados_generados': 0, 'baseline_sha256': file_hash(Path(baseline)/'pares_entrenamiento.jsonl'), 'codigo_sha256': file_hash(__file__), 'advertencia': 'Extraccion heuristica, no validacion juridica. revision_only no es train. No entrenar directamente estas relaciones.'}
    (out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--raw-dir', required=True); p.add_argument('--baseline', required=True); p.add_argument('--out-dir', required=True)
    a = p.parse_args(); build(a.raw_dir, a.baseline, a.out_dir)
