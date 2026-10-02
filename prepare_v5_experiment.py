"""Variante experimental débil y auditoría; no convierte heurísticas en aprobación."""
import argparse,json,html,re
from collections import Counter
from pathlib import Path
from dataset_utils import read_jsonl,write_jsonl,fingerprint,file_hash
from augment_pairs import augment
from extract_sections import plain

PAST_ACTIONS=re.compile(r'\b(?:sustrajo|sustrajeron|golpeo|golpearon|agredio|agredieron|amenazo|amenazaron|disparo|dispararon|entrego|entregaron|falsifico|vendio|compro|engano|violo|abuso|lesiono|mato|apunalo|atropello|exigio|deposito|transfirio|recibio|retuvo|ingreso|obligo|incumplio|oculto|transporto|suministro|murio|fallecio|causo|apodero)\b')

def eligible(row,catalog,excluded):
    return (row['split']=='train' and row['articulo'] in catalog and row['articulo'] not in excluded
            and row['distancia_cita']<=250 and row['senales_facticas']>=2
            and len(PAST_ACTIONS.findall(plain(row['hechos'])))>=2
            and row['senales_procesales']<=1 and row['papel_cita']=='condena' and len(row['hechos'])>=150)

def prepare(directory,v4_dir):
    directory,v4_dir=Path(directory),Path(v4_dir)
    baseline=read_jsonl(directory/'pares_entrenamiento.jsonl');v4=read_jsonl(v4_dir/'pares_entrenamiento.jsonl')
    if baseline!=v4:raise ValueError('Baseline alterado')
    v5_manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    v4_manifest=json.loads((v4_dir/'manifest.json').read_text(encoding='utf-8'))
    if v5_manifest['source_files']!=v4_manifest['source_files']:raise ValueError('Crudos distintos del v4; auditar antes de comparar')
    catalog=json.loads((directory/'catalogo_cp.json').read_text(encoding='utf-8'))
    proposals=read_jsonl(directory/'propuestas_secciones.jsonl')
    excluded=set(json.loads((Path(__file__).parent/'data/reference/politica_dataset.json').read_text(encoding='utf-8'))['excluir_etiquetas'])
    held=[r for r in baseline if r['split']!='train'];held_groups={r['grupo_id'] for r in held};held_text={fingerprint(r['hechos']) for r in held}
    added=[];seen={(fingerprint(r['hechos']),r['articulo']) for r in baseline}
    for row in proposals:
        if not eligible(row,catalog,excluded):continue
        if row['grupo_id'] in held_groups or fingerprint(row['hechos']) in held_text:raise ValueError('Fuga experimental')
        key=(fingerprint(row['hechos']),row['articulo'])
        if key in seen:continue
        seen.add(key);added.append({**row,'articulo_texto':catalog[row['articulo']],
                                  'supervision':'débil experimental; rol y factualidad heurísticos; pendiente de revisión'})
    variant=directory/'experimental_debil';variant.mkdir(exist_ok=False)
    all_rows=baseline+added;write_jsonl(variant/'pares_entrenamiento.jsonl',all_rows)
    write_jsonl(variant/'pares_aumentados.jsonl',augment(all_rows))
    (variant/'catalogo_cp.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'rows':len(all_rows),'added_weak_train_pairs':len(added),'splits':dict(Counter(r['split'] for r in all_rows)),
              'supervision':'débil experimental; ninguna nueva etiqueta jurídicamente aprobada',
              'selection_rule':'train, CP no general, distancia<=250, señales fácticas>=2, acciones en pasado>=2, procesales<=1, papel_cita=condena heurístico, longitud>=150',
              'baseline_sha256':file_hash(directory/'pares_entrenamiento.jsonl'),'dataset_sha256':file_hash(variant/'pares_aumentados.jsonl'),
              'validation_test_unchanged':held==[r for r in all_rows if r['split']!='train']}
    (variant/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    audit={'baseline_exact_v4':True,'raw_hashes_exact_v4':True,'catalog_exact_v4':catalog==json.loads((v4_dir/'catalogo_cp.json').read_text(encoding='utf-8')),
           'proposals':len(proposals),'proposal_articles':len({r['articulo'] for r in proposals}),
           'train_review_candidates':sum(r['split']=='train' and r['articulo'] in catalog for r in proposals),
           'experimental':manifest,'pending_missing_catalog':dict(Counter(r['articulo'] for r in proposals if r['articulo'] not in catalog))}
    (directory/'auditoria_v5.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    # Muestra legible: conservar hechos completos y evidencia, sin botones de aprobación automática.
    selected=[r for r in proposals if r['split']=='train' and r['articulo'] in catalog and r['articulo'] not in excluded][:60]
    cards=[]
    for row in selected:
        esc=lambda value:html.escape(str(value))
        cards.append(f'<article><h2>CP {esc(row["articulo"])} · {esc(row["candidate_id"][:12])}</h2><p>Rol heurístico: {esc(row["papel_cita"])}; soporte train v4: {row["soporte_train_v4"]}</p><p>{esc(row["hechos"])}</p><details><summary>Evidencia de cita</summary><p>{esc(row["evidencia_cita"])}</p><pre>{esc(row["candidate_id"])}</pre></details><a href="{esc(row["fuente_url"])}">Fuente TSJ</a></article>')
    page='<!doctype html><meta charset="utf-8"><title>Revisión v5</title><style>body{max-width:1000px;margin:32px auto;font:17px/1.6 sans-serif;background:#fafafa}article{background:white;padding:20px;margin:24px 0;border:1px solid #ccc}pre{white-space:pre-wrap}h2{font-size:20px}</style><h1>Propuestas v5 pendientes</h1><p>Hechos y rol de cita son propuestas heurísticas. Esta página no aprueba etiquetas. Las fuentes deben comprobarse.</p>'+''.join(cards)
    (directory/'revision_muestra.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='pending_missing_catalog'},ensure_ascii=False,indent=2));return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--v4-dir',required=True);a=p.parse_args();prepare(a.run_dir,a.v4_dir)
