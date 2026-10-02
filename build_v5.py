"""v5 conserva evaluación v4 y genera propuestas, sin inventar aprobación jurídica."""
import argparse
import json
from collections import Counter,defaultdict
from pathlib import Path
from dataset_utils import file_hash,fingerprint,grouped_split,read_jsonl,write_jsonl
from extract_sections import section_candidates
from augment_pairs import augment

def build(raw_dir,v4_dir,out_dir):
    raw_dir,v4_dir,out_dir=map(Path,(raw_dir,v4_dir,out_dir))
    catalog=json.loads((v4_dir/'catalogo_cp.json').read_text(encoding='utf-8'))
    baseline=read_jsonl(v4_dir/'pares_entrenamiento.jsonl')
    previous=read_jsonl(v4_dir/'pares_candidatos.jsonl')
    # Los mismos grupos v4 se extienden a TODOS los candidatos, no solo 300 filas.
    assignment={}
    for row in grouped_split([r for r in previous if r['articulo'] in catalog]):
        assignment[str(row['fuente_id'])]=(row['split'],row['grupo_id'])
    for row in baseline:
        if assignment[str(row['fuente_id'])]!=(row['split'],row['grupo_id']):
            raise ValueError('Particiones v4 incompatibles; no continuar')
    out_dir.mkdir(parents=True,exist_ok=False)
    rows=[];errors=[];sources={};seen={};held={fingerprint(r['hechos']) for r in baseline if r['split']!='train'}
    for index,path in enumerate(sorted(raw_dir.glob('*.json')),1):
        sources[path.name]=file_hash(path)
        try:
            data=json.loads(path.read_text(encoding='utf-8-sig'))
            for row in section_candidates(data):
                split,group=assignment.get(str(row['fuente_id']),('review_only',fingerprint(str(row['fuente_id']))))
                row.update(split=split,grupo_id=group)
                row['motivo']='sin_catalogo_cp' if row['articulo'] not in catalog else 'requiere_revision_juridica'
                if split=='train' and fingerprint(row['hechos']) in held:
                    row['split']='review_only';row['motivo']='texto_compartido_con_evaluacion'
                seen.setdefault(fingerprint(row['hechos']),set()).add(row['split']);rows.append(row)
        except (ValueError,TypeError,KeyError) as exc:errors.append({'archivo':path.name,'error':str(exc)})
        if index%2000==0:print(f'v5: {index} documentos',flush=True)
    # Propuestas duplicadas entre splits jamás serán añadidas automáticamente.
    for row in rows:
        if len(seen[fingerprint(row['hechos'])]-{'review_only'})>1:
            row['split']='review_only';row['motivo']='texto_compartido_entre_particiones'
    train_counts=Counter(r['articulo'] for r in baseline if r['split']=='train')
    for row in rows:
        row['soporte_train_v4']=train_counts[row['articulo']]
        row['prioridad_revision']=row['prioridad']+(8 if train_counts[row['articulo']]<5 else 0)
    rows.sort(key=lambda r:(r['split']!='train',-r['prioridad_revision'],r['candidate_id']))
    write_jsonl(out_dir/'propuestas_secciones.jsonl',rows)
    write_jsonl(out_dir/'cola_revision_train.jsonl',[r for r in rows if r['split']=='train' and r['articulo'] in catalog])
    write_jsonl(out_dir/'errores_extraccion.jsonl',errors)
    # Baseline débil conservado para comparar; no fingir que v5 ya fue anotado.
    write_jsonl(out_dir/'pares_entrenamiento.jsonl',baseline)
    write_jsonl(out_dir/'pares_aumentados.jsonl',augment(baseline))
    (out_dir/'catalogo_cp.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'version':'v5','baseline':'v4 congelado','supervision':'débil; nuevas propuestas NO incorporadas sin revisión',
              'rows':len(baseline),'splits':dict(Counter(r['split'] for r in baseline)),
              'proposals':len(rows),'proposal_splits':dict(Counter(r['split'] for r in rows)),
              'proposal_roles':dict(Counter(r['papel_cita'] for r in rows)),'errors':len(errors),
              'v4_dataset_sha256':file_hash(v4_dir/'pares_entrenamiento.jsonl'),
              'source_files':sources,'code_hashes':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    (out_dir/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (out_dir/'INSTRUCCIONES.md').write_text('''# Dataset v5\n\nLos 300 pares v4 y sus particiones se conservan. Las nuevas propuestas están pendientes: no son datos aprobados.\n\nRevisar cola_revision_train.jsonl, completar candidate_id, decision (aprobar/rechazar), articulos_aprobados (lista), revisor y justificacion en revisiones.jsonl. Aprobar solo si los hechos justifican los artículos y todos los positivos conocidos están incluidos. Se conservan los originales para auditar.\n\nEjecutar apply_reviews.py --run-dir RUTA --reviews revisiones.jsonl --output-dir RUTA_NUEVA. La herramienta solo incorpora propuestas de train y rechaza fugas. No modifica validation/test ni el v4.\n\nEl catálogo no fue declarado jurídicamente vigente. Verificar versiones antes de uso final. No subir manifiestos con datos personales a repositorios públicos.\n''',encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('source_files','code_hashes')},ensure_ascii=False,indent=2))
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',required=True);p.add_argument('--v4-dir',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();build(a.raw_dir,a.v4_dir,a.output_dir)
