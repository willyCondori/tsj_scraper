"""Una ejecución nueva y trazable; nunca sobrescribe el dataset original."""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from dataset_utils import article_number,file_hash,fingerprint,grouped_split,load_catalog,read_jsonl,write_jsonl
from extract_pairs import extract
from augment_pairs import augment

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',default='data/raw');p.add_argument('--catalog',default='data/reference/articulos_codigo_penal.json')
    p.add_argument('--output-root',default='data/processed/runs');p.add_argument('--run-name');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--candidate-file',help='Reutilizar candidatos verificados de un run anterior')
    p.add_argument('--policy',default=str(Path(__file__).parent/'data/reference/politica_dataset.json'))
    a=p.parse_args(); catalog=load_catalog(a.catalog)
    directory=Path(a.output_root)/(a.run_name or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    directory.mkdir(parents=True,exist_ok=False)
    sources={path.name:file_hash(path) for path in sorted(Path(a.raw_dir).glob('*.json'))}
    policy=json.loads(Path(a.policy).read_text(encoding='utf-8'))
    excluded={article_number(n) for n in policy.get('excluir_etiquetas',[])}
    if a.candidate_file:
        previous=Path(a.candidate_file).parent/'manifest.json'
        if not previous.exists(): raise ValueError('Candidatos sin manifest de procedencia')
        if json.loads(previous.read_text(encoding='utf-8'))['source_files']!=sources:
            raise ValueError('Los JSON crudos cambiaron; vuelve a extraer candidatos')
        rows=read_jsonl(a.candidate_file);metadata={}
        for row in rows:
            source=str(row['fuente_id'])
            if source not in metadata:
                metadata[source]=json.loads((Path(a.raw_dir)/f'{source}.json').read_text(encoding='utf-8'))
            raw=metadata[source]
            row.update({key:raw.get(key) for key in ('nro_expediente','fecha_emision')})
            row['resultado_resolucion']=raw.get('formas_resoluciones')
        write_jsonl(directory/'pares_candidatos.jsonl',rows)
    else: rows=extract(a.raw_dir,directory)
    accepted=[]; quarantine=[]; duplicates=[];seen=set()
    for row in rows:
        number=article_number(row['articulo']); row['articulo']=number
        reason='sin_catalogo_cp' if number not in catalog else ('etiqueta_general_separada' if number in excluded else ('requiere_revision_contenido' if row['tipo_contenido'] not in ('factico','mixto') else None))
        if reason: quarantine.append({**row,'motivo':reason});continue
        key=(fingerprint(row['hechos']),number)
        if key in seen: duplicates.append({**row,'motivo':'duplicado_texto_articulo'})
        else: seen.add(key);accepted.append(row)
    # Agrupar ANTES de quitar duplicados conserva vínculos entre fuentes compartidas.
    all_split=grouped_split([r for r in rows if r['articulo'] in catalog],a.seed)
    splits={(str(r['fuente_id']),r['articulo'],r['hechos']):(r['grupo_id'],r['split']) for r in all_split}
    for row in accepted:
        row['grupo_id'],row['split']=splits[(str(row['fuente_id']),row['articulo'],row['hechos'])]
        row['articulo_texto']=catalog[row['articulo']]
    write_jsonl(directory/'pares_entrenamiento.jsonl',accepted)
    write_jsonl(directory/'pares_revision.jsonl',quarantine)
    write_jsonl(directory/'pares_duplicados.jsonl',duplicates)
    write_jsonl(directory/'pares_aumentados.jsonl',augment(accepted))
    (directory/'catalogo_cp.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'seed':a.seed,'catalog_sha256':file_hash(a.catalog),'source_files':sources,'rows':len(accepted),
        'policy':policy,'policy_sha256':file_hash(a.policy),
        'candidate_input_sha256':file_hash(a.candidate_file) if a.candidate_file else None,
        'quarantine':len(quarantine),'duplicates':len(duplicates),'splits':dict(Counter(r['split'] for r in accepted)),
        'supervision':'débil; etiquetas extraídas de citas, revisión pendiente',
        'code_hashes':{path.name:file_hash(path) for path in Path(__file__).parent.glob('*.py')}}
    (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Run: {directory.resolve()}\nSplits: {manifest["splits"]}\nRevisión: {len(quarantine)}')

if __name__=='__main__': main()
