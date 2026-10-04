"""Limpieza mínima reversible y aplicación estricta de revisiones en train."""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from dataset_utils import file_hash, fingerprint, read_jsonl, write_jsonl

HEADER=re.compile(r'^\s*(?:(?:[IVX]+|\d+)(?:\.\d+)*[.)]?\s*)?(?:CONSIDERANDO|VISTOS|ACTUACIONES PROCESALES VINCULADAS AL RECURSO|ANTECEDENTES(?: PROCESALES)?|RELACI[ÓO]N DE LOS HECHOS|HECHOS PROBADOS)\s*[:.\-]?\s*',re.I)

def clean_prefix(text):
    match=HEADER.match(text)
    if not match:return text
    result=text[match.end():]
    # No eliminar cuerpos de oraciones ni resumir. Preservar exactamente el resto.
    return result if len(result)>=100 else text

def build(source, output, reviews=None):
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError('No sobrescribir referencia')
    rows=read_jsonl(source/'pares_aumentados.jsonl')
    manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    for name,key in [('pares_aumentados.jsonl','dataset_sha256'),('catalogo_cp.json','catalog_sha256')]:
        if file_hash(source/name)!=manifest[key]:raise ValueError('Referencia modificada: '+name)
    catalog=json.loads((source/'catalogo_cp.json').read_text(encoding='utf-8'))
    indexed={};original_queries={fingerprint(r['hechos']) for r in rows if r['split']=='train'}
    for r in read_jsonl(reviews) if reviews else []:
        if r.get('estado_revision')!='aprobado':continue
        if r.get('split')!='train' or r.get('query_id') not in original_queries:raise ValueError('Revisión no pertenece a train')
        if any(not r.get(k) for k in ('revisor','justificacion','evidencia_fuente','hechos_aprobados','articulos_aprobados')):raise ValueError('Revisión aprobada incompleta')
        if r.get('papel_confirmado') not in ('condena','aplicacion_confirmada'):raise ValueError('No aprobar citas aisladas como aplicación')
        if len(r['hechos_aprobados'])<100:raise ValueError('Hechos revisados demasiado cortos')
        labels=r['articulos_aprobados']
        if not isinstance(labels,list) or not labels or len(labels)!=len(set(labels)) or any(n not in catalog for n in labels):raise ValueError('Artículos revisados inválidos')
        if r['query_id'] in indexed:raise ValueError('Una revisión por consulta, incluyendo todos sus artículos')
        indexed[r['query_id']]=r
    output.mkdir(parents=True,exist_ok=True)
    result=[];audit=[];used=set()
    for row in rows:
        if row['split']!='train':result.append(row);continue
        key=fingerprint(row['hechos']);revision=indexed.get(key)
        if revision:
            if key in used:continue
            used.add(key)
            for number in revision['articulos_aprobados']:
                result.append({**row,'hechos':revision['hechos_aprobados'],'articulo':number,'articulo_texto':catalog[number],
                  'estado_revision':'aprobado','revision_evidencia':revision,'metodo_texto':'revision_documental'})
            audit.append({'query_id':key,'accion':'revision_documental','revision':revision})
        else:
            clean=clean_prefix(row['hechos'])
            new=dict(row)
            if clean!=row['hechos']:
                new.update(hechos=clean,hechos_originales=row['hechos'],metodo_texto='solo_encabezado_inicial')
                audit.append({'query_id':key,'fuente_id':row['fuente_id'],'articulo':row['articulo'],'antes':row['hechos'],'despues':clean})
            result.append(new)
    # Colisiones tras limpieza no se resuelven mezclando particiones.
    seen={}
    for r in result:
        key=fingerprint(r['hechos'])
        if key in seen and seen[key]!=r['split']:raise ValueError('Limpieza produce fuga entre particiones')
        seen[key]=r['split']
    assert [r for r in result if r['split']!='train']==[r for r in rows if r['split']!='train']
    write_jsonl(output/'pares_aumentados.jsonl',result)
    write_jsonl(output/'auditoria_cambios_train.jsonl',audit)
    shutil.copy2(source/'catalogo_cp.json',output/'catalogo_cp.json')
    summary={'version':'v6_controlada_1','dataset_sha256':file_hash(output/'pares_aumentados.jsonl'),
      'catalog_sha256':file_hash(output/'catalogo_cp.json'),'origen_sha256':manifest['dataset_sha256'],
      'splits':dict(Counter(r['split'] for r in result)),'cambios_train':len(audit),'revisiones_aprobadas':len(indexed),
      'validation_test_sin_cambios':True,'catalog_entries':len(catalog),'supervision':'débil salvo revisiones documentales explícitas'}
    (output/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--reviews')
    a=p.parse_args();print(json.dumps(build(a.source,a.output,a.reviews),ensure_ascii=False,indent=2))
