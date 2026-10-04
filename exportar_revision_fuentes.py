"""Expedientes de revisión con fuente completa; inferencias pendientes de aprobación."""
import json
import html
from collections import defaultdict,Counter
from pathlib import Path
from dataset_utils import read_jsonl,write_jsonl,fingerprint,file_hash
from extract_pairs import clean_document
from extract_structured_v7 import extract_document

def export(source,raw_dir,out):
    source,raw_dir,out=Path(source),Path(raw_dir),Path(out)
    out.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(source/'pares_aumentados.jsonl');grouped=defaultdict(list)
    # No abrir fuentes de test ni producir propuestas para sus etiquetas.
    for r in rows:
        if r['split']!='test':grouped[str(r['fuente_id'])].append(r)
    reviews=[];errors=[];roles=Counter();norms=Counter();cards=[]
    for source_id,items in grouped.items():
        path=raw_dir/(source_id+'.json')
        if not path.is_file():errors.append({'fuente_id':source_id,'error':'fuente_ausente'});continue
        raw=json.loads(path.read_text(encoding='utf-8-sig'))
        text=clean_document(raw.get('contenido'))
        split=items[0]['split'];structured=extract_document(raw,split)
        for relation in structured['relaciones']:
            relation['identificador_norma_articulo']=relation['norma_id']+':'+relation['articulo']
            norms[relation['norma_id']]+=1
            roles[relation['papel_propuesto']]+=1
        # Evidencia de hechos, decisiones y citas separada; ningún vínculo se aprueba por cercanía.
        document={'fuente_id':source_id,'split':split,'fuente_sha256':file_hash(path),
          'nro_resolucion':raw.get('nro_resolucion'),'nro_expediente':raw.get('nro_expediente'),
          'url_pdf_escaneado':raw.get('url_pdf_escaneado'),'texto_fuente':text,**structured}
        (out/(source_id+'.json')).write_text(json.dumps(document,ensure_ascii=False,indent=2),encoding='utf-8')
        seen=set()
        for row in items:
            key=fingerprint(row['hechos'])
            if key in seen:continue
            seen.add(key)
            labels=sorted({r['articulo'] for r in items if fingerprint(r['hechos'])==key})
            reviews.append({'query_id':key,'fuente_id':source_id,'split':split,'grupo_id':row['grupo_id'],
              'hechos_originales':row['hechos'],'articulos_actuales':labels,'documento_evidencia':source_id+'.json',
              'estado_revision':'pendiente','hechos_aprobados':'','articulos_aprobados':[],
              'papel_confirmado':'','persona_conducta_confirmada':'','revisor':'','justificacion':'','evidencia_fuente':''})
        esc=html.escape
        cards.append('<article><h2>'+esc(source_id+' '+str(raw.get('nro_resolucion')))+' — '+esc(split)+'</h2><p>No se aprobaron relaciones automáticamente.</p><details><summary>Hechos, decisiones y relaciones candidatas</summary><pre>'+esc(json.dumps(structured,ensure_ascii=False,indent=2))+'</pre></details><details><summary>Fuente completa</summary><pre>'+esc(text)+'</pre></details></article>')
    write_jsonl(out/'revisiones_documentales.jsonl',reviews);write_jsonl(out/'errores.jsonl',errors)
    summary={'documentos':len(grouped)-len(errors),'consultas_para_revision':len(reviews),'errores':len(errors),
      'normas_detectadas':dict(norms),'papeles_heuristicos':dict(roles),'fuentes_test_abiertas':0,
      'relaciones_aprobadas':0,'aviso':'Roles y hechos son propuestas heurísticas; verificar resolución y aplicación antes de aprobar.'}
    (out/'resumen.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'revision_fuentes.html').write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1100px;margin:auto;font:16px system-ui}article{border:1px solid #aaa;padding:20px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Revisión documental train/validation</h1>'+''.join(cards),encoding='utf-8')
    return summary

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--raw-dir',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(export(a.source,a.raw_dir,a.output),ensure_ascii=False,indent=2))
