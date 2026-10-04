"""Incorporar exclusivamente anotaciones explícitas y trazables de train."""
import argparse,json
from pathlib import Path
from dataset_utils import fingerprint,read_jsonl,write_jsonl,article_number
from augment_pairs import augment

def apply(run_dir,reviews_path,output_dir):
    directory,output=Path(run_dir),Path(output_dir)
    catalog=json.loads((directory/'catalogo_cp.json').read_text(encoding='utf-8'))
    rows=read_jsonl(directory/'pares_entrenamiento.jsonl')
    candidates={r['candidate_id']:r for r in read_jsonl(directory/'propuestas_secciones.jsonl')}
    held=[r for r in rows if r['split']!='train'];held_text={fingerprint(r['hechos']) for r in held}
    held_sources={str(r['fuente_id']) for r in held};held_groups={r['grupo_id'] for r in held}
    seen=set();added=0
    for review in read_jsonl(reviews_path):
        cid=review['candidate_id']
        if cid in seen:raise ValueError('Revisión duplicada')
        seen.add(cid);candidate=candidates[cid]
        if candidate['split']!='train':raise ValueError('Solo incorporar candidatos train; evaluación congelada')
        if review['decision']=='rechazar':
            rows=[r for r in rows if r.get('candidate_id')!=cid]
            continue
        if review['decision']!='aprobar':raise ValueError('Decisión inválida')
        if not review.get('revisor') or not review.get('justificacion'):raise ValueError('Falta revisor o justificación')
        if candidate['split']!='train':raise ValueError('Solo incorporar candidatos train; evaluación congelada')
        text=review.get('hechos_corregidos') or candidate['hechos']
        if not isinstance(text,str) or len(text.strip())<100:raise ValueError('Hechos insuficientes')
        if fingerprint(text) in held_text or str(candidate['fuente_id']) in held_sources or candidate['grupo_id'] in held_groups:
            raise ValueError('Fuga a evaluación')
        labels=review.get('articulos_aprobados')
        if not isinstance(labels,list) or not labels:raise ValueError('Falta lista de artículos revisados')
        # Una revisión sustituye todas las etiquetas débiles de esa propuesta.
        rows=[r for r in rows if r.get('candidate_id')!=cid]
        for number in dict.fromkeys(article_number(n) for n in labels):
            if number not in catalog:raise ValueError('Artículo sin catálogo; corregir referencia primero')
            rows.append({**candidate,'hechos':text,'articulo':number,'articulo_texto':catalog[number],
                         'estado_revision':'aprobado','revisor':review['revisor'],'justificacion':review['justificacion'],
                         'articulos_aprobados':labels});added+=1
    # Relaciones transitivas nuevas: comprobar también documentos y expedientes.
    assignments={}
    for row in rows:
        keys=[('text',fingerprint(row['hechos'])),('source',str(row['fuente_id'])),('group',row['grupo_id'])]
        if row.get('documento_hash'):keys.append(('document',row['documento_hash']))
        if row.get('nro_expediente'):keys.append(('case',str(row['nro_expediente']).strip().casefold()))
        for key in keys:
            if key in assignments and assignments[key]!=row['split']:raise ValueError('Fuga de grupo tras revisión')
            assignments[key]=row['split']
    unique={}
    for row in rows:
        key=(fingerprint(row['hechos']),row['articulo'])
        if key not in unique or row.get('estado_revision')=='aprobado':unique[key]=row
    rows=list(unique.values());output.mkdir(parents=True,exist_ok=False)
    if [r for r in rows if r['split']!='train']!=held:raise ValueError('Evaluación alterada tras revisión')
    write_jsonl(output/'pares_entrenamiento.jsonl',rows);write_jsonl(output/'pares_aumentados.jsonl',augment(rows))
    (output/'catalogo_cp.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2),encoding='utf-8')
    write_jsonl(output/'revisiones_aplicadas.jsonl',read_jsonl(reviews_path))
    print(f'Pares revisados incorporados: {added}; total deduplicado: {len(rows)}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--reviews',required=True);p.add_argument('--output-dir',required=True)
    a=p.parse_args();apply(a.run_dir,a.reviews,a.output_dir)
