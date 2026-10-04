"""Comparación exploratoria fija de recuperación y reordenación, solo validation."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from dataset_utils import fingerprint, write_jsonl
from entrenamiento_final.group_metrics import group_metrics

RERANKER='cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'

def prepare_queries(rows):
    if not rows or any(r['split']!='validation' for r in rows):
        raise ValueError('Solo validation; test no permitido')
    queries={};truth=defaultdict(set)
    for r in rows:
        key=fingerprint(r['hechos']);queries[key]=r['hechos'];truth[key].add(r['articulo'])
    return queries,truth

def reorder(candidate_ids, cross_scores, policy):
    order=np.argsort(-np.asarray(cross_scores),kind='stable')
    if policy=='cross_encoder':return [candidate_ids[i] for i in order]
    if policy=='rrf':
        cross_rank=np.empty(len(order),dtype=int);cross_rank[order]=np.arange(1,len(order)+1)
        fusion=1/(60+np.arange(1,len(order)+1))+1/(60+cross_rank)
        return [candidate_ids[i] for i in np.argsort(-fusion,kind='stable')]
    if policy=='e5':return list(candidate_ids)
    raise ValueError('Política desconocida')

def measure(rows,records):
    from sklearn.metrics import f1_score
    from sklearn.preprocessing import MultiLabelBinarizer
    truth=[set(r['positivos_conocidos']) for r in records]
    pred=[{r['top10'][0]} for r in records]
    labels=sorted(set().union(*truth,*pred));encoder=MultiLabelBinarizer(classes=labels)
    y=encoder.fit_transform(truth);p=encoder.transform(pred)
    result={'queries':len(records),'accuracy@1':float(np.mean([r['acierto_top1'] for r in records]))}
    for avg in ('macro','micro','weighted'):result['f1_'+avg+'_top1']=float(f1_score(y,p,average=avg,zero_division=0))
    for k in (3,5,10):
        result['recall@'+str(k)]=float(np.mean([len(set(r['top10'][:k])&set(r['positivos_conocidos']))/len(r['positivos_conocidos']) for r in records]))
    result['por_expediente']=group_metrics(rows,records)
    return result

def compare(model,cross,rows,catalog,out_dir,sensitivity_ids):
    queries,truth=prepare_queries(rows);keys=list(queries);numbers=sorted(catalog)
    corpus=model.encode(['passage: '+catalog[n] for n in numbers],normalize_embeddings=True,batch_size=16,show_progress_bar=True)
    anchors=model.encode(['query: '+queries[k] for k in keys],normalize_embeddings=True,batch_size=16,show_progress_bar=True)
    scores=anchors@corpus.T
    candidates=[[numbers[i] for i in np.argsort(-s,kind='stable')[:10]] for s in scores]
    pairs=[(queries[key],catalog[n]) for key,ns in zip(keys,candidates) for n in ns]
    # Sin prefijos E5: el cross-encoder recibe (consulta, artículo).
    raw_logits=np.asarray(cross.predict(pairs,batch_size=8,show_progress_bar=True))
    if raw_logits.shape not in ((len(pairs),),(len(pairs),1)):
        raise ValueError('Forma inesperada de scores: '+str(raw_logits.shape))
    if not np.isfinite(raw_logits).all():raise ValueError('Scores no finitos')
    logits=raw_logits.reshape(len(keys),10)
    result={};out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    subset=[r for r in rows if fingerprint(r['hechos']) in sensitivity_ids]
    all_records={}
    for policy in ('e5','cross_encoder','rrf'):
        records=[]
        for i,key in enumerate(keys):
            ranked=reorder(candidates[i],logits[i],policy)
            records.append({'query_id':key,'hechos':queries[key],'positivos_conocidos':sorted(truth[key]),
               'top10':ranked,'acierto_top1':ranked[0] in truth[key],
               'candidatos_e5':candidates[i],'scores_cross_encoder':[float(x) for x in logits[i]],
               'textos_candidatos':{n:catalog[n] for n in candidates[i]},
               'fuentes':[{'fuente_id':r['fuente_id'],'cita_original':r.get('cita_original'),'grupo_id':r['grupo_id']} for r in rows if fingerprint(r['hechos'])==key],
               'estado_revision':'pendiente'})
        write_jsonl(out/(policy+'_detalle.jsonl'),records)
        result[policy]={'referencia_v6':measure(rows,records),'sensibilidad':measure(subset,[r for r in records if r['query_id'] in sensitivity_ids])}
        all_records[policy]=records
    changes=[]
    for i,key in enumerate(keys):
        before=all_records['e5'][i]
        for policy in ('cross_encoder','rrf'):
            after=all_records[policy][i]
            if before['top10'][0]!=after['top10'][0]:
                changes.append({'query_id':key,'politica':policy,'antes':before['top10'][0],
                 'despues':after['top10'][0],'acierto_antes':before['acierto_top1'],'acierto_despues':after['acierto_top1'],'hechos':queries[key]})
    write_jsonl(out/'cambios_top1.jsonl',changes)
    token_lengths=[len(cross.tokenizer(a,b,truncation=False)['input_ids']) for a,b in pairs] if hasattr(cross,'tokenizer') else []
    result['diagnostico']={'pares':len(pairs),'scores_min':float(logits.min()),'scores_max':float(logits.max()),
      'queries_scores_constantes':int(np.sum(np.ptp(logits,axis=1)<1e-8)),
      'fraccion_pares_mas_512':float(np.mean(np.array(token_lengths)>512)) if token_lengths else None,
      'accuracy_orden_inverso_solo_diagnostico':float(np.mean([candidates[i][int(np.argmin(logits[i]))] in truth[key] for i,key in enumerate(keys)])),
      'aviso':'El orden inverso es un diagnóstico, no una política seleccionada para mejorar validation.'}
    write_jsonl(out/'pares_y_scores.jsonl',[{'query_id':key,'indice_query':i,'indice_candidato':j,'articulo':n,'hechos':queries[key],'texto_articulo':catalog[n],'score_cross':float(logits[i,j])} for i,key in enumerate(keys) for j,n in enumerate(candidates[i])])
    result['protocolo']={'candidatos':10,'reranker':RERANKER,'max_length_pair':512,
      'advertencia':'Pares largos se truncan; scores no son probabilidades. Solo validation, etiquetas débiles. Resultados exploratorios; no se adopta un ganador automáticamente.'}
    (out/'metricas.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
