"""Entrenamiento reproducible con checkpoints completos; compatible con notebook Colab."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from dataset_utils import file_hash,fingerprint,read_jsonl

BASE_MODEL='intfloat/multilingual-e5-base'

def query_text(text):
    return 'query: '+text

def passage_text(text):
    return 'passage: '+text
os.environ.setdefault('USE_TF','0')

def validate_data(rows,catalog):
    grouped=defaultdict(list); seen_text={}; seen_source={}; seen_group={}
    positives=defaultdict(set)
    for row in rows:
        split=row['split'];number=row['articulo'];text=row['hechos']
        if split not in ('train','validation','test'): raise ValueError(f'Split inválido: {split}')
        if not text or number not in catalog: raise ValueError('Hecho vacío o artículo sin catálogo')
        if split!='train' and row.get('tipo')!='original': raise ValueError('Aumento fuera de train')
        for value,seen in ((fingerprint(text),seen_text),(str(row['fuente_id']),seen_source),(row['grupo_id'],seen_group)):
            if value in seen and seen[value]!=split: raise ValueError('Fuga entre splits')
            seen[value]=split
        grouped[split].append(row); positives[fingerprint(text)].add(number)
    # MNRL de un positivo: no enfrentar anclas con varios artículos conocidos.
    ambiguous={key for key,values in positives.items() if len(values)>1}
    # Conservar pares multietiqueta: la pérdida enmascara TODOS sus positivos conocidos.
    for split in ('train','validation','test'):
        if not grouped[split]: raise ValueError(f'Split {split} vacío; ampliar datos/revisar grupos')
    if len({catalog[r['articulo']] for r in grouped['train']})<2: raise ValueError('Se necesitan dos positivos diferentes en train')
    print(f'Anclas multietiqueta conservadas: {len(ambiguous)}; pares train: {len(grouped["train"])}')
    return grouped

def ranking(model,rows,catalog,batch_size=32,seen_articles=None,prediction_path=None):
    import numpy as np
    numbers=sorted(catalog);corpus=model.encode([passage_text(catalog[n]) for n in numbers],normalize_embeddings=True,batch_size=batch_size,show_progress_bar=True)
    queries={};positive=defaultdict(set)
    for row in rows:
        key=fingerprint(row['hechos']);queries[key]=row['hechos'];positive[key].add(row['articulo'])
    keys=list(queries);scores=model.encode([query_text(queries[k]) for k in keys],normalize_embeddings=True,batch_size=batch_size,show_progress_bar=True)
    totals=defaultdict(float);by_article=defaultdict(list); true_sets=[]; predicted_sets=[];records=[];cohorts=defaultdict(list)
    for start in range(0,len(keys),batch_size):
        orders=np.argsort(-(scores[start:start+batch_size]@corpus.T),axis=1)
        for offset,order in enumerate(orders):
            relevant=positive[keys[start+offset]];ranked=[numbers[i] for i in order]
            first=min(ranked.index(n)+1 for n in relevant);totals['mrr']+=1/first
            true_sets.append(relevant);predicted_sets.append({ranked[0]});totals['accuracy@1']+=float(ranked[0] in relevant)
            dcg=sum(1/np.log2(rank+2) for rank,n in enumerate(ranked[:10]) if n in relevant)
            ideal=sum(1/np.log2(rank+2) for rank in range(min(10,len(relevant))))
            totals['ndcg@10']+=dcg/ideal
            record={'query_id':keys[start+offset],'hechos':queries[keys[start+offset]],'positivos_conocidos':sorted(relevant),
                    'top10':ranked[:10],'acierto_top1':ranked[0] in relevant,'primer_positivo_rank':first}
            if seen_articles is not None:
                cohort='con_positivos_no_vistos' if relevant-set(seen_articles) else 'solo_positivos_vistos'
                cohorts[cohort].append(float(ranked[0] in relevant));record['cohorte']=cohort
            records.append(record)
            for k in (1,3,5):
                retrieved=set(ranked[:k]);hits=len(retrieved&relevant)
                totals[f'recall@{k}']+=hits/len(relevant);totals[f'precision@{k}']+=hits/min(k,len(numbers))
            for number in relevant: by_article[number].append(float(number in ranked[:3]))
    result={key:value/len(keys) for key,value in totals.items()}
    result['macro_article_recall@3']=float(np.mean([np.mean(v) for v in by_article.values()]))
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score, precision_score, recall_score
    labels=sorted(set().union(*true_sets,*predicted_sets))
    mlb=MultiLabelBinarizer(classes=labels)
    truth=mlb.fit_transform(true_sets);predictions=mlb.transform(predicted_sets)
    for avg in ('micro','macro','weighted'):
        result[f'f1_{avg}_top1']=float(f1_score(truth,predictions,average=avg,zero_division=0))
        result[f'precision_{avg}_top1']=float(precision_score(truth,predictions,average=avg,zero_division=0))
        result[f'recall_{avg}_top1']=float(recall_score(truth,predictions,average=avg,zero_division=0))
    from group_metrics import group_metrics
    result['por_expediente']=group_metrics(rows,records)
    result['queries']=len(keys);result['candidates']=len(numbers)
    result['por_articulo_recall@3']={number:{'soporte_queries':len(values),'recall@3':float(np.mean(values))} for number,values in by_article.items()}
    if cohorts:result['cohortes']={name:{'queries':len(values),'accuracy@1':float(np.mean(values))} for name,values in cohorts.items()}
    if prediction_path:
        from dataset_utils import write_jsonl
        write_jsonl(prediction_path,records)
    return result


import torch
from torch import nn
from torch.nn import functional as F

class MultiPositiveRankingLoss(nn.Module):
    """Contraste query→artículo: distribuir probabilidad entre positivos conocidos.

    label[0] es el ID del artículo de esta fila; label[1:] marca todos los
    artículos válidos para su ancla. No se consideran negativos otros positivos.
    """
    def __init__(self,model,scale=20.0,number_count=None,article_weights=None):
        super().__init__();self.model=model;self.scale=scale;self.number_count=number_count
        self.register_buffer('article_weights',torch.tensor(article_weights,dtype=torch.float32) if article_weights is not None else None)
    @staticmethod
    def objective(scores,labels,number_count=None,article_weights=None):
        count=number_count or labels.shape[1]-1
        document_ids=[labels[:,0].long()]
        for column in range(1+count,labels.shape[1]):document_ids.append(labels[:,column].long())
        document_ids=torch.cat(document_ids)
        positive_mask=labels[:,1:1+count].bool()[:,document_ids]
        if not positive_mask.diagonal().all():raise ValueError('Falta positivo diagonal')
        log_probs=F.log_softmax(scores.float(),dim=1)
        # Cada positivo conocido recibe peso; no basta optimizar el más fácil.
        per_anchor=-(log_probs.masked_fill(~positive_mask,0).sum(1)/positive_mask.sum(1))
        if article_weights is not None:
            positives=labels[:,1:1+count].float()
            weights=(positives@article_weights.to(scores.device))/positives.sum(1)
            return (per_anchor*weights).sum()/weights.sum()
        return per_anchor.mean()
    def forward(self,sentence_features,labels):
        anchors=F.normalize(self.model(sentence_features[0])['sentence_embedding'],dim=1)
        documents=F.normalize(self.model(sentence_features[1])['sentence_embedding'],dim=1)
        if len(sentence_features)>2:
            documents=torch.cat([documents]+[F.normalize(self.model(features)['sentence_embedding'],dim=1) for features in sentence_features[2:]],dim=0)
        return self.objective(anchors@documents.T*self.scale,labels,self.number_count,self.article_weights)
    def get_config_dict(self):return {'scale':self.scale,'positives':'all-known-labels','number_count':self.number_count,'class_balanced':self.article_weights is not None}


def train(run_dir,output_dir,epochs=12,batch_size=16,max_seq_length=512,seed=42,initial_model=None,approved_only=False,learning_rate=2e-5,scale=20.0,patience=3,evaluate_test=False,negative_reviews=None,selection_metric='f1_macro',class_balance_loss=False):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments,losses
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import InformationRetrievalEvaluator,SentenceEvaluator
    from transformers import set_seed,EarlyStoppingCallback
    from transformers.trainer_utils import get_last_checkpoint
    if learning_rate<=0 or scale<=0 or patience<1 or selection_metric not in ('f1_macro','f1_macro_group','ndcg','mrr'): raise ValueError('Hiperparámetros inválidos')
    directory=Path(run_dir);output=Path(output_dir)
    if '/drive/' in output.as_posix().casefold(): raise ValueError('Checkpoints deben ser locales; guardado en Drive desactivado')
    output.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(directory/'pares_aumentados.jsonl');catalog=json.loads((directory/'catalogo_cp.json').read_text(encoding='utf-8'))
    if approved_only: rows=[r for r in rows if r.get('estado_revision')=='aprobado']
    data=validate_data(rows,catalog);set_seed(seed)
    config={'dataset_sha256':file_hash(directory/'pares_aumentados.jsonl'),'catalog_sha256':file_hash(directory/'catalogo_cp.json'),
        'recipe':'multipositive-e5-base-v1','prefixes':{'query':'query: ','passage':'passage: '},'learning_rate':learning_rate,'scale':scale,'patience':patience,'base_model':BASE_MODEL,'initial_model':initial_model,'epochs':epochs,'batch_size':batch_size,'max_seq_length':max_seq_length,'seed':seed,'approved_only':approved_only,
        'negative_reviews_sha256':file_hash(negative_reviews) if negative_reviews else None,'selection_metric':selection_metric,'class_balance_loss':class_balance_loss}
    config_path=output/'run_config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Configuración/datos diferentes: usa una carpeta de modelo nueva')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    checkpoint=get_last_checkpoint(str(output))
    if checkpoint:
        required=['trainer_state.json','optimizer.pt','scheduler.pt']
        missing=[name for name in required if not (Path(checkpoint)/name).exists()]
        if missing: raise ValueError(f'Checkpoint incompleto: {missing}')
    model=SentenceTransformer(checkpoint or initial_model or BASE_MODEL)
    architecture=model[0].auto_model.config
    capacity=int(architecture.max_position_embeddings)
    if architecture.model_type in ('roberta','xlm-roberta'):
        capacity-=int(architecture.pad_token_id or 0)+1
    print('Capacidad de arquitectura:',capacity,'Límite predeterminado del tokenizer:',model.tokenizer.model_max_length)
    if max_seq_length>capacity: raise ValueError(f'max_seq_length supera capacidad {capacity}')
    model.max_seq_length=max_seq_length
    import numpy as np
    token_stats={}
    for name,texts in [('hechos',list({query_text(r['hechos']) for r in rows})),('articulos',list({passage_text(t) for t in catalog.values()}))]:
        lengths=np.array([len(model.tokenizer(t,truncation=False)['input_ids']) for t in texts])
        token_stats[name]={'p50':float(np.percentile(lengths,50)),'p95':float(np.percentile(lengths,95)),
            'max':int(lengths.max()),'truncated_fraction':float((lengths>max_seq_length).mean())}
    (output/'tokens.json').write_text(json.dumps(token_stats,indent=2),encoding='utf-8');print(token_stats)
    queries={};relevant=defaultdict(set)
    for row in data['validation']:
        key=fingerprint(row['hechos']);queries[key]=row['hechos'];relevant[key].add(row['articulo'])
    evaluator=InformationRetrievalEvaluator({k:query_text(v) for k,v in queries.items()},{k:passage_text(v) for k,v in catalog.items()},dict(relevant),name='validation',
        accuracy_at_k=[1,3,5],precision_recall_at_k=[1,3,5],mrr_at_k=[10],ndcg_at_k=[10],map_at_k=[10],batch_size=batch_size)
    if selection_metric!='ndcg':
        parent_evaluator=evaluator
        class ValidationMetricsEvaluator(SentenceEvaluator):
            def __init__(self):
                super().__init__();self.primary_metric='validation_'+selection_metric;self.greater_is_better=True
            def __call__(self,model,output_path=None,epoch=-1,steps=-1):
                result=parent_evaluator(model,output_path=output_path,epoch=epoch,steps=steps)
                validation_metrics=ranking(model,data['validation'],catalog,batch_size)
                result[self.primary_metric]=(validation_metrics['por_expediente']['f1_macro_por_expediente'] if selection_metric=='f1_macro_group' else validation_metrics['f1_macro_top1' if selection_metric=='f1_macro' else 'mrr'])
                return result
        evaluator=ValidationMetricsEvaluator()
    numbers=sorted(catalog);number_ids={number:i for i,number in enumerate(numbers)}
    known_positives=defaultdict(set)
    same_text=defaultdict(set)
    for number,text in catalog.items():same_text[fingerprint(text)].add(number)
    for row in data['train']:
        known_positives[fingerprint(row['hechos'])].update(same_text[fingerprint(catalog[row['articulo']])])
    labels=[]
    for row in data['train']:
        positives=known_positives[fingerprint(row['hechos'])]
        labels.append([number_ids[row['articulo']]]+[int(n in positives) for n in numbers])
    columns={'anchor':[query_text(r['hechos']) for r in data['train']],
        'positive':[passage_text(catalog[r['articulo']]) for r in data['train']]}
    if negative_reviews:
        reviewed={}
        for row in read_jsonl(negative_reviews):
            if row.get('estado_revision')!='aprobado' or not row.get('revisor') or not row.get('justificacion'):
                raise ValueError('Negativos sin revisión explícita')
            if row['query_id'] in reviewed:raise ValueError('Revisión de negativos duplicada')
            negatives=row['negativos_aprobados']
            if not isinstance(negatives,list) or not negatives or len(set(negatives))!=len(negatives):raise ValueError('Lista de negativos inválida')
            if any(n not in catalog for n in negatives):raise ValueError('Negativo sin catálogo')
            reviewed[row['query_id']]=negatives
        sizes={len(reviewed.get(fingerprint(r['hechos']),[])) for r in data['train']}
        if len(sizes)!=1 or 0 in sizes:raise ValueError('Revisar mismo número de negativos para TODAS las anclas train')
        for index,row in enumerate(data['train']):
            negatives=reviewed[fingerprint(row['hechos'])]
            if set(negatives)&known_positives[fingerprint(row['hechos'])]:raise ValueError('Negativo es positivo conocido')
            labels[index].extend(number_ids[n] for n in negatives)
        for offset in range(next(iter(sizes))):
            columns[f'negative_{offset+1}']=[passage_text(catalog[reviewed[fingerprint(r['hechos'])][offset]]) for r in data['train']]
    columns['label']=labels
    train_dataset=Dataset.from_dict(columns)
    article_weights=None
    if class_balance_loss:
        from collections import Counter
        frequency=Counter(n for positives in known_positives.values() for n in positives)
        maximum=max(frequency.values())
        article_weights=[min(4.0,(maximum/max(1,frequency[n]))**.5) for n in numbers]
    # Establecer el nombre exacto de la métrica con el evaluador instalado.
    if not (output/'baseline_validation.json').exists():
        base=SentenceTransformer(BASE_MODEL);base.max_seq_length=max_seq_length
        baseline=ranking(base,data['validation'],catalog,batch_size)
        (output/'baseline_validation.json').write_text(json.dumps(baseline,indent=2),encoding='utf-8')
        del base
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    evaluator(model,output_path=str(output));metric=evaluator.primary_metric
    args=SentenceTransformerTrainingArguments(output_dir=str(output),num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,weight_decay=.01,warmup_ratio=.1,fp16=torch.cuda.is_available(),bf16=False,
        eval_strategy='epoch',save_strategy='epoch',save_total_limit=1,
        save_only_model=False,load_best_model_at_end=True,metric_for_best_model=metric,greater_is_better=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,seed=seed,data_seed=seed,report_to='none',logging_steps=5,gradient_checkpointing=True)
    trainer=SentenceTransformerTrainer(model=model,args=args,train_dataset=train_dataset,
        loss=MultiPositiveRankingLoss(model,scale=scale,number_count=len(numbers),article_weights=article_weights),evaluator=evaluator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)])
    trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model(str(output/'final'))  # Guardado final desactivado hasta evaluar.
    seen_articles={r['articulo'] for r in data['train']}
    metrics={'validation':ranking(model,data['validation'],catalog,batch_size,seen_articles,output/'predicciones_validation.jsonl'),
        'supervision':'aprobada' if approved_only else 'débil: candidatos extraídos de citas'}
    if (directory/'validation_referencia_v6.jsonl').exists():
        metrics['validation_referencia_v6']=ranking(model,read_jsonl(directory/'validation_referencia_v6.jsonl'),catalog,batch_size,seen_articles,output/'predicciones_validation_referencia.jsonl')
    if evaluate_test: metrics['test']=ranking(model,data['test'],catalog,batch_size,seen_articles,output/'predicciones_test.jsonl')
    metrics['best_checkpoint']=trainer.state.best_model_checkpoint
    metrics['epochs_completed']=trainer.state.epoch
    metrics['train_pairs']=len(data['train'])
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2));return model,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-seq-length',type=int,default=512)
    p.add_argument('--initial-model');p.add_argument('--approved-only',action='store_true')
    p.add_argument('--learning-rate',type=float,default=2e-5);p.add_argument('--scale',type=float,default=20);p.add_argument('--patience',type=int,default=3);p.add_argument('--seed',type=int,default=42);p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--negative-reviews')
    p.add_argument('--selection-metric',choices=['f1_macro','f1_macro_group','ndcg','mrr'],default='f1_macro')
    p.add_argument('--class-balance-loss',action='store_true')
    a=p.parse_args();train(a.run_dir,a.output_dir,a.epochs,a.batch_size,a.max_seq_length,initial_model=a.initial_model,approved_only=a.approved_only,learning_rate=a.learning_rate,scale=a.scale,patience=a.patience,seed=a.seed,evaluate_test=a.evaluate_test,negative_reviews=a.negative_reviews,selection_metric=a.selection_metric,class_balance_loss=a.class_balance_loss)
