"""Entrenamiento reproducible con checkpoints completos; compatible con notebook Colab."""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from dataset_utils import file_hash,fingerprint,read_jsonl

BASE_MODEL='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
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
    grouped['train']=[r for r in grouped['train'] if fingerprint(r['hechos']) not in ambiguous]
    for split in ('train','validation','test'):
        if not grouped[split]: raise ValueError(f'Split {split} vacío; ampliar datos/revisar grupos')
    if len({catalog[r['articulo']] for r in grouped['train']})<2: raise ValueError('Se necesitan dos positivos diferentes en train')
    print(f'Anclas multietiqueta excluidas de MNRL: {len(ambiguous)}')
    return grouped

def ranking(model,rows,catalog,batch_size=32):
    import numpy as np
    numbers=sorted(catalog);corpus=model.encode([catalog[n] for n in numbers],normalize_embeddings=True,batch_size=batch_size,show_progress_bar=True)
    queries={};positive=defaultdict(set)
    for row in rows:
        key=fingerprint(row['hechos']);queries[key]=row['hechos'];positive[key].add(row['articulo'])
    keys=list(queries);scores=model.encode([queries[k] for k in keys],normalize_embeddings=True,batch_size=batch_size,show_progress_bar=True)
    totals=defaultdict(float);by_article=defaultdict(list)
    for start in range(0,len(keys),batch_size):
        orders=np.argsort(-(scores[start:start+batch_size]@corpus.T),axis=1)
        for offset,order in enumerate(orders):
            relevant=positive[keys[start+offset]];ranked=[numbers[i] for i in order]
            first=min(ranked.index(n)+1 for n in relevant);totals['mrr']+=1/first
            for k in (1,3,5):
                retrieved=set(ranked[:k]);hits=len(retrieved&relevant)
                totals[f'recall@{k}']+=hits/len(relevant);totals[f'precision@{k}']+=hits/min(k,len(numbers))
            for number in relevant: by_article[number].append(float(number in ranked[:3]))
    result={key:value/len(keys) for key,value in totals.items()}
    result['macro_article_recall@3']=float(np.mean([np.mean(v) for v in by_article.values()]))
    result['queries']=len(keys);result['candidates']=len(numbers)
    return result

def train(run_dir,output_dir,epochs=4,batch_size=32,max_seq_length=256,seed=42,initial_model=None,approved_only=False):
    import torch
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer,SentenceTransformerTrainer,SentenceTransformerTrainingArguments,losses
    from sentence_transformers.training_args import BatchSamplers
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from transformers import set_seed
    from transformers.trainer_utils import get_last_checkpoint
    directory=Path(run_dir);output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(directory/'pares_aumentados.jsonl');catalog=json.loads((directory/'catalogo_cp.json').read_text(encoding='utf-8'))
    if approved_only: rows=[r for r in rows if r.get('estado_revision')=='aprobado']
    data=validate_data(rows,catalog);set_seed(seed)
    config={'dataset_sha256':file_hash(directory/'pares_aumentados.jsonl'),'catalog_sha256':file_hash(directory/'catalogo_cp.json'),
        'base_model':BASE_MODEL,'initial_model':initial_model,'epochs':epochs,'batch_size':batch_size,'max_seq_length':max_seq_length,'seed':seed,'approved_only':approved_only}
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
    capacity=min(int(model[0].auto_model.config.max_position_embeddings),int(model.tokenizer.model_max_length))
    if max_seq_length>capacity: raise ValueError(f'max_seq_length supera capacidad {capacity}')
    model.max_seq_length=max_seq_length
    import numpy as np
    token_stats={}
    for name,texts in [('hechos',list({r['hechos'] for r in rows})),('articulos',list(set(catalog.values())))]:
        lengths=np.array([len(model.tokenizer(t,truncation=False)['input_ids']) for t in texts])
        token_stats[name]={'p50':float(np.percentile(lengths,50)),'p95':float(np.percentile(lengths,95)),
            'max':int(lengths.max()),'truncated_fraction':float((lengths>max_seq_length).mean())}
    (output/'tokens.json').write_text(json.dumps(token_stats,indent=2),encoding='utf-8');print(token_stats)
    queries={};relevant=defaultdict(set)
    for row in data['validation']:
        key=fingerprint(row['hechos']);queries[key]=row['hechos'];relevant[key].add(row['articulo'])
    evaluator=InformationRetrievalEvaluator(queries,catalog,dict(relevant),name='validation',
        accuracy_at_k=[1,3,5],precision_recall_at_k=[1,3,5],mrr_at_k=[10],ndcg_at_k=[10],map_at_k=[10],batch_size=batch_size)
    train_dataset=Dataset.from_dict({'anchor':[r['hechos'] for r in data['train']],
        'positive':[catalog[r['articulo']] for r in data['train']]})
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
        learning_rate=2e-5,warmup_ratio=.1,fp16=torch.cuda.is_available(),bf16=False,
        eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
        save_only_model=False,load_best_model_at_end=True,metric_for_best_model=metric,greater_is_better=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,seed=seed,data_seed=seed,report_to='none',logging_steps=50)
    trainer=SentenceTransformerTrainer(model=model,args=args,train_dataset=train_dataset,
        loss=losses.MultipleNegativesRankingLoss(model),evaluator=evaluator)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output/'final'))
    metrics={'validation':ranking(model,data['validation'],catalog,batch_size),
        'test':ranking(model,data['test'],catalog,batch_size),
        'supervision':'aprobada' if approved_only else 'débil: candidatos extraídos de citas'}
    (output/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False,indent=2));return model,metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=4);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--max-seq-length',type=int,default=256)
    p.add_argument('--initial-model');p.add_argument('--approved-only',action='store_true')
    a=p.parse_args();train(a.run_dir,a.output_dir,a.epochs,a.batch_size,a.max_seq_length,initial_model=a.initial_model,approved_only=a.approved_only)
