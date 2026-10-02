"""BM25/RRF con el mismo encoder; minado de propuestas solo sobre train."""
import argparse,json,math,re
from collections import Counter,defaultdict
from pathlib import Path
from dataset_utils import fingerprint,read_jsonl,write_jsonl

def tokens(text):return re.findall(r'\w+',text.casefold(),flags=re.UNICODE)

class BM25:
    def __init__(self,documents,k1=1.5,b=.75):
        self.docs=[Counter(tokens(t)) for t in documents];self.k1=k1;self.b=b
        self.lengths=[sum(d.values()) for d in self.docs];self.avg=sum(self.lengths)/max(1,len(self.docs))
        df=Counter(term for d in self.docs for term in d)
        self.idf={t:math.log(1+(len(self.docs)-n+.5)/(n+.5)) for t,n in df.items()}
    def scores(self,query):
        result=[]
        for doc,length in zip(self.docs,self.lengths):
            value=0
            for term in set(tokens(query)):
                frequency=doc[term]
                if frequency:value+=self.idf.get(term,0)*frequency*(self.k1+1)/(frequency+self.k1*(1-self.b+self.b*length/max(self.avg,1)))
            result.append(value)
        return result

def rrf(first,second,weight=.5,k=60):
    scores=defaultdict(float)
    for ranking,w in ((first,weight),(second,1-weight)):
        for rank,item in enumerate(ranking,1):scores[item]+=w/(k+rank)
    return sorted(scores,key=lambda item:(-scores[item],item))

def retrieve(run_dir,model_path,output,mode='validation',weight=.5):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    directory=Path(run_dir);catalog=json.loads((directory/'catalogo_cp.json').read_text(encoding='utf-8'))
    numbers=sorted(catalog);rows=[r for r in read_jsonl(directory/'pares_entrenamiento.jsonl') if r['split']==mode]
    queries={};gold=defaultdict(set)
    for row in rows:key=fingerprint(row['hechos']);queries[key]=row;gold[key].add(row['articulo'])
    model=SentenceTransformer(model_path);model.max_seq_length=512
    texts=[catalog[n] for n in numbers];corpus=model.encode(texts,normalize_embeddings=True,show_progress_bar=True)
    embeddings=model.encode([r['hechos'] for r in queries.values()],normalize_embeddings=True,show_progress_bar=True)
    bm=BM25(texts);result=[]
    for index,(key,row) in enumerate(queries.items()):
        dense=np.argsort(-(embeddings[index]@corpus.T)).tolist()
        bm_scores=bm.scores(row['hechos'])
        lexical=sorted(range(len(numbers)),key=lambda i:(-bm_scores[i],i))
        fused=rrf(dense,lexical,weight)
        record={'query_id':key,'fuente_id':row['fuente_id'],'split':mode,'hechos':row['hechos'],
                'positivos_conocidos':sorted(gold[key]),'dense_top10':[numbers[i] for i in dense[:10]],
                'bm25_top10':[numbers[i] for i in lexical[:10]],'hybrid_top10':[numbers[i] for i in fused[:10]]}
        if mode=='train':
            # No afirmar negativos: los positivos conocidos pueden ser incompletos.
            record['negativos_propuestos']=[numbers[i] for i in fused[:30] if numbers[i] not in gold[key]][:4]
            record['estado_revision']='pendiente';record['advertencia']='ausencia de cita no demuestra irrelevancia'
        result.append(record)
    write_jsonl(output,result)
    if mode!='train':
        metrics={}
        for variant in ('dense','bm25','hybrid'):
            metrics[variant]={'accuracy@1':sum(r[f'{variant}_top10'][0] in gold[r['query_id']] for r in result)/len(result),
                'recall@5':sum(len(set(r[f'{variant}_top10'][:5])&gold[r['query_id']])/len(gold[r['query_id']]) for r in result)/len(result)}
        print(json.dumps(metrics,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['train','validation','test'],default='validation');p.add_argument('--dense-weight',type=float,default=.5)
    a=p.parse_args()
    if not 0<=a.dense_weight<=1:raise ValueError('Peso fuera de 0–1')
    retrieve(a.run_dir,a.model,a.output,a.mode,a.dense_weight)
