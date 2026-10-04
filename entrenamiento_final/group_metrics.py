"""Métricas por consulta y con igual peso por expediente; sin consultar test."""
from collections import defaultdict
from dataset_utils import fingerprint

def group_metrics(rows, records):
    import numpy as np
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.metrics import f1_score
    memberships = defaultdict(set)
    for row in rows:
        memberships[fingerprint(row['hechos'])].add(str(row['grupo_id']))
    if any(len(v) != 1 for v in memberships.values()):
        raise ValueError('Consulta asociada a varios grupos: revisar agrupación')
    counts = defaultdict(int)
    for record in records:
        counts[next(iter(memberships[record['query_id']]))] += 1
    weights = [1 / counts[next(iter(memberships[r['query_id']]))] for r in records]
    truth = [set(r['positivos_conocidos']) for r in records]
    pred = [{r['top10'][0]} for r in records]
    labels = sorted(set().union(*truth, *pred))
    encoder = MultiLabelBinarizer(classes=labels)
    y = encoder.fit_transform(truth); p = encoder.transform(pred)
    result = {'expedientes': len(counts), 'consultas': len(records),
              'accuracy_top1_por_expediente': float(np.average([r['acierto_top1'] for r in records], weights=weights))}
    for avg in ('macro', 'micro', 'weighted'):
        result['f1_' + avg + '_por_expediente'] = float(f1_score(y, p, average=avg, sample_weight=weights, zero_division=0))
    return result
