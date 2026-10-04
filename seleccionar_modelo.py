"""Selección conservadora en validation original. Nunca se consulta test."""
def choose(reference, candidate, reference_checkpoint, candidate_checkpoint):
    if reference['queries']!=candidate['queries'] or reference['candidates']!=candidate['candidates']:
        raise ValueError('Evaluaciones no comparables')
    required=['f1_macro_top1','accuracy@1','recall@3']
    reasons=[]
    if candidate['f1_macro_top1']<=reference['f1_macro_top1']+1e-9:reasons.append('F1 macro no mejora')
    for metric in required[1:]:
        if candidate[metric]+1e-9<reference[metric]:reasons.append(metric+' disminuye')
    metric='f1_macro_por_expediente'
    if candidate['por_expediente'][metric]+1e-9<reference['por_expediente'][metric]:reasons.append('F1 macro por expediente disminuye')
    return {'seleccion':'referencia' if reasons else 'candidato',
      'checkpoint_seleccionado':str(reference_checkpoint if reasons else candidate_checkpoint),
      'motivos':reasons or ['Mejora F1 macro sin deteriorar los criterios de control'],
      'validacion':'original v6; decisión exploratoria, no test final',
      'guardado_final_autorizado_por_codigo':False}
