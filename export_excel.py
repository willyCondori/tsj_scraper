"""Exportar las salidas de un run para revisión; no transforma etiquetas."""
import argparse
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from dataset_utils import read_jsonl

def export(directory):
    import pandas as pd
    directory=Path(directory)
    originals=read_jsonl(directory/'pares_entrenamiento.jsonl')
    augmented=read_jsonl(directory/'pares_aumentados.jsonl')
    full_review=read_jsonl(directory/'pares_revision.jsonl')
    groups=defaultdict(list)
    for row in full_review:groups[row.get('motivo')].append(row)
    for group in groups.values():group.sort(key=lambda r:hashlib.sha256(json.dumps(r,ensure_ascii=False).encode('utf-8')).hexdigest())
    review=[];index=0
    while len(review)<1000:
        added=False
        for group in groups.values():
            if index<len(group):review.append(group[index]);added=True
            if len(review)==1000:break
        if not added:break
        index+=1
    frame=pd.DataFrame(augmented)
    if frame.empty: raise ValueError('Run sin pares entrenables; revisar pares_revision.jsonl')
    stats=frame.groupby(['articulo','split'],dropna=False).agg(filas=('hechos','size'),
        resoluciones=('fuente_id','nunique'),textos=('hechos','nunique')).reset_index()
    output=directory/'dataset_completo.xlsx'
    if output.exists(): raise FileExistsError('El Excel ya existe; usa otro run o renombra el archivo para conservarlo')
    with pd.ExcelWriter(output,engine='openpyxl') as writer:
        for name,rows in [('Pares_originales',originals),('Pares_aumentados',augmented),('Muestra_revision',review)]:
            df=pd.DataFrame(rows)
            if 'articulo' in df: df['articulo']=df['articulo'].astype(str)
            for column in df:
                df[column]=df[column].map(lambda v:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v)
            df.to_excel(writer,sheet_name=name,index=False)
        stats.to_excel(writer,sheet_name='Estadisticas',index=False)
        writer.sheets['Estadisticas']['G1']='Revisión';writer.sheets['Estadisticas']['H1']='Filas'
        writer.sheets['Estadisticas']['G2']='Total en pares_revision.jsonl';writer.sheets['Estadisticas']['H2']=len(full_review)
        writer.sheets['Estadisticas']['G3']='Muestra en este Excel';writer.sheets['Estadisticas']['H3']=len(review)
        for ws in writer.book:
            ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
            for col in ws.columns:
                header=col[0].value
                ws.column_dimensions[col[0].column_letter].width=90 if header=='hechos' else (65 if header=='articulo_texto' else 24)
    print(output.resolve())
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);a=p.parse_args();export(a.run_dir)
