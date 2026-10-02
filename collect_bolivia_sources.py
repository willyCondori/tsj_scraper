"""Importar páginas oficiales verificadas como evidencia pendiente, no etiquetas."""
import argparse,json,hashlib
from datetime import datetime,timezone
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.parse import urlparse
from extract_pairs import clean_document
from dataset_utils import write_jsonl

SOURCES=[
    'https://www.fiscalia.gob.bo/comunicacion/noticias/sentencia-de-30-anos-de-carcel-para-autor-de-feminicidio-en-la-paz',
    'https://fiscalia.gob.bo/comunicacion/noticias/fiscalia-dos-sujetos-son-sentenciados-a-30-y-20-anos-de-prision-por-feminicidio-y-feminicidio-en-grado-de-tentativa-en-tarija',
]

def collect(output_dir):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=False);rows=[];errors=[]
    for url in SOURCES:
        try:
            req=Request(url,headers={'User-Agent':'TSJ-dataset-research/5.0 (public-source-validation)'})
            with urlopen(req,timeout=40) as response:
                final=response.url
                if urlparse(final).hostname not in ('fiscalia.gob.bo','www.fiscalia.gob.bo'):raise ValueError('Redirección fuera de fuente oficial')
                content=response.read();html=content.decode(response.headers.get_content_charset() or 'utf-8',errors='replace')
            key=hashlib.sha256(url.encode()).hexdigest()
            (output/f'{key}.html').write_text(html,encoding='utf-8')
            rows.append({'source_id':f'fge:{key}','url':url,'final_url':final,'fetched_at':datetime.now(timezone.utc).isoformat(),
                         'sha256':hashlib.sha256(content).hexdigest(),'contenido':clean_document(html),
                         'estado_revision':'pendiente','split':'review_only','norma':None,'articulos':[],
                         'advertencia':'Página completa: segmentar noticia, verificar caso y etiqueta antes de incorporar. Puede contener navegación.'})
        except Exception as exc:errors.append({'url':url,'error':str(exc)})
    write_jsonl(output/'fuentes_fiscalia_pendientes.jsonl',rows);write_jsonl(output/'errores_fuentes.jsonl',errors)
    print(json.dumps({'descargadas':len(rows),'errores':errors},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();collect(a.output_dir)
