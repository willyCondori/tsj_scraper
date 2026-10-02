"""Extracción de candidatos con evidencia. Una cita no constituye una etiqueta validada."""
import argparse
import html
import json
import re
import unicodedata
from pathlib import Path
from dataset_utils import NUMBER, article_number, fingerprint, normalize_text, write_jsonl

CITATION = re.compile(
    rf'\b(?:art[íi]culos?|arts?)\.?\s*(?P<numeros>{NUMBER}(?:\s*[º°])?'
    rf'(?:\s*(?:,\s*(?:y\s+)?|y|e|al|a|[-–])\s*{NUMBER})*)'
    r'\s*,?\s*(?:todos\s+)?(?:del|de\s+la)?\s*'
    r'(?P<norma>c[óo]digo\s+penal\b|c\.?\s*p(?![\w]|\s*\.\s*p(?:\.|\b))\.?)', re.I)
GENERIC_CITATION = re.compile(rf'\b(?:art[íi]culos?|arts?)\.?\s*{NUMBER}(?:\s*(?:,|y|al|a|[-–])\s*{NUMBER})*(?:\s*[º°])?', re.I)
FACTUAL = ['sustrajo','sustrajeron','se apodero','golpeo','agredio','amenazo','disparo','portaba','entrego dinero','causo la muerte','fallecio','murio','vendio','falsifico','se hizo entrega']
PROCEDURAL = ['apelacion restringida','precedente contradictorio','admisibilidad','admisible','inadmisible','tribunal de alzada','tribunal de casacion','defecto absoluto','defectos absolutos','nulidad','reposicion del juicio','fundamentacion de la sentencia']

def clean_document(text):
    text = re.sub(r'<(script|style)\b[^>]*>.*?</\1>', ' ', str(text or ''), flags=re.I|re.S)
    text = re.sub(r'</?(?:p|div|br|li|h[1-6])\b[^>]*>', '\n', text, flags=re.I)
    text = html.unescape(re.sub(r'<[^>]+>', ' ', text))
    return '\n'.join(normalize_text(line) for line in text.splitlines() if normalize_text(line))

def extraer_articulos_citados(text):
    result=[]
    for match in CITATION.finditer(text):
        tokens=list(re.finditer(NUMBER,match['numeros'],re.I)); numbers=[]
        for index,token in enumerate(tokens):
            number=article_number(token[0]); numbers.append(number)
            if index:
                sep=match['numeros'][tokens[index-1].end():token.start()].strip()
                prev=article_number(tokens[index-1][0])
                if sep in ('-','–','al','a') and prev.isdigit() and number.isdigit():
                    if 0 < int(number)-int(prev) <= 50:
                        numbers.extend(str(n) for n in range(int(prev)+1,int(number)))
        for number in dict.fromkeys(numbers):
            result.append({'numero':number,'inicio':match.start(),'fin':match.end(),'cita':match[0]})
    return result

def prepare_context(text):
    # Primero quitar todas las citas en el documento: ninguna puede quedar truncada por la ventana.
    masked=CITATION.sub(lambda m:' '*(m.end()-m.start()),text)
    masked=GENERIC_CITATION.sub(lambda m:' '*(m.end()-m.start()),masked)
    boundaries=[m.end() for m in re.finditer(r'[.!?]\s+|\n+',masked)]
    return masked,boundaries

def extraer_ventana_hecho(text, start, end, context=None):
    masked,boundaries=context if context is not None else prepare_context(text)
    left=max(0,start-650); right=min(len(text),end+250)
    # Expandir a límites de oración/párrafo; como máximo 200 caracteres adicionales.
    before=[b for b in boundaries if max(0,left-200)<=b<=left]
    after=[b for b in boundaries if right<=b<=min(len(text),right+200)]
    if before: left=before[-1]
    elif left:
        while left < start and not masked[left-1].isspace(): left+=1
    if after: right=after[0]
    elif right<len(text):
        while right>end and not masked[right-1].isspace(): right-=1
    return normalize_text(masked[left:right])

def clasificar_contenido_hechos(text):
    plain=''.join(c for c in unicodedata.normalize('NFKD',text.lower()) if not unicodedata.combining(c))
    count=lambda words:sum(len(re.findall(r'(?<!\w)'+re.escape(w)+r'(?!\w)',plain)) for w in words)
    f,p=count(FACTUAL),count(PROCEDURAL)
    kind='procesal' if p>=2 and p>2*f else ('factico' if f>0 and p==0 else ('mixto' if f>0 else 'indeterminado'))
    return {'tipo_contenido':kind,'señales_facticas':f,'señales_procesales':p}

def procesar_archivo(path):
    data=json.loads(Path(path).read_text(encoding='utf-8-sig')); text=clean_document(data.get('contenido'))
    source=data.get('id')
    if source is None: raise ValueError('Resolución sin id')
    rows=[];citations=extraer_articulos_citados(text)
    context=prepare_context(text) if citations else None
    document_hash=fingerprint(text)
    for citation in citations:
        facts=extraer_ventana_hecho(text,citation['inicio'],citation['fin'],context)
        if len(facts)<150: continue
        rows.append({'hechos':facts,'articulo':citation['numero'],'norma':'Codigo Penal',
            'fuente_id':source,'nro_resolucion':data.get('nro_resolucion'),'sala':data.get('sala'),
            'nro_expediente':data.get('nro_expediente'),'fecha_emision':data.get('fecha_emision'),
            'resultado_resolucion':data.get('formas_resoluciones'),
            'materia':data.get('materia'),'tipo_proceso':data.get('procesos'),
            'documento_hash':document_hash,'cita_original':citation['cita'],
            'cita_inicio':citation['inicio'],'cita_fin':citation['fin'],
            'estado_revision':'pendiente',**clasificar_contenido_hechos(facts)})
    return rows

def extract(raw_dir,out_dir):
    paths=sorted(Path(raw_dir).glob('*.json'))
    if not paths: raise ValueError(f'No hay JSON en {raw_dir}')
    rows=[];errors=[]
    for index,path in enumerate(paths,1):
        try: rows.extend(procesar_archivo(path))
        except (ValueError,TypeError,KeyError) as exc: errors.append({'archivo':path.name,'error':str(exc)})
        if index%1000==0: print(f'Extracción: {index}/{len(paths)}',flush=True)
    write_jsonl(Path(out_dir)/'pares_candidatos.jsonl',rows)
    write_jsonl(Path(out_dir)/'errores_extraccion.jsonl',errors)
    print(f'Candidatos: {len(rows)}; errores: {len(errors)}')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--raw-dir',default='data/raw'); parser.add_argument('--out-dir',required=True)
    args=parser.parse_args(); extract(args.raw_dir,args.out_dir)
