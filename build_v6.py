"""Ampliación trazable de supervisión débil; evaluación v4 congelada."""
import argparse,json,re,html,shutil
from collections import Counter,defaultdict
from pathlib import Path
from dataset_utils import fingerprint,file_hash,normalize_text,read_jsonl,write_jsonl,grouped_split
from extract_sections import plain,section_candidates
from augment_pairs import augment

# Verbos en pasado, no nombres de delitos o expresiones de aplicación de normas.
ACTIONS=re.compile(r'\b(?:sustrajo|sustrajeron|golpeo|golpearon|agredio|agredieron|amenazo|amenazaron|disparo|dispararon|entrego|entregaron|falsifico|falsificaron|vendio|vendieron|compro|engano|viol[oó]|abuso|lesiono|mato|apunalo|atropello|exigio|deposito|transfirio|recibio|retuvo|ingreso|obligo|oculto|transporto|suministro|murio|fallecio|causo|apodero|conducia|condujo|condujeron|impacto|embistio|sometio|despojo|agarr[oó]|propino|introdujo|pateo|sac[oó]|forzo|forcejeo|arrastro|arrastraron|accedio|penetro|sujeto|distribuyo|intercepto|sustraido|golpeado|agredida|agredido|amenazada|amenazado)\b')
PROCEDURE=re.compile(r'\b(?:casacion|apelacion|precedente|agravio|nulidad|alzada|doctrina legal|inadmisib\w*|fundamentacion|recurrente)\b')
THEORY=re.compile(r'\b(?:tipo penal exige|bien juridico|elementos constitutivos|doctrina legal|jurisprudencia establece|se entiende por|elemento subjetivo|configuracion del delito|sujeto activo|tipicidad|no se ha violado|no se ha vulnerado)\b')
NEGATION=re.compile(r'\b(?:no se (?:demostro|acredito|probo)|no (?:golpeo|agredio|violo|sustrajo)|inexistencia del hecho|hecho no existio)\b')

def clean_facts(text):
    """Mantiene oraciones originales; no genera paráfrasis ni nuevos hechos."""
    parts=re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])',text)
    retained=[]
    for part in parts:
        norm=plain(part)
        if THEORY.search(norm):continue
        if len(PROCEDURE.findall(norm))>=2 and not ACTIONS.search(norm):continue
        retained.append(part)
    return normalize_text(' '.join(retained))

def quality(row,catalog,excluded):
    if row.get('split')!='train':return None,'fuera_train'
    if row['articulo'] not in catalog:return None,'sin_catalogo_cp'
    if row['articulo'] in excluded or int(re.match(r'\d+',row['articulo'])[0])<100:return None,'etiqueta_general_separada'
    if row.get('papel_cita') in ('absolucion','precedente','ambiguo'):return None,'papel_no_elegible'
    text=clean_facts(row['hechos']);norm=plain(text)
    # Evitar sustantivos y referencias a violaciones de normas como acciones.
    actions=sum(1 for m in ACTIONS.finditer(norm) if m[0] not in ('abuso','sujeto','violo') or (m[0]=='violo' and re.match(r'\s+a\b',norm[m.end():])))
    procedural=len(PROCEDURE.findall(norm))
    if not 180<=len(text)<=3600:return None,'longitud'
    if actions<2:return None,'pocas_acciones'
    if procedural>2 or procedural>actions:return None,'contenido_procesal'
    if THEORY.search(norm) or NEGATION.search(norm):return None,'doctrina_o_negacion'
    if row.get('distancia_cita',99999)>400:return None,'cita_lejana'
    role=row.get('papel_cita')
    if role=='no_identificado' and (row.get('seccion')!='hechos' or actions<3 or procedural):return None,'rol_sin_evidencia'
    tier='conservador' if role=='condena' and row['distancia_cita']<=250 and procedural<=1 else 'ampliado'
    return {**row,'hechos_original':row['hechos'],'hechos':text,'articulo_texto':catalog[row['articulo']],
            'señales_facticas':actions,'señales_procesales':procedural,'acciones_pasado':actions,
            'nivel_seleccion':tier,'tipo_contenido':'factico_heuristico',
            'estado_revision':'pendiente','supervision':'débil; asociación por cita, sin aprobación jurídica',
            'metodo':'v6_filtrado_oraciones','transformacion':'selección extractiva de oraciones, sin generación'},None

def shingles(text):
    words=re.findall(r'\w+',plain(text));return {tuple(words[i:i+5]) for i in range(max(0,len(words)-4))}

def close_to_held(text,held):
    current=shingles(text)
    if not current:return False
    for other in held:
        # Contención también detecta ventanas recortadas de un texto de evaluación.
        if other and len(current&other)/min(len(current),len(other))>=.75:return True
    return False

def official_additions(pages_path,baseline):
    pages=json.loads(Path(pages_path).read_text(encoding='utf-8'));cleaned=[];starts=[];offset=0
    for i,page in enumerate(pages):
        lines=[line for line in page.splitlines() if not re.fullmatch(r'\s*(?:CÓDIGO PENAL|Código Penal|Fiscalía General del Estado|\d{1,3})\s*',line)]
        text='\n'.join(lines)+'\n';starts.append((offset,i+1));cleaned.append(text);offset+=len(text)
    text=''.join(cleaned)
    pattern=re.compile(r'(?m)^Artículo\s+(\d+)\s*(bis|ter|quater|quinquies|sexies|septies)?\s*[.º°]*\s*\(',re.I)
    matches=list(pattern.finditer(text));entries=defaultdict(list)
    for i,m in enumerate(matches):
        number=str(int(m[1]))+((' '+m[2].lower()) if m[2] else '')
        if int(m[1])>363:continue
        body=text[m.start():matches[i+1].start() if i+1<len(matches) else len(text)]
        body=re.split(r'(?m)^(?:CAPÍTULO|TÍTULO|LIBRO|DISPOSICIONES|LEY N[°º])\b',body)[0]
        page=max(p for start,p in starts if start<=m.start())
        entries[number].append((normalize_text(body),page))
    additions={};provenance=[]
    for number,values in entries.items():
        if number in baseline or len(values)!=1:continue
        body,page=values[0]
        if len(body)<100 or len(body)>18000:continue
        additions[number]=body
        provenance.append({'articulo':number,'contenido':body,'pagina_pdf':page,
            'source_url':'https://fiscalia.gob.bo/marco-legal/leyes/codigo-penal',
            'download_url':'https://files.mp.gob.bo/v1/file/download/68a37c965afca6cadfcaa724',
            'estado_revision':'extracción de publicación oficial; vigencia temporal no certificada'})
    return additions,provenance

def build(v4,v5,out,pages,pdf,raw_dir):
    v4,v5,out=map(Path,(v4,v5,out));baseline=read_jsonl(v4/'pares_entrenamiento.jsonl')
    if baseline!=read_jsonl(v5/'pares_entrenamiento.jsonl'):raise ValueError('Baseline v5 distinto de v4')
    catalog=json.loads((v4/'catalogo_cp.json').read_text(encoding='utf-8'))
    excluded=set(json.loads((Path(__file__).parent/'data/reference/politica_dataset.json').read_text(encoding='utf-8'))['excluir_etiquetas'])
    additions,provenance=official_additions(pages,catalog)
    expanded={**catalog,**additions};stats=Counter();missing=Counter();eligible=[]
    held=[r for r in baseline if r['split']!='train'];held_text=[shingles(r['hechos']) for r in held]
    blocked={key:set() for key in ('fuente_id','grupo_id','documento_hash','nro_expediente')}
    # Extiende el bloqueo a TODAS las propuestas de validation/test/review_only.
    for line in (v5/'propuestas_secciones.jsonl').open(encoding='utf-8'):
        r=json.loads(line)
        if r['split']!='train':
            for key in blocked:
                value=r.get(key)
                if value and (key!='nro_expediente' or re.search(r'\d{4}',str(value))):blocked[key].add(str(value))
    for r in held:
        for key in blocked:
            value=r.get(key)
            if value and (key!='nro_expediente' or re.search(r'\d{4}',str(value))):blocked[key].add(str(value))
    assignments={str(r['fuente_id']):(r['split'],r['grupo_id']) for r in grouped_split([r for r in read_jsonl(v4/'pares_candidatos.jsonl') if r['articulo'] in catalog])}
    manifest_v4=json.loads((v4/'manifest.json').read_text(encoding='utf-8'))
    for row in baseline:
        if assignments[str(row['fuente_id'])]!=(row['split'],row['grupo_id']):raise ValueError('Partición original incompatible')
    errors=[];files=sorted(Path(raw_dir).glob('*.json'))
    if set(p.name for p in files)!=set(manifest_v4['source_files']):raise ValueError('Cambió el inventario de crudos')
    def proposals():
        for index,path in enumerate(files,1):
            if file_hash(path)!=manifest_v4['source_files'][path.name]:raise ValueError('Crudo modificado: '+path.name)
            try:
                data=json.loads(path.read_text(encoding='utf-8-sig'))
                for row in section_candidates(data,max_blocks=12,overlap_first=True):
                    split,group=assignments.get(str(row['fuente_id']),('review_only',fingerprint(str(row['fuente_id']))))
                    row.update(split=split,grupo_id=group,metodo='secciones_v6',motivo='requiere_revision_juridica')
                    yield row
            except (ValueError,TypeError,KeyError) as exc:errors.append({'archivo':path.name,'error':str(exc)})
            if index%3000==0:print(f'Extracción v6: {index}/{len(files)} documentos',flush=True)
    for original in proposals():
        selected,reason=quality(original,expanded,excluded)
        if reason:
            stats[reason]+=1
            if reason=='sin_catalogo_cp':missing[original['articulo']]+=1
            continue
        if any(str(selected.get(k)) in values for k,values in blocked.items()):stats['vinculo_evaluacion']+=1;continue
        if close_to_held(selected['hechos'],held_text):stats['similitud_evaluacion']+=1;continue
        # Nuevos artículos generales no entran al aprendizaje de tipos delictivos.
        if selected['articulo'] in additions and int(re.match(r'\d+',selected['articulo'])[0])<100:
            stats['nuevo_articulo_general']+=1;continue
        selected['catalogo_origen']='fiscalia_ampliacion' if selected['articulo'] in additions else 'v4'
        eligible.append(selected)
    eligible.sort(key=lambda r:(r['nivel_seleccion']!='conservador',-r['acciones_pasado'],r['distancia_cita'],r['candidate_id']))
    seen={(fingerprint(re.sub(r'[^\w\s]',' ',r['hechos'])),r['articulo']) for r in baseline};counts=Counter();accepted=[]
    for row in eligible:
        key=(fingerprint(re.sub(r'[^\w\s]',' ',row['hechos'])),row['articulo'])
        if key in seen:stats['duplicado']+=1;continue
        source=(str(row['fuente_id']),row['articulo'])
        if counts[source]>=2:stats['limite_por_fuente_articulo']+=1;continue
        seen.add(key);counts[source]+=1;accepted.append(row)
    conservative=[r for r in accepted if r['nivel_seleccion']=='conservador' and r['articulo'] in catalog]
    core=baseline+conservative;large=baseline+accepted
    if [r for r in large if r['split']!='train']!=held:raise ValueError('Evaluación alterada')
    out.mkdir(parents=True,exist_ok=False)
    for directory,rows,cat in [(out,core,catalog),(out/'experimental_ampliado',large,expanded)]:
        directory.mkdir(exist_ok=True)
        write_jsonl(directory/'pares_entrenamiento.jsonl',rows);write_jsonl(directory/'pares_aumentados.jsonl',augment(rows))
        (directory/'catalogo_cp.json').write_text(json.dumps(cat,ensure_ascii=False,indent=2),encoding='utf-8')
        manifest={'version':'v6','rows':len(rows),'splits':dict(Counter(r['split'] for r in rows)),
                  'added_weak_train_pairs':len(rows)-len(baseline),'supervision':'débil; etiquetas pendientes',
                  'catalog_entries':len(cat),'validation_test_unchanged':True,
                  'baseline_sha256':file_hash(v4/'pares_entrenamiento.jsonl'),
                  'proposal_input_sha256':file_hash(v5/'propuestas_secciones.jsonl'),
                  'dataset_sha256':file_hash(directory/'pares_aumentados.jsonl'),
                  'catalog_sha256':file_hash(directory/'catalogo_cp.json'),
                  'code_sha256':file_hash(__file__),'selection_rule':'ver build_v6.py; verbos pasados, límites procesales, rol, cercanía y controles de fuga'}
        manifest['source_manifest_sha256']=file_hash(v4/'manifest.json')
        manifest['extractor_sha256']=file_hash(Path(__file__).parent/'extract_sections.py')
        (directory/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        write_jsonl(directory/'propuestas_secciones.jsonl',accepted if directory!=out else [r for r in accepted if r['articulo'] in catalog])
    write_jsonl(out/'nuevos_pares_pendientes.jsonl',accepted)
    write_jsonl(out/'errores_extraccion.jsonl',errors)
    write_jsonl(out/'catalogo_adiciones_fuente.jsonl',provenance)
    by_article=defaultdict(list)
    for row in accepted:by_article[row['articulo']].append(row)
    review=[r for article in sorted(by_article,key=lambda a:(sum(b['articulo']==a and b['split']=='train' for b in baseline),a)) for r in by_article[article][:15]]
    write_jsonl(out/'cola_revision_prioritaria.jsonl',review)
    write_jsonl(out/'plantilla_revisiones.jsonl',[{'candidate_id':r['candidate_id'],'decision':None,'articulos_aprobados':[],
               'revisor':'','justificacion':'','hechos_corregidos':None} for r in review])
    audit={'baseline_rows':len(baseline),'conservative_rows':len(core),'expanded_rows':len(large),
           'added_by_tier':dict(Counter(r['nivel_seleccion'] for r in accepted)),
           'added_by_role':dict(Counter(r['papel_cita'] for r in accepted)),
           'new_train_articles':sorted({r['articulo'] for r in accepted}-{r['articulo'] for r in baseline if r['split']=='train'}),
           'unique_train_queries':len({fingerprint(r['hechos']) for r in large if r['split']=='train'}),
           'train_articles':len({r['articulo'] for r in large if r['split']=='train'}),
           'official_catalog_additions':len(additions),'excluded_counts':dict(stats),'remaining_missing_catalog':dict(missing),
           'review_queue':len(review),'validation_test_identical':True,'catalog_baseline_preserved':True,'extraction_errors':len(errors),
           'official_pdf_sha256':file_hash(pdf),'near_overlap_threshold':.75,
           'note':'No certifica aplicabilidad ni vigencia temporal; ampliación de catálogo cambia la dificultad de recuperación.'}
    (out/'auditoria_v6.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    source_dir=out/'fuentes';source_dir.mkdir();shutil.copy2(pdf,source_dir/'codigo_penal_fiscalia.pdf')
    cards=[]
    for r in review[:100]:
        esc=lambda v:html.escape(str(v))
        cards.append(f'<article><h2>CP {esc(r["articulo"])} — {esc(r["nivel_seleccion"])}</h2><p>{esc(r["hechos"])}</p><p>Rol heurístico: {esc(r["papel_cita"])}</p><details><summary>Evidencia</summary><p>{esc(r["evidencia_cita"])}</p></details><p><a href="{esc(r["fuente_url"])}">Resolución TSJ</a></p><code>{esc(r["candidate_id"])}</code></article>')
    (out/'revision_muestra.html').write_text('<!doctype html><meta charset="utf-8"><title>Revisión v6</title><style>body{max-width:1000px;margin:30px auto;font:17px/1.6 sans-serif}article{border:1px solid #ccc;padding:20px;margin:20px 0}code{overflow-wrap:anywhere}</style><h1>Pares v6 pendientes de revisión</h1><p>Esta selección no aprueba etiquetas. Contrastar relato, papel de cita y versión temporal de la norma.</p>'+''.join(cards),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('remaining_missing_catalog','new_train_articles')},ensure_ascii=False,indent=2))
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--v4-dir',required=True);p.add_argument('--v5-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--official-pages',required=True);p.add_argument('--official-pdf',required=True);p.add_argument('--raw-dir',required=True)
    a=p.parse_args();build(a.v4_dir,a.v5_dir,a.output_dir,a.official_pages,a.official_pdf,a.raw_dir)
