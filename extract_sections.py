"""Propuestas de hechos por secciones. Las heurísticas nunca aprueban etiquetas."""
import re
import unicodedata
from dataset_utils import fingerprint, normalize_text
from extract_pairs import clean_document, extraer_articulos_citados, prepare_context

FACTS = re.compile(r'\b(?:sustraj\w*|apoder\w*|golpe\w*|agredi\w*|agred\w*|amenaz\w*|dispar\w*|portab\w*|entreg\w*|falsific\w*|vend\w*|compr\w*|engañ\w*|viol\w*|abus\w*|lesion\w*|mat\w*|apuñal\w*|impact\w*|conduci\w*|atropell\w*|exigi\w*|deposit\w*|transfer\w*|recibi\w*|reten\w*|ingres\w*|oblig\w*|incumpli\w*|tocami\w*|penetr\w*|ocult\w*|transport\w*|suministr\w*|murio|fallecio|causo)\b', re.I)
PROCESS = re.compile(r'\b(?:casacion|apelacion|inadmisib\w*|precedente|agravio|nulidad|alzada|fundamentacion|recurso|doctrina legal)\b', re.I)
FACT_HEADING = re.compile(r'(?:hechos\s+(?:probados|acreditados|facticos)|relacion\s+(?:factica|de\s+(?:los\s+)?hechos)|descripcion\s+de\s+(?:los\s+)?hechos|antecedentes\s+(?:facticos|del\s+hecho))',re.I)
LEGAL_HEADING = re.compile(r'(?:fundamentos\s+(?:juridicos|legales)|analisis\s+(?:juridico|del\s+caso)|doctrina\s+legal|por\s+tanto)',re.I)

def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFKD',text.casefold()) if not unicodedata.combining(c))

def citation_role(text):
    text=plain(text)
    rules={'absolucion':r'absolvi\w*|absuelv\w*|absoluc\w*',
           'condena':r'conden\w*|declar\w*\s+(?:autor|culpable)',
           'acusacion':r'acus\w*|imput\w*', 'precedente':r'precedente|auto\s+supremo\s+\d'}
    found=[name for name,pattern in rules.items() if re.search(pattern,text)]
    return found[0] if len(found)==1 else ('ambiguo' if found else 'no_identificado')

def fact_blocks(text):
    """Offsets sobre clean_document. Bloques completos, no ventana pegada a la cita."""
    # Encabezados incluso en documentos que han perdido saltos de línea.
    boundaries={0,len(text)}
    for match in re.finditer(r'\n+|(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÑ])',text): boundaries.add(match.end())
    points=sorted(boundaries);blocks=[];section='no_identificada'
    for start,end in zip(points,points[1:]):
        raw=text[start:end].strip();norm=plain(raw)
        if FACT_HEADING.search(norm[:220]): section='hechos'
        elif LEGAL_HEADING.search(norm[:160]): section='juridica'
        factual=len(FACTS.findall(norm));procedural=len(PROCESS.findall(norm))
        # Señales de verbos no bastan para validar: solo priorizan revisión.
        if len(raw)<100 or factual<1:continue
        if procedural>max(2,2*factual):continue
        score=3*factual-procedural+(5 if section=='hechos' else 0)
        blocks.append({'inicio':start,'fin':end,'texto':raw,'seccion':section,
                       'senales_facticas':factual,'senales_procesales':procedural,'prioridad':score})
    return blocks

def section_candidates(data,max_blocks=4,overlap_first=False):
    text=clean_document(data.get('contenido'));citations=extraer_articulos_citados(text)
    if not citations:return []
    masked,_=prepare_context(text);blocks=sorted(fact_blocks(text),key=lambda b:(-b['prioridad'],b['inicio']))[:max_blocks]
    result=[];seen=set()
    for block in blocks:
        facts=normalize_text(masked[block['inicio']:block['fin']])
        if len(facts)<100:continue
        # Cercanía es evidencia propuesta, nunca prueba de aplicabilidad.
        distance=lambda c: max(block['inicio']-c['fin'],c['inicio']-block['fin'],0)
        # En v6 una cita dentro del bloque tiene distancia cero y precede a las externas.
        nearest=sorted(citations,key=distance if overlap_first else lambda c: min(abs(c['inicio']-block['fin']),abs(c['fin']-block['inicio'])))
        anchor=nearest[0]
        associated=[c for c in nearest if c['inicio']==anchor['inicio']]
        for citation in associated:
            key=(fingerprint(facts),citation['numero'])
            if key in seen:continue
            seen.add(key)
            evidence=text[max(0,citation['inicio']-220):min(len(text),citation['fin']+220)]
            row={'hechos':facts,'articulo':citation['numero'],'norma':'Codigo Penal',
                 'fuente_id':data['id'],'fuente_url':f'https://genesis.tsj.bo/jurisprudencia/{data["id"]}',
                 'documento_hash':fingerprint(text),'nro_resolucion':data.get('nro_resolucion'),
                 'nro_expediente':data.get('nro_expediente'),'fecha_emision':data.get('fecha_emision'),
                 'resultado_resolucion':data.get('formas_resoluciones'),'sala':data.get('sala'),
                 'hechos_inicio':block['inicio'],'hechos_fin':block['fin'],'seccion':block['seccion'],
                 'senales_facticas':block['senales_facticas'],'senales_procesales':block['senales_procesales'],
                 'prioridad':block['prioridad'],'papel_cita':citation_role(evidence),
                 'cita_original':citation['cita'],'cita_inicio':citation['inicio'],'cita_fin':citation['fin'],
                 'evidencia_cita':evidence,'distancia_cita':distance(citation),
                 'estado_revision':'pendiente','metodo':'secciones_v5','tipo_contenido':'candidato_factico'}
            row['candidate_id']=fingerprint(f'{data["id"]}|{block["inicio"]}|{block["fin"]}|{citation["numero"]}')
            result.append(row)
    return result
