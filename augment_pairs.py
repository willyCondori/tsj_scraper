"""Variantes opcionales, únicamente en train y con sustituciones revisadas."""
import argparse
import json
import random
import re
from pathlib import Path
from dataset_utils import fingerprint, read_jsonl, write_jsonl

def augment(rows, replacements=None, seed=42):
    rng=random.Random(seed); result=[]; seen=set()
    for row in rows:
        original={**row,'tipo':'original','parent_id':fingerprint(row['hechos']+'|'+row['articulo'])}
        result.append(original); seen.add((fingerprint(row['hechos']),row['articulo']))
    for row in list(result):
        if row['split']!='train' or row.get('estado_revision')!='aprobado': continue
        for source,variants in (replacements or {}).items():
            if not variants: continue
            text=re.sub(r'(?<!\w)'+re.escape(source)+r'(?!\w)',lambda m:rng.choice(variants),row['hechos'],flags=re.I)
            key=(fingerprint(text),row['articulo'])
            if key in seen: continue
            seen.add(key); result.append({**row,'hechos':text,'tipo':'parafraseado','regla_aumento':source})
            break
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--synonyms')
    a=p.parse_args();directory=Path(a.run_dir)
    replacements=json.loads(Path(a.synonyms).read_text(encoding='utf-8')) if a.synonyms else {}
    write_jsonl(directory/'pares_aumentados.jsonl',augment(read_jsonl(directory/'pares_entrenamiento.jsonl'),replacements))
