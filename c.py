"""Resumen de distribución de un run; reemplaza imports obsoletos del aumentador."""
import argparse
from collections import Counter
from pathlib import Path
from dataset_utils import read_jsonl

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',required=True);args=parser.parse_args()
    rows=read_jsonl(Path(args.run_dir)/'pares_aumentados.jsonl')
    for column in ('tipo_contenido','tipo','split','articulo'):
        print(column,Counter(row.get(column) for row in rows))
