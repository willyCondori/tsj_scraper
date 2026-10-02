import sys
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extract_pairs import extraer_articulos_citados,clasificar_contenido_hechos,clean_document,extraer_ventana_hecho
from dataset_utils import article_number,grouped_split,load_catalog
from augment_pairs import augment
from finetune_embeddings import validate_data

class PipelineTests(unittest.TestCase):
    def test_notebook_checkpoint_return_order(self):
        notebook=json.loads((Path(__file__).resolve().parents[1]/'Modelo_fin_checkpoint_corregido.ipynb').read_text(encoding='utf-8'))
        text=next(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code' and 'def buscar_ultimo_checkpoint()' in ''.join(c['source']))
        node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='buscar_ultimo_checkpoint')
        with tempfile.TemporaryDirectory() as directory:
            context={'os':os,'DRIVE_PATH':directory}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'checkpoint','exec'),context)
            self.assertEqual(context['buscar_ultimo_checkpoint'](),(None,0))
            (Path(directory)/'checkpoint-epoch-1').mkdir();(Path(directory)/'checkpoint-epoch-3').mkdir()
            result=context['buscar_ultimo_checkpoint']()
            self.assertEqual(result,(str(Path(directory)/'checkpoint-epoch-3'),3))
    def test_citations(self):
        for text,expected in [('arts. 251-253 del CP',['251','253','252']),('art. 335 del CP establece',['335']),
            ('art. 335 del CP. persona declaró',['335']),
            ('art. 416 del CPP',[]),('art. 416 del C.P.P.',[]),('art. 308 septies del Código Penal',['308 septies']),
            ('arts. 282, 283 y 287, todos del Código Penal',['282','283','287'])]:
            self.assertEqual([a['numero'] for a in extraer_articulos_citados(text)],expected)
    def test_normalization(self):
        self.assertEqual(article_number('308bis'),'308 bis');self.assertEqual(article_number(335.0),'335')
        with self.assertRaises(ValueError): article_number('308 bis basura')
    def test_classification(self):
        self.assertEqual(clasificar_contenido_hechos('inadmisible')['señales_procesales'],1)
        self.assertEqual(clasificar_contenido_hechos('El día del recurso')['tipo_contenido'],'indeterminado')
    def test_clean_and_mask(self):
        self.assertEqual(clean_document('<p>A&nbsp;B</p><p>C</p>'),'A B\nC')
        text='La persona sustrajo el objeto. '*25+'arts. 282, 283 y 287, todos del Código Penal. '+'Luego declaró. '*20
        citation=extraer_articulos_citados(text)[0]
        self.assertNotIn('282',extraer_ventana_hecho(text,citation['inicio'],citation['fin']))
    def test_transitive_groups(self):
        rows=[{'fuente_id':1,'hechos':'A','articulo':'1'},{'fuente_id':2,'hechos':'A','articulo':'2'},
            {'fuente_id':2,'hechos':'B','articulo':'3'},{'fuente_id':3,'hechos':'B','articulo':'4'}]
        result=grouped_split(rows)
        self.assertEqual(len({r['grupo_id'] for r in result}),1)
        self.assertEqual(result,grouped_split(list(reversed(rows))))
    def test_case_groups(self):
        rows=[{'fuente_id':i,'hechos':str(i),'articulo':'335','nro_expediente':'La Paz 29/2012'} for i in (1,2)]
        self.assertEqual(len({r['grupo_id'] for r in grouped_split(rows)}),1)
    def test_augment_train_only(self):
        rows=[{'hechos':'la víctima declaró','articulo':'335','split':split,'estado_revision':'aprobado'} for split in ['train','validation','test']]
        result=augment(rows,{'víctima':['persona agraviada']})
        self.assertTrue(all(r['split']=='train' for r in result if r['tipo']=='parafraseado'))
    def test_catalog_cp_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'catalog.json';path.write_text('[{"numero_articulo":"308bis","norma_sigla":"Ley 1173","contenido":"otro"},{"numero_articulo":"335","norma_sigla":"CP","contenido":"estafa"}]',encoding='utf-8')
            self.assertEqual(load_catalog(path),{'335':'estafa'})
    def test_leak_detection(self):
        rows=[{'hechos':'mismo','articulo':'335','fuente_id':i,'grupo_id':str(i),'split':split,'tipo':'original'} for i,split in enumerate(['train','test'])]
        with self.assertRaisesRegex(ValueError,'Fuga'):validate_data(rows,{'335':'estafa'})

if __name__=='__main__':unittest.main()
