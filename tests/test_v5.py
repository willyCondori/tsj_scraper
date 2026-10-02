import json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from extract_sections import section_candidates,citation_role
from dataset_utils import write_jsonl
from apply_reviews import apply
from retrieval_tools import BM25,rrf
from prepare_v5_experiment import eligible

class V5Tests(unittest.TestCase):
    def test_facts_separate_from_citation(self):
        facts='HECHOS PROBADOS: El acusado sustrajo un vehículo estacionado en la puerta de la casa, lo transportó a otra ciudad y lo vendió a una tercera persona.'
        data={'id':1,'contenido':f'<p>{facts}</p><p>Se acusa por el art. 331 del Código Penal.</p>'}
        rows=section_candidates(data)
        self.assertEqual(rows[0]['articulo'],'331')
        self.assertIn('sustrajo',rows[0]['hechos'])
        self.assertNotIn('331',rows[0]['hechos'])
        self.assertEqual(rows[0]['estado_revision'],'pendiente')
        self.assertEqual(rows[0]['papel_cita'],'acusacion')
    def test_ambiguous_roles_not_approved(self):
        self.assertEqual(citation_role('La acusación fue presentada y la sentencia absolvió al acusado'),'ambiguo')
    def test_cpp_excluded(self):
        self.assertEqual(section_candidates({'id':1,'contenido':'El acusado golpeó a la víctima y el recurso menciona el art. 416 del CPP.'}),[])
    def test_bm25_and_rrf(self):
        scores=BM25(['sustracción vehículo','documento falsificado']).scores('vehículo')
        self.assertGreater(scores[0],scores[1]);self.assertEqual(rrf([0,1],[1,0],weight=1),[0,1])
    def test_weak_variant_requires_actions_not_only_offense_names(self):
        row={'split':'train','articulo':'1','distancia_cita':0,'senales_facticas':3,
             'senales_procesales':0,'papel_cita':'condena','hechos':'La sentencia menciona falsificación y transporte, pero no describe acciones. '*3}
        self.assertFalse(eligible(row,{'1':'norma'},set()))
        row['hechos']='El acusado golpeó a la víctima y la amenazó durante el hecho, en el domicilio donde se encontraba, según la descripción del relato de hechos probado en la sentencia.'
        self.assertTrue(eligible(row,{'1':'norma'},set()))
        row['split']='test';self.assertFalse(eligible(row,{'1':'norma'},set()))
    def test_reviews_frozen_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'catalogo_cp.json').write_text('{"1":"norma"}')
            base={'hechos':'texto evaluación '*12,'articulo':'1','fuente_id':2,'grupo_id':'eval','split':'test'}
            write_jsonl(root/'pares_entrenamiento.jsonl',[base])
            candidate={'candidate_id':'a','hechos':'nuevos hechos '*12,'articulo':'1','fuente_id':1,'grupo_id':'train','split':'test'}
            write_jsonl(root/'propuestas_secciones.jsonl',[candidate])
            review={'candidate_id':'a','decision':'aprobar','articulos_aprobados':['1'],'revisor':'r','justificacion':'evidencia'}
            write_jsonl(root/'reviews.jsonl',[review])
            with self.assertRaisesRegex(ValueError,'evaluación congelada'):apply(root,root/'reviews.jsonl',root/'out')
            candidate['split']='train';candidate['hechos']=base['hechos']
            write_jsonl(root/'propuestas_secciones.jsonl',[candidate])
            with self.assertRaisesRegex(ValueError,'Fuga'):apply(root,root/'reviews.jsonl',root/'out')
            candidate['hechos']='nuevos hechos '*12
            write_jsonl(root/'propuestas_secciones.jsonl',[candidate])
            apply(root,root/'reviews.jsonl',root/'out')
            result=[json.loads(line) for line in (root/'out/pares_entrenamiento.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(result[0],base);self.assertEqual(result[1]['estado_revision'],'aprobado')

if __name__=='__main__':unittest.main()
