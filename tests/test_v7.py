import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_structured_v7 import citations, extract_document

class StructuredTests(unittest.TestCase):
    def test_norm_identity(self):
        rows=list(citations('art. 251 del Código Penal; arts. 124 y 173 del CPP; art. 5 de la Ley 348'))
        self.assertEqual([(r['norma_id'],r['articulo']) for r in rows], [('CP','251'),('CPP','124'),('CPP','173'),('LEY:348','5')])
    def test_evidence_and_no_automatic_positive(self):
        text='HECHOS PROBADOS: El acusado golpeó a la víctima en la calle y causó lesiones durante la noche.\nFue condenado conforme al art. 271 del Código Penal.'
        r=extract_document({'id':1,'contenido':text},'test')
        self.assertEqual(len(r['hechos']),1)
        self.assertNotIn('HECHOS PROBADOS',r['hechos'][0]['hechos_limpios'])
        self.assertEqual(r['relaciones'][0]['papel_propuesto'],'condena')
        self.assertFalse(r['relaciones'][0]['usable_entrenamiento'])
        self.assertIsNone(r['relaciones'][0]['hecho_id'])
        self.assertEqual(r['relaciones'][0]['split'],'test')
        for x in r['decisiones']: self.assertEqual(text[x['inicio']:x['fin']],x['texto'])
    def test_precedent_and_unresolved(self):
        r=extract_document({'id':2,'contenido':'El Auto Supremo 123 cita el art. 251 del CP. Se invoca el art. 124.'})
        self.assertEqual(r['relaciones'][0]['papel_propuesto'],'precedente_o_documento_citado')
        self.assertTrue(r['citas_sin_norma'])
        self.assertEqual(r['relaciones'][0]['split'],'revision_only')
    def test_negation_preserved(self):
        r=extract_document({'id':3,'contenido':'El testigo declaró que el acusado no golpeó a la víctima durante la noche en el lugar indicado.'})
        self.assertIn('no golpeó',r['hechos'][0]['hechos_limpios'])
    def test_procedural_violation_not_factual_action(self):
        r=extract_document({'id':4,'contenido':'El Auto de Vista violó el debido proceso al devolver obrados y resolver el recurso de apelación.'})
        self.assertEqual(r['hechos'],[])

if __name__=='__main__': unittest.main()
