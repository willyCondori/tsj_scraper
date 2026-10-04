import sys,unittest,json,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from build_v6 import quality,close_to_held,shingles,official_additions
from dataset_utils import write_jsonl,read_jsonl
from apply_reviews import apply
from extract_sections import section_candidates

class V6Tests(unittest.TestCase):
    def row(self):
        return {'split':'train','articulo':'331','hechos':('El acusado sustrajo el vehículo de la víctima y lo vendió a un tercero en otra ciudad. '+ 'Los testigos identificaron el vehículo estacionado frente al domicilio y relataron las circunstancias del hecho.'), 'papel_cita':'condena','seccion':'hechos','distancia_cita':0}
    def test_actions_not_offense_nouns(self):
        r=self.row();r['hechos']='El sujeto fue condenado por abuso de confianza y estafa, conforme a las normas que rigen la sentencia y sus efectos para el acusado. '*2
        self.assertEqual(quality(r,{'331':'texto'},set())[1],'pocas_acciones')
    def test_pending_and_evaluation_excluded(self):
        r=self.row();selected,_=quality(r,{'331':'texto'},set());self.assertEqual(selected['estado_revision'],'pendiente')
        r['split']='test';self.assertEqual(quality(r,{'331':'texto'},set())[1],'fuera_train')
    def test_absolution_excluded(self):
        r=self.row();r['papel_cita']='absolucion';self.assertIsNone(quality(r,{'331':'texto'},set())[0])
    def test_containment_leak(self):
        text=' '.join('palabra'+str(i) for i in range(100));fragment=' '.join(text.split()[10:40])
        self.assertTrue(close_to_held(fragment,[shingles(text)]))
    def test_citation_inside_block_precedes_external(self):
        facts='HECHOS PROBADOS: '+('El acusado sustrajo el vehículo y lo vendió a otro comprador, '*8)+'art. 331 del Código Penal '+('durante la noche transportó el vehículo sustraído, '*8)
        rows=section_candidates({'id':1,'contenido':'<p>'+facts+'</p><p>art. 251 del Código Penal.</p>'},overlap_first=True)
        self.assertEqual(rows[0]['articulo'],'331')
    def test_catalog_preserves_old_text(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'pages.json';p.write_text(json.dumps(['Artículo 251. (Homicidio). '+'texto '*25+'\nArtículo 252 bis. (Feminicidio). '+'contenido '*30]),encoding='utf-8')
            additions,source=official_additions(p,{'251':'original'})
            self.assertNotIn('251',additions);self.assertIn('252 bis',additions);self.assertEqual(source[0]['pagina_pdf'],1)
    def test_review_replaces_or_rejects_weak_labels(self):
        for decision in ('aprobar','rechazar'):
            with self.subTest(decision=decision),tempfile.TemporaryDirectory() as t:
                p=Path(t);r={**self.row(),'candidate_id':'new','fuente_id':1,'grupo_id':'train','estado_revision':'pendiente'}
                held={**r,'candidate_id':'held','fuente_id':2,'grupo_id':'test','split':'test','hechos':'otro texto de evaluación '*15}
                (p/'catalogo_cp.json').write_text(json.dumps({'331':'uno','332':'dos'}))
                write_jsonl(p/'pares_entrenamiento.jsonl',[r,held]);write_jsonl(p/'propuestas_secciones.jsonl',[r])
                write_jsonl(p/'reviews.jsonl',[{'candidate_id':'new','decision':decision,'articulos_aprobados':['332'],'revisor':'test','justificacion':'comprobación','hechos_corregidos':None}])
                apply(p,p/'reviews.jsonl',p/'out');rows=read_jsonl(p/'out/pares_entrenamiento.jsonl')
                self.assertEqual([x for x in rows if x['split']=='test'],[held])
                new=[x for x in rows if x['split']=='train']
                if decision=='rechazar':self.assertEqual(new,[])
                else:self.assertEqual(new[0]['articulo'],'332');self.assertEqual(new[0]['estado_revision'],'aprobado')

if __name__=='__main__':unittest.main()
