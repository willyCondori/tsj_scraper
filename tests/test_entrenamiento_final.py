import ast
import json
import unittest
from pathlib import Path
from seleccionar_modelo import choose
from mejorar_dataset import clean_prefix

ROOT=Path(__file__).resolve().parents[1]

class FinalTrainingTests(unittest.TestCase):
    def test_notebook_syntax_and_embedded_code(self):
        notebook=json.loads((ROOT/'Entrenamiento_final_E5_base.ipynb').read_text(encoding='utf-8'))
        all_code=''
        for cell in notebook['cells']:
            if cell['cell_type']!='code':continue
            source=''.join(cell['source']);all_code+=source
            if not source.startswith('%'):compile(source,'notebook','exec')
            if "(CODE/'train_e5_base.py').write_text" in source:
                tree=ast.parse(source)
                embedded=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='write_text']
                values=[ast.literal_eval(n.args[0]) for n in embedded]
                self.assertIn((ROOT/'entrenamiento_final/train_e5_base.py').read_text(encoding='utf-8'),values)
        self.assertIn('REENTRENAR=False',all_code)
        self.assertIn('EVALUAR_TEST=False',all_code)
        self.assertIn('# modelo.save(',all_code)
        self.assertIn("selection_metric='f1_macro'",all_code)

    def test_reported_reference_and_rollback(self):
        record=json.loads((ROOT/'entrenamiento_final/resultados_validation_reportados.json').read_text(encoding='utf-8'))
        self.assertEqual(record['referencia']['queries'],16)
        self.assertAlmostEqual(record['referencia']['f1_macro_top1'],.6556776556776556)
        decision=choose(record['referencia'],record['candidato'],'checkpoint-126','candidate')
        self.assertEqual(decision['seleccion'],'referencia')
        self.assertFalse(record['test_evaluado_en_esta_prueba'])

    def test_limited_cleaning_preserves_negation(self):
        body='El acusado no firmó el documento y no recibió el dinero. '+('Se examinó la evidencia. '*5)
        self.assertEqual(clean_prefix('CONSIDERANDO: '+body),body)

    def test_final_model_identity(self):
        tree=ast.parse((ROOT/'entrenamiento_final/train_e5_base.py').read_text(encoding='utf-8'))
        model=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='BASE_MODEL' for t in n.targets)]
        self.assertEqual(model,['intfloat/multilingual-e5-base'])

if __name__=='__main__':unittest.main()
