"""Búsqueda pequeña: selección exclusiva por validación, test no consultado."""
import argparse,gc,json
from pathlib import Path
from train_v5 import train

def experiment(run_dir,output_root,epochs=12,batch_size=16,compare_balance=True):
    import torch
    root=Path(output_root);results=[]
    # Mismo seed/modelo/dataset: aislar LR antes de probar otro factor.
    for rate in (5e-6,1e-5,2e-5):
        output=root/f'lr_{rate:g}_seed_42'
        model,metrics=train(run_dir,output,epochs=epochs,batch_size=batch_size,learning_rate=rate,evaluate_test=False)
        results.append({'learning_rate':rate,'class_balance_loss':False,'validation':metrics['validation'],'checkpoint':metrics['best_checkpoint']})
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    # Macro F1 priorizado; MRR y accuracy desempatan. No seleccionar con test.
    results.sort(key=lambda r:(r['validation']['f1_macro_top1'],r['validation']['mrr'],r['validation']['accuracy@1']),reverse=True)
    if compare_balance:
        rate=results[0]['learning_rate']
        model,metrics=train(run_dir,root/f'lr_{rate:g}_balanced_seed_42',epochs=epochs,batch_size=batch_size,learning_rate=rate,evaluate_test=False,class_balance_loss=True)
        results.append({'learning_rate':rate,'class_balance_loss':True,'validation':metrics['validation'],'checkpoint':metrics['best_checkpoint']})
        del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        results.sort(key=lambda r:(r['validation']['f1_macro_top1'],r['validation']['mrr'],r['validation']['accuracy@1']),reverse=True)
    (root/'comparacion_validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Selección provisional solo por validación:',results[0]);return results

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-dir',required=True);p.add_argument('--output-root',required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--skip-balance',action='store_true')
    a=p.parse_args();experiment(a.run_dir,a.output_root,a.epochs,a.batch_size,compare_balance=not a.skip_balance)
