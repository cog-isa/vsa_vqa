import os
import json
import sys
sys.path.append('/home/mshaban/docker_workspace/visdial-ns-vqa/ns-vqa/reason')
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils
import torch

def find_clevr_question_type(out_mod):
    """Find CLEVR question type according to program modules"""
    if out_mod == 'count':
        q_type = 'count'
    elif out_mod == 'exist':
        q_type = 'exist'
    elif out_mod in ['equal_integer', 'greater_than', 'less_than']:
        q_type = 'compare_num'
    elif out_mod in ['equal_size', 'equal_color', 'equal_material', 'equal_shape']:
        q_type = 'compare_attr'
    elif out_mod.startswith('query'):
        q_type = 'query'
    return q_type


def check_program(pred, gt):
    """Check if the input programs matches"""
    # ground truth programs have a start token as the first entry
    for i in range(len(pred)):
        if pred[i] != gt[i+1]:
            return False
        if pred[i] == 2:
            break
    return True

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
executor = get_executor(opt)
model = Seq2seqParser(opt)

print('| running test')
stats = {
    'count': 0,
    'count_tot': 0,
    'exist': 0,
    'exist_tot': 0,
    'compare_num': 0,
    'compare_num_tot': 0,
    'compare_attr': 0,
    'compare_attr_tot': 0,
    'query': 0,
    'query_tot': 0,
    'correct_ans': 0,
    'correct_prog': 0,
    'total': 0
}

debug = 0 # DEBUG
debug_rounds = 10
for x, y, ans, idx in loader:
    with torch.no_grad():
        model.set_input(x, y)
        pred_program = model.parse()
        if debug:
            print('QUESTION:')
            for i in range(16):
                q_token = executor.vocab['question_idx_to_token'][x[0][i].item()]
                if q_token == "<END>":
                    break
                if q_token == "<START>":
                    continue
                print(q_token, end=' ')

            print('\nPROGRAM: ')
            for i in range(27):
                pg_token = executor.vocab['program_idx_to_token'][pred_program[0][i].item()]
                if pg_token == "<END>":
                    break
                print(pg_token)
            #debug_rounds -= 1
            #if debug_rounds == 0:
            #    exit(0)

    y_np, pg_np, idx_np, ans_np = y.detach().numpy(), pred_program.detach().numpy(), idx.detach().numpy(), ans.detach().numpy()

    
    for i in range(pg_np.shape[0]):
        pred_ans, _ = executor.run(pg_np[i], idx_np[i], 'val', guess=True)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

        if debug:
            print('PROGRAM ANSWER:')
            print(pred_ans)
            print('GT ANS:')
            print(gt_ans)

        q_type = find_clevr_question_type(executor.vocab['program_idx_to_token'][y_np[i][1]])
        if pred_ans == gt_ans:
            stats[q_type] += 1
            stats['correct_ans'] += 1
        if check_program(pg_np[i], y_np[i]):
            stats['correct_prog'] += 1

        stats['%s_tot' % q_type] += 1
        stats['total'] += 1
    print('| %d/%d questions processed, accuracy %f' % (stats['total'], len(loader.dataset), stats['correct_ans'] / stats['total']))

result = {
    'count_acc': stats['count'] / stats['count_tot'],
    'exist_acc': stats['exist'] / stats['exist_tot'],
    'program_acc': stats['correct_prog'] / stats['total'],
    'overall_acc': stats['correct_ans'] / stats['total']
}
print(result)
print(stats['count'])
print(stats['count_tot'])
print(stats['exist'])
print(stats['exist_tot'])
print(stats['correct_ans'])
print(stats['total'])

utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    