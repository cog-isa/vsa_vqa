import os
import json
import sys
sys.path.append('/home/mshaban/docker_workspace/ns-vqa/reason_split_heads')
from vsa_clevr.vsa_reasoner import VSAReasoner
from vsa_clevr.vsa_scene_parser import VSASceneParser 
from options.test_options import TestOptions
from datasets import get_dataloader
from executors import get_executor
from models.parser import Seq2seqParser
import utils.utils as utils
import torch


DESCR = {
    'attribute': ['attribute', 'color', 'size', 'shape', 'material', 'coordinates'],
    'color': ['purple', 'blue', 'brown', 'cyan', 'yellow', 'red', 'gray', 'green'],
    'size': ['small', 'large'],
    'shape': ['sphere', 'cylinder', 'cube'],
    'material': ['metal', 'rubber'],
    'coordinates': ['x_coord', 'y_coord', 'z_coord']
}


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

HD_DIM = 30000
VSA_TYPE = 'polar'
THR = 16

opt = TestOptions().parse()
loader = get_dataloader(opt, 'val')
parser = VSASceneParser('../data/attr_net/results/clevr_val_scenes_zerotrained.json', dim=HD_DIM, vsa_type=VSA_TYPE, thr=THR, descr=DESCR)
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
for x, y, ans, idx in loader:
    with torch.no_grad():
        model.set_input(x, y)
        pred_program = model.parse()

    y_np, pg_np, idx_np, ans_np = y.detach().numpy(), pred_program.detach().numpy(), idx.detach().numpy(), ans.detach().numpy()

    for i in range(pg_np.shape[0]):
        scene = parser.parse(idx_np[i])
        executor = VSAReasoner(scene)
        
        pred_ans, _ = executor.run(pg_np[i], scene, guess=True)
        gt_ans = executor.vocab['answer_idx_to_token'][ans_np[i]]

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
    'compare_num_acc': stats['compare_num'] / stats ['compare_num_tot'],
    'compare_attr_acc': stats['compare_attr'] / stats['compare_attr_tot'],
    'query_acc': stats['query'] / stats['query_tot'],
    'program_acc': stats['correct_prog'] / stats['total'],
    'overall_acc': stats['correct_ans'] / stats['total']
}
print(result)

utils.mkdirs(os.path.dirname(opt.save_result_path))
with open(opt.save_result_path, 'w') as fout:
    json.dump(result, fout)
print('| result saved to %s' % opt.save_result_path)
    