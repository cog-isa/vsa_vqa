import random
import json
import utils.utils as utils
from vsa_clevr.vsa import bind, bundle, cyclicsh, ItemMemory
from vsa_clevr.vsa import generate as hd
from vsa_clevr import vsa
import numpy as np

OBJECTS = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


ANSWER_CANDIDATES = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'yes', 'no'}


class VSAReasoner:
    """Symbolic program executor"""

    def __init__(self, out, vocab_json='../data/raw/visdial/raw/visdial_vocab.json'):
        self.vocab = utils.load_vocab(vocab_json)
        self.objects = OBJECTS
        self.answer_candidates = ANSWER_CANDIDATES
        self.category = out['attrs']['category']
        self.attrs = out['attrs']['attribute']
        self.objects = out['attrs']['object']

        self.modules = {}
        self._register_modules()
    
    def run(self, x, scene, guess=False, debug=False):
        ans, temp = None, None

        # Find the length of the program sequence before the '<END>' token
        length = 0
        for k in range(len(x)):
            l = len(x) - k
            if self.vocab['program_idx_to_token'][x[l-1]] == '<END>':
                length = l
        if length == 0:
            return 'error', []
        
        scene = scene['scene_vec']
        #print(index)
        #print(scene)
        self.exe_trace = []
        tokens = []
        obj_type = None

        self.exe_trace = []
        for j in range(length):
            i = length - 1 - j
            #print(x[i])
            token = self.vocab['program_idx_to_token'][x[i]]
            if token == 'scene':
                temp = ans
                ans = scene
            elif token.startswith('filter_type'):
                try:
                    token, obj_type = token.split('[')
                    
                    for obj in OBJECTS:
                        if obj.startswith(obj_type):
                            obj_type = obj
                            break

                except:
                    ans = 'error'
                    break

                if obj_type[-1] == ']':
                    obj_type = obj_type[:-1]

            if token in self.modules:
                module = self.modules[token]
                if token.startswith('same') or token.startswith('relate'):
                    ans = module(ans, scene)

                    tokens.append('{} {} {}'.format(token, ans, scene))
                elif token.startswith('filter_type'):
                    ans = module(ans, obj_type)
                    
                    tokens.append('{} {} {}'.format(token, ans, obj_type))
                else:
                    ans = module(ans, temp)

                    tokens.append('{} {} {}'.format(token, ans, temp))
                if ans == 'error':
                    tokens.append('{}'.format(token))
                    
                    break
            self.exe_trace.append(ans)
            if debug:
                print(token)
                print('ans:')
                self._print_debug_message(ans)
                print('temp: ')
                self._print_debug_message(temp)
                print()
        ans = str(ans)

        if ans == 'error' and guess:
            final_module = self.vocab['program_idx_to_token'][x[0]]
            if final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])

        return ans, tokens

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        print(obj)
    
    def _register_modules(self):
        self.modules['count'] = self.count
        self.modules['poping'] = self.poping
        self.modules['filter_type'] = self.filter_type
        self.modules['filter_up'] = self.filter_up
        self.modules['filter_down'] = self.filter_down
        self.modules['exist'] = self.exist
        self.modules['filter_left'] = self.filter_left
        self.modules['filter_right'] = self.filter_right
        self.modules['filter_center'] = self.filter_center
        self.modules['relate_up'] = self.relate_up
        self.modules['relate_down'] = self.relate_down
        self.modules['relate_left'] = self.relate_left
        self.modules['relate_right'] = self.relate_right
        self.modules['sort_up'] = self.sort_up
        self.modules['sort_down'] = self.sort_down
        self.modules['sort_left'] = self.sort_left
        self.modules['sort_right'] = self.sort_right
        self.modules['query_type'] = self.query_type
        self.modules['intersect'] = self.intersect
        self.modules['union'] = self.union
        self.modules['query_first'] = self.query_first

    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return 'error'

    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return 'error'

    def poping(self, scene, obj):
        if type(scene) == list and type(obj) == dict:
            output = []
            for o in scene:
                if o != obj:
                    output.append(o)
            return output
        return 'error'

    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return 'error'

    def filter_type(self, scene, t, prev=None):
        #print('cat: {}'.format(self.category.get_im_name()))
        #print(scene)
        
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.category, t)
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def get_objects(self, scene, attr, value, thr=0.05):
        attr_hd = self.attrs.get_vector(attr.get_im_name())
        value_hd = attr.get_vector(value)
        obj_n = bind(scene, bind(attr_hd, value_hd))
        _, obj_idx = np.where(self.objects.search(obj_n, distance=False) > thr)

        obj = []
        for i in obj_idx:
            obj.append(self.objects.get_name(i))

        return obj

    # def filter_type(self, scene, t):
    #     #print('category: {}'.format(t))
    #     if type(scene) == list:
    #         output = []
    #         for o in scene:
    #             #print(o)
    #             if o['category'] == t:
             
    #                output.append(o)
    #         #print('output: {}'.format(output))
    #         return output
    #     return 'error'

    def filter_up(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['2d_coords'][1] < 0.5:
                    output.append(o)
            return output
        return 'error'

    def filter_down(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['2d_coords'][1] >= 0.5:
                    output.append(o)
            return output
        return 'error'

    def filter_left(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['2d_coords'][0] < 0.5:
                    output.append(o)
            return output
        return 'error'

    def filter_right(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o['2d_coords'][0] >= 0.5:
                    output.append(o)
            return output
        return 'error'

    def filter_center(self, scene, _):
        eps = 0.1
        if type(scene) == list:
            output = []
            for o in scene:
                if abs(o['2d_coords'][0] - 0.5) < eps and abs(o['2d_coords'][1]- 0.5) < eps:
                    output.append(o)
            return output
        return 'error'

    def relate_up(self, scene, obj):
        if type(scene) == list and type(obj) == dict:
            output = []
            y = obj['2d_coords'][1]
            for o in scene:
                if o['2d_coords'][1] < y:
                    output.append(o)
            return output
        return 'error'

    def relate_down(self, scene, obj):
        if type(scene) == list and type(obj) == dict:
            output = []
            y = obj['2d_coords'][1]
            for o in scene:
                if o['2d_coords'][1] > y:
                    output.append(o)
            return output
        return 'error'

    def relate_left(self, scene, obj):
        if type(scene) == list and type(obj) == dict:
            output = []
            x = obj['2d_coords'][0]
            for o in scene:
                if o['2d_coords'][0] < x:
                    output.append(o)
            return output
        return 'error'

    def relate_right(self, scene, obj):
        if type(scene) == list and type(obj) == dict:
            output = []
            x = obj['2d_coords'][0]
            for o in scene:
                if o['2d_coords'][0] > x:
                    output.append(o)
            return output
        return 'error'

    def sort_up(self, scene, _):
        if type(scene) == list:
            scene.sort(key=lambda x: x['2d_coords'][1])
            return scene
        return 'error'

    def sort_down(self, scene, _):
        if type(scene) == list:
            scene.sort(key=lambda x: x['2d_coords'][1])
            return scene[::-1]
        return 'error'

    def sort_left(self, scene, _):
        if type(scene) == list:
            scene.sort(key=lambda x: x['2d_coords'][0])
            return scene
        return 'error'

    def sort_right(self, scene, _):
        if type(scene) == list:
            scene.sort(key=lambda x: x['2d_coords'][0])
            return scene[::-1]
        return 'error'

    def query_first(self, scene, _):
        if type(scene) == list:
            if len(scene) == 0:
                return None
            return scene[0]
        return 'error'

    def query_type(self, obj, _):
        if type(obj) == dict:
            return obj['category']
        return 'error'