import numpy as np

from vsa_clevr.vsa import bind, bundle, cyclicsh
import utils.utils as utils
import json 
import random

CLEVR_COLORS = ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow']
CLEVR_MATERIALS = ['rubber', 'metal']
CLEVR_SHAPES = ['cube', 'cylinder', 'sphere']
CLEVR_SIZES = ['large', 'small']

CLEVR_ANSWER_CANDIDATES = {
    'count': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
    'equal_color': ['yes', 'no'],
    'equal_integer': ['yes', 'no'],
    'equal_material': ['yes', 'no'],
    'equal_shape': ['yes', 'no'],
    'equal_size': ['yes', 'no'],
    'exist': ['yes', 'no'],
    'greater_than': ['yes', 'no'],
    'less_than': ['yes', 'no'],
    'query_color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
    'query_material': ['metal', 'rubber'],
    'query_size': ['small', 'large'],
    'query_shape': ['cube', 'cylinder', 'sphere'],
    'same_color': ['yes', 'no'],
    'same_material': ['yes', 'no'],
    'same_size': ['yes', 'no'],
    'same_shape': ['yes', 'no']
}

class VSAReasoner:
    def __init__(self, out):
        self.attrs = out['attrs']['attribute']
        self.color = out['attrs']['color']
        self.size = out['attrs']['size']
        self.shape = out['attrs']['shape']
        self.material = out['attrs']['material']
        self.objects = out['attrs']['object']
        # self.boolean = out['attrs']['boolean']
        # self.zero = out['attrs']['zero']
        self.coordinates = out['attrs']['coordinates']
        self.x_coordinates = out['attrs']['x_coord']
        self.y_coordinates = out['attrs']['y_coord']
        self.z_coordinates = out['attrs']['z_coord']

        self.vocab = utils.load_vocab('../data/reason/clevr_h5/clevr_vocab.json')

        self.answer_candidates = CLEVR_ANSWER_CANDIDATES
        
        self.colors = CLEVR_COLORS
        self.materials = CLEVR_MATERIALS
        self.shapes = CLEVR_SHAPES
        self.sizes = CLEVR_SIZES
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES

        self.modules = {}
        self._register_modules()

    def _register_modules(self):
        self.modules['count'] = self.count
        self.modules['equal_color'] = self.equal_color
        self.modules['equal_integer'] = self.equal_integer
        self.modules['equal_material'] = self.equal_material
        self.modules['equal_shape'] = self.equal_shape
        self.modules['equal_size'] = self.equal_size
        self.modules['exist'] = self.exist
        self.modules['filter_color[blue]'] = self.filter_blue
        self.modules['filter_color[brown]'] = self.filter_brown
        self.modules['filter_color[cyan]'] = self.filter_cyan
        self.modules['filter_color[gray]'] = self.filter_gray
        self.modules['filter_color[green]'] = self.filter_green
        self.modules['filter_color[purple]'] = self.filter_purple
        self.modules['filter_color[red]'] = self.filter_red
        self.modules['filter_color[yellow]'] = self.filter_yellow
        self.modules['filter_material[rubber]'] = self.filter_rubber
        self.modules['filter_material[metal]'] = self.filter_metal
        self.modules['filter_shape[cube]'] = self.filter_cube
        self.modules['filter_shape[cylinder]'] = self.filter_cylinder
        self.modules['filter_shape[sphere]'] = self.filter_sphere
        self.modules['filter_size[large]'] = self.filter_large
        self.modules['filter_size[small]'] = self.filter_small
        self.modules['greater_than'] = self.greater_than
        self.modules['less_than'] = self.less_than
        self.modules['intersect'] = self.intersect
        self.modules['query_color'] = self.query_color
        self.modules['query_material'] = self.query_material
        self.modules['query_shape'] = self.query_shape
        self.modules['query_size'] = self.query_size
        self.modules['relate[behind]'] = self.relate_behind
        self.modules['relate[front]'] = self.relate_front
        self.modules['relate[left]'] = self.relate_left
        self.modules['relate[right]'] = self.relate_right
        self.modules['same_color'] = self.same_color
        self.modules['same_material'] = self.same_material
        self.modules['same_shape'] = self.same_shape
        self.modules['same_size'] = self.same_size
        self.modules['union'] = self.union
        self.modules['unique'] = self.unique

    def run(self, x, scene, guess=False, debug=False):
        ans, temp, prev = None, None, None

        # Find the length of the program sequence before the '<END>' token
        length = 0
        for k in range(len(x)):
            l = len(x) - k
            if self.vocab['program_idx_to_token'][x[l-1]] == '<END>':
                length = l
        if length == 0:
            return 'error', []

        scene = scene['scene_vec']

        self.exe_trace = []
        tokens = []

        for j in range(length):
            i = length - 1 - j
            token = self.vocab['program_idx_to_token'][x[i]]
            if token == 'scene':
                if temp is not None:
                    ans = 'error'
                    break
                temp = ans
                ans = scene

                tokens.append('scene')
            elif token in self.modules:
                module = self.modules[token]
                # try:
                if token.startswith('same') or token.startswith('relate'):
                    try:
                        ans = module(ans, scene)
                    except:
                        print(scene)
                        print(tokens)

                    tokens.append('{} {} {}'.format(token, ans, scene))
                elif token.startswith('filter'):
                    ans = module(scene, '_', ans)

                    tokens.append('{} {} {}'.format(token, scene, ans))
                elif token.startswith('query'):
                    ans = module(scene, ans, '_')
                    
                    tokens.append('{} {} {}'.format(token, scene, ans))
                else:
                    ans = module(ans, temp)
                    
                    tokens.append('{} {} {}'.format(token, ans, temp))
                if ans == 'error':
                    tokens.append('{}'.format(token))
                    
                    break
                # except Exception as e:
                #             print(e)
                #             print(token)
                #             print(scene)
                #             print(tokens)
            self.exe_trace.append(ans)

        ans = str(ans)

        if ans == 'error' and guess:
            final_module = self.vocab['program_idx_to_token'][x[0]]
            if final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])

        return ans, tokens

    def count(self, scene, _):
        if type(scene) == list:
            result = len(scene)
            return result
        else:
            return 'error'


    def equal_color(self, color1, color2):
        if type(color1) == str and color1 in self.colors and type(color2) == str and color2 in self.colors:
            if color1 == color2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_material(self, material1, material2):
        if type(material1) == str and material1 in self.materials and type(material2) == str and material2 in self.materials:
            if material1 == material2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def equal_shape(self, shape1, shape2):
        if type(shape1) == str and shape1 in self.shapes and type(shape2) == str and shape2 in self.shapes:
            if shape1 == shape2:
                return 'yes'
            else:
                return 'no'
        return 'error'
    
    def equal_size(self, size1, size2):
        if type(size1) == str and size1 in self.sizes and type(size2) == str and size2 in self.sizes:
            if size1 == size2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def exist(self, scene, _):
        if scene is None or scene == []:
            result = 'no'
        else:
            result = 'yes'
        return result

    def filter_blue(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'blue')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_brown(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'brown')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_cyan(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'cyan')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_gray(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'gray')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_green(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'green')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_purple(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'purple')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_red(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'red')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_yellow(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.color, 'yellow')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_rubber(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.material, 'rubber')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_metal(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.material, 'metal')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_cube(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.shape, 'cube')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_cylinder(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.shape, 'cylinder')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_sphere(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.shape, 'sphere')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_large(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.size, 'large')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def filter_small(self, scene, _, prev=None):
        if type(scene) == np.ndarray:
            result = self.get_objects(scene, self.size, 'small')
            if prev is not None:
                result = self.intersect(result, prev)
            return result
        else:
            return 'error'

    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return 'yes'
            else:
                return 'no'
        return 'error'

    def intersect(self, scene1, scene2):
        try:
            inter = set(scene1).intersection(set(scene2))
            return list(inter)
        except:
            return scene1

    def query_color(self, scene, obj, _):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        result = self.get_attr_value(scene, self.color, obj)
        return result

    def query_material(self, scene, obj, _):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        result = self.get_attr_value(scene, self.material, obj)
        return result

    def query_shape(self, scene, obj, _):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        result = self.get_attr_value(scene, self.shape, obj)
        return result

    def query_size(self, scene, obj, _):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        result = self.get_attr_value(scene, self.size, obj)
        return result

    def relate_behind(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'

        y_coord_hd = self.coordinates.get_vector('y_coord')
        obj_hd = self.objects.get_vector(obj)
        temp = bind(y_coord_hd, obj_hd)

        obj_coord_noisy = bind(scene, temp)
        obj_coord_id = np.argmin(self.y_coordinates.search(obj_coord_noisy))
        obj_coord_name = self.y_coordinates.get_name(obj_coord_id)
        obj_coord_hd = self.y_coordinates.get_vector(obj_coord_name)

        temp_coords = []

        for i in range(self.objects.item_count):
            coord_temp = cyclicsh(obj_coord_hd, i, inverse=True)
            temp_coords.append(coord_temp)

        relate_bundle = bundle(temp_coords)
        temp = bind(y_coord_hd, relate_bundle)

        objects_noisy = bind(scene, temp)
        objects_temp = self.objects.search(objects_noisy, distance=False)
        objects_idx = np.where(objects_temp > 0.02)[1]

        result = []
        for i in objects_idx:
            result.append(self.objects.get_name(int(i)))

        if obj in result:
            result.remove(obj)

        return result

    def relate_front(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'

        y_coord_hd = self.coordinates.get_vector('y_coord')
        obj_hd = self.objects.get_vector(obj)
        temp = bind(y_coord_hd, obj_hd)

        obj_coord_noisy = bind(scene, temp)
        obj_coord_id = np.argmin(self.y_coordinates.search(obj_coord_noisy))
        obj_coord_name = self.y_coordinates.get_name(obj_coord_id)
        obj_coord_hd = self.y_coordinates.get_vector(obj_coord_name)

        temp_coords = []

        for i in range(self.objects.item_count):
            coord_temp = cyclicsh(obj_coord_hd, i, inverse=False)
            temp_coords.append(coord_temp)

        relate_bundle = bundle(temp_coords)
        temp = bind(y_coord_hd, relate_bundle)

        objects_noisy = bind(scene, temp)
        objects_temp = self.objects.search(objects_noisy, distance=False)
        objects_idx = np.where(objects_temp > 0.02)[1]

        result = []
        for i in objects_idx:
            result.append(self.objects.get_name(int(i)))

        if obj in result:
            result.remove(obj)

        return result

    def relate_left(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        x_coord_hd = self.coordinates.get_vector('x_coord')
        obj_hd = self.objects.get_vector(obj)
        temp = bind(x_coord_hd, obj_hd)

        obj_coord_noisy = bind(scene, temp)
        obj_coord_id = np.argmin(self.x_coordinates.search(obj_coord_noisy))
        obj_coord_name = self.x_coordinates.get_name(obj_coord_id)
        obj_coord_hd = self.x_coordinates.get_vector(obj_coord_name)

        temp_coords = []

        for i in range(self.objects.item_count):
            coord_temp = cyclicsh(obj_coord_hd, i, inverse=True)
            temp_coords.append(coord_temp)

        relate_bundle = bundle(temp_coords)
        temp = bind(x_coord_hd, relate_bundle)

        objects_noisy = bind(scene, temp)
        objects_temp = self.objects.search(objects_noisy, distance=False)
        objects_idx = np.where(objects_temp > 0.02)[1]

        result = []
        for i in objects_idx:
            result.append(self.objects.get_name(int(i)))

        if obj in result:
            result.remove(obj)

        return result

    def relate_right(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        x_coord_hd = self.coordinates.get_vector('x_coord')
        obj_hd = self.objects.get_vector(obj)
        temp = bind(x_coord_hd, obj_hd)

        obj_coord_noisy = bind(scene, temp)
        obj_coord_id = np.argmin(self.x_coordinates.search(obj_coord_noisy))
        obj_coord_name = self.x_coordinates.get_name(obj_coord_id)
        obj_coord_hd = self.x_coordinates.get_vector(obj_coord_name)

        temp_coords = []

        for i in range(self.objects.item_count):
            coord_temp = cyclicsh(obj_coord_hd, i, inverse=False)
            temp_coords.append(coord_temp)

        relate_bundle = bundle(temp_coords)
        temp = bind(x_coord_hd, relate_bundle)

        objects_noisy = bind(scene, temp)
        objects_temp = self.objects.search(objects_noisy, distance=False)
        objects_idx = np.where(objects_temp > 0.02)[1]

        result = []
        for i in objects_idx:
            result.append(self.objects.get_name(int(i)))

        if obj in result:
            result.remove(obj)

        return result

    def same_color(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        color = self.query_color(scene, obj, '_')
        result = self.get_objects(scene, self.color, color)

        try:
            result.remove(obj)
        except:
            result = result

        if result == None:
            return []

        return result

    def same_material(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        material = self.query_material(scene, obj, '_')
        result = self.get_objects(scene, self.material, material)

        try:
            result.remove(obj)
        except:
            result = result

        if result == None:
            return []

        return result

    def same_shape(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        shape = self.query_shape(scene, obj, '_')
        result = self.get_objects(scene, self.shape, shape)

        try:
            result.remove(obj)
        except:
            result = result

        if result == None:
            return []

        return result

    def same_size(self, obj, scene):
        if type(scene) != np.ndarray or type(obj) != str or not obj.startswith('obj'):
           return 'error'
         
        size = self.query_size(scene, obj, '_')
        result = self.get_objects(scene, self.size, size)

        try:
            result.remove(obj)
        except:
            result = result

        if result == None:
            return []

        return result

    def union(self, scene1, scene2):
        try:
            un = set(scene1).union(set(scene2))
            return list(un)
        except:
            return scene1

    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            ans = scene[0]
            if isinstance(ans, str):
                return ans
            else:
                return 'error'
        else:
           return 'error'

    # def unique(self, scene, _):
    #     if len(scene) == 0:
    #         return 'error'

    #     ans = scene[0]
    #     if isinstance(ans, str):
    #         return ans
    #     else:
    #         return 'object0'

    def get_objects(self, scene, attr, value, thr=0.05):
        attr_hd = self.attrs.get_vector(attr.get_im_name())
        value_hd = attr.get_vector(value)
        obj_n = bind(scene, bind(attr_hd, value_hd))
        _, obj_idx = np.where(self.objects.search(obj_n, distance=False) > thr)

        obj = []
        for i in obj_idx:
            obj.append(self.objects.get_name(i))

        return obj

    def get_attr_value(self, scene, attr, obj):
        obj_hd = self.objects.get_vector(obj)
        attr_hd = self.attrs.get_vector(attr.get_im_name())
        attr_value_n = bind(scene, bind(obj_hd, attr_hd))
        attr_value_idx = np.argmin(attr.search(attr_value_n))
        attr_value = attr.get_name(attr_value_idx)

        return attr_value