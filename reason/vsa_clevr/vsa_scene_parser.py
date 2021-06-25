from vsa_clevr.vsa import bind, bundle, cyclicsh, ItemMemory
from vsa_clevr.vsa import generate as hd
from vsa_clevr import vsa
import json
from collections import defaultdict

class VSASceneParser:
    def __init__(self, path, dim, vsa_type, thr, objects):
        with open(path) as json_file:
            scenes_dict = json.load(json_file)['scenes']
        
        scenes = defaultdict(dict)

        for s in scenes_dict:
            scenes[s['image_index']] = s

        self.scenes = scenes

        self.dim = dim
        self.vsa_type = vsa_type
        self.thr = thr
        self.objects = objects

        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        im_dict = {}

        # Create Item Memory which contains HD vector for every possible object
        category_im = ItemMemory(name='category', d=self.dim)
        category_im.append_batch(self.objects, d=self.dim)

        attr_im = ItemMemory(name='attribute', d=self.dim)
        attr_im.append_batch(['category'], d=self.dim)

        im_dict['category'] = category_im
        im_dict['attribute'] = attr_im

        self.im_dict = im_dict


    def parse(self, scene_idx):
        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        current_scene = self.scenes[scene_idx]  # Dict
        image_index = scene_idx
        
        if current_scene == {}:
            scene_im = {}

            scene_im['object'] = ItemMemory(name='object', d=self.dim)

            scene_vec = hd()

            scene_im.update(self.im_dict)

            output_data = {
                'attrs': scene_im,
                'scene_vec': scene_vec,
                'current_scene': {},
                'image_index': image_index
            }

            return output_data

        

        # Object description
        current_scene_objs = {}
        count_obj = 0
        for obj in current_scene['objects']:
            object_name = 'object' + str(count_obj)
            current_scene_objs[object_name] = {
                'category': obj['category']
            }

            count_obj += 1

        scene_im = {}

        scene_im['object'] = ItemMemory(name='object', d=self.dim)

        scene_vec = self.scene_vector_polar(current_scene_objs, scene_im)

        scene_im.update(self.im_dict)

        output_data = {
            'attrs': scene_im,
            'scene_vec': scene_vec,
            'current_scene': current_scene_objs,
            'image_index': image_index
        }

        return output_data


    def scene_vector_polar(self, current_scene_objs, scene_im):
        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)


        scene_vec_raw = []  # Set of objects in the scene and other scene properties

        obj_num = 0
        for obj_name, attrs in current_scene_objs.items():
            obj_vec_raw = []  # Set of object attribute vectors to be bundled

            scene_im['object'].append(obj_name, hd())

            for key, value in attrs.items():
                attr_vec = self.im_dict['attribute'].get_vector(key)
                value_vec = self.im_dict[key].get_vector(value)

                obj_vec_raw.append(bind(attr_vec, value_vec))  # Replenish the description of an object

            obj_vec = bundle(obj_vec_raw)  # Get one-vector description of an object
            obj_vec = bind(scene_im['object'].get_vector(obj_name), obj_vec)
            scene_vec_raw.append(obj_vec)  # Replenish the description of a scene

            obj_num += 1
            
        return bundle(scene_vec_raw, thr=self.thr)

    def get_memory(self):
        return self.im_dict

    def get_vsa_settings(self):
        return self.dim, self.vsa_type
