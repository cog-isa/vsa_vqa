from vsa_clevr.vsa import bind, bundle, cyclicsh, ItemMemory
from vsa_clevr.vsa import generate as hd
from vsa_clevr import vsa
import json


class VSASceneParser:
    def __init__(self, path, dim, vsa_type, thr, descr):
        with open(path) as json_file:
            scenes = json.load(json_file)

        self.scenes = scenes['scenes']
        self.dim = dim
        self.vsa_type = vsa_type
        self.thr = thr
        self.descr = descr

        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        im_dict = {}

        for attr_type in self.descr:
            temp_im = ItemMemory(name=attr_type, d=self.dim)
            temp_im.append_batch(self.descr[attr_type], d=self.dim)
            im_dict[attr_type] = temp_im

        self.im_dict = im_dict

    def parse(self, scene_idx):
        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        current_scene = self.scenes[scene_idx]  # Dict
        image_filename = current_scene['image_filename']

        # Object description
        current_scene_objs = {}
        count_obj = 0
        for obj in current_scene['objects']:
            object_name = 'object' + str(count_obj)
            current_scene_objs[object_name] = {
                'color': obj['color'],
                'size': obj['size'],
                'shape': obj['shape'],
                'material': obj['material'],
                'position': obj['3d_coords']
            }

            count_obj += 1

        scene_im = {}
        # Coordinates
        scene_im['x_coord'], scene_im['y_coord'], scene_im['z_coord'] = self.process_objects_coordinates(current_scene)
        scene_im['object'] = ItemMemory(name='object', d=self.dim)

        scene_vec = self.scene_vector_polar(current_scene_objs, scene_im, current_scene)

        scene_im.update(self.im_dict)

        output_data = {
            'attrs': scene_im,
            'scene_vec': scene_vec,
            'current_scene': current_scene_objs,
            'image_filename': current_scene['image_filename']
        }

        return output_data

    def process_objects_coordinates(self, current_scene):
        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        x_coords = []
        y_coords = []
        z_coords = []

        for obj in current_scene['objects']:
            d3_coords = obj['3d_coords']
            x_coords.append(d3_coords[0])
            y_coords.append(d3_coords[1])
            z_coords.append(d3_coords[2])

        x_coords.sort()
        y_coords.sort()
        z_coords.sort()

        x_0 = hd()
        x_vec = {}
        shift = 0
        for x in x_coords:
            x_vec[str(x)] = cyclicsh(x_0, shift)
            shift += 1

        y_0 = hd()
        y_vec = {}
        shift = 0
        for y in y_coords:
            y_vec[str(y)] = cyclicsh(y_0, shift)
            shift += 1

        z_0 = hd()
        z_vec = {}
        shift = 0
        for z in z_coords:
            z_vec[str(z)] = cyclicsh(z_0, shift)
            shift += 1

        x_coords_item_memory = ItemMemory(name='x_coords', d=self.dim)
        for key, value in x_vec.items():
            x_coords_item_memory.append(key, value)

        y_coords_item_memory = ItemMemory(name='y_coords', d=self.dim)
        for key, value in y_vec.items():
            y_coords_item_memory.append(key, value)

        z_coords_item_memory = ItemMemory(name='z_coords', d=self.dim)
        for key, value in z_vec.items():
            z_coords_item_memory.append(key, value)

        return x_coords_item_memory, y_coords_item_memory, z_coords_item_memory

    def scene_vector_polar(self, current_scene_objs, scene_im, current_scene):
        vsa.set_type(self.vsa_type)
        vsa.set_dimension(self.dim)

        skipped_attributes = ['rotation', 'pixel_coords', 'embedding', 'position']

        scene_vec_raw = []  # Set of objects in the scene and other scene properties

        obj_x_coords_raw = []  # Set of x object coordinates
        obj_y_coords_raw = []  # Set of y object coordinates
        obj_z_coords_raw = []  # Set of z object coordinates

        obj_num = 0
        for obj_name, attrs in current_scene_objs.items():
            obj_vec_raw = []  # Set of object attribute vectors to be bundled

            scene_im['object'].append(obj_name, hd())

            for key, value in attrs.items():
                if key in skipped_attributes:
                    continue
                else:
                    attr_vec = self.im_dict['attribute'].get_vector(key)
                    value_vec = self.im_dict[key].get_vector(value)

                    obj_vec_raw.append(bind(attr_vec, value_vec))  # Replenish the description of an object

            obj_vec = bundle(obj_vec_raw)  # Get one-vector description of an object
            obj_vec = bind(scene_im['object'].get_vector(obj_name), obj_vec)
            scene_vec_raw.append(obj_vec)  # Replenish the description of a scene

            # Add coordinates
            obj_discr = current_scene['objects'][obj_num]

            obj_x_coord = str(obj_discr['3d_coords'][0])
            obj_x_coord_vec = scene_im['x_coord'].get_vector(obj_x_coord)
            # Replenish the set of x object coordinates
            obj_x_coords_raw.append(bind(obj_x_coord_vec, scene_im['object'].get_vector(obj_name)))

            obj_y_coord = str(obj_discr['3d_coords'][1])
            obj_y_coord_vec = scene_im['y_coord'].get_vector(obj_y_coord)
            # Replenish the set of y object coordinates
            obj_y_coords_raw.append(bind(obj_y_coord_vec, scene_im['object'].get_vector(obj_name)))

            obj_z_coord = str(obj_discr['3d_coords'][2])
            obj_z_coord_vec = scene_im['z_coord'].get_vector(obj_z_coord)
            # Replenish the set of z object coordinates
            obj_z_coords_raw.append(bind(obj_z_coord_vec, scene_im['object'].get_vector(obj_name)))

            obj_num += 1
            
        obj_x_coord_bundle = bind(self.im_dict['coordinates'].get_vector('x_coord'), bundle(obj_x_coords_raw, thr=self.thr))
        obj_y_coord_bundle = bind(self.im_dict['coordinates'].get_vector('y_coord'), bundle(obj_y_coords_raw, thr=self.thr))
        obj_z_coord_bundle = bind(self.im_dict['coordinates'].get_vector('z_coord'), bundle(obj_z_coords_raw, thr=self.thr))

        scene_vec_raw.append(obj_x_coord_bundle)
        scene_vec_raw.append(obj_y_coord_bundle)
        scene_vec_raw.append(obj_z_coord_bundle)

        return bundle(scene_vec_raw, thr=self.thr)

    def get_memory(self):
        return self.im_dict

    def get_vsa_settings(self):
        return self.dim, self.vsa_type
