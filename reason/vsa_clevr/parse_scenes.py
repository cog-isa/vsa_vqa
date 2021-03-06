import argparse
import pickle

from vsa_scene_parser import VSASceneParser

parser = argparse.ArgumentParser()

parser.add_argument("--clevr_scenes", help="path to a scenes description")
parser.add_argument("--scene_idx", help="index of a scene")

HD_DIM = 20000
VSA_TYPE = 'polar'
THR = 6

DESCR = {
    'attribute': ['attribute', 'color', 'size', 'shape', 'material', 'coordinates'],
    'color': ['purple', 'blue', 'brown', 'cyan', 'yellow', 'red', 'gray', 'green'],
    'size': ['small', 'large'],
    'shape': ['sphere', 'cylinder', 'cube'],
    'material': ['metal', 'rubber'],
    'coordinates': ['x_coord', 'y_coord', 'z_coord']
}


def main(args):
    if args.clevr_scenes:
        path_to_clevr_scenes = args.clevr_scenes
    if args.scene_idx:
        scene_idx = int(args.scene_idx)

    vsa_parser = VSASceneParser(path_to_clevr_scenes, dim=HD_DIM, vsa_type=VSA_TYPE, thr=THR, descr=DESCR)

    output = vsa_parser.parse(scene_idx)

    # print(output)
    output_file_name = 'parsed_scenes' + str(scene_idx) + '.pickle'
    with open(output_file_name, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
