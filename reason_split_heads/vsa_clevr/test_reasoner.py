import pickle
import json
from vsa_reasoner import VSAReasoner

with open('parsed_scene_0.pickle', 'rb') as f:
    data = pickle.load(f)

scene = data['scene_vec']

reas = VSAReasoner(data)

# Question 1
# Is there a big brown object of the same shape as the green thing?
# Answer: yes
# START
out = reas.filter_green(scene, '_')
out = reas.unique(out, '_')
out = reas.same_shape(out, scene)
out = reas.filter_large(scene, '_', prev=out)
out = reas.filter_brown(scene, '_', prev=out)
out = reas.exist(out, '_')
print('OK')
# END




