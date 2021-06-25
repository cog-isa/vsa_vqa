import numpy as np
import scipy.spatial as spat

global VECTOR_SYMBOLIC_ARCHITECTURE_TYPE
global VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION

VECTOR_SYMBOLIC_ARCHITECTURE_TYPE = 'polar'
# VECTOR_SYMBOLIC_ARCHITECTURE_TYPE = 'binary'
VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION = 1000


def set_type(vsa_type='binary'):
    """Set VSA type."""
    
    implemented_types = ['binary', 'polar']
    
    if vsa_type in implemented_types:
        global VECTOR_SYMBOLIC_ARCHITECTURE_TYPE
        VECTOR_SYMBOLIC_ARCHITECTURE_TYPE = vsa_type
    else:
        raise ValueError('VSA type is unknown. Choose "binary" or "polar".')
        

def get_type():
    """Print VSA type."""
    
    return globals()['VECTOR_SYMBOLIC_ARCHITECTURE_TYPE']
        

def set_dimension(vsa_dimension=1000):
    """Set VSA dimension."""
    
    global VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION
    VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION = vsa_dimension
    

def get_dimension():
    """Print VSA dim."""
    
    return globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']


def generate():
    """Generate a random hypervector of dimension dim.
    
    Default vsa type -- 'binary'
    Default dimension dim -- 1000
    """
    
    vsa_type = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_TYPE']
    dim = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']
    
    if vsa_type == 'binary':
        vector = np.random.choice([0.0, 1.0], size=dim).reshape(1, -1)
    elif vsa_type == 'polar':
        vector = np.random.choice([-1.0, 1.0], size=dim).reshape(1, -1)
        
    return vector


def polarized(hd_vec):
    """Convert a given vector to a vector with values -1.0 and 1.0."""
    
    hd_vec[hd_vec > 0] = 1.0
    hd_vec[hd_vec < 0] = -1.0
    hd_vec[hd_vec == 0] = np.random.choice([-1.0, 1.0])
    return hd_vec


def binarized(hd_vec):
    """Convert a given vector to a vector with values 0.0 and 1.0."""
    
    hd_vec[hd_vec > 0] = 1.0
    hd_vec[hd_vec < 0] = 0.0
    return hd_vec


def cyclicsh(hd_vec, shift=1, inverse=False):
    """Shift the hypervector by the shift value."""
    
    if inverse:
        shift = -1 * shift
    return np.roll(hd_vec, shift=shift, axis=1)


def bind(hd_vec_1, hd_vec_2):
    """Bind two hypervectors.
    
    Default vsa type -- 'binary'
    """
    
    vsa_type = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_TYPE']
    
    if vsa_type == 'binary':
        vector = np.logical_xor(hd_vec_1, hd_vec_2).astype(float)
    elif vsa_type == 'polar':
        try:
            vector = hd_vec_1 * hd_vec_2
        except:
            print(hd_vec_1)
    else:
        print(vsa_type)
        
    return vector


def bundle(*hd_vectors, thr=1):
    """Bundling - threshold sum.

    Input - list of vectors to be bundle
    Default vsa type -- 'binary'
    'thr' is used only when a vsa type is 'polar'
    """

    vsa_type = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_TYPE']

    hd_vectors = np.array(hd_vectors).squeeze(0)  # Get rid of a dimension introduced by a star operator
    result = np.array(hd_vectors).sum(axis=0)

    if vsa_type == 'binary':
        thr = hd_vectors.shape[0] // 2  # To perform majority sum

        result[result >= thr] = 1
        result[result < thr] = 0

    elif vsa_type == 'polar':

        result[result > thr] = thr
        result[result < -thr] = -thr

    return result


class ItemMemory:
    """Store noiseless hypervectors.
    
    Default vsa type -- 'binary'
    Default dimension dim -- 1000
    """
    
    def __init__(self, name, d=1000):
        # self.dim = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_DIMENSION']
        self.item_memory_name = name
        self.dim = d
        self.item_count = 0
        self.names = []
        self.memory = np.zeros((1, self.dim))

    def append(self, name, hd_vec):
        """Add hypervector to a memory."""

        if not self.item_count:
            self.memory[self.item_count, :] = hd_vec
        else:
            self.memory = np.append(self.memory, hd_vec, axis=0)

        self.names.append(name)

        self.item_count += 1

    def append_batch(self, names, d=1000):
        """Add batch of hypervectors."""

        set_dimension(vsa_dimension=d)

        for name in names:
            self.append(name, hd_vec=generate())

    def search(self, hd_vec_query, distance=True):
        """Return distances from query hypervector to every hypervector in the item memory."""
        
        result = np.zeros((1, self.item_count))

        vsa_type = globals()['VECTOR_SYMBOLIC_ARCHITECTURE_TYPE']
    
        if vsa_type == 'binary':
            for i in range(self.item_count):
                if distance:
                    result[0, i] = spat.distance.hamming(self.memory[i, :], hd_vec_query)
                else:
                    result[0, i] = 1 - spat.distance.hamming(self.memory[i, :], hd_vec_query)

        elif vsa_type == 'polar':
            for i in range(self.item_count):
                if distance:
                    result[0, i] = spat.distance.cosine(self.memory[i, :], hd_vec_query)
                else:
                    result[0, i] = 1 - spat.distance.cosine(self.memory[i, :], hd_vec_query)
        
        return result

    def get_names(self):
        """Get names of stored entities."""
        
        return self.names

    def get_im_name(self):
        """Get name of ItemMemory."""
        return self.item_memory_name

    def get_name(self, index):
        """Get name of entity by its index."""
        
        return self.names[index]

    def get_vector(self, name):
        """Get stored hypervector by name of entity it represents."""
        
        idx = self.names.index(name)
        return self.memory[idx, :].reshape(1, -1)
    
    def get_dim(self):
        """Get dim."""
        
        return self.dim
