import h5py
import numpy as np


def load_from_hdf5(filename):
    """Loads an HDF5 file and returns it as a dictionary with the original structure."""
    def recursively_load_dict(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):  # If item is a group, load it as a dictionary
                result[key] = recursively_load_dict(item)
            else:  # Otherwise, load it as an array
                data = item[()]
                # Decode byte strings if necessary
                if isinstance(data, np.ndarray) and data.dtype.kind == 'S':  # Byte strings in NumPy arrays
                    data = np.array(data, dtype=str)
                elif isinstance(data, bytes):  # Single byte string
                    data = data.decode('utf-8')
                result[key] = data
        
        # Load any lists stored as attributes
        for key, item in group.attrs.items():
            result[key] = list(item) if isinstance(item, np.ndarray) else item
        
        return result
    
    with h5py.File(filename, 'r') as f:
        return recursively_load_dict(f)