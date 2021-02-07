import numpy as np


def get_items(obj, values):
    """
    Get items according the type of obj to adapt the various input format. 
    If the type of obj is dict, we can merely obtain items from obj; 
    otherwise, items combines obj and value.
    """
    if isinstance(obj, dict):
        return obj.items()
    else:
        return zip(obj, values)

def dict_to(dict, rtype):
    """Convert dict to other format including list, array, dict, etc."""
    if rtype == 'list':
        return list(dict.values())
    elif rtype == 'array':
        return np.array(list(dict.values()))
    elif rtype == 'dict':
        return dict

def to_batch(networks, batch_size):
    pass
