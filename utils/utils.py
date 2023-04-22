import os


def get_dimensions(path):
    dims = []
    dims_size = 100000
    for p in os.listdir(path):
        subpath = os.path.join(path, p)
        aux_dims = [
            int(s.split("_")[0])
            for s in os.listdir(subpath)
            if os.path.isdir(os.path.join(subpath, s))
        ]
        aux_dims.sort()

        if len(aux_dims) < dims_size:
            dims = aux_dims

    return dims
