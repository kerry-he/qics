import scipy as sp
import h5py

from cones import *
from solver import model, solver

def read_problem(file_name):
    # Read data from file
    with h5py.File(file_name, 'r') as f:
        # Auxiliary problem information
        description = f.attrs['description']
        offset = f.attrs['offset']
        print("Now solving: ", description)


        # Objective and constraint matrices
        c = f['/data/c'][:]
        b = f['/data/b'][:]
        h = f['/data/h'][:]

        if f['/data/A'].attrs['sparse']:
            (A_v, A_i, A_j) = f['/data/A'][:]
            A_shape         = f['/data/A'].attrs['shape']
            A               = sp.sparse.coo_array((A_v, (A_i, A_j)), A_shape).todense()
        else:
            A = f['/data/A'][:]

        if f['/data/G'].attrs['sparse']:
            (G_v, G_i, G_j) = f['/data/G'][:]
            G_shape         = f['/data/G'].attrs['shape']
            G               = sp.sparse.coo_array((G_v, (G_i, G_j)), G_shape).todense()
        else:
            G = f['/data/A'][:]


        # List of cones
        cones = []
        for i in range(len(f['/cones'])):
            cone_type = f['/cones/' + str(i)].asstr()[()]

            if cone_type == 'qre':
                n       = f['/cones/' + str(i)].attrs['n']
                complex = f['/cones/' + str(i)].attrs['complex']
                cones.append(quantrelentr.QuantRelEntropy(n))
            elif cone_type == 'nn':
                dim       = f['/cones/' + str(i)].attrs['dim']
                cones.append(nonnegorthant.NonNegOrthant(dim))
            elif cone_type == 'psd':
                n       = f['/cones/' + str(i)].attrs['n']
                complex = f['/cones/' + str(i)].attrs['complex']
                cones.append(possemidefinite.PosSemiDefinite(n))                

    return model.Model(c, A, b, G, h, cones=cones, offset=offset)


if __name__ == "__main__":
    # Input into model and solve
    file_name = "problems/rel_entr_entanglement/ree_8_8_rand.hdf5"

    model = read_problem(file_name)
    solver = solver.Solver(model)
    solver.solve()