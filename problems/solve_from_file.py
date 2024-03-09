import numpy as np
import scipy as sp

from cones import *
from solver import model, solver

def read_problem(file_name):
    # Read data from file
    f = sp.io.loadmat(file_name)

    # Auxiliary problem information
    description = f['description'][0]
    offset = f['offset'][0, 0]
    print("Now solving: ", description)


    # Objective and constraint matrices
    c = np.array(f['c'])
    b = np.array(f['b'])
    h = np.array(f['h'])

    A = np.array(f['A'].todense())
    G = np.array(f['G'].todense())


    # List of cones
    cones = []
    for i in range(len(f['cones'][0])):
        cone_i = f['cones'][0][i][0, 0]
        cone_type = cone_i['type']

        if cone_type == 'qre':
            n         = cone_i['n'][0, 0]
            hermitian = bool(cone_i['hermitian'][0, 0])
            cones.append(quantrelentr.Cone(n, hermitian=hermitian))
        elif cone_type == 'nn':
            dim = cone_i['dim'][0, 0]
            cones.append(nonnegorthant.Cone(dim))
        elif cone_type == 'psd':
            n         = cone_i['n'][0, 0]
            hermitian = bool(cone_i['hermitian'][0, 0])
            cones.append(possemidefinite.Cone(n, hermitian=hermitian))                

    return model.Model(c, A, b, G, h, cones=cones, offset=offset)


if __name__ == "__main__":
    # Input into model and solve
    file_name = "problems/quant_key_rate/DMCV_10_60_05_35.mat"

    model = read_problem(file_name)
    solver = solver.Solver(model)
    solver.solve()