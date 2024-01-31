import SparseArrays
import HDF5

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

function read_problem(file_name)
    # Read data from file
    f = HDF5.h5open(file_name, "r")

    # Auxiliary problem information
    description = HDF5.attributes(f)["description"][]
    offset = HDF5.attributes(f)["offset"][]
    print("Now solving: ", description)

    # Objective and constraint matrices
    c = vec(f["/data/c"][])
    b = vec(f["/data/b"][])
    h = vec(f["/data/h"][])

    if Bool(HDF5.attributes(f["/data/A"])["sparse"][])
        A_v     = f["/data/A"][][:, 1]
        A_i     = f["/data/A"][][:, 2] .+ 1
        A_j     = f["/data/A"][][:, 3] .+ 1
        A_shape = HDF5.attributes(f["/data/A"])["shape"][]
        A       = SparseArrays.sparse(A_i, A_j, A_v, A_shape[1], A_shape[2])
    else
        A = f["/data/A"][]
    end

    if Bool(HDF5.attributes(f["/data/G"])["sparse"][])
        G_v     = f["/data/G"][][:, 1]
        G_i     = f["/data/G"][][:, 2] .+ 1
        G_j     = f["/data/G"][][:, 3] .+ 1
        G_shape = HDF5.attributes(f["/data/G"])["shape"][]
        G       = SparseArrays.sparse(G_i, G_j, G_v, G_shape[1], G_shape[2])
    else
        G = f["/data/G"][]
    end

    # List of cones
    cones = Cones.Cone{T}[]
    total_dim = 1
    for i in 0:(length(HDF5.read(f["/cones"])) - 1)
        cone_type = HDF5.read(f["/cones/" * string(i)])

        if cone_type == "qre"
            n       = HDF5.attributes(f["/cones/" * string(i)])["n"][]
            dim     = HDF5.attributes(f["/cones/" * string(i)])["dim"][]
            complex = HDF5.attributes(f["/cones/" * string(i)])["complex"][]
            push!(cones, Cones.EpiTrRelEntropyTri{Float64}(Int64(dim)))

            # Swap X and Y around as Hypatia uses (t, Y, X) ordering for QRE
            vn = Int(n*(n+1) / 2)
            for k in 1:vn
                Base.swaprows!(G, total_dim + k, total_dim + vn + k)
                h[total_dim + k], h[total_dim + vn + k] = h[total_dim + vn + k], h[total_dim + k]
            end
        elseif cone_type == "nn"
            dim     = HDF5.attributes(f["/cones/" * string(i)])["dim"][]
            push!(cones, Cones.Nonnegative{T}(Int64(dim)))        
        end

        total_dim += dim
    end

    close(f)

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=offset)
end

# Input into model and solve
file_name = "problems/ea_rate_distortion/ea-rd_2_ef.hdf5"
T = Float64

model = read_problem(file_name)
solver = Solvers.Solver{T}(verbose = true)
Solvers.load(solver, model)
Solvers.solve(solver)