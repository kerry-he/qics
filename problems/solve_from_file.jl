import SparseArrays
import MAT

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

T = Float64

function read_problem(file_name)
    # Read data from file
    f = MAT.matopen(file_name)

    # Auxiliary problem information
    description = MAT.read(f, "description")
    offset = MAT.read(f, "offset")
    is_complex = MAT.read(f, "complex")
    print("Now solving: ", description)

    # Objective and constraint matrices
    c = vec(MAT.read(f, "c"))
    b = vec(MAT.read(f, "b"))
    A = collect(MAT.read(f, "A"))

    if ~is_complex
        h = vec(MAT.read(f, "h"))
        G = collect(MAT.read(f, "G"))
        cones_raw = MAT.read(f, "cones")
    else
        h = vec(MAT.read(f, "h_alt"))
        G = collect(MAT.read(f, "G_alt"))
        cones_raw = MAT.read(f, "cones_alt")
    end


    # List of cones
    cones = Cones.Cone{T}[]
    total_dim = 1
    for cone_i in cones_raw
        cone_type = cone_i["type"]

        if cone_type == "qre"
            n       = cone_i["n"]
            dim     = cone_i["dim"]
            push!(cones, Cones.EpiTrRelEntropyTri{Float64}(Int64(dim)))

            # Swap X and Y around as Hypatia uses (t, Y, X) ordering for QRE
            vn = Int(n*(n+1) / 2)
            for k in 1:vn
                Base.swaprows!(G, total_dim + k, total_dim + vn + k)
                h[total_dim + k], h[total_dim + vn + k] = h[total_dim + vn + k], h[total_dim + k]
            end
        elseif cone_type == "nn"
            dim = cone_i["dim"]
            push!(cones, Cones.Nonnegative{T}(Int64(dim)))
        elseif cone_type == "psd"
            dim = cone_i["dim"]
            push!(cones, Cones.PosSemidefTri{T, T}(Int64(dim)))         
        end

        total_dim += dim
    end

    close(f)

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=offset)
end

# Input into model and solve
file_name = "problems/quant_key_rate/qkd_DMCV_10_60_05_35.mat"

model = read_problem(file_name)
solver = Solvers.Solver{T}(verbose = true)
Solvers.load(solver, model)
Solvers.solve(solver)

println("Opt value: ", Solvers.get_primal_obj(solver))
println("Solve time: ", Solvers.get_solve_time(solver))
println("Abs gap: ", (Solvers.get_primal_obj(solver) - Solvers.get_dual_obj(solver)) / Solvers.get_tau(solver))