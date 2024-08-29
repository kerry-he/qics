import SparseArrays
using LinearAlgebra

import Hypatia
import Hypatia.Cones
import Hypatia.Solvers

import CSV
import DelimitedFiles

T = Float64
R = Complex{T}

function _swap_rows(G, cones)
    total_dim = 1
    for (k, cone) in enumerate(cones)
        if cone isa Cones.EpiTrRelEntropyTri
            for k in 1:cone.vw_dim
                Base.swaprows!(G, total_dim + k, total_dim + cone.vw_dim + k)
            end
        elseif cone isa Cones.EpiRelEntropy
            for k in 1:cone.w_dim
                Base.swaprows!(G, total_dim + k, total_dim + cone.w_dim + k)
            end
        end
        total_dim += Cones.dimension(cone)
    end
    return G
end

function save_and_print_stats(solver, problem, csv_name)
    worst_gap = min(solver.gap / solver.point.tau[], abs(solver.primal_obj_t - solver.dual_obj_t))
    max_tau_obj = max(solver.point.tau[], min(abs(solver.primal_obj_t), abs(solver.dual_obj_t)))
    total_time = Solvers.get_solve_time(solver)
    opt_val = (Solvers.get_primal_obj(solver) + Solvers.get_dual_obj(solver)) / 2

    CSV.write(csv_name, (
        problem = [problem],
        solver  = ["hypatia"],
        status  = [string(solver.status)],
        optval  = [opt_val],
        time    = [total_time],
        iter    = [Solvers.get_num_iters(solver)],
        gap     = [worst_gap / max_tau_obj],
        pfeas   = [solver.x_feas],
        dfeas   = [max(solver.y_feas, solver.z_feas)]
    ), writeheader = false, append = true, sep = ',')
end

function save_and_print_fail(exception, problem, csv_name)
    CSV.write(csv_name, (
        problem = [problem],
        solver  = ["hypatia"],
        status  = [string(exception)]
    ), writeheader = false, append = true, sep = ',')
end

function read_problem(filename)
    # Read data from file
    fd = open(filename, "r")
    cones = Cones.Cone{T}[]
    obj_offset = 0.0
    use_G = false
    totalvars = 0
    totalconstr = 0
    objsense = 1
    A_idxs = 1:0

    function _read_cone(cone, sz)
        if cone == "L+"
            return Cones.Nonnegative{T}(sz)
        elseif cone == "Q"
            return Cones.EpiNormEucl{T}(sz)
        elseif cone == "SVECPSD"
            return Cones.PosSemidefTri{T, T}(sz)
        elseif cone == "HVECPSD"
            return Cones.PosSemidefTri{T, R}(sz)
        elseif cone == "CRE"
            return Cones.EpiRelEntropy{T}(sz)
        elseif cone == "SVECQE"
            n = div(isqrt(1 + 8 * (sz - 2)), 2)
            return Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, T}, T}(
                Cones.NegEntropySSF(),
                n
            )
        elseif cone == "HVECQE"
            n = isqrt(sz - 2)
            return Cones.EpiPerSepSpectral{Cones.MatrixCSqr{T, R}, T}(
                Cones.NegEntropySSF(),
                n
            )
        elseif cone == "SVECQRE"
            return Cones.EpiTrRelEntropyTri{T, T}(sz)
        elseif cone == "HVECQRE"
            return Cones.EpiTrRelEntropyTri{T, R}(sz)
        end
    end

    while !eof(fd)
        line = readline(fd)
        startswith(line,"#") && continue # comments
        length(line) == 0 && continue # blank lines

        # new block

        if startswith(line,  "VER")
            nextline = readline(fd)
            @assert startswith(nextline, "4")
            continue
        end

        if startswith(line, "OBJSENSE")
            nextline = readline(fd)
            if strip(nextline) == "MIN"
                objsense = 1
            else
                objsense = -1
            end
            continue
        end

        if startswith(line, "VAR")
            nextline = readline(fd)
            totalvars, lines = split(nextline)
            totalvars = parse(Int, strip(totalvars))
            lines = parse(Int, strip(lines))

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))
                if cone == "F"
                    @assert lines == 1
                    @assert totalvars == sz
                    use_G = true
                else
                    push!(cones, _read_cone(cone, sz))
                    use_G = false
                end
            end
            
            continue
        end

        if startswith(line, "CON")
            nextline = readline(fd)
            totalconstr, lines = split(nextline)
            totalconstr = parse(Int,strip(totalconstr))
            lines = parse(Int,strip(lines))
            total_cone_dim = 0

            for k in 1:lines
                nextline = readline(fd)
                cone, sz = split(nextline)
                sz = parse(Int,strip(sz))
                if cone == "L="
                    A_idxs = total_cone_dim+1:total_cone_dim+sz
                else
                    push!(cones, _read_cone(cone, sz))
                end
                total_cone_dim += sz
            end

            continue
        end

        if startswith(line, "OBJACOORD")
            c = zeros(T, totalvars, 1)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                v = parse(Float64,strip(v))
                c[i] = v
            end
        end

        if startswith(line, "OBJBCOORD")
            nextline = readline(fd)
            obj_offset = parse(Float64, strip(nextline))
        end

        if startswith(line, "ACOORD")
            A = zeros(T, totalconstr, totalvars)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, j, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                j = parse(Int,strip(j)) + 1
                v = parse(Float64,strip(v))
                A[i, j] = v
            end
        end

        if startswith(line, "BCOORD")
            b = zeros(T, totalconstr, 1)
            nextline = readline(fd)
            lines = parse(Int, strip(nextline))
            for k in 1:lines
                nextline = readline(fd)
                i, v = split(nextline)
                i = parse(Int,strip(i)) + 1
                v = parse(Float64,strip(v))
                b[i] = v
            end
        end
    end

    if use_G
        # Need to split A into [-G; -A] and b into [-h; -b]
        # and swap G, h
        c *= objsense
        G_idxs = filter(x -> x ∉ A_idxs, 1:size(A, 1))
        G = -_swap_rows(A[G_idxs, :], cones)
        h = -_swap_rows(b[G_idxs, :], cones)
        A = A[A_idxs, :]
        b = b[A_idxs, :]
    else
        # No G, just need to swap c and A
        c = _swap_rows(c, cones) * objsense
        A = _swap_rows(A', cones)'
        G = -one(T) * I
        h = zeros(T, totalvars)
    end
    c = c[:, 1]
    b = b[:, 1]
    h = h[:, 1]

    return Hypatia.Models.Model{T}(c, A, b, G, h, cones, obj_offset=obj_offset)

    close(fd)

end

# Input into model and solve
folder = "./qreps/"
fnames = readdir(folder)
pushfirst!(fnames, fnames[1])

csv_name = "data_hypatia.csv"

header = [
    "problem",
    "solver",
    "status",
    "optval",
    "time",
    "iter",
    "gap",
    "pfeas",
    "dfeas"
]
# header = reshape(header, 1, length(header))
# DelimitedFiles.writedlm(csv_name, header, ',')

for fname in fnames
    try
        model = read_problem(folder * fname)
        solver = Solvers.Solver{T}(verbose = true, time_limit = 3600, tol_rel_opt=1e-8, tol_feas=1e-8)
        Solvers.load(solver, model)
        Solvers.solve(solver)
        save_and_print_stats(solver, fname, csv_name)
    catch exception
        save_and_print_fail(exception, fname, csv_name)
    end
end